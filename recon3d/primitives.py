"""Stage 7: geometric primitive fitting.

Replaces raw traced polylines (normalised 0..1, origin top-left) with
recognisable parametric primitives where the fit is reliable.  The original
polyline is always preserved in ``fallback_points``; when no candidate
primitive fits within ``cfg.primitives.max_fit_error_norm`` the path keeps a
``bezier`` (open) / ``closed_region`` (closed) type that simply wraps the
simplified source curve.

All fitting is deterministic (algebraic least squares, no RNG).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .schemas import (
    EvidenceSource,
    GeometricPrimitive,
    PrimitiveType,
    TraceLayer,
    VectorPath,
)

_TWO_PI = 2.0 * math.pi
_CLOSED_FALLBACK = PrimitiveType.CLOSED_REGION
_OPEN_FALLBACK = PrimitiveType.BEZIER


# ---------------------------------------------------------------------------
# small geometry helpers
# ---------------------------------------------------------------------------

def _rdp(P: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of an (open) polyline."""
    n = len(P)
    if n <= 2:
        return P.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a, b = P[i0], P[i1]
        ab = b - a
        L = math.hypot(ab[0], ab[1])
        seg = P[i0 + 1:i1]
        if L < 1e-12:
            d = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            # perpendicular distance to line a->b
            d = np.abs(ab[0] * (a[1] - seg[:, 1]) - (a[0] - seg[:, 0]) * ab[1]) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i0 + 1 + k] = True
            stack.append((i0, i0 + 1 + k))
            stack.append((i0 + 1 + k, i1))
    return P[keep]


def _seg_distances(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Min distance from each point in P to any segment A[i]->B[i]."""
    ab = B - A
    denom = np.einsum("ij,ij->i", ab, ab)
    best = np.full(len(P), np.inf)
    for i in range(len(A)):
        if denom[i] < 1e-24:
            d = np.hypot(P[:, 0] - A[i, 0], P[:, 1] - A[i, 1])
        else:
            t = ((P[:, 0] - A[i, 0]) * ab[i, 0] + (P[:, 1] - A[i, 1]) * ab[i, 1]) / denom[i]
            t = np.clip(t, 0.0, 1.0)
            dx = P[:, 0] - (A[i, 0] + t * ab[i, 0])
            dy = P[:, 1] - (A[i, 1] + t * ab[i, 1])
            d = np.hypot(dx, dy)
        best = np.minimum(best, d)
    return best


def _boundary_error(P: np.ndarray, poly: np.ndarray) -> float:
    """Mean distance of P to the closed polygon boundary ``poly``."""
    A = poly
    B = np.roll(poly, -1, axis=0)
    return float(np.mean(_seg_distances(P, A, B)))


def _angular_coverage(P: np.ndarray, center: np.ndarray) -> Tuple[float, float, float]:
    """Fraction of the full turn covered by P around center.

    Returns (coverage, start_angle_deg, sweep_deg).
    """
    ang = np.arctan2(P[:, 1] - center[1], P[:, 0] - center[0])
    ang = np.sort(ang)
    gaps = np.diff(ang)
    wrap = ang[0] + _TWO_PI - ang[-1]
    all_gaps = np.concatenate([gaps, [wrap]])
    k = int(np.argmax(all_gaps))
    max_gap = float(all_gaps[k])
    coverage = 1.0 - max_gap / _TWO_PI
    # arc starts right after the largest gap
    start = float(ang[(k + 1) % len(ang)])
    start_deg = math.degrees(start) % 360.0
    sweep_deg = coverage * 360.0
    return coverage, start_deg, sweep_deg


# ---------------------------------------------------------------------------
# algebraic fits
# ---------------------------------------------------------------------------

def _fit_line_pca(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Total-least-squares line.  Returns (p0, p1, mean_perp_err, angle_deg)."""
    c = P.mean(axis=0)
    U, S, _ = np.linalg.svd(P - c, full_matrices=False)
    d = U[0] * S[0]  # projections along principal axis
    v = np.array([1.0, 0.0])
    # principal axis = right singular vector 1
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    v = Vt[0]
    t = (P - c) @ v
    p0 = c + t.min() * v
    p1 = c + t.max() * v
    perp = np.abs((P[:, 0] - c[0]) * v[1] - (P[:, 1] - c[1]) * v[0])
    angle = math.degrees(math.atan2(v[1], v[0])) % 180.0
    return p0, p1, float(perp.mean()), angle


def _fit_circle_pratt(P: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    """Pratt algebraic circle fit (Chernov). Returns (center, r, mean_err)."""
    x = P[:, 0]
    y = P[:, 1]
    xm, ym = float(x.mean()), float(y.mean())
    u = x - xm
    v = y - ym
    z = u * u + v * v
    Mxx = float(np.mean(u * u))
    Myy = float(np.mean(v * v))
    Mxy = float(np.mean(u * v))
    Mxz = float(np.mean(u * z))
    Myz = float(np.mean(v * z))
    Mzz = float(np.mean(z * z))
    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy * Mxy
    Var_z = Mzz - Mz * Mz
    A2 = 4.0 * Cov_xy - 3.0 * Mz * Mz - Var_z
    A1 = Var_z * Mz + 4.0 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
    A0 = (Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy)
          - Var_z * Cov_xy)
    A22 = A2 + A2
    xn, yn = 0.0, A0
    for _ in range(50):
        Dy = A1 + xn * (A22 + 16.0 * xn * xn)
        if abs(Dy) < 1e-300:
            break
        xo = xn
        xn = xn - yn / Dy
        if xn == xo:
            break
        yo = yn
        yn = A0 + xn * (A1 + xn * (A2 + 4.0 * xn * xn))
        if abs(yn) >= abs(yo):
            break
    DET = xn * xn - xn * Mz + Cov_xy
    if abs(DET) < 1e-300:
        return None
    Xc = (Mxz * (Myy - xn) - Myz * Mxy) / DET / 2.0
    Yc = (Myz * (Mxx - xn) - Mxz * Mxy) / DET / 2.0
    r2 = Xc * Xc + Yc * Yc + Mz
    if r2 <= 0 or not np.isfinite(r2):
        return None
    r = math.sqrt(r2)
    center = np.array([Xc + xm, Yc + ym])
    err = float(np.mean(np.abs(np.hypot(P[:, 0] - center[0], P[:, 1] - center[1]) - r)))
    return center, r, err


def _fit_ellipse_fitzgibbon(P: np.ndarray) -> Optional[Dict[str, Any]]:
    """Fitzgibbon direct least-squares ellipse fit, implemented in numpy.

    Returns dict(center, radii, rotation_degrees) or None when the conic is
    not an ellipse / is degenerate.
    """
    if len(P) < 5:
        return None
    x = P[:, 0]
    y = P[:, 1]
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ T
    C = np.array([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.solve(C, M))
    except np.linalg.LinAlgError:
        return None
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    # The ellipse eigenpair is identified by the conic discriminant
    # 4ac - b^2 > 0. The eigenvalue itself can be a tiny negative number for
    # exact (zero-noise) ellipse data, so do not require strict positivity.
    a1 = None
    fallback = None
    for i in range(3):
        v = eigvecs[:, i]
        cond = 4.0 * v[0] * v[2] - v[1] * v[1]
        if cond > 0:
            if eigvals[i] > 0:
                a1 = v
                break
            if fallback is None or eigvals[i] > fallback[0]:
                fallback = (eigvals[i], v)
    if a1 is None:
        if fallback is None:
            return None
        a1 = fallback[1]
    a2 = T @ a1
    a, b, c = float(a1[0]), float(a1[1]), float(a1[2])
    d, e, f = float(a2[0]), float(a2[1]), float(a2[2])
    Amat = np.array([[a, b / 2.0], [b / 2.0, c]])
    try:
        center = np.linalg.solve(Amat, -np.array([d, e]) / 2.0)
    except np.linalg.LinAlgError:
        return None
    Fc = f + 0.5 * (d * center[0] + e * center[1])
    w, V = np.linalg.eigh(Amat)
    # ellipse iff A is definite and Fc has the opposite sign of its eigenvalues
    if not (np.all(w > 0) or np.all(w < 0)):
        return None
    q = -Fc / w
    if np.any(q <= 0):
        return None
    radii = np.sqrt(q)
    if not np.all(np.isfinite(radii)) or radii.min() < 1e-7:
        return None
    if radii.max() / radii.min() > 100.0:
        return None
    # rotation of the major axis
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    V = V[:, order]
    rot = math.degrees(math.atan2(V[1, 0], V[0, 0])) % 180.0
    return {
        "center": center,
        "radii": radii,
        "rotation_degrees": rot,
    }


def _cross_check_ellipse(P: np.ndarray, fit: Dict[str, Any]) -> Dict[str, Any]:
    """Cross-check Fitzgibbon result with cv2.fitEllipse; prefer cv2 when the
    two disagree strongly (Fitzgibbon can drift on sparse/noisy data)."""
    if len(P) < 5:
        return fit
    try:
        (cx, cy), (w, h), ang = cv2.fitEllipse(P.astype(np.float32).reshape(-1, 1, 2))
    except cv2.error:
        return fit
    c_cv = np.array([cx, cy])
    r_cv = np.array([max(w, h) / 2.0, min(w, h) / 2.0])
    if not np.all(np.isfinite(r_cv)) or r_cv.min() < 1e-7:
        return fit
    c_f = fit["center"]
    r_f = fit["radii"]
    scale = max(float(r_f.max()), 1e-9)
    center_rel = float(np.hypot(*(c_cv - c_f))) / scale
    radii_rel = float(np.abs(r_cv - r_f).max()) / scale
    if center_rel > 0.2 or radii_rel > 0.2:
        rot_cv = (90.0 + ang) % 180.0 if w >= h else ang % 180.0
        return {"center": c_cv, "radii": r_cv, "rotation_degrees": rot_cv}
    return fit


def _ellipse_error(P: np.ndarray, center: np.ndarray, radii: np.ndarray,
                   rot_deg: float) -> float:
    """Approximate mean point-to-ellipse distance (radial approximation)."""
    th = math.radians(rot_deg)
    ct, st = math.cos(th), math.sin(th)
    Q = P - center
    qx = Q[:, 0] * ct + Q[:, 1] * st
    qy = -Q[:, 0] * st + Q[:, 1] * ct
    sx = qx / radii[0]
    sy = qy / radii[1]
    m = np.hypot(sx, sy)
    m = np.maximum(m, 1e-12)
    ex = radii[0] * sx / m
    ey = radii[1] * sy / m
    return float(np.mean(np.hypot(qx - ex, qy - ey)))


# ---------------------------------------------------------------------------
# polygon / rectangle analysis
# ---------------------------------------------------------------------------

def _approx_corners(P: np.ndarray) -> np.ndarray:
    peri = float(np.sum(np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T)))
    if peri < 1e-12:
        return P[:0]
    contour = (P * 4096.0).astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, 0.02 * peri * 4096.0, True)
    return approx.reshape(-1, 2) / 4096.0


def _edge_info(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    angs = np.degrees(np.arctan2(edges[:, 1], edges[:, 0]))
    return lengths, angs


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    lu, lv = np.hypot(*u), np.hypot(*v)
    if lu < 1e-12 or lv < 1e-12:
        return 180.0
    c = float(np.dot(u, v) / (lu / 1.0) / lv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _try_rectangle(P: np.ndarray, corners: np.ndarray, max_err: float
                   ) -> Optional[Tuple[Dict[str, Any], float]]:
    if len(corners) != 4:
        return None
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    if lengths.min() < 1e-6:
        return None
    # opposite edges parallel, adjacent perpendicular
    for i in range(4):
        a = _angle_between(edges[i], edges[(i + 1) % 4])
        if abs(a - 90.0) > 3.0:
            return None
    center = corners.mean(axis=0)
    rot = math.degrees(math.atan2(edges[0][1], edges[0][0])) % 180.0
    err = _boundary_error(P, corners)
    if err > max_err:
        return None
    params = {
        "center": [float(center[0]), float(center[1])],
        "size": [float(lengths[0]), float(lengths[1])],
        "rotation_degrees": float(rot),
        "corners": [[float(x), float(y)] for x, y in corners],
    }
    return params, err


def _try_rounded_rectangle(P: np.ndarray, corners: np.ndarray, max_err: float
                           ) -> Optional[Tuple[Dict[str, Any], float]]:
    if len(corners) != 8:
        return None
    lengths, _ = _edge_info(corners)
    longs = lengths[0::2]
    shorts = lengths[1::2]
    if longs.min() < 1.5 * max(shorts.max(), 1e-12):
        return None
    if longs.std() > 0.2 * longs.mean() or shorts.std() > 0.5 * shorts.mean():
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(
        (P * 4096.0).astype(np.float32).reshape(-1, 1, 2))
    center = np.array([cx / 4096.0, cy / 4096.0])
    size = [float(w / 4096.0), float(h / 4096.0)]
    hw, hh = size[0] / 2.0, size[1] / 2.0
    th = math.radians(ang)
    ct, st = math.cos(th), math.sin(th)
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    rect = np.column_stack([local[:, 0] * ct - local[:, 1] * st,
                            local[:, 0] * st + local[:, 1] * ct]) + center
    err = _boundary_error(P, rect)
    if err > max_err:
        return None
    params = {
        "center": [float(center[0]), float(center[1])],
        "size": size,
        "rotation_degrees": float(ang % 180.0),
        "corner_radius": float(shorts.mean() / 2.0),
    }
    return params, err


def _try_regular_polygon(P: np.ndarray, corners: np.ndarray, max_err: float
                         ) -> Optional[Tuple[Dict[str, Any], float]]:
    n = len(corners)
    if n < 3 or n > 12:
        return None
    lengths, _ = _edge_info(corners)
    if lengths.min() < 1e-6:
        return None
    if lengths.std() > 0.05 * lengths.mean():
        return None
    center = corners.mean(axis=0)
    radii = np.hypot(corners[:, 0] - center[0], corners[:, 1] - center[1])
    if radii.std() > 0.05 * radii.mean():
        return None
    # interior angles must match a regular n-gon
    expected = (n - 2) * 180.0 / n
    for i in range(n):
        u = corners[(i - 1) % n] - corners[i]
        v = corners[(i + 1) % n] - corners[i]
        if abs(_angle_between(u, v) - expected) > 4.0:
            return None
    err = _boundary_error(P, corners)
    if err > max_err:
        return None
    rot = math.degrees(math.atan2(corners[0][1] - center[1],
                                  corners[0][0] - center[0]))
    params = {
        "center": [float(center[0]), float(center[1])],
        "radius": float(radii.mean()),
        "n": int(n),
        "rotation_degrees": float(rot % 360.0),
    }
    return params, err


# ---------------------------------------------------------------------------
# symmetry test
# ---------------------------------------------------------------------------

def _mirror_residual(P: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """Reflect P about its principal axis; mean distance of reflected points
    back onto the source polyline.  Returns (residual, axis_point, axis_deg)."""
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    v = Vt[0]
    n = np.array([-v[1], v[0]])
    d = (P - c) @ n
    R = P - 2.0 * np.outer(d, n)
    A = P[:-1]
    B = P[1:]
    res = float(np.mean(_seg_distances(R, A, B)))
    return res, c, math.degrees(math.atan2(v[1], v[0])) % 180.0


# ---------------------------------------------------------------------------
# per-path fitting
# ---------------------------------------------------------------------------

def _confidence(fit_error: float, max_err: float, n_points: int,
                coverage: float = 1.0, fallback: bool = False) -> float:
    if fallback:
        return 0.3
    base = max(0.0, 1.0 - fit_error / max(max_err, 1e-9))
    cnt = min(1.0, n_points / 30.0)
    cov = 1.0 if coverage >= 0.9 else min(1.0, coverage / 0.35)
    conf = (0.55 + 0.45 * base) * (0.7 + 0.3 * cnt) * (0.75 + 0.25 * cov)
    return round(float(max(0.05, min(0.99, conf))), 4)


def _fit_single_path(path: VectorPath, cfg: PipelineConfig) -> GeometricPrimitive:
    pcfg = cfg.primitives
    max_err = pcfg.max_fit_error_norm
    min_arc = pcfg.min_arc_coverage
    P = np.asarray(path.points, dtype=float).reshape(-1, 2) if path.points else np.zeros((0, 2))
    fallback = [(float(x), float(y)) for x, y in path.points]
    n = len(P)

    ptype = _CLOSED_FALLBACK if path.closed else _OPEN_FALLBACK
    params: Dict[str, Any] = {"points": [[float(x), float(y)] for x, y in P]}
    fit_error = 0.0
    coverage = 1.0
    accepted = False

    simplified = _rdp(P, 0.3 * max_err) if n > 2 else P

    if n == 1:
        ptype = PrimitiveType.POINT
        params = {"point": [float(P[0][0]), float(P[0][1])]}
        accepted = True
    elif n == 2:
        ptype = PrimitiveType.LINE
        params = {"p0": [float(P[0][0]), float(P[0][1])],
                  "p1": [float(P[1][0]), float(P[1][1])]}
        accepted = True
    elif n >= 3:
        # --- 1. line segment (open paths only) ---------------------------
        if not path.closed:
            p0, p1, lerr, _ = _fit_line_pca(P)
            if lerr <= max_err:
                ptype = PrimitiveType.LINE
                params = {"p0": [float(p0[0]), float(p0[1])],
                          "p1": [float(p1[0]), float(p1[1])]}
                fit_error = lerr
                accepted = True

        # --- 2. circle / circular arc ------------------------------------
        ellipse_fit: Optional[Dict[str, Any]] = None
        ellipse_err = math.inf
        if not accepted:
            circ = _fit_circle_pratt(P)
            if circ is not None:
                center, r, cerr = circ
                if cerr <= max_err:
                    ellipse_fit = _fit_ellipse_fitzgibbon(P)
                    if ellipse_fit is not None:
                        ellipse_fit = _cross_check_ellipse(P, ellipse_fit)
                        ellipse_err = _ellipse_error(
                            P, ellipse_fit["center"], ellipse_fit["radii"],
                            ellipse_fit["rotation_degrees"])
                    # parsimony guard: only keep the circle when the ellipse
                    # is not dramatically better
                    if not (ellipse_err < 0.4 * cerr):
                        coverage, start_deg, sweep_deg = _angular_coverage(P, center)
                        if coverage >= 0.9:
                            ptype = PrimitiveType.CIRCLE
                            params = {"center": [float(center[0]), float(center[1])],
                                      "radius": float(r)}
                            fit_error = cerr
                            accepted = True
                        elif coverage >= min_arc:
                            ptype = PrimitiveType.CIRCULAR_ARC
                            params = {"center": [float(center[0]), float(center[1])],
                                      "radius": float(r),
                                      "start_angle_deg": float(start_deg),
                                      "sweep_deg": float(sweep_deg)}
                            fit_error = cerr
                            accepted = True

        # --- 3. rectangle / rounded rectangle / regular polygon ----------
        if not accepted and path.closed:
            corners = _approx_corners(P)
            for detector, t in ((_try_rectangle, PrimitiveType.RECTANGLE),
                                (_try_rounded_rectangle, PrimitiveType.ROUNDED_RECTANGLE),
                                (_try_regular_polygon, PrimitiveType.REGULAR_POLYGON)):
                got = detector(P, corners, max_err)
                if got is not None:
                    params, fit_error = got
                    ptype = t
                    accepted = True
                    break

        # --- 4. ellipse / elliptical arc ---------------------------------
        if not accepted:
            if ellipse_fit is None:
                ellipse_fit = _fit_ellipse_fitzgibbon(P)
                if ellipse_fit is not None:
                    ellipse_fit = _cross_check_ellipse(P, ellipse_fit)
            if ellipse_fit is not None:
                if ellipse_err is math.inf:
                    ellipse_err = _ellipse_error(
                        P, ellipse_fit["center"], ellipse_fit["radii"],
                        ellipse_fit["rotation_degrees"])
                if ellipse_err <= max_err:
                    th = math.radians(ellipse_fit["rotation_degrees"])
                    ct, st = math.cos(th), math.sin(th)
                    Q = P - ellipse_fit["center"]
                    qx = Q[:, 0] * ct + Q[:, 1] * st
                    qy = -Q[:, 0] * st + Q[:, 1] * ct
                    ang_pts = np.degrees(np.arctan2(qy / ellipse_fit["radii"][1],
                                                    qx / ellipse_fit["radii"][0]))
                    ang_sorted = np.sort(np.radians(ang_pts))
                    gaps = np.diff(ang_sorted)
                    wrap = ang_sorted[0] + _TWO_PI - ang_sorted[-1]
                    max_gap = float(max(gaps.max(), wrap))
                    coverage = 1.0 - max_gap / _TWO_PI
                    c = ellipse_fit["center"]
                    base = {"center": [float(c[0]), float(c[1])],
                            "radii": [float(ellipse_fit["radii"][0]),
                                      float(ellipse_fit["radii"][1])],
                            "rotation_degrees": float(ellipse_fit["rotation_degrees"])}
                    if path.closed or coverage >= 0.9:
                        ptype = PrimitiveType.ELLIPSE
                        params = base
                        fit_error = ellipse_err
                        accepted = True
                    elif coverage >= min_arc:
                        ptype = PrimitiveType.ELLIPTICAL_ARC
                        base["start_angle_deg"] = float(
                            math.degrees(ang_sorted[(int(np.argmax(
                                np.concatenate([gaps, [wrap]]))) + 1) % len(ang_sorted)]) % 360.0)
                        base["sweep_deg"] = float(coverage * 360.0)
                        params = base
                        fit_error = ellipse_err
                        accepted = True

        # --- 5. polyline (open, few keypoints) ---------------------------
        if not accepted and not path.closed:
            if 3 <= len(simplified) <= 12:
                A = simplified[:-1]
                B = simplified[1:]
                perr = float(np.mean(_seg_distances(P, A, B)))
                if perr <= max_err:
                    ptype = PrimitiveType.POLYLINE
                    params = {"points": [[float(x), float(y)] for x, y in simplified]}
                    fit_error = perr
                    accepted = True

        # --- 6. symmetric spline -----------------------------------------
        if not accepted:
            res, axis_pt, axis_deg = _mirror_residual(P)
            if res <= 0.5 * max_err:
                ptype = PrimitiveType.SYMMETRIC_SPLINE
                params = {
                    "points": [[float(x), float(y)] for x, y in simplified],
                    "axis": {"point": [float(axis_pt[0]), float(axis_pt[1])],
                             "angle_degrees": float(axis_deg)},
                }
                fit_error = res
                accepted = True

    if not accepted:
        # fallback keeps the raw (lightly simplified) curve
        A = simplified[:-1]
        B = simplified[1:]
        fit_error = float(np.mean(_seg_distances(P, A, B))) if n > 2 else 0.0
        params = {"points": [[float(x), float(y)] for x, y in simplified]}

    return GeometricPrimitive(
        id="",  # filled by caller (needs index)
        type=ptype,
        params=params,
        fit_error=round(float(fit_error), 6),
        source_path=path.path_id,
        source_layer=path.source_layer,
        confidence=_confidence(fit_error, max_err, n, coverage, fallback=not accepted),
        fallback_points=fallback,
    )


def fit_primitives(layers: List[TraceLayer], cfg: PipelineConfig) -> List[GeometricPrimitive]:
    """Fit every traced path with the best parametric primitive."""
    out: List[GeometricPrimitive] = []
    counters: Dict[Tuple[str, str], int] = {}
    for layer in layers:
        for path in layer.paths:
            if cfg.primitives.enabled:
                prim = _fit_single_path(path, cfg)
            else:
                ptype = (_CLOSED_FALLBACK if path.closed else _OPEN_FALLBACK)
                points = [[float(x), float(y)] for x, y in path.points]
                prim = GeometricPrimitive(
                    id="pending", type=ptype, params={"points": points},
                    fit_error=0.0, source_path=path.path_id,
                    source_layer=path.source_layer, confidence=0.3,
                    fallback_points=[tuple(point) for point in points],
                    source=EvidenceSource.FITTED_FROM_OBSERVATION)
            key = (layer.name.value, prim.type.value)
            idx = counters.get(key, 0)
            counters[key] = idx + 1
            prim.id = "%s_%s_%d" % (layer.name.value, prim.type.value, idx)
            out.append(prim)
    return out
