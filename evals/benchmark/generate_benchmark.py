"""Benchmark dataset generator driver (runs OUTSIDE Blender).

For every registered case this writes a spec JSON, launches Blender in
background mode with blender_build_case.py, then post-processes the renders
into the final ground-truth layout:

    evals/benchmark/dataset/<case_id>/
        input.png        rendered reference image
        mask.png         ground-truth binary mask (0/255)
        bbox.json        tight foreground bbox (pixels)
        camera.json      exact intrinsics + extrinsics + GT encodings
        dimensions.json  known physical object dimensions
        parts.json       part hierarchy + semantic labels + construction family
        depth.png        16-bit normalised depth (see camera.json encoding)
        normals.png      camera-space normals (see camera.json encoding)
        meta.json        difficulty, tags, known failure flags
        reference.glb    ground-truth 3D model
        build_spec.json  spec the case was built from (reproducibility)
        blender_log.txt  captured Blender stdout/stderr

CLI:
    python -m evals.benchmark.generate_benchmark \
        --blender /Applications/Blender.app/Contents/MacOS/Blender \
        --out evals/benchmark/dataset --cases all [--workers 1] [--resolution 640]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

BUILD_SCRIPT = str(Path(__file__).resolve().parent / "blender_build_case.py")
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

# construction family tags come from GOAL.md's test groups
FAMILY_TAGS = ("revolution", "extruded", "primitive_assembly",
               "radial", "mirrored", "freeform", "sweep")

# ---------------------------------------------------------------------------
# case registry
# ---------------------------------------------------------------------------
# parts: (id, label, parent, construction, major) — the reference part
# hierarchy used to score semantic decomposition and part recall.

CASES: List[Dict[str, Any]] = [
    dict(id="wheel_01", builder="wheel", difficulty="easy",
         tags=["revolution", "radial"], failure_flags=[],
         dims={"diameter": 0.664, "width": 0.184, "unit": "m",
               "note": "17-inch class wheel + tyre"},
         parts=[("tyre", "tyre", None, "revolve", True),
                ("rim", "rim", None, "revolve", True),
                ("spokes", "spoke", "rim", "radial_array", True),
                ("hub", "hub", "rim", "revolve", True)]),
    dict(id="wheel_02", builder="wheel", difficulty="medium",
         tags=["revolution", "radial", "mirrored"], failure_flags=["soft_shadow"],
         dims={"diameter": 0.664, "width": 0.184, "unit": "m"},
         parts=[("tyre", "tyre", None, "revolve", True),
                ("rim", "rim", None, "revolve", True),
                ("spokes", "spoke", "rim", "radial_array", True),
                ("hub", "hub", "rim", "revolve", True)]),
    dict(id="bottle_01", builder="bottle", difficulty="easy",
         tags=["revolution"], failure_flags=[],
         dims={"height": 0.316, "diameter": 0.120, "unit": "m"},
         parts=[("body", "bottle_body", None, "revolve", True),
                ("cap", "cap", "body", "primitive", True)]),
    dict(id="bottle_02", builder="bottle", difficulty="hard",
         tags=["revolution"], failure_flags=["reflective_surface", "clutter",
                                             "severe_perspective"],
         dims={"height": 0.316, "diameter": 0.120, "unit": "m"},
         parts=[("body", "bottle_body", None, "revolve", True),
                ("cap", "cap", "body", "primitive", True)]),
    dict(id="vase_01", builder="vase", difficulty="medium",
         tags=["revolution", "freeform"], failure_flags=["curved_profile"],
         dims={"height": 0.30, "diameter": 0.18, "unit": "m"},
         parts=[("body", "vase_body", None, "revolve", True)]),
    dict(id="mug_01", builder="mug", difficulty="easy",
         tags=["revolution", "sweep"], failure_flags=["meaningful_hole"],
         dims={"height": 0.095, "diameter": 0.086, "unit": "m"},
         parts=[("body", "mug_body", None, "revolve", True),
                ("handle", "handle", "body", "sweep", True)]),
    dict(id="mug_02", builder="mug", difficulty="medium",
         tags=["revolution", "sweep"], failure_flags=["meaningful_hole",
                                                      "soft_shadow"],
         dims={"height": 0.095, "diameter": 0.086, "unit": "m"},
         parts=[("body", "mug_body", None, "revolve", True),
                ("handle", "handle", "body", "sweep", True)]),
    dict(id="gear_01", builder="gear", difficulty="easy",
         tags=["radial", "extruded"], failure_flags=["center_hole"],
         dims={"outer_diameter": 0.130, "thickness": 0.016, "unit": "m"},
         parts=[("gear_body", "gear_body", None, "extrude", True),
                ("teeth", "tooth", "gear_body", "radial_array", True),
                ("center_bore", "center_bore", "gear_body", "boolean", False)]),
    dict(id="gear_02", builder="gear", difficulty="hard",
         tags=["radial", "extruded"],
         failure_flags=["reflective_surface", "severe_perspective", "clutter"],
         dims={"outer_diameter": 0.130, "thickness": 0.016, "unit": "m"},
         parts=[("gear_body", "gear_body", None, "extrude", True),
                ("teeth", "tooth", "gear_body", "radial_array", True),
                ("center_bore", "center_bore", "gear_body", "boolean", False)]),
    dict(id="bracket_01", builder="bracket", difficulty="medium",
         tags=["extruded", "primitive_assembly"], failure_flags=["mounting_holes"],
         dims={"leg_length": 0.160, "thickness": 0.048, "unit": "m"},
         parts=[("bracket", "bracket_body", None, "extrude", True),
                ("hole_a", "mounting_hole", "bracket", "boolean", True),
                ("hole_b", "mounting_hole", "bracket", "boolean", True)]),
    dict(id="sign_01", builder="sign", difficulty="easy",
         tags=["extruded"], failure_flags=["relief_detail"],
         dims={"width": 0.420, "height": 0.270, "unit": "m"},
         parts=[("plate", "plate", None, "extrude", True),
                ("star_relief", "logo_relief", "plate", "extrude", True),
                ("ring_relief", "border_relief", "plate", "revolve", False),
                ("foot", "stand_foot", None, "primitive", False)]),
    dict(id="box_01", builder="box_enclosure", difficulty="easy",
         tags=["primitive_assembly"], failure_flags=[],
         dims={"width": 0.208, "depth": 0.148, "height": 0.140, "unit": "m"},
         parts=[("body", "enclosure_body", None, "primitive", True),
                ("lid", "lid", "body", "primitive", True),
                ("feet", "rubber_foot", "body", "primitive", False)]),
    dict(id="crate_01", builder="crate", difficulty="medium",
         tags=["primitive_assembly", "mirrored"],
         failure_flags=["thin_slats", "repeated_parts"],
         dims={"width": 0.400, "depth": 0.400, "height": 0.304, "unit": "m"},
         parts=[("bottom", "bottom_panel", None, "primitive", True),
                ("posts", "corner_post", None, "primitive", True),
                ("slats", "side_slat", "posts", "primitive", True)]),
    dict(id="lamp_01", builder="desk_lamp", difficulty="medium",
         tags=["primitive_assembly"], failure_flags=["thin_arms"],
         dims={"height": 0.430, "base_diameter": 0.192, "unit": "m"},
         parts=[("base", "base", None, "revolve", True),
                ("lower_arm", "lower_arm", "base", "primitive", True),
                ("upper_arm", "upper_arm", "lower_arm", "primitive", True),
                ("shade", "shade", "upper_arm", "primitive", True),
                ("bulb", "bulb", "shade", "primitive", False)]),
    dict(id="chair_01", builder="chair", difficulty="medium",
         tags=["mirrored", "primitive_assembly"],
         failure_flags=["thin_legs", "occluded_rear_legs"],
         dims={"height": 0.536, "seat_width": 0.320, "unit": "m"},
         parts=[("seat", "seat", None, "primitive", True),
                ("legs", "leg", "seat", "primitive", True),
                ("backrest", "backrest", "seat", "primitive", True),
                ("back_slats", "back_slat", "backrest", "primitive", False)]),
    dict(id="table_01", builder="table", difficulty="easy",
         tags=["mirrored", "primitive_assembly"], failure_flags=["thin_legs"],
         dims={"width": 0.560, "depth": 0.400, "height": 0.355, "unit": "m"},
         parts=[("top", "tabletop", None, "primitive", True),
                ("legs", "leg", "top", "primitive", True)]),
    dict(id="knob_01", builder="knob", difficulty="easy",
         tags=["revolution"], failure_flags=[],
         dims={"height": 0.045, "diameter": 0.042, "unit": "m"},
         parts=[("body", "knob_body", None, "revolve", True),
                ("base_ring", "base_ring", "body", "revolve", False)]),
    dict(id="pipe_elbow_01", builder="pipe_elbow", difficulty="medium",
         tags=["sweep"], failure_flags=["curved_axis"],
         dims={"centreline_radius": 0.125, "pipe_diameter": 0.045, "unit": "m"},
         parts=[("pipe", "pipe", None, "sweep", True),
                ("flange_a", "flange", "pipe", "revolve", True),
                ("flange_b", "flange", "pipe", "revolve", True)]),
]

REQUIRED_FILES = ("input.png", "mask.png", "bbox.json", "camera.json",
                  "dimensions.json", "parts.json", "depth.png", "normals.png",
                  "meta.json")


# ---------------------------------------------------------------------------
# per-case build
# ---------------------------------------------------------------------------

def build_case(case: Dict[str, Any], blender: str, out_root: Path,
               resolution: int, timeout: int = 600) -> Dict[str, Any]:
    """Run one case end-to-end; returns a status dict (never raises)."""
    case_id = case["id"]
    case_dir = out_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    # remove stale outputs so a failed build can never pass verification on
    # files from a previous run
    for name in ("input.png", "mask.png", "mask_raw.png", "depth.png",
                 "normals.png", "reference.glb", "camera.json", "bbox.json",
                 "dimensions.json", "parts.json", "meta.json"):
        (case_dir / name).unlink(missing_ok=True)
    seed = abs(hash(case_id)) % (2 ** 31)
    spec = {"case_id": case_id, "builder": case["builder"],
            "difficulty": case["difficulty"], "seed": seed,
            "resolution": resolution}
    spec_path = case_dir / "build_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    log_path = case_dir / "blender_log.txt"
    started = time.time()
    try:
        proc = subprocess.run(
            [blender, "--background", "--factory-startup",
             "--python", BUILD_SCRIPT, "--", str(spec_path), str(case_dir)],
            capture_output=True, text=True, timeout=timeout)
        log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        if proc.returncode != 0 or "BUILD_OK" not in proc.stdout:
            return {"case_id": case_id, "ok": False,
                    "error": "blender exited rc=%d without BUILD_OK; see blender_log.txt"
                             % proc.returncode}
    except subprocess.TimeoutExpired:
        log_path.write_text("TIMEOUT after %ds" % timeout)
        return {"case_id": case_id, "ok": False, "error": "blender timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"case_id": case_id, "ok": False, "error": repr(exc)}

    try:
        _postprocess(case, case_dir)
    except Exception as exc:  # noqa: BLE001
        return {"case_id": case_id, "ok": False,
                "error": "postprocess failed: %r" % (exc,)}
    return {"case_id": case_id, "ok": True,
            "seconds": round(time.time() - started, 1)}


def _postprocess(case: Dict[str, Any], case_dir: Path) -> None:
    """Threshold the mask, compute the bbox, write the remaining GT files."""
    raw = cv2.imread(str(case_dir / "mask_raw.png"), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError("mask_raw.png missing or unreadable")
    mask = (raw > 127).astype(np.uint8) * 255
    fg = int((mask > 0).sum())
    if fg < 100:
        raise RuntimeError("mask nearly empty (%d px)" % fg)
    cv2.imwrite(str(case_dir / "mask.png"), mask)
    (case_dir / "mask_raw.png").unlink(missing_ok=True)

    ys, xs = np.where(mask > 0)
    bbox = {"x0": int(xs.min()), "y0": int(ys.min()),
            "x1": int(xs.max()) + 1, "y1": int(ys.max()) + 1,
            "source": "rendered_mask", "image_size": [int(mask.shape[1]),
                                                      int(mask.shape[0])]}
    (case_dir / "bbox.json").write_text(json.dumps(bbox, indent=2))

    (case_dir / "dimensions.json").write_text(json.dumps(case["dims"], indent=2))

    parts = {
        "object_id": case["id"],
        "construction_family": case["tags"],
        "parts": [
            {"id": pid, "label": label, "parent": parent,
             "construction": construction, "major": major}
            for (pid, label, parent, construction, major) in case["parts"]
        ],
        "hierarchy_edges": [
            {"child": pid, "parent": parent}
            for (pid, _, parent, _, _) in case["parts"] if parent
        ],
        "major_part_ids": [pid for (pid, _, _, _, major) in case["parts"] if major],
    }
    (case_dir / "parts.json").write_text(json.dumps(parts, indent=2))

    meta = {"case_id": case["id"], "difficulty": case["difficulty"],
            "tags": case["tags"], "known_failure_flags": case["failure_flags"],
            "generator": "evals/benchmark/generate_benchmark.py"}
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def verify_case(case_dir: Path) -> List[str]:
    """Return a list of problems with a generated case (empty = good)."""
    problems: List[str] = []
    for name in REQUIRED_FILES:
        if not (case_dir / name).exists():
            problems.append("missing %s" % name)
    if problems:
        return problems
    mask = cv2.imread(str(case_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        problems.append("mask.png unreadable")
    else:
        frac = float((mask > 0).mean())
        if frac < 0.005:
            problems.append("mask nearly empty (%.4f)" % frac)
        if frac > 0.95:
            problems.append("mask fills frame (%.4f)" % frac)
    depth = cv2.imread(str(case_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    if depth is None or int(depth.max()) == 0:
        problems.append("depth.png empty")
    normals = cv2.imread(str(case_dir / "normals.png"), cv2.IMREAD_UNCHANGED)
    if normals is None or int(normals.max()) == 0:
        problems.append("normals.png empty")
    for jf in ("camera.json", "parts.json", "meta.json", "bbox.json",
               "dimensions.json"):
        try:
            json.loads((case_dir / jf).read_text())
        except Exception as exc:  # noqa: BLE001
            problems.append("%s invalid: %r" % (jf, exc))
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="generate_benchmark")
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "dataset"))
    ap.add_argument("--cases", default="all",
                    help="'all' or comma-separated case ids")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--check-only", action="store_true",
                    help="only verify an existing dataset")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.cases.strip().lower() == "all":
        selected = CASES
    else:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        known = {c["id"] for c in CASES}
        unknown = wanted - known
        if unknown:
            print("unknown cases: %s (known: %s)" % (sorted(unknown), sorted(known)))
            return 2
        selected = [c for c in CASES if c["id"] in wanted]

    results: List[Dict[str, Any]] = []
    if not args.check_only:
        if not Path(args.blender).exists():
            print("blender not found at %s" % args.blender)
            return 2
        workers = max(1, args.workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(build_case, case, args.blender, out_root,
                                args.resolution): case for case in selected}
            for fut in concurrent.futures.as_completed(futs):
                res = fut.result()
                results.append(res)
                state = "OK " if res["ok"] else "FAIL"
                print("[%s] %-16s %s" % (state, res["case_id"],
                                         res.get("error", "%ss" % res.get("seconds"))))

    # verification pass
    n_ok = 0
    for case in selected:
        problems = verify_case(out_root / case["id"])
        if problems:
            print("[VERIFY-FAIL] %s: %s" % (case["id"], "; ".join(problems)))
        else:
            n_ok += 1
    print("verified %d/%d cases with complete ground truth" % (n_ok, len(selected)))
    return 0 if n_ok == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
