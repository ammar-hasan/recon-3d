"""Stage 1-6 acceptance tests on synthetic images (no network / no rembg).

Set RECON3D_TEST_REMBG=1 to enable the optional rembg backend test (first run
downloads model weights).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from recon3d import (
    crop,
    input_manager,
    preprocess,
    segmentation,
    svg_cleanup,
    vectorize,
)
from recon3d.config import PipelineConfig
from recon3d.input_manager import InputError
from recon3d.schemas import InputSpec
from recon3d.svg_cleanup import _point_to_polyline_dist

# ---------------------------------------------------------------------------
# synthetic image helpers
# ---------------------------------------------------------------------------

W, H = 640, 480


def smooth_noise_bg(rng: np.random.Generator, w: int = W, h: int = H) -> np.ndarray:
    noise = rng.integers(100, 200, size=(h // 8, w // 8, 3), dtype=np.uint8)
    bg = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(bg, (0, 0), 5)


def circle_scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    img = smooth_noise_bg(rng)
    truth = np.zeros((H, W), np.uint8)
    cv2.circle(img, (320, 240), 80, (30, 90, 160), -1)
    cv2.circle(truth, (320, 240), 80, 1, -1)
    return img, truth


def rect_with_hole_scene(seed: int = 1):
    rng = np.random.default_rng(seed)
    img = smooth_noise_bg(rng)
    truth = np.zeros((H, W), np.uint8)
    cv2.rectangle(img, (220, 150), (420, 330), (40, 120, 60), -1)
    cv2.rectangle(truth, (220, 150), (420, 330), 1, -1)
    # meaningful hole showing background through the part
    hole_patch = smooth_noise_bg(np.random.default_rng(99), 60, 60)
    img[210:270, 290:350] = hole_patch
    truth[210:270, 290:350] = 0
    return img, truth


def two_objects_scene(seed: int = 2):
    rng = np.random.default_rng(seed)
    img = smooth_noise_bg(rng)
    t1 = np.zeros((H, W), np.uint8)
    t2 = np.zeros((H, W), np.uint8)
    cv2.circle(img, (180, 240), 70, (30, 90, 160), -1)
    cv2.circle(t1, (180, 240), 70, 1, -1)
    cv2.circle(img, (470, 240), 45, (160, 40, 40), -1)
    cv2.circle(t2, (470, 240), 45, 1, -1)
    return img, t1, t2


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union else 0.0


def make_cfg(tmp_path: Path, **seg_kw) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.crop.canvas_size = 512
    cfg.crop.padding_px = 24
    for k, v in seg_kw.items():
        setattr(cfg.segmentation, k, v)
    return cfg


def run_seg(img: np.ndarray, truth: np.ndarray, tmp_path: Path, cfg: PipelineConfig,
            name: str = "img.png", **spec_kw):
    path = tmp_path / name
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    spec = InputSpec(image_paths=[str(path)], output_dir=str(tmp_path / "proj"), **spec_kw)
    bundle = input_manager.load_input(spec)
    seg_dir = tmp_path / "proj" / "segmentation"
    seg = segmentation.segment(bundle, str(seg_dir), cfg)
    pred = (cv2.imread(str(seg.mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    return seg, pred, bundle


# ---------------------------------------------------------------------------
# Stage 1: input handling
# ---------------------------------------------------------------------------

def test_load_png_jpeg_webp(tmp_path):
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
    files = {}
    Image.fromarray(arr).save(tmp_path / "a.png")
    Image.fromarray(arr).save(tmp_path / "b.jpg", quality=95)
    Image.fromarray(arr).save(tmp_path / "c.webp")
    rgba = np.dstack([arr, np.full((60, 80), 200, np.uint8)])
    Image.fromarray(rgba, "RGBA").save(tmp_path / "d.png")
    Image.fromarray(arr[:, :, 0]).save(tmp_path / "e.png")  # grayscale
    for f in ("a.png", "b.jpg", "c.webp", "d.png", "e.png"):
        spec = InputSpec(image_paths=[str(tmp_path / f)], output_dir=str(tmp_path / "o"))
        bundle = input_manager.load_input(spec)
        assert len(bundle.images) == 1
        assert Path(bundle.images[0].path).name == "original.png"
        assert bundle.images[0].width == 80 and bundle.images[0].height == 60
        assert len(bundle.images[0].sha256) == 64
        files[f] = bundle.images[0]
    assert files["d.png"].channels == 4
    assert files["e.png"].channels == 1


def test_rembg_undersegmentation_rescue_accepts_containing_subject(monkeypatch):
    rgb = np.full((100, 100, 3), 245, np.uint8)
    seed = np.zeros((100, 100), np.uint8)
    seed[45:55, 45:55] = 1
    enclosing = np.zeros_like(seed)
    enclosing[40:60, 32:67] = 1

    monkeypatch.setattr(segmentation, "_grabcut",
                        lambda *args, **kwargs: enclosing.copy())
    monkeypatch.setattr(
        segmentation, "_expand_edge_bounded_enclosure",
        lambda image, mask: (mask, {"accepted": False}),
    )
    recovered, diagnostics = segmentation._rescue_undersegmented_rembg(
        rgb, seed, PipelineConfig())
    assert diagnostics["accepted"] is True
    assert np.array_equal(recovered, enclosing)


def test_rembg_undersegmentation_rescue_rejects_unrelated_component(monkeypatch):
    rgb = np.full((100, 100, 3), 245, np.uint8)
    seed = np.zeros((100, 100), np.uint8)
    seed[10:20, 10:20] = 1
    unrelated = np.zeros_like(seed)
    unrelated[30:80, 30:70] = 1
    monkeypatch.setattr(segmentation, "_grabcut",
                        lambda *args, **kwargs: unrelated.copy())
    recovered, diagnostics = segmentation._rescue_undersegmented_rembg(
        rgb, seed, PipelineConfig())
    assert diagnostics["accepted"] is False
    assert np.array_equal(recovered, seed)


def test_exif_orientation_applied(tmp_path):
    img = Image.new("RGB", (40, 20), (10, 20, 30))
    img.putpixel((0, 0), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW for display
    img.save(tmp_path / "rot.jpg", exif=exif, quality=95, subsampling=0)
    spec = InputSpec(image_paths=[str(tmp_path / "rot.jpg")], output_dir=str(tmp_path / "o"))
    bundle = input_manager.load_input(spec)
    li = bundle.images[0]
    assert li.exif_orientation_applied
    assert (li.width, li.height) == (20, 40)
    out = np.asarray(Image.open(li.path).convert("RGB"))
    # orientation 6 rotates 90 CW: original top-left lands at top-right
    assert out[0, -1, 0] > 150 and out[0, -1, 2] < 100


def test_corrupt_and_unsupported_rejected(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"this is not a png at all")
    spec = InputSpec(image_paths=[str(tmp_path / "bad.png")], output_dir=str(tmp_path / "o"))
    with pytest.raises(InputError, match="corrupt"):
        input_manager.load_input(spec)

    Image.new("RGB", (10, 10)).save(tmp_path / "anim.gif")
    spec = InputSpec(image_paths=[str(tmp_path / "anim.gif")], output_dir=str(tmp_path / "o"))
    with pytest.raises(InputError, match="unsupported"):
        input_manager.load_input(spec)

    with pytest.raises(InputError, match="not found"):
        input_manager.load_input(
            InputSpec(image_paths=[str(tmp_path / "missing.png")],
                      output_dir=str(tmp_path / "o"))
        )


def test_hint_validation(tmp_path):
    img, _ = circle_scene()
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # invalid box -> warning + cleared
    spec = InputSpec(image_paths=[str(p)], output_dir=str(tmp_path / "o1"),
                     box=(2000, 2000, 2100, 2100))
    bundle = input_manager.load_input(spec)
    assert bundle.spec.box is None
    assert any("bounding box" in w for w in bundle.warnings)

    # off-image point -> warning + cleared
    spec = InputSpec(image_paths=[str(p)], output_dir=str(tmp_path / "o2"),
                     point=(5000.0, 10.0))
    bundle = input_manager.load_input(spec)
    assert bundle.spec.point is None

    # empty mask -> warning + cleared
    Image.fromarray(np.zeros((H, W), np.uint8)).save(tmp_path / "empty.png")
    spec = InputSpec(image_paths=[str(p)], output_dir=str(tmp_path / "o3"),
                     mask_path=str(tmp_path / "empty.png"))
    bundle = input_manager.load_input(spec)
    assert bundle.spec.mask_path is None
    assert any("empty" in w for w in bundle.warnings)

    # size-mismatched mask -> clear error
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(tmp_path / "small.png")
    spec = InputSpec(image_paths=[str(p)], output_dir=str(tmp_path / "o4"),
                     mask_path=str(tmp_path / "small.png"))
    with pytest.raises(InputError, match="does not match"):
        input_manager.load_input(spec)

    # calibrated multiview hints must align one-to-one with images
    spec = InputSpec(
        image_paths=[str(p), str(p)], output_dir=str(tmp_path / "o5"),
        view_azimuths_deg=[0.0])
    with pytest.raises(InputError, match="one angle per image"):
        input_manager.load_input(spec)

    spec = spec.model_copy(update={"view_azimuths_deg": [10.0, 55.0]})
    bundle = input_manager.load_input(spec)
    assert bundle.spec.view_azimuths_deg == [10.0, 55.0]


# ---------------------------------------------------------------------------
# Stage 2: segmentation
# ---------------------------------------------------------------------------

def test_segment_circle_classical_iou(tmp_path):
    img, truth = circle_scene()
    cfg = make_cfg(tmp_path, backend="threshold")
    seg, pred, _ = run_seg(img, truth, tmp_path, cfg)
    assert seg.backend == "classical"
    assert iou(pred, truth) >= 0.95
    assert Path(seg.rgba_path).exists() and Path(seg.original_path).exists()
    assert 0.0 < seg.confidence <= 1.0
    x0, y0, x1, y1 = seg.bbox
    assert abs((x1 - x0) - 160) <= 8 and abs((y1 - y0) - 160) <= 8


def test_segment_grabcut_box_guided(tmp_path):
    img, truth = circle_scene()
    cfg = make_cfg(tmp_path, backend="grabcut")
    box = (320 - 95, 240 - 95, 320 + 95, 240 + 95)
    seg, pred, _ = run_seg(img, truth, tmp_path, cfg, box=box)
    assert seg.backend == "grabcut"
    assert iou(pred, truth) >= 0.90


def test_point_selects_target_object(tmp_path):
    img, t1, t2 = two_objects_scene()
    cfg = make_cfg(tmp_path, backend="grabcut")
    seg, pred, _ = run_seg(img, t2, tmp_path, cfg, point=(470.0, 240.0))
    assert iou(pred, t2) >= 0.90
    assert iou(pred, t1) < 0.05  # other object not selected


def test_user_mask_backend(tmp_path):
    img, truth = circle_scene()
    cfg = make_cfg(tmp_path, backend="user_mask")
    mp = tmp_path / "mask.png"
    cv2.imwrite(str(mp), truth * 255)
    seg, pred, _ = run_seg(img, truth, tmp_path, cfg, mask_path=str(mp))
    assert seg.backend == "user_mask"
    assert iou(pred, truth) >= 0.99


def test_meaningful_hole_preserved(tmp_path):
    img, truth = rect_with_hole_scene()
    cfg = make_cfg(tmp_path, backend="threshold")
    seg, pred, _ = run_seg(img, truth, tmp_path, cfg)
    assert iou(pred, truth) >= 0.90
    hole_pred = pred[215:265, 295:345]
    assert hole_pred.mean() <= 0.05  # hole stays open
    assert seg.diagnostics["hole_count"] >= 1


@pytest.mark.skipif(os.environ.get("RECON3D_TEST_REMBG") != "1",
                    reason="rembg downloads model weights; enable explicitly")
def test_rembg_backend_optional(tmp_path):
    img, truth = circle_scene()
    cfg = make_cfg(tmp_path, backend="rembg")
    seg, pred, _ = run_seg(img, truth, tmp_path, cfg)
    assert seg.backend == "rembg"
    assert iou(pred, truth) >= 0.90


# ---------------------------------------------------------------------------
# Stage 3: crop + coordinate integrity
# ---------------------------------------------------------------------------

def test_crop_roundtrip_and_coverage(tmp_path):
    img, truth = rect_with_hole_scene()
    cfg = make_cfg(tmp_path, backend="threshold")
    seg, pred, _ = run_seg(img, truth, tmp_path, cfg)
    meta, crop_rgba, crop_mask = crop.make_crop(seg, str(tmp_path / "proj" / "segmentation"), cfg)

    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [W, H], size=(200, 2))
    errs = []
    for x, y in pts:
        u, v = meta.to_crop(x, y)
        xr, yr = meta.to_original(u, v)
        errs.append(np.hypot(xr - x, yr - y))
    assert max(errs) < 0.1
    assert np.mean(errs) < 0.01

    # aspect preserved: uniform scale
    u0, v0 = meta.to_crop(100, 100)
    u1, v1 = meta.to_crop(110, 100)
    u2, v2 = meta.to_crop(100, 110)
    assert abs(np.hypot(u1 - u0, v1 - v0) - np.hypot(u2 - u2, v2 - v0)) < 1e-9

    # full object inside the crop canvas (no clipping)
    cm = cv2.imread(str(crop_mask), cv2.IMREAD_GRAYSCALE) > 0
    src_area = float((pred > 0).sum())
    scale2 = meta.scale ** 2
    assert abs(cm.sum() / scale2 - src_area) / src_area < 0.02
    assert (tmp_path / "proj" / "segmentation" / "crop_metadata.json").exists()


# ---------------------------------------------------------------------------
# Stage 4: preprocessing
# ---------------------------------------------------------------------------

@pytest.fixture()
def prepped(tmp_path):
    img, truth = rect_with_hole_scene()
    cfg = make_cfg(tmp_path, backend="threshold")
    seg, pred, _ = run_seg(img, truth, tmp_path, cfg)
    seg_dir = tmp_path / "proj" / "segmentation"
    meta, crop_rgba, crop_mask = crop.make_crop(seg, str(seg_dir), cfg)
    layers = preprocess.preprocess(crop_rgba, crop_mask, str(tmp_path / "prep"), cfg)
    return cfg, crop_rgba, crop_mask, layers


def test_preprocess_outputs(prepped, tmp_path):
    cfg, crop_rgba, crop_mask, layers = prepped
    for p in (layers.silhouette_path, layers.color_quantized_path,
              layers.structural_edges_path, layers.details_path,
              layers.lighting_normalized_path):
        assert Path(p).exists(), p

    sil = cv2.imread(layers.silhouette_path, cv2.IMREAD_GRAYSCALE) > 0
    cm = cv2.imread(crop_mask, cv2.IMREAD_GRAYSCALE) > 0
    assert iou(sil.astype(np.uint8), cm.astype(np.uint8)) >= 0.97

    # hole survives into the silhouette
    yy, xx = np.nonzero(cm)
    cy, cx = int(yy.mean()), int(xx.mean())
    hole = sil[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5]
    assert hole.mean() < 0.5  # centre region (hole) is open

    edges = cv2.imread(layers.structural_edges_path, cv2.IMREAD_GRAYSCALE)
    assert edges[cm].max() > 0  # some geometric edges found inside object
    assert edges[~cm].max() == 0  # no edges in background


def test_preprocess_deterministic(prepped, tmp_path):
    cfg, crop_rgba, crop_mask, layers = prepped
    layers2 = preprocess.preprocess(crop_rgba, crop_mask, str(tmp_path / "prep2"), cfg)
    for a, b in ((layers.silhouette_path, layers2.silhouette_path),
                 (layers.color_quantized_path, layers2.color_quantized_path),
                 (layers.structural_edges_path, layers2.structural_edges_path),
                 (layers.details_path, layers2.details_path),
                 (layers.lighting_normalized_path, layers2.lighting_normalized_path)):
        assert Path(a).read_bytes() == Path(b).read_bytes(), a


# ---------------------------------------------------------------------------
# Stage 5 + 6: vectorize + cleanup
# ---------------------------------------------------------------------------

def _run_to_traces(tmp_path: Path, vec_backend: str):
    img, truth = rect_with_hole_scene()
    cfg = make_cfg(tmp_path, backend="threshold")
    cfg.vectorize.backend = vec_backend
    seg, pred, _ = run_seg(img, truth, tmp_path, cfg)
    seg_dir = tmp_path / "proj" / "segmentation"
    meta, crop_rgba, crop_mask = crop.make_crop(seg, str(seg_dir), cfg)
    img_layers = preprocess.preprocess(crop_rgba, crop_mask, str(tmp_path / "prep"), cfg)
    traces_dir = tmp_path / "traces"
    layers = vectorize.vectorize(img_layers, str(traces_dir), cfg)
    return cfg, layers, traces_dir


def test_vectorize_produces_four_layers(tmp_path):
    cfg, layers, traces_dir = _run_to_traces(tmp_path, "contour")
    assert len(layers) == 4
    names = {l.name.value for l in layers}
    assert names == {"silhouette", "color_regions", "structural_edges", "details"}
    sil = next(l for l in layers if l.name.value == "silhouette")
    assert len(sil.paths) >= 2  # outer + hole
    assert sil.image_size == (cfg.crop.canvas_size, cfg.crop.canvas_size)
    for f in ("silhouette.svg", "color_regions.svg",
              "structural_edges.svg", "details.svg"):
        assert (traces_dir / f).exists()
    closed = [p for p in sil.paths if p.closed]
    assert closed and all(len(p.points) >= 3 for p in closed)


def test_vectorize_vtracer_backend(tmp_path):
    vtracer = pytest.importorskip("vtracer")
    cfg, layers, traces_dir = _run_to_traces(tmp_path, "vtracer")
    sil = next(l for l in layers if l.name.value == "silhouette")
    assert sil.stats["backend"] == "vtracer"
    assert len(sil.paths) >= 2  # outer rect + hole subpath
    areas = sorted(abs(p.area) for p in sil.paths if p.closed)
    assert areas[-1] > 0.1 * cfg.crop.canvas_size ** 2 * 0.5  # big outer contour


def test_cleanup_reduction_shape_and_holes(tmp_path):
    cfg, layers, _ = _run_to_traces(tmp_path, "contour")
    cleaned = svg_cleanup.cleanup_layers(layers, str(tmp_path / "cleaned"), cfg)
    sil_raw = next(l for l in layers if l.name.value == "silhouette")
    sil = next(l for l in cleaned if l.name.value == "silhouette")

    raw_pts = sil_raw.stats["raw_point_count"]
    new_pts = sil.stats["cleaned_point_count"]
    assert 1 - new_pts / raw_pts >= 0.70

    # shape deviation: mean distance raw points -> cleaned polyline
    w, h = sil.image_size
    xs = [p.points for p in sil.paths if p.closed]
    assert xs
    diag = max(
        np.hypot(max(x for x, _ in pts) - min(x for x, _ in pts),
                 max(y for _, y in pts) - min(y for _, y in pts))
        for pts in xs
    )
    raw_by_id = {p.path_id: p for p in sil_raw.paths}
    devs = []
    for p in sil.paths:
        src = raw_by_id.get(p.path_id)
        if src is None or not p.closed:
            continue
        poly = np.asarray(p.points)
        for q in src.points[::4]:  # subsample raw points
            devs.append(_point_to_polyline_dist(
                np.array([q[0] / w, q[1] / h]), poly, p.closed))
    assert np.mean(devs) <= 0.005 * diag
    assert np.max(devs) <= 0.02 * diag

    # hole retained with hierarchy + winding
    holes = [p for p in sil.paths if p.is_hole]
    assert holes and all(p.parent_path_id for p in holes)
    parent = next(p for p in sil.paths if p.path_id == holes[0].parent_path_id)
    assert parent.area > 0 and holes[0].area < 0
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for p in sil.paths for x, y in p.points)

    # cleaned artefacts written
    assert (tmp_path / "cleaned" / "cleaned_silhouette.json").exists()
    assert (tmp_path / "cleaned" / "cleaned_silhouette.svg").exists()


def test_cleanup_open_paths_stay_open(tmp_path):
    from recon3d.schemas import TraceLayer, TraceLayerName, VectorPath

    line = VectorPath(
        path_id="edge_000", source_layer=TraceLayerName.STRUCTURAL_EDGES,
        closed=False, points=[(10.0 + i, 10.0 + 0.5 * np.sin(i)) for i in range(200)],
    )
    layer = TraceLayer(name=TraceLayerName.STRUCTURAL_EDGES,
                       svg_path="x.svg", paths=[line], image_size=(256, 256))
    cfg = PipelineConfig()
    out = svg_cleanup.cleanup_layers([layer], str(tmp_path), cfg)
    p = out[0].paths[0]
    assert not p.closed
    assert len(p.points) < len(line.points)
    # coordinates are normalised to 0..1 over image_size
    assert p.points[0] == pytest.approx((line.points[0][0] / 256, line.points[0][1] / 256), abs=1e-3)
    assert p.points[-1] == pytest.approx((line.points[-1][0] / 256, line.points[-1][1] / 256), abs=1e-3)
