from pathlib import Path

import cv2
import numpy as np

from recon3d import (camera, constraints, depth, primitives, segmentation,
                     semantic_parts, uncertainty)
from recon3d.config import PipelineConfig
from recon3d.schemas import (
    CropMetadata,
    EvidenceSource,
    GeometricPrimitive,
    InputSpec,
    InputBundle,
    LoadedImage,
    PrimitiveType,
    SegmentationResult,
    SketchGraph,
    TraceLayer,
    TraceLayerName,
    VectorPath,
)


def _primitive() -> GeometricPrimitive:
    return GeometricPrimitive(
        id="p", type=PrimitiveType.RECTANGLE,
        params={"points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
        source_path="path", source_layer=TraceLayerName.SILHOUETTE,
        fallback_points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
    )


def test_primitive_constraint_and_semantic_ablation_controls():
    cfg = PipelineConfig()
    cfg.primitives.enabled = False
    layer = TraceLayer(
        name=TraceLayerName.SILHOUETTE, svg_path="trace.svg",
        image_size=(100, 100), paths=[VectorPath(
            path_id="path", points=[(0.1, 0.1), (0.9, 0.1),
                                    (0.9, 0.9), (0.1, 0.9)],
            closed=True, source_layer=TraceLayerName.SILHOUETTE)])
    fitted = primitives.fit_primitives([layer], cfg)
    assert fitted[0].type == PrimitiveType.CLOSED_REGION
    assert fitted[0].confidence == 0.3

    cfg.constraints.enabled = False
    assert constraints.detect_constraints(fitted, cfg) == []

    cfg.semantics.enabled = False
    graph = semantic_parts.decompose_parts(
        SketchGraph(primitives=fitted), "",
        InputSpec(image_paths=["image.png"], target_label="gear"), cfg)
    assert [part.part_class for part in graph.parts] == ["object"]
    assert graph.stats["semantic_backend"] == "disabled_ablation"


def test_camera_ablation_returns_explicit_unknown_fallback():
    cfg = PipelineConfig()
    cfg.camera.enabled = False
    seg = SegmentationResult(
        mask_path="m", rgba_path="r", original_path="i", confidence=1.0,
        backend="test", bbox=(0, 0, 10, 10), coverage=1.0,
        selection_source=EvidenceSource.DIRECTLY_OBSERVED)
    estimate = camera.estimate_camera(
        SketchGraph(primitives=[_primitive()]), seg,
        CropMetadata(source_image_size=(10, 10), source_bbox=(0, 0, 10, 10),
                     padding=0, output_size=(10, 10), scale=1.0,
                     offset=(0.0, 0.0)),
        InputSpec(image_paths=["image.png"]), cfg)
    assert estimate.focal_length_px.source == EvidenceSource.UNKNOWN
    assert estimate.focal_length_px.confidence == 0.0
    assert "disabled" in estimate.notes[0]


def test_required_ablation_configs_load():
    root = Path("evals/ablations")
    names = (
        "no_vtracer.yaml", "no_svg_simplification.yaml",
        "no_primitive_fitting.yaml", "no_constraint_detection.yaml",
        "no_semantic_part_reasoning.yaml", "no_camera_estimation.yaml",
        "no_background_removal.yaml", "no_depth.yaml", "no_normals.yaml",
        "no_depth_normals.yaml", "no_refinement.yaml",
        "no_uncertainty_tracking.yaml",
    )
    for name in names:
        PipelineConfig.from_yaml(root / name)


def test_background_removal_ablation_uses_full_frame(tmp_path):
    image_path = tmp_path / "input.png"
    cv2.imwrite(str(image_path), np.full((20, 30, 3), 255, np.uint8))
    spec = InputSpec(image_paths=[str(image_path)])
    bundle = InputBundle(
        spec=spec, images=[LoadedImage(
            path=str(image_path), width=30, height=20, sha256="test")])
    cfg = PipelineConfig()
    cfg.segmentation.background_removal_enabled = False
    result = segmentation.segment(bundle, str(tmp_path / "seg"), cfg)
    assert result.backend == "disabled_background_removal"
    assert result.bbox == (0, 0, 30, 20)
    assert result.coverage == 1.0


def test_depth_and_normals_can_be_disabled_independently(tmp_path):
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[4:28, 4:28, :3] = 180
    rgba[4:28, 4:28, 3] = 255
    mask = np.zeros((32, 32), np.uint8)
    mask[4:28, 4:28] = 255
    rgba_path, mask_path = tmp_path / "rgba.png", tmp_path / "mask.png"
    cv2.imwrite(str(rgba_path), rgba)
    cv2.imwrite(str(mask_path), mask)

    cfg = PipelineConfig()
    cfg.depth.depth_enabled = False
    normals_only = depth.estimate_depth(
        str(rgba_path), str(mask_path), SketchGraph(),
        str(tmp_path / "normals_only"), cfg)
    assert normals_only.backend == "normals_only"
    assert normals_only.depth_path is None
    assert Path(normals_only.normals_path).is_file()

    cfg = PipelineConfig()
    cfg.depth.normals_enabled = False
    depth_only = depth.estimate_depth(
        str(rgba_path), str(mask_path), SketchGraph(),
        str(tmp_path / "depth_only"), cfg)
    assert depth_only.backend == "depth_only"
    assert Path(depth_only.depth_path).is_file()
    assert depth_only.normals_path is None


def test_uncertainty_ablation_uniforms_confidence_but_preserves_source():
    graph = SketchGraph(
        primitives=[_primitive()], uncertainty={"physical_scale": "unknown"})
    graph.primitives[0].confidence = 0.2
    graph.primitives[0].source = EvidenceSource.FITTED_FROM_OBSERVATION
    uniform = uncertainty.disable_tracking(graph)
    assert uniform.primitives[0].confidence == 1.0
    assert uniform.primitives[0].source == EvidenceSource.FITTED_FROM_OBSERVATION
    assert uniform.uncertainty == {}
