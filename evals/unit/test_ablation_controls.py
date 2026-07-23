from pathlib import Path

from recon3d import camera, constraints, primitives, semantic_parts
from recon3d.config import PipelineConfig
from recon3d.schemas import (
    CropMetadata,
    EvidenceSource,
    GeometricPrimitive,
    InputSpec,
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
        "no_depth_normals.yaml", "no_refinement.yaml",
    )
    for name in names:
        PipelineConfig.from_yaml(root / name)
