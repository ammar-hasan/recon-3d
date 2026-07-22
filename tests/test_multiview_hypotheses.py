"""Phase 6/7 tests: cross-view fusion and hypothesis auditability."""
from __future__ import annotations

import numpy as np

from recon3d import multiview_refinement
from recon3d.config import PipelineConfig
from recon3d.hypotheses import evaluate_hypotheses
from recon3d.multiview import _match_parts
from recon3d.schemas import (
    ConstraintType,
    BlenderManifest,
    ConstructionPlan,
    EvidenceSource,
    GeometricConstraint,
    GeometricPrimitive,
    MultiViewObservation,
    MultiViewResult,
    OperatorCategory,
    PrimitiveType,
    SemanticPart,
    SketchGraph,
    TraceLayerName,
    Visibility,
)


def _rect(pid, x0, y0, x1, y1):
    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return GeometricPrimitive(
        id=pid, type=PrimitiveType.CLOSED_REGION,
        params={"points": points}, fallback_points=points,
        source_path="path_" + pid, source_layer=TraceLayerName.SILHOUETTE,
        confidence=0.9,
    )


def test_cross_view_matching_uses_class_and_relative_geometry():
    pa, pb = _rect("pa", 0.1, 0.1, 0.8, 0.8), _rect("pb", 0.2, 0.2, 0.3, 0.5)
    qa, qb = _rect("qa", 0.12, 0.11, 0.82, 0.81), _rect("qb", 0.22, 0.2, 0.32, 0.5)
    primary = SketchGraph(primitives=[pa, pb], parts=[
        SemanticPart(id="body", part_class="gear_body", primitive_ids=["pa"], confidence=0.8),
        SemanticPart(id="tooth", part_class="tooth", primitive_ids=["pb"], confidence=0.7),
    ])
    secondary = SketchGraph(primitives=[qa, qb], parts=[
        SemanticPart(id="other_body", part_class="gear_body", primitive_ids=["qa"], confidence=0.8),
        SemanticPart(id="other_tooth", part_class="tooth", primitive_ids=["qb"], confidence=0.7),
    ])
    matches = _match_parts(primary, secondary, "view_001", 0.85)
    assert {(m.primary_part_id, m.secondary_part_id) for m in matches} == {
        ("body", "other_body"), ("tooth", "other_tooth")}
    assert all(m.source == EvidenceSource.FITTED_FROM_OBSERVATION for m in matches)


def test_hypotheses_are_scored_rejectable_and_do_not_change_primitives():
    body = _rect("body_curve", 0.2, 0.1, 0.8, 0.9)
    hidden = _rect("hidden_curve", 0.3, 0.3, 0.7, 0.7)
    graph = SketchGraph(
        primitives=[body, hidden],
        constraints=[GeometricConstraint(
            type=ConstraintType.MIRROR_SYMMETRY,
            entities=[body.id], params={"axis": [0.5, 0.0, 0.5, 1.0]},
            confidence=0.8)],
        parts=[
            SemanticPart(
                id="body", part_class="bottle_body", primitive_ids=[body.id],
                selected_operator=OperatorCategory.REVOLVE.value,
                confidence=0.8),
            SemanticPart(
                id="hidden", part_class="appendage", primitive_ids=[hidden.id],
                selected_operator=OperatorCategory.FREEFORM.value,
                visibility=Visibility.PARTIAL, confidence=0.4),
        ],
    )
    before = [p.model_dump() for p in graph.primitives]
    fused, report = evaluate_hypotheses(graph, MultiViewResult(enabled=False),
                                         PipelineConfig())
    assert report.accepted_ids
    assert report.rejected_ids
    assert [p.model_dump() for p in fused.primitives] == before
    accepted = [c for c in report.candidates if c.accepted]
    assert all(c.source == EvidenceSource.GENERATED_HYPOTHESIS for c in accepted)
    assert all(c.confidence <= 0.5 for c in accepted)
    assert any(c.hypothesis_type == "occlusion_completion" and not c.accepted
               for c in report.candidates)
    inferred = fused.parts[0].inferred_geometry
    assert any(v.source == EvidenceSource.GENERATED_HYPOTHESIS
               for key, v in inferred.items() if key.startswith("accepted_"))


def test_multiview_and_hypothesis_config_round_trip(tmp_path):
    cfg = PipelineConfig()
    cfg.multiview.max_views = 3
    cfg.hypotheses.acceptance_threshold = 0.6
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = PipelineConfig.from_yaml(path)
    assert loaded.multiview.max_views == 3
    assert loaded.hypotheses.acceptance_threshold == 0.6


def test_joint_multiview_refinement_updates_only_inferred_depth(monkeypatch,
                                                                tmp_path):
    plan = ConstructionPlan(object_id="box", parts=[],
                            metadata={"global_scale": [1.1, 0.9, 1.0]})
    result = MultiViewResult(enabled=True, observations=[
        MultiViewObservation(view_id="view_001", image_path="secondary.png")
    ])
    cfg = PipelineConfig()

    monkeypatch.setattr(
        multiview_refinement, "_candidate_grid",
        lambda *_: ([{"filename": "candidate.png"}],
                    {"view_000": np.ones((8, 8), np.uint8),
                     "view_001": np.ones((8, 8), np.uint8)}))
    script = tmp_path / "blender" / "render_multiview_candidates.py"
    script.parent.mkdir(parents=True)
    script.write_text("# safe test script\n")
    monkeypatch.setattr(multiview_refinement, "_write_script",
                        lambda *_: script)
    monkeypatch.setattr(
        multiview_refinement.runner, "run_blender",
        lambda *_args, **_kwargs: BlenderManifest(
            blend_path=str(tmp_path / "blender" / "scene.blend"),
            script_path=str(script), success=True))
    monkeypatch.setattr(
        multiview_refinement, "_score_candidates",
        lambda *_: ({
            1.0: {
                "view_000": {"yaw_deg": 0.0, "silhouette_iou": 0.90,
                             "render_path": "primary.png"},
                "view_001": {"yaw_deg": 30.0, "silhouette_iou": 0.62,
                             "render_path": "baseline.png"}},
            1.5: {
                "view_000": {"yaw_deg": 0.0, "silhouette_iou": 0.898,
                             "render_path": "primary_15.png"},
                "view_001": {"yaw_deg": 40.0, "silhouette_iou": 0.75,
                             "render_path": "eligible.png"}},
            2.0: {
                "view_000": {"yaw_deg": 0.0, "silhouette_iou": 0.70,
                             "render_path": "primary_bad.png"},
                "view_001": {"yaw_deg": 45.0, "silhouette_iou": 0.83,
                             "render_path": "best_secondary.png"}},
        }, {"candidate_count": 3, "rendered_candidate_count": 3}))

    refined, solved, rebuild = multiview_refinement.refine_multiview_geometry(
        plan, result, str(tmp_path), cfg)

    assert rebuild
    assert plan.metadata["global_scale"] == [1.1, 0.9, 1.0]
    assert refined.metadata["global_scale"] == [1.1, 0.9, 1.5]
    assert solved.relative_camera_poses["view_001"].value == [0.0, -40.0, 0.0]
    assert (solved.relative_camera_poses["view_001"].source
            == EvidenceSource.FITTED_FROM_OBSERVATION)
    assert not solved.joint_optimization[
        "primary_observed_geometry_overwritten"]
    assert solved.joint_optimization["best_depth_scale"] == 1.5
