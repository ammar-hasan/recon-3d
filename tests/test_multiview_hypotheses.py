"""Phase 6/7 tests: cross-view fusion and hypothesis auditability."""
from __future__ import annotations

from recon3d.config import PipelineConfig
from recon3d.hypotheses import evaluate_hypotheses
from recon3d.multiview import _match_parts
from recon3d.schemas import (
    ConstraintType,
    EvidenceSource,
    GeometricConstraint,
    GeometricPrimitive,
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
