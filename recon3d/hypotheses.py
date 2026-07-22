"""Phase 7: explicit, scored, rejectable hidden-geometry hypotheses."""
from __future__ import annotations

from typing import List, Optional, Tuple

from .config import PipelineConfig
from .schemas import (
    ConstraintType,
    EvidenceSource,
    EvidencedValue,
    HypothesisCandidate,
    HypothesisReport,
    MultiViewResult,
    OperatorCategory,
    SketchGraph,
    Visibility,
)


def evaluate_hypotheses(
    graph: SketchGraph,
    multiview: Optional[MultiViewResult],
    cfg: PipelineConfig,
) -> Tuple[SketchGraph, HypothesisReport]:
    """Propose hidden geometry, score it, and retain a complete audit trail.

    Accepted candidates annotate ``inferred_geometry`` only.  They never
    modify primitives, constraints, observed profiles, or selected operators.
    """
    if not cfg.hypotheses.enabled:
        return graph, HypothesisReport()
    fused = graph.model_copy(deep=True)
    candidates: List[HypothesisCandidate] = []
    match_counts = {}
    if multiview is not None:
        for match in multiview.matches:
            match_counts[match.primary_part_id] = match_counts.get(match.primary_part_id, 0) + 1

    mirror_parts = set()
    for constraint in fused.constraints:
        if constraint.type == ConstraintType.MIRROR_SYMMETRY:
            entities = set(constraint.entities)
            for part in fused.parts:
                if entities.intersection(part.primitive_ids):
                    mirror_parts.add(part.id)

    def add(part, kind, proposal, base, evidence, reject=None):
        score = min(1.0, base + 0.08 * min(match_counts.get(part.id, 0), 3))
        reasons = list(reject or [])
        accepted = score >= cfg.hypotheses.acceptance_threshold and not reasons
        idx = sum(1 for c in candidates if c.part_id == part.id)
        candidate = HypothesisCandidate(
            id="hyp_%s_%s_%d" % (part.id, kind, idx),
            part_id=part.id, hypothesis_type=kind, proposal=proposal,
            score=score,
            confidence=min(cfg.hypotheses.max_confidence, 0.5 * score),
            accepted=accepted, supporting_evidence=evidence,
            rejection_reasons=reasons or ([] if accepted else [
                "score %.3f below acceptance threshold %.3f"
                % (score, cfg.hypotheses.acceptance_threshold)
            ]),
        )
        candidates.append(candidate)
        if accepted:
            part.inferred_geometry["accepted_%s" % candidate.id] = EvidencedValue(
                value=proposal, source=EvidenceSource.GENERATED_HYPOTHESIS,
                confidence=candidate.confidence,
                note="accepted after evidence scoring; does not overwrite observed geometry",
            )

    for part in fused.parts:
        op = part.selected_operator or ""
        support = ["semantic part class '%s'" % part.part_class]
        if match_counts.get(part.id):
            support.append("matched in %d secondary view(s)" % match_counts[part.id])
        if op == OperatorCategory.REVOLVE.value:
            add(part, "proposed_cross_section",
                {"method": "mirror observed radial profile about the revolution axis"},
                0.62, support + ["selected revolve operator"])
        elif op in (OperatorCategory.EXTRUDE.value,
                    OperatorCategory.PRIMITIVE.value,
                    OperatorCategory.SWEEP.value):
            add(part, "hidden_side_completion",
                {"method": "continue the selected operator to the unobserved rear surface"},
                0.56, support + ["selected %s operator" % op])

        if part.id in mirror_parts:
            add(part, "mirror_completion",
                {"method": "reflect observed geometry across fitted mirror axis"},
                0.70, support + ["observed mirror-symmetry constraint"])
        elif part.visibility == Visibility.PARTIAL:
            add(part, "occlusion_completion",
                {"method": "complete occluded extent using parent-part continuity"},
                0.42, support,
                reject=["no observed mirror or multiview constraint supports the hidden extent"]
                       if not match_counts.get(part.id) else [])

    report = HypothesisReport(
        candidates=candidates,
        accepted_ids=[c.id for c in candidates if c.accepted],
        rejected_ids=[c.id for c in candidates if not c.accepted],
        scoring_policy={
            "acceptance_threshold": cfg.hypotheses.acceptance_threshold,
            "maximum_hypothesis_confidence": cfg.hypotheses.max_confidence,
            "secondary_view_support_bonus_per_view": 0.08,
            "observed_geometry_overwrite_allowed": False,
        },
    )
    fused.stats = dict(fused.stats)
    fused.stats.update({
        "hypothesis_candidate_count": len(candidates),
        "hypothesis_accepted_count": len(report.accepted_ids),
        "hypothesis_rejected_count": len(report.rejected_ids),
    })
    return fused, report
