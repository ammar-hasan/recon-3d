import json
from pathlib import Path

import pytest

from evals.ablations.run_matrix import summarize_matrix


def _dashboard(path: Path, silhouette: float, runtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": [{
        "case_id": "box_01", "passed_mvp": True,
        "silhouette_iou": silhouette, "major_part_recall": 1.0,
        "editability_score": 1.0, "blender_execution_success": True,
        "glb_valid": True, "overall_score": 0.9,
        "pipeline_seconds": runtime,
    }]}))


def test_summarize_matrix_computes_matched_deltas(tmp_path):
    baseline = tmp_path / "baseline.json"
    _dashboard(baseline, 0.9, 10.0)
    results = tmp_path / "results"
    _dashboard(results / "no_refinement" / "dashboard.json", 0.8, 4.0)
    summary = summarize_matrix(
        baseline, results, ["no_refinement"], ["box_01"])
    row = summary["ablations"][0]
    assert summary["matrix_complete"] is True
    assert row["delta_vs_full"]["silhouette_iou_mean"] == pytest.approx(-0.1)
    assert row["delta_vs_full"]["pipeline_seconds_mean"] == pytest.approx(-6.0)
    assert row["human_preference_delta"] is None


def test_summarize_matrix_rejects_invalid_full_baseline(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"cases": [{
        "case_id": "box_01", "silhouette_iou": None,
        "blender_execution_success": False}]}))
    with pytest.raises(ValueError, match="baseline"):
        summarize_matrix(baseline, tmp_path / "results", [], ["box_01"])
