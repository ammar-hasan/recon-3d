import json

from evals.perceptual.summarize import REQUIRED_PASSES, collect_case, summarize


def _project(tmp_path, name, shaded=0.9, clay=0.7):
    project = tmp_path / name
    validation = project / "validation"
    validation.mkdir(parents=True)
    (validation / "metrics.json").write_text(json.dumps({
        "perceptual_similarity": shaded,
        "color_region_agreement": 0.8,
        "clay_silhouette_iou": clay,
        "silhouette_iou": 0.85,
        "depth_correlation": 0.5,
        "feature_alignment_error_px": 2.0,
    }))
    for render_pass in REQUIRED_PASSES:
        (validation / render_pass).write_bytes(b"x")
    return project


def test_collect_case_enforces_clay_geometry_rule(tmp_path):
    project = _project(tmp_path, "case")
    case = collect_case("case", project)
    assert case["all_required_passes"]
    assert case["geometry_compensation_flag"]


def test_summary_aggregates_render_and_metric_coverage(tmp_path):
    first = _project(tmp_path, "a", shaded=0.9, clay=0.9)
    second = _project(tmp_path, "b", shaded=0.7, clay=0.8)
    (second / "validation" / "render_materialid.png").unlink()
    summary = summarize({"a": first, "b": second})
    assert summary["aggregate"]["mean_shaded_ssim"] == 0.8
    assert summary["aggregate"]["required_pass_coverage"][
        "render_materialid.png"] == 0.5
