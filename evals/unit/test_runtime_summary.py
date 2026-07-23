import json

from evals.runtime.summarize import summarize


def test_runtime_summary_supports_instrumented_and_legacy_manifests(tmp_path):
    dataset = tmp_path / "dataset"
    projects = tmp_path / "projects"
    for case, difficulty in (("easy_01", "easy"), ("hard_01", "hard")):
        (dataset / case).mkdir(parents=True)
        (projects / case).mkdir(parents=True)
        (dataset / case / "meta.json").write_text(json.dumps({
            "case_id": case, "difficulty": difficulty}))
    (projects / "easy_01" / "manifest.json").write_text(json.dumps({
        "timings_seconds": {"total": 10.0, "planning": 2.0},
        "resource_usage": {"peak_process_rss_mb": 128.0},
    }))
    (projects / "hard_01" / "manifest.json").write_text(json.dumps({
        "started_at": "2026-07-23T00:00:00",
        "finished_at": "2026-07-23T00:00:30",
    }))
    report = summarize(str(projects), str(dataset), str(tmp_path / "out"))
    assert report["total_runtime_seconds"]["mean"] == 20.0
    assert report["runtime_by_difficulty"]["easy"]["median"] == 10.0
    assert report["runtime_by_difficulty"]["hard"]["median"] == 30.0
    assert report["instrumented_stage_timing_cases"] == 1
    assert report["peak_rss_cases"] == 1
    assert (tmp_path / "out" / "suite.md").is_file()
