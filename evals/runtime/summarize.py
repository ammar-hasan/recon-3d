"""Summarize Eval 27 evidence from reconstruction manifests."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np


def _duration(manifest):
    timed = (manifest.get("timings_seconds") or {}).get("total")
    if timed is not None:
        return float(timed), "instrumented"
    try:
        started = datetime.datetime.fromisoformat(manifest["started_at"])
        finished = datetime.datetime.fromisoformat(manifest["finished_at"])
        return (finished - started).total_seconds(), "wall_clock_timestamp"
    except Exception:
        return None, "not_measured"


def _stats(values):
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return {key: None for key in ("count", "mean", "median", "p90", "min", "max")}
    return {
        "count": int(len(data)), "mean": float(np.mean(data)),
        "median": float(np.median(data)), "p90": float(np.percentile(data, 90)),
        "min": float(np.min(data)), "max": float(np.max(data)),
    }


def summarize(projects_root: str, dataset: str, out_dir: str):
    projects = Path(projects_root)
    dataset_root = Path(dataset)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for meta_path in sorted(dataset_root.glob("*/meta.json")):
        case = meta_path.parent.name
        manifest_path = projects / case / "manifest.json"
        if not manifest_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        seconds, source = _duration(manifest)
        rows.append({
            "case": case, "difficulty": meta.get("difficulty", "unknown"),
            "total_seconds": seconds, "timing_source": source,
            "stage_timings_seconds": manifest.get("timings_seconds") or {},
            "peak_process_rss_mb": (manifest.get("resource_usage") or {}).get(
                "peak_process_rss_mb"),
            "gpu_memory_mb": (manifest.get("resource_usage") or {}).get(
                "gpu_memory_mb"),
        })
    measured = [row for row in rows if row["total_seconds"] is not None]
    by_difficulty = {}
    for difficulty in sorted({row["difficulty"] for row in measured}):
        by_difficulty[difficulty] = _stats([
            row["total_seconds"] for row in measured
            if row["difficulty"] == difficulty])
    report = {
        "suite": "eval27_runtime_resources",
        "total_runtime_seconds": _stats([
            row["total_seconds"] for row in measured]),
        "runtime_by_difficulty": by_difficulty,
        "instrumented_stage_timing_cases": sum(
            bool(row["stage_timings_seconds"]) for row in rows),
        "peak_rss_cases": sum(row["peak_process_rss_mb"] is not None for row in rows),
        "gpu_memory_cases": sum(row["gpu_memory_mb"] is not None for row in rows),
        "cases": rows,
    }
    (output / "suite.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    total = report["total_runtime_seconds"]
    lines = ["# Eval 27 — Runtime and Resources", "",
             "- cases: %d" % len(rows),
             "- mean total runtime: **%.1f s**" % total["mean"],
             "- median total runtime: **%.1f s**" % total["median"],
             "- p90 total runtime: **%.1f s**" % total["p90"],
             "- stage-instrumented cases: %d" % report["instrumented_stage_timing_cases"],
             "- peak-RSS cases: %d" % report["peak_rss_cases"],
             "- GPU-memory cases: %d" % report["gpu_memory_cases"], "",
             "| Difficulty | N | Mean s | Median s | P90 s |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for difficulty, stats in by_difficulty.items():
        lines.append("| %s | %d | %.1f | %.1f | %.1f |" % (
            difficulty, stats["count"], stats["mean"], stats["median"], stats["p90"]))
    lines += ["", "| Case | Difficulty | Seconds | Source |",
              "| --- | --- | ---: | --- |"]
    for row in rows:
        lines.append("| %s | %s | %.1f | %s |" % (
            row["case"], row["difficulty"], row["total_seconds"],
            row["timing_source"]))
    (output / "suite.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-root", default="projects/e2e_final")
    parser.add_argument("--dataset", default="evals/benchmark/dataset")
    parser.add_argument("--out", default="evals/results_runtime_summary")
    args = parser.parse_args()
    report = summarize(args.projects_root, args.dataset, args.out)
    print(json.dumps(report["total_runtime_seconds"], indent=2, sort_keys=True))
    return 0 if report["total_runtime_seconds"]["count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
