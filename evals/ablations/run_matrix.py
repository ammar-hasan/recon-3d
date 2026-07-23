"""Resumable matched ablation-matrix runner and dashboard summarizer."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


ABLATIONS = OrderedDict([
    ("no_background_removal", "no_background_removal.yaml"),
    ("no_vtracer", "no_vtracer.yaml"),
    ("no_svg_simplification", "no_svg_simplification.yaml"),
    ("no_primitive_fitting", "no_primitive_fitting.yaml"),
    ("no_constraint_detection", "no_constraint_detection.yaml"),
    ("no_semantic_part_reasoning", "no_semantic_part_reasoning.yaml"),
    ("no_camera_estimation", "no_camera_estimation.yaml"),
    ("no_depth", "no_depth.yaml"),
    ("no_normals", "no_normals.yaml"),
    ("no_refinement", "no_refinement.yaml"),
    ("no_uncertainty_tracking", "no_uncertainty_tracking.yaml"),
])


def _mean(values: Iterable[object]) -> Optional[float]:
    numeric = [float(value) for value in values
               if value is not None and not isinstance(value, str)]
    return float(np.mean(numeric)) if numeric else None


def _case_metrics(cases: List[Dict]) -> Dict[str, Optional[float]]:
    return {
        "case_count": len(cases),
        "mvp_pass_rate": _mean(bool(case.get("passed_mvp")) for case in cases),
        "silhouette_iou_mean": _mean(case.get("silhouette_iou") for case in cases),
        "part_recall_mean": _mean(case.get("major_part_recall") for case in cases),
        "editability_score_mean": _mean(case.get("editability_score") for case in cases),
        "blender_success_rate": _mean(
            case.get("blender_execution_success") for case in cases),
        "glb_valid_rate": _mean(case.get("glb_valid") for case in cases),
        "overall_score_mean": _mean(case.get("overall_score") for case in cases),
        "pipeline_seconds_mean": _mean(case.get("pipeline_seconds") for case in cases),
    }


def summarize_matrix(baseline_dashboard: Path, results_root: Path,
                     ablation_names: List[str], case_ids: List[str]) -> Dict:
    baseline = json.loads(baseline_dashboard.read_text())
    wanted = set(case_ids)
    baseline_cases = [case for case in baseline.get("cases", [])
                      if not wanted or case.get("case_id") in wanted]
    baseline_metrics = _case_metrics(baseline_cases)
    if (len(baseline_cases) != len(case_ids)
            or baseline_metrics["silhouette_iou_mean"] is None
            or baseline_metrics["blender_success_rate"] != 1.0):
        raise ValueError(
            "matched full baseline is incomplete or invalid for requested cases")
    rows = []
    missing = []
    delta_keys = (
        "mvp_pass_rate", "silhouette_iou_mean", "part_recall_mean",
        "editability_score_mean", "blender_success_rate", "glb_valid_rate",
        "overall_score_mean", "pipeline_seconds_mean")
    for name in ablation_names:
        dashboard_path = results_root / name / "dashboard.json"
        if not dashboard_path.is_file():
            missing.append(name)
            continue
        dashboard = json.loads(dashboard_path.read_text())
        cases = [case for case in dashboard.get("cases", [])
                 if not wanted or case.get("case_id") in wanted]
        metrics = _case_metrics(cases)
        deltas = {}
        for key in delta_keys:
            current, base = metrics.get(key), baseline_metrics.get(key)
            deltas[key] = (current - base
                           if current is not None and base is not None else None)
        rows.append({
            "ablation": name,
            "metrics": metrics,
            "delta_vs_full": deltas,
            # EVAL.md requires human preference, but it cannot be inferred
            # from automated signals. Preserve the missing evidence explicitly.
            "human_preference_delta": None,
        })
    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "case_ids": case_ids,
        "required_ablation_count": len(ABLATIONS),
        "completed_ablation_count": len(rows),
        "matrix_complete": len(rows) == len(ablation_names) and not missing,
        "baseline": baseline_metrics,
        "ablations": rows,
        "missing_ablations": missing,
        "human_preference_status": "not_measured",
    }


def _markdown(summary: Dict) -> str:
    lines = [
        "# Matched Ablation Matrix", "",
        "- cases: " + ", ".join(summary["case_ids"]),
        "- completed: %d/%d" % (
            summary["completed_ablation_count"],
            summary["required_ablation_count"]),
        "- human preference: not measured", "",
        "| Ablation | Silhouette Δ | Part recall Δ | Editability Δ | "
        "Blender success Δ | Runtime Δs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["ablations"]:
        delta = row["delta_vs_full"]
        fmt = lambda value: "-" if value is None else "%.3f" % value
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            row["ablation"], fmt(delta["silhouette_iou_mean"]),
            fmt(delta["part_recall_mean"]),
            fmt(delta["editability_score_mean"]),
            fmt(delta["blender_success_rate"]),
            fmt(delta["pipeline_seconds_mean"])))
    return "\n".join(lines) + "\n"


def _dashboard_has_cases(path: Path, case_ids: List[str]) -> bool:
    if not path.is_file():
        return False
    try:
        actual = {case.get("case_id") for case in json.loads(path.read_text()).get("cases", [])}
    except Exception:
        return False
    return set(case_ids) <= actual


def _run_one(name: str, config_name: Optional[str], args, repo: Path) -> Dict:
    out_dir = Path(args.results_root) / name
    dashboard = out_dir / "dashboard.json"
    if args.resume and _dashboard_has_cases(dashboard, args.case_ids):
        return {"ablation": name, "status": "reused", "dashboard": str(dashboard)}
    project_dir = Path(args.projects_root) / name
    cmd = [
        sys.executable, str(repo / "evals" / "e2e" / "run_e2e.py"),
        "--cases", ",".join(args.case_ids),
        "--dataset", str(Path(args.dataset).resolve()),
        "--projects-root", str(project_dir.resolve()),
        "--out", str(out_dir.resolve()),
        # absolute() preserves the virtualenv launcher symlink; resolve()
        # dereferences it to the system interpreter and loses site-packages.
        "--python", str(Path(args.python).absolute()),
        "--timeout", str(args.timeout), "--skip-baseline", "--skip-unguided",
    ]
    if config_name:
        cmd += ["--config", str(
            (repo / "evals" / "ablations" / config_name).resolve())]
    if args.skip_blender_probe:
        cmd.append("--skip-blender")
    if args.rescore:
        cmd.append("--skip-pipeline")
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix_runner.log").write_text(
        (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""))
    status = "completed" if proc.returncode == 0 else "failed"
    if proc.returncode == 0 and dashboard.is_file():
        data = json.loads(dashboard.read_text())
        returncodes = [case.get("pipeline_returncode")
                       for case in data.get("cases", [])]
        if any(code not in (None, 0) for code in returncodes):
            status = "completed_with_pipeline_failures"
    return {"ablation": name,
            "status": status,
            "returncode": proc.returncode, "dashboard": str(dashboard)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="box_01,bottle_01,gear_01")
    parser.add_argument("--ablations", default="all")
    parser.add_argument("--dataset", default="evals/benchmark/dataset")
    parser.add_argument("--projects-root", default="projects/ablation_matrix")
    parser.add_argument("--results-root", default="evals/results_ablation_matrix")
    parser.add_argument("--baseline-dashboard", default=None,
                        help="reuse an existing matched full-run dashboard")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rescore", action="store_true",
                        help="reuse project artifacts but rebuild dashboards")
    parser.add_argument("--skip-blender-probe", action="store_true")
    args = parser.parse_args()
    args.case_ids = [case.strip() for case in args.cases.split(",") if case.strip()]
    if args.ablations == "all":
        names = list(ABLATIONS)
    else:
        names = [name.strip() for name in args.ablations.split(",") if name.strip()]
        unknown = set(names) - set(ABLATIONS)
        if unknown:
            parser.error("unknown ablations: %s" % sorted(unknown))

    repo = Path(__file__).resolve().parents[2]
    outcomes = []
    tasks = [("full", None)] if not args.baseline_dashboard else []
    tasks += [(name, ABLATIONS[name]) for name in names]
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(
            _run_one, name, config, args, repo) for name, config in tasks]
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            print("[%s] %s" % (outcome["status"], outcome["ablation"]), flush=True)

    results_root = Path(args.results_root)
    baseline_dashboard = (Path(args.baseline_dashboard)
                          if args.baseline_dashboard
                          else results_root / "full" / "dashboard.json")
    summary = summarize_matrix(
        baseline_dashboard, results_root, names, args.case_ids)
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "matrix.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    (results_root / "matrix.md").write_text(_markdown(summary))
    print("matrix: %d/%d complete" % (
        summary["completed_ablation_count"], len(names)))
    return 0 if all(outcome["status"] != "failed" for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
