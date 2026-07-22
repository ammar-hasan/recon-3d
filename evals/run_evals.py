"""Umbrella evaluation CLI (EVAL.md levels A/B/C).

    python -m evals.run_evals --level unit|stage|e2e|all [--out evals/results]

Runs the requested pytest suites and/or the e2e runner, writes
evals/results/summary_<timestamp>.yaml and prints a compact dashboard table.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]

LEVEL_SUITES = {
    "unit": "evals/unit",
    "stage": "evals/stage",
}


def run_pytest_suite(name: str, suite_path: str, python: str) -> Dict[str, Any]:
    """Run one pytest suite; parse the summary line for counts."""
    proc = subprocess.run(
        [python, "-m", "pytest", suite_path, "-q", "--tb=short"],
        capture_output=True, text=True, cwd=str(ROOT))
    tail = (proc.stdout or "").strip().splitlines()
    summary_line = tail[-1] if tail else ""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "error": 0}
    # parse e.g. "34 passed, 1 xfailed in 0.61s"
    for chunk in summary_line.split(","):
        words = chunk.strip().split()
        if len(words) >= 2 and words[0].isdigit() and words[1] in counts:
            counts[words[1]] = int(words[0])
    ok = proc.returncode == 0
    return {"level": name, "suite": suite_path, "ok": ok,
            "returncode": proc.returncode, "counts": counts,
            "summary_line": summary_line,
            "output_tail": "\n".join(tail[-25:])}


def run_e2e(args: argparse.Namespace, python: str) -> Dict[str, Any]:
    cmd = [python, "-m", "evals.e2e.run_e2e",
           "--cases", args.cases,
           "--dataset", args.dataset,
           "--projects-root", args.projects_root,
           "--out", str(Path(args.out) / "e2e"),
           "--workers", str(args.workers)]
    if args.skip_blender:
        cmd.append("--skip-blender")
    if args.skip_pipeline:
        cmd.append("--skip-pipeline")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    dashboard_path = Path(args.out) / "e2e" / "dashboard.json"
    dashboard = None
    if dashboard_path.exists():
        try:
            dashboard = json.loads(dashboard_path.read_text())
        except Exception:  # noqa: BLE001
            dashboard = None
    return {"level": "e2e", "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "output_tail": "\n".join((proc.stdout or "").strip().splitlines()[-25:]),
            "dashboard": dashboard}


def print_table(results: List[Dict[str, Any]]) -> None:
    print("\n%-8s %-6s %-40s" % ("level", "ok", "details"))
    print("-" * 60)
    for r in results:
        if r["level"] == "e2e":
            dash = r.get("dashboard") or {}
            s = dash.get("summary", {})
            details = ("mvp %s/%s, silhouette_iou=%s, baseline_iou=%s"
                       % (s.get("n_passed_mvp"), s.get("n_cases"),
                          _f(s.get("silhouette_iou_mean")),
                          _f(s.get("baseline_svg_extrusion_iou_mean"))))
        else:
            c = r["counts"]
            details = ("%d passed, %d failed, %d skipped, %d xfailed"
                       % (c["passed"], c["failed"], c["skipped"], c["xfailed"]))
        print("%-8s %-6s %-40s" % (r["level"],
                                   "OK" if r["ok"] else "FAIL", details))


def _f(v: Any) -> str:
    return "-" if v is None else ("%.3f" % float(v))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="run_evals")
    ap.add_argument("--level", default="all",
                    choices=["unit", "stage", "e2e", "all"])
    ap.add_argument("--out", default="evals/results")
    ap.add_argument("--cases", default="all")
    ap.add_argument("--dataset",
                    default=str(ROOT / "evals" / "benchmark" / "dataset"))
    ap.add_argument("--projects-root", default="projects/e2e")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-blender", action="store_true")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    levels = ["unit", "stage", "e2e"] if args.level == "all" else [args.level]
    for level in levels:
        if level in LEVEL_SUITES:
            print(">>> running level %s (%s)" % (level, LEVEL_SUITES[level]))
            res = run_pytest_suite(level, LEVEL_SUITES[level], args.python)
            print("    %s" % res["summary_line"])
            results.append(res)
        elif level == "e2e":
            print(">>> running level e2e")
            results.append(run_e2e(args, args.python))

    print_table(results)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": ts,
        "level": args.level,
        "results": [{k: v for k, v in r.items() if k != "dashboard"}
                    for r in results],
        "e2e_summary": ((results[-1].get("dashboard") or {}).get("summary")
                        if results and results[-1]["level"] == "e2e" else None),
        "all_ok": all(r["ok"] for r in results),
    }
    path = out_dir / ("summary_%s.yaml" % ts)
    path.write_text(yaml.safe_dump(summary, sort_keys=False))
    print("\nsummary written to %s" % path)
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
