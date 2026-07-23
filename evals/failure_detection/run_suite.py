"""Generate and score a deterministic difficult-input suite for Eval 28."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from recon3d.input_quality import assess_input_quality
from recon3d.schemas import (EvidenceSource, InputBundle, InputSpec, LoadedImage,
                             SchemaIO, SegmentationResult)


def _sharp(size=256, color=(35, 95, 190), variant=0):
    image = np.full((size, size, 3), 242, np.uint8)
    margin = max(8, size // 5)
    cv2.rectangle(image, (margin, margin), (size - margin - 1, size - margin - 1),
                  color, -1)
    cv2.line(image, (margin, margin + 8 + variant % 13),
             (size - margin - 1, size - margin - 9), (255, 255, 255), 3)
    cv2.circle(image, (size // 2, size // 2), max(4, size // 9),
               (15, 15, 15), 3)
    return image


def _case_specs():
    transparent = np.dstack([_sharp(), np.full((256, 256), 128, np.uint8)])
    blurred = cv2.GaussianBlur(_sharp(), (51, 51), 18)
    textureless = np.full((256, 256, 3), 128, np.uint8)
    small = _sharp()
    red = np.full((256, 256, 3), (0, 0, 255), np.uint8)
    green = np.full((256, 256, 3), (0, 255, 0), np.uint8)
    difficult = [
        ("transparent_object", [transparent], None, (48, 48, 208, 208)),
        ("mirror", [_sharp(variant=1)], "mirror object", (48, 48, 208, 208)),
        ("heavy_occlusion", [_sharp(variant=2)], "heavily occluded object", (48, 48, 208, 208)),
        ("very_low_resolution", [_sharp(size=48)], None, (6, 6, 42, 42)),
        ("target_smaller_than_64px", [small], None, (108, 108, 148, 148)),
        ("severe_blur", [blurred], None, (48, 48, 208, 208)),
        ("overlapping_instances", [_sharp(variant=3)], "multiple overlapping identical objects", (48, 48, 208, 208)),
        ("unknown_target", [_sharp(variant=4)], "unknown target", (48, 48, 208, 208)),
        ("texture_without_boundary", [textureless], "texture only", (48, 48, 208, 208)),
        ("conflicting_views", [red, green], None, (48, 48, 208, 208)),
    ]
    easy = []
    for index in range(20):
        color = (30 + index * 7 % 180, 55 + index * 11 % 170,
                 75 + index * 13 % 160)
        easy.append(("easy_%02d" % index,
                     [_sharp(color=color, variant=index)], None,
                     (48, 48, 208, 208)))
    return difficult, easy


def _write_case(root: Path, name: str, images, description, bbox):
    case = root / "cases" / name
    case.mkdir(parents=True, exist_ok=True)
    paths = []
    loaded = []
    for index, image in enumerate(images):
        path = case / ("input.png" if index == 0 else "view_%03d.png" % index)
        cv2.imwrite(str(path), image)
        paths.append(str(path))
        loaded.append(LoadedImage(
            path=str(path), width=image.shape[1], height=image.shape[0],
            sha256="synthetic-%s-%d" % (name, index),
            channels=1 if image.ndim == 2 else image.shape[2]))
    x0, y0, x1, y1 = bbox
    mask = np.zeros(images[0].shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = 255
    mask_path = case / "object_mask.png"
    cv2.imwrite(str(mask_path), mask)
    seg = SegmentationResult(
        mask_path=str(mask_path), rgba_path=str(case / "object_rgba.png"),
        original_path=paths[0], confidence=0.9, backend="synthetic_eval",
        bbox=bbox, coverage=float(np.mean(mask > 0)),
        selection_source=EvidenceSource.DIRECTLY_OBSERVED)
    bundle = InputBundle(
        spec=InputSpec(image_paths=paths, description=description,
                       output_dir=str(case)),
        images=loaded)
    assessment = assess_input_quality(bundle, seg)
    assessment_path = SchemaIO.save_json(
        assessment, case / "quality_assessment.json")
    return assessment, assessment_path


def run(out_dir: str) -> dict:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    difficult, easy = _case_specs()
    rows = []
    for expected_failure, specs in ((True, difficult), (False, easy)):
        for name, images, description, bbox in specs:
            assessment, path = _write_case(
                root, name, images, description, bbox)
            detected = assessment.risk == "high"
            status = "partial_success" if detected else "success"
            rows.append({
                "case": name,
                "expected_failure": expected_failure,
                "detected_unreliable": detected,
                "risk": assessment.risk,
                "status_policy": status,
                "signal_codes": [signal.code for signal in assessment.signals],
                "recommendation_count": len(assessment.recommendations),
                "assessment_path": path,
                "partial_artifact_preserved": Path(path).is_file(),
            })
    hard = [row for row in rows if row["expected_failure"]]
    clean = [row for row in rows if not row["expected_failure"]]
    detection = sum(row["detected_unreliable"] for row in hard) / len(hard)
    false_failure = sum(row["detected_unreliable"] for row in clean) / len(clean)
    graceful = sum(
        row["detected_unreliable"]
        and row["status_policy"] == "partial_success"
        and row["partial_artifact_preserved"]
        and row["recommendation_count"] > 0
        for row in hard) / len(hard)
    misleading = sum(
        row["detected_unreliable"] and row["status_policy"] == "success"
        for row in hard) / len(hard)
    metrics = {
        "impossible_case_detection_rate": detection,
        "false_failure_rate_on_easy_cases": false_failure,
        "graceful_partial_output_rate": graceful,
        "misleading_success_claim_rate": misleading,
    }
    acceptance = {
        "impossible_case_detection_rate": detection >= 0.90,
        "false_failure_rate_on_easy_cases": false_failure <= 0.05,
        "graceful_partial_output_rate": graceful >= 0.90,
        "misleading_success_claim_rate": misleading == 0.0,
    }
    report = {"suite": "eval28_difficult_inputs", "metrics": metrics,
              "acceptance": acceptance, "passed": all(acceptance.values()),
              "cases": rows}
    (root / "suite.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# Eval 28 — Failure Detection", "",
             "- difficult cases: %d" % len(hard),
             "- easy controls: %d" % len(clean),
             "- suite pass: **%s**" % ("yes" if report["passed"] else "no"), "",
             "| Metric | Result | Target | Pass |",
             "| --- | ---: | ---: | --- |"]
    targets = {
        "impossible_case_detection_rate": ">= 0.90",
        "false_failure_rate_on_easy_cases": "<= 0.05",
        "graceful_partial_output_rate": ">= 0.90",
        "misleading_success_claim_rate": "0.00",
    }
    for key, value in metrics.items():
        lines.append("| %s | %.3f | %s | %s |" % (
            key, value, targets[key], "PASS" if acceptance[key] else "FAIL"))
    lines += ["", "| Case | Expected | Risk | Signals |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append("| %s | %s | %s | %s |" % (
            row["case"], "difficult" if row["expected_failure"] else "easy",
            row["risk"], ", ".join(row["signal_codes"]) or "none"))
    (root / "suite.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="evals/results_failure_detection")
    args = parser.parse_args()
    report = run(args.out)
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
