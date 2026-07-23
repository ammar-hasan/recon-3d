"""Evidence-backed preflight checks for unreliable reconstruction inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .schemas import (EvidenceSource, InputBundle, InputQualityAssessment,
                      InputQualitySignal, SegmentationResult)


_EXPLICIT_HAZARDS = {
    "transparent": ("transparent_object", "Provide an opaque reference or depth/multiview evidence."),
    "glass": ("transparent_object", "Provide an opaque reference or depth/multiview evidence."),
    "mirror": ("mirror_or_reflective", "Provide diffuse cross-polarized views or a clean silhouette mask."),
    "reflective": ("mirror_or_reflective", "Provide diffuse cross-polarized views or a clean silhouette mask."),
    "occluded": ("heavy_occlusion", "Provide an unoccluded view of the hidden structure."),
    "heavy occlusion": ("heavy_occlusion", "Provide an unoccluded view of the hidden structure."),
    "overlapping": ("overlapping_instances", "Provide a target mask or an isolated reference."),
    "unknown target": ("unknown_target", "Provide a target label, point, box, or mask."),
    "texture only": ("texture_without_boundary", "Provide a silhouette or geometry-bearing side view."),
    "conflicting views": ("conflicting_views", "Verify that every view depicts the same object and calibration."),
}


def _signal(code: str, severity: str, evidence: str, recommendation: str,
            confidence: float, source: EvidenceSource = EvidenceSource.DIRECTLY_OBSERVED
            ) -> InputQualitySignal:
    return InputQualitySignal(
        code=code, severity=severity, evidence=evidence,
        recommendation=recommendation, confidence=confidence, source=source)


def _dedupe(signals: Iterable[InputQualitySignal]) -> list[InputQualitySignal]:
    best = {}
    rank = {"low": 0, "medium": 1, "high": 2}
    for signal in signals:
        previous = best.get(signal.code)
        if previous is None or rank[signal.severity] > rank[previous.severity]:
            best[signal.code] = signal
    return [best[key] for key in sorted(best)]


def assess_input_quality(bundle: InputBundle,
                         seg: Optional[SegmentationResult] = None
                         ) -> InputQualityAssessment:
    """Assess only observable or explicitly supplied reliability hazards.

    The checks are deliberately conservative: absence of a warning is not a
    claim that hidden geometry is recoverable. High risk means the pipeline
    must not report an unqualified reconstruction success.
    """
    signals: list[InputQualitySignal] = []
    primary = bundle.images[0]
    min_side = min(primary.width, primary.height)
    if min_side < 64:
        signals.append(_signal(
            "very_low_resolution", "high", "minimum image side is %d px" % min_side,
            "Provide an image with at least 128 px on its shorter side.", 1.0))
    elif min_side < 128:
        signals.append(_signal(
            "low_resolution", "medium", "minimum image side is %d px" % min_side,
            "Provide a higher-resolution reference.", 1.0))

    if seg is not None:
        x0, y0, x1, y1 = seg.bbox
        bbox_max = max(x1 - x0, y1 - y0)
        if bbox_max < 64:
            signals.append(_signal(
                "target_smaller_than_64px", "high",
                "segmented target maximum extent is %d px" % bbox_max,
                "Provide a tighter, higher-resolution image of the target.", 1.0))
    else:
        x0, y0, x1, y1 = 0, 0, primary.width, primary.height

    image = cv2.imread(primary.path, cv2.IMREAD_UNCHANGED)
    if image is not None:
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            partial = float(np.mean((alpha > 0) & (alpha < 255)))
            if partial >= 0.01:
                signals.append(_signal(
                    "transparent_object", "high",
                    "%.1f%% of pixels have partial alpha" % (100.0 * partial),
                    "Provide an opaque reference or depth/multiview evidence.",
                    min(1.0, 0.7 + partial)))
        if image.ndim == 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        roi = gray[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
        if roi.size:
            blur_variance = float(cv2.Laplacian(roi, cv2.CV_64F).var())
            if blur_variance < 8.0:
                signals.append(_signal(
                    "severe_blur", "high", "Laplacian variance is %.2f" % blur_variance,
                    "Provide a sharply focused reference.", 0.9))
            elif blur_variance < 25.0:
                signals.append(_signal(
                    "blur", "medium", "Laplacian variance is %.2f" % blur_variance,
                    "Provide a sharper reference.", 0.75))
            edges = cv2.Canny(roi, 60, 160)
            edge_fraction = float(np.mean(edges > 0))
            if edge_fraction < 0.002:
                signals.append(_signal(
                    "texture_without_boundary", "high",
                    "geometric edge fraction is %.4f" % edge_fraction,
                    "Provide a silhouette or geometry-bearing side view.", 0.85))

    text = " ".join(filter(None, [bundle.spec.target_label,
                                    bundle.spec.description])).lower()
    for phrase, (code, recommendation) in _EXPLICIT_HAZARDS.items():
        if phrase in text:
            signals.append(_signal(
                code, "high", "user description contains '%s'" % phrase,
                recommendation, 1.0, EvidenceSource.USER_SUPPLIED))

    if len(bundle.images) > 1:
        primary_rgb = cv2.imread(bundle.images[0].path, cv2.IMREAD_COLOR)
        for index, loaded in enumerate(bundle.images[1:], start=1):
            other = cv2.imread(loaded.path, cv2.IMREAD_COLOR)
            if primary_rgb is None or other is None:
                continue
            hist_a = cv2.calcHist([primary_rgb], [0, 1], None, [16, 16],
                                  [0, 256, 0, 256])
            hist_b = cv2.calcHist([other], [0, 1], None, [16, 16],
                                  [0, 256, 0, 256])
            cv2.normalize(hist_a, hist_a)
            cv2.normalize(hist_b, hist_b)
            distance = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))
            if distance > 0.85:
                signals.append(_signal(
                    "conflicting_views", "high",
                    "view %d color-histogram distance is %.3f" % (index, distance),
                    "Verify that every view depicts the same object and calibration.",
                    min(1.0, distance)))

    signals = _dedupe(signals)
    if any(signal.severity == "high" for signal in signals):
        risk = "high"
    elif signals:
        risk = "medium"
    else:
        risk = "low"
    recommendations = sorted({signal.recommendation for signal in signals})
    return InputQualityAssessment(
        risk=risk, unreliable_input_detected=risk == "high",
        signals=signals, recommendations=recommendations)
