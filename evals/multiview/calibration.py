"""Train and apply a prospective held-out-silhouette calibrator.

Hidden-geometry confidence and the probability that an unseen silhouette will
pass are different quantities.  This module fits the latter from a calibration
cohort and serializes the complete model so a disjoint evaluation cohort can be
scored without refitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.optimize import minimize

from evals.metrics import brier_score, expected_calibration_error


FEATURE_NAMES = ("primary_silhouette_iou", "operational_risk_rank")
RISK_RANKS = {"low": 0.0, "medium": 1.0, "high": 2.0, "unknown": 3.0}
DEFAULT_L2_CANDIDATES = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def feature_vector(case: Dict) -> np.ndarray:
    risk = str(case.get("unseen_view_risk", "unknown"))
    return np.asarray([
        float(case["primary_silhouette_iou"]),
        RISK_RANKS.get(risk, RISK_RANKS["unknown"]),
    ], dtype=np.float64)


def _fit(X: np.ndarray, y: np.ndarray, l2: float) -> Dict:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (X - mean) / scale

    def objective(params: np.ndarray) -> float:
        logits = np.clip(params[0] + normalized @ params[1:], -30.0, 30.0)
        loss = np.sum(np.logaddexp(0.0, logits) - y * logits)
        return float(loss + 0.5 * l2 * np.sum(params[1:] ** 2))

    result = minimize(
        objective, np.zeros(normalized.shape[1] + 1), method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError("silhouette calibrator optimization failed: %s"
                           % result.message)
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "intercept": float(result.x[0]),
        "coefficients": result.x[1:].tolist(),
        "l2": float(l2),
    }


def predict_probability(model: Dict, case: Dict) -> float:
    if "constant_probability" in model:
        return float(model["constant_probability"])
    vector = feature_vector(case)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    logit = float(model["intercept"] + ((vector - mean) / scale) @ coefficients)
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))))


def _leave_one_out_brier(X: np.ndarray, y: np.ndarray, l2: float) -> float:
    probabilities: List[float] = []
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        candidate = _fit(X[keep], y[keep], l2)
        case = {
            "primary_silhouette_iou": float(X[index, 0]),
            "unseen_view_risk": next(
                name for name, rank in RISK_RANKS.items()
                if rank == float(X[index, 1])),
        }
        probabilities.append(predict_probability(candidate, case))
    values = np.asarray(probabilities)
    return float(np.mean((values - y) ** 2))


def fit_calibrator(cases: Sequence[Dict],
                   l2_candidates: Iterable[float] = DEFAULT_L2_CANDIDATES) -> Dict:
    if len(cases) < 6:
        raise ValueError("calibration requires at least six cases")
    X = np.asarray([feature_vector(case) for case in cases])
    y = np.asarray([float(case["silhouette_pass"]) for case in cases])
    if len(np.unique(y)) < 2:
        raise ValueError("calibration cohort must contain passes and failures")
    candidates = [float(value) for value in l2_candidates]
    if not candidates or any(value <= 0.0 for value in candidates):
        raise ValueError("l2 candidates must be positive")
    scores = {value: _leave_one_out_brier(X, y, value)
              for value in candidates}
    selected = min(candidates, key=lambda value: (scores[value], value))
    model = _fit(X, y, selected)
    model.update({
        "schema_version": 1,
        "model_type": "l2_logistic_regression",
        "target": "heldout_silhouette_iou_gte_0_75",
        "feature_names": list(FEATURE_NAMES),
        "selection": "leave_one_out_brier_on_calibration_cohort",
        "candidate_l2_brier": {str(key): value for key, value in scores.items()},
        "training_case_ids": [str(case["case_id"]) for case in cases],
        "training_case_count": len(cases),
        "training_pass_rate": float(y.mean()),
    })
    return model


def cross_validate_calibrator(cases: Sequence[Dict], n_folds: int = 3) -> Dict:
    """Return deterministic stratified out-of-fold calibration evidence."""
    if n_folds < 2:
        raise ValueError("cross-validation requires at least two folds")
    if len(cases) < n_folds * 2:
        raise ValueError("too few cases for requested cross-validation folds")
    folds: List[List[Dict]] = [[] for _ in range(n_folds)]
    for outcome in (False, True):
        group = sorted(
            (case for case in cases if bool(case["silhouette_pass"]) is outcome),
            key=lambda case: hashlib.sha256(
                str(case["case_id"]).encode("utf-8")).hexdigest(),
        )
        for index, case in enumerate(group):
            folds[index % n_folds].append(case)

    predictions = []
    for fold_index, test_cases in enumerate(folds):
        test_ids = {str(case["case_id"]) for case in test_cases}
        train_cases = [case for case in cases
                       if str(case["case_id"]) not in test_ids]
        model = fit_calibrator(train_cases)
        for case in test_cases:
            predictions.append({
                "case_id": str(case["case_id"]),
                "fold": fold_index,
                "silhouette_pass": bool(case["silhouette_pass"]),
                "silhouette_pass_probability": predict_probability(model, case),
                "training_case_ids": model["training_case_ids"],
                "selected_l2": model["l2"],
            })
    predictions.sort(key=lambda item: item["case_id"])
    probabilities = [item["silhouette_pass_probability"]
                     for item in predictions]
    outcomes = [item["silhouette_pass"] for item in predictions]
    positives = [p for p, outcome in zip(probabilities, outcomes) if outcome]
    negatives = [p for p, outcome in zip(probabilities, outcomes) if not outcome]
    auc = float(np.mean([
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives for negative in negatives
    ]))
    return {
        "method": "deterministic_stratified_out_of_fold",
        "fold_assignment": "sha256(case_id), round-robin within outcome",
        "fold_count": n_folds,
        "case_count": len(predictions),
        "expected_calibration_error_5_bin": expected_calibration_error(
            probabilities, outcomes, n_bins=5),
        "brier_score": brier_score(probabilities, outcomes),
        "roc_auc": auc,
        "mean_probability": float(np.mean(probabilities)),
        "observed_pass_rate": float(np.mean(outcomes)),
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True,
                        help="calibration-cohort suite JSON")
    parser.add_argument("--out", required=True, help="model JSON path")
    parser.add_argument(
        "--cross-validation-out", default=None,
        help="optional deterministic out-of-fold evidence JSON")
    args = parser.parse_args()
    suite = json.loads(Path(args.suite).read_text())
    model = fit_calibrator(suite["cases"])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, sort_keys=True))
    if args.cross_validation_out:
        cv_output = Path(args.cross_validation_out)
        cv_output.parent.mkdir(parents=True, exist_ok=True)
        cv_output.write_text(json.dumps(
            cross_validate_calibrator(suite["cases"]),
            indent=2, sort_keys=True))
    print(json.dumps(model, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
