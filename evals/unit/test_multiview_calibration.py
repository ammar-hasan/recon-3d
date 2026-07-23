import pytest

from evals.multiview.calibration import (
    cross_validate_calibrator,
    fit_calibrator,
    predict_probability,
)


def _cases():
    return [
        {"case_id": "low_a", "primary_silhouette_iou": 0.95,
         "unseen_view_risk": "low", "silhouette_pass": True},
        {"case_id": "low_b", "primary_silhouette_iou": 0.90,
         "unseen_view_risk": "low", "silhouette_pass": True},
        {"case_id": "medium_a", "primary_silhouette_iou": 0.85,
         "unseen_view_risk": "medium", "silhouette_pass": True},
        {"case_id": "medium_b", "primary_silhouette_iou": 0.75,
         "unseen_view_risk": "medium", "silhouette_pass": False},
        {"case_id": "high_a", "primary_silhouette_iou": 0.70,
         "unseen_view_risk": "high", "silhouette_pass": False},
        {"case_id": "high_b", "primary_silhouette_iou": 0.60,
         "unseen_view_risk": "high", "silhouette_pass": False},
    ]


def test_calibrator_serializes_training_provenance_and_predicts():
    model = fit_calibrator(_cases(), l2_candidates=[0.5, 1.0])
    assert model["training_case_count"] == 6
    assert model["training_case_ids"] == [case["case_id"] for case in _cases()]
    assert model["l2"] in (0.5, 1.0)
    low = predict_probability(model, _cases()[0])
    high = predict_probability(model, _cases()[-1])
    assert 0.0 <= high < low <= 1.0


def test_calibrator_rejects_too_small_or_one_class_cohort():
    with pytest.raises(ValueError, match="at least six"):
        fit_calibrator(_cases()[:5])
    one_class = [dict(case, silhouette_pass=True) for case in _cases()]
    with pytest.raises(ValueError, match="passes and failures"):
        fit_calibrator(one_class)


def test_cross_validation_predictions_never_train_on_their_case():
    cases = _cases() + [dict(case, case_id=case["case_id"] + "_copy")
                        for case in _cases()]
    evidence = cross_validate_calibrator(cases, n_folds=2)
    assert evidence["case_count"] == 12
    assert 0.0 <= evidence["expected_calibration_error_5_bin"] <= 1.0
    assert 0.0 <= evidence["brier_score"] <= 1.0
    assert 0.0 <= evidence["roc_auc"] <= 1.0
    for prediction in evidence["predictions"]:
        assert prediction["case_id"] not in prediction["training_case_ids"]
