"""Level A unit tests for primitive-error, colour, similarity and
calibration functions in evals.metrics."""
from __future__ import annotations

import numpy as np
import pytest

from evals import metrics as m


# ---------------------------------------------------------------------------
# primitive parameter errors
# ---------------------------------------------------------------------------

class TestCircleParamError:
    def test_perfect_fit(self):
        r = m.circle_param_error((0.5, 0.5), 0.2, (0.5, 0.5), 0.2)
        assert r["center_rel_error"] == 0.0
        assert r["radius_rel_error"] == 0.0

    def test_radius_error(self):
        r = m.circle_param_error((0.5, 0.5), 0.21, (0.5, 0.5), 0.20)
        assert r["radius_rel_error"] == pytest.approx(0.05)

    def test_center_error_normalised_by_radius(self):
        r = m.circle_param_error((0.6, 0.5), 0.2, (0.5, 0.5), 0.2)
        assert r["center_rel_error"] == pytest.approx(0.5)

    def test_zero_ref_radius_raises(self):
        with pytest.raises(ValueError):
            m.circle_param_error((0, 0), 1.0, (0, 0), 0.0)


class TestEllipseParamError:
    def test_perfect(self):
        r = m.ellipse_param_error((0.3, 0.2), 45.0, (0.3, 0.2), 45.0)
        assert r["major_axis_rel_error"] == 0.0
        assert r["minor_axis_rel_error"] == 0.0
        assert r["rotation_error_deg"] == 0.0

    def test_axes_sorted(self):
        # swapped major/minor in the fit must not inflate the axis error
        r = m.ellipse_param_error((0.2, 0.3), 0.0, (0.3, 0.2), 0.0)
        assert r["major_axis_rel_error"] == pytest.approx(0.0)
        assert r["minor_axis_rel_error"] == pytest.approx(0.0)

    def test_rotation_mod_180(self):
        r = m.ellipse_param_error((0.3, 0.2), 179.0, (0.3, 0.2), 1.0)
        assert r["rotation_error_deg"] == pytest.approx(2.0)

    def test_axis_rel_error(self):
        r = m.ellipse_param_error((0.33, 0.18), 0.0, (0.30, 0.20), 0.0)
        assert r["major_axis_rel_error"] == pytest.approx(0.10)
        assert r["minor_axis_rel_error"] == pytest.approx(0.10)


class TestLineErrors:
    def test_angle_diff_wraps(self):
        assert m.angle_diff_deg(350.0, 10.0) == pytest.approx(-20.0)

    def test_angle_diff_mod_180(self):
        assert abs(m.angle_diff_deg(5.0, 184.0, period=180.0)) == pytest.approx(1.0)

    def test_line_angle_error(self):
        assert m.line_angle_error_deg(0.0, 179.0) == pytest.approx(1.0)
        assert m.line_angle_error_deg(30.0, 30.0) == 0.0

    def test_endpoint_error_direct(self):
        e = m.line_endpoint_error((0, 0), (10, 0), (0, 0), (10, 0))
        assert e == 0.0

    def test_endpoint_error_swapped(self):
        # undirected segment: swapped endpoints are the same segment
        e = m.line_endpoint_error((10, 0), (0, 0), (0, 0), (10, 0))
        assert e == 0.0

    def test_bad_period_raises(self):
        with pytest.raises(ValueError):
            m.angle_diff_deg(1.0, 2.0, period=0.0)


# ---------------------------------------------------------------------------
# colour / image similarity
# ---------------------------------------------------------------------------

class TestColorAndImage:
    def test_delta_e_identical(self):
        assert m.color_delta_e76((120, 40, 200), (120, 40, 200)) == pytest.approx(0.0)

    def test_delta_e_black_white(self):
        d = m.color_delta_e76((0, 0, 0), (255, 255, 255))
        assert d == pytest.approx(100.0, abs=1.0)  # L 0 -> 100

    def test_delta_e_float_input(self):
        assert m.color_delta_e76((1.0, 1.0, 1.0), (255, 255, 255)) == pytest.approx(0.0, abs=1e-6)

    def test_ssim_identical(self):
        rng = np.random.RandomState(0)
        img = (rng.rand(64, 64) * 255).astype(np.uint8)
        assert m.ssim(img, img) == pytest.approx(1.0)

    def test_ssim_shape_mismatch(self):
        with pytest.raises(ValueError):
            m.ssim(np.zeros((10, 10), np.uint8), np.zeros((10, 20), np.uint8))

    def test_pearson_identical(self):
        x = np.arange(20, dtype=float)
        assert m.pearson_correlation(x, x) == pytest.approx(1.0)

    def test_pearson_anticorrelated(self):
        x = np.arange(20, dtype=float)
        assert m.pearson_correlation(x, -x) == pytest.approx(-1.0)

    def test_pearson_constant_is_zero(self):
        assert m.pearson_correlation(np.ones(10), np.arange(10.0)) == 0.0

    def test_spearman_monotonic(self):
        x = np.arange(15, dtype=float)
        assert m.spearman_correlation(x, x ** 2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# detection / calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_ece_perfect(self):
        conf = [0.9] * 10
        correct = [True] * 9 + [False]
        assert m.expected_calibration_error(conf, correct, n_bins=10) == pytest.approx(0.0)

    def test_ece_known(self):
        # all predictions in the 0.4-0.5 bin, all wrong
        conf = [0.45] * 8
        correct = [False] * 8
        assert m.expected_calibration_error(conf, correct, n_bins=10) == pytest.approx(0.45)

    def test_ece_confidence_one_in_last_bin(self):
        assert m.expected_calibration_error([1.0], [True]) == pytest.approx(0.0)

    def test_ece_validates_input(self):
        with pytest.raises(ValueError):
            m.expected_calibration_error([1.2], [True])
        with pytest.raises(ValueError):
            m.expected_calibration_error([], [])
        with pytest.raises(ValueError):
            m.expected_calibration_error([0.5], [True, False])

    def test_brier_perfect(self):
        assert m.brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)

    def test_brier_known(self):
        assert m.brier_score([0.5], [True]) == pytest.approx(0.25)

    def test_selective_accuracy(self):
        r = m.selective_accuracy([0.9, 0.8, 0.2, 0.1],
                                 [True, False, True, True], 0.5)
        assert r["coverage"] == pytest.approx(0.5)
        assert r["accuracy"] == pytest.approx(0.5)

    def test_prf_counts(self):
        r = m.precision_recall_f1_counts(tp=8, fp=2, fn=2)
        assert r["precision"] == pytest.approx(0.8)
        assert r["recall"] == pytest.approx(0.8)

    def test_prf_zero_divisions(self):
        r = m.precision_recall_f1_counts(tp=0, fp=0, fn=0)
        assert r["precision"] == 1.0 and r["recall"] == 1.0

    def test_match_sets(self):
        r = m.match_sets(["a", "b", "x"], ["a", "b", "c"])
        assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 1

    def test_match_sets_with_key(self):
        r = m.match_sets(["Tyre", "RIM"], ["tyre", "rim"],
                         key=lambda s: s.lower())
        assert r["tp"] == 2


# ---------------------------------------------------------------------------
# depth / normals
# ---------------------------------------------------------------------------

class TestDepthNormals:
    def test_abs_rel_zero(self):
        d = np.full((10, 10), 2.0)
        assert m.depth_abs_rel_error(d, d) == pytest.approx(0.0)

    def test_abs_rel_known(self):
        ref = np.full((10, 10), 2.0)
        pred = np.full((10, 10), 3.0)
        assert m.depth_abs_rel_error(pred, ref) == pytest.approx(0.5)

    def test_abs_rel_masked(self):
        ref = np.ones((4, 4))
        pred = np.ones((4, 4)) * 2.0
        mask = np.zeros((4, 4), bool)
        mask[0, 0] = True
        pred[0, 0] = 1.0
        assert m.depth_abs_rel_error(pred, ref, mask=mask) == pytest.approx(0.0)

    def test_normals_identical(self):
        n = np.zeros((5, 5, 3))
        n[..., 2] = 1.0
        r = m.normals_angular_error_deg(n, n)
        assert r["mean_deg"] == pytest.approx(0.0)
        assert r["pct_below_11_25"] == 1.0

    def test_normals_90deg(self):
        a = np.zeros((4, 4, 3)); a[..., 0] = 1.0
        b = np.zeros((4, 4, 3)); b[..., 1] = 1.0
        r = m.normals_angular_error_deg(a, b)
        assert r["mean_deg"] == pytest.approx(90.0)
        assert r["pct_below_30"] == 0.0

    def test_normals_zero_vector_ignored(self):
        a = np.zeros((2, 2, 3)); a[..., 2] = 1.0; a[0, 0] = 0.0
        b = np.zeros((2, 2, 3)); b[..., 2] = 1.0
        r = m.normals_angular_error_deg(a, b)
        assert r["mean_deg"] == pytest.approx(0.0)
