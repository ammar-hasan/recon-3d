"""Level A unit tests for evals.metrics mask/contour functions."""
from __future__ import annotations

import numpy as np
import pytest

from evals import metrics as m


def _rect_mask(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


# ---------------------------------------------------------------------------
# mask_iou
# ---------------------------------------------------------------------------

class TestMaskIoU:
    def test_identical_masks(self):
        a = _rect_mask((100, 100), 10, 10, 50, 50)
        assert m.mask_iou(a, a) == pytest.approx(1.0)

    def test_known_overlap(self):
        a = _rect_mask((100, 100), 0, 0, 40, 40)    # 1600 px
        b = _rect_mask((100, 100), 20, 0, 60, 40)   # 1600 px, overlap 800
        # inter = 20*40 = 800, union = 2400
        assert m.mask_iou(a, b) == pytest.approx(800.0 / 2400.0)

    def test_disjoint(self):
        a = _rect_mask((100, 100), 0, 0, 20, 20)
        b = _rect_mask((100, 100), 60, 60, 90, 90)
        assert m.mask_iou(a, b) == 0.0

    def test_both_empty_is_one(self):
        z = np.zeros((50, 50), np.uint8)
        assert m.mask_iou(z, z) == 1.0

    def test_one_empty_is_zero(self):
        a = _rect_mask((50, 50), 0, 0, 10, 10)
        assert m.mask_iou(a, np.zeros((50, 50), np.uint8)) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            m.mask_iou(np.zeros((10, 10)), np.zeros((10, 20)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            m.mask_iou(np.zeros((10, 10, 3)), np.zeros((10, 10, 3)))


# ---------------------------------------------------------------------------
# boundary f-score
# ---------------------------------------------------------------------------

class TestBoundaryFScore:
    def test_identical(self):
        a = _rect_mask((100, 100), 20, 20, 60, 60)
        assert m.boundary_f_score(a, a, tolerance_px=2.0) == pytest.approx(1.0)

    def test_small_shift_high_score(self):
        a = _rect_mask((100, 100), 20, 20, 60, 60)
        b = _rect_mask((100, 100), 21, 21, 61, 61)
        assert m.boundary_f_score(a, b, tolerance_px=2.0) > 0.9

    def test_large_shift_low_score(self):
        a = _rect_mask((100, 100), 0, 0, 20, 20)
        b = _rect_mask((100, 100), 70, 70, 90, 90)
        assert m.boundary_f_score(a, b, tolerance_px=2.0) < 0.05

    def test_both_empty(self):
        z = np.zeros((50, 50), np.uint8)
        assert m.boundary_f_score(z, z) == 1.0

    def test_one_empty(self):
        a = _rect_mask((50, 50), 5, 5, 25, 25)
        assert m.boundary_f_score(a, np.zeros((50, 50), np.uint8)) == 0.0

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            m.boundary_f_score(np.zeros((5, 5)), np.zeros((5, 5)), -1.0)


# ---------------------------------------------------------------------------
# mask helpers
# ---------------------------------------------------------------------------

class TestMaskHelpers:
    def test_hole_count_none(self):
        a = _rect_mask((100, 100), 10, 10, 60, 60)
        assert m.hole_count(a) == 0

    def test_hole_count_one(self):
        a = _rect_mask((100, 100), 10, 10, 80, 80)
        a[30:50, 30:50] = 0
        assert m.hole_count(a) == 1

    def test_hole_touching_border_not_counted(self):
        a = _rect_mask((100, 100), 10, 10, 80, 80)
        a[30:50, 0:30] = 0  # opens to the image border
        assert m.hole_count(a) == 0

    def test_bbox(self):
        a = _rect_mask((100, 100), 12, 34, 56, 78)
        assert m.mask_bbox(a) == (12, 34, 56, 78)

    def test_bbox_empty(self):
        assert m.mask_bbox(np.zeros((10, 10), np.uint8)) is None

    def test_coverage_full(self):
        a = _rect_mask((100, 100), 10, 10, 50, 50)
        assert m.mask_coverage(a, (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_coverage_half(self):
        a = _rect_mask((100, 100), 0, 0, 100, 100)
        assert m.mask_coverage(a, (0, 0, 50, 100)) == pytest.approx(0.5)

    def test_precision_recall_f1_perfect(self):
        a = _rect_mask((50, 50), 5, 5, 25, 25)
        r = m.mask_precision_recall_f1(a, a)
        assert r["precision"] == r["recall"] == r["f1"] == pytest.approx(1.0)

    def test_precision_recall_f1_known(self):
        ref = _rect_mask((100, 100), 0, 0, 40, 40)     # 1600
        pred = _rect_mask((100, 100), 0, 0, 40, 20)    # 800, all inside ref
        r = m.mask_precision_recall_f1(pred, ref)
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(0.5)
        assert r["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)


# ---------------------------------------------------------------------------
# chamfer distance
# ---------------------------------------------------------------------------

class TestChamfer:
    def test_identical_contours_zero(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
        assert m.chamfer_distance(pts, pts) == pytest.approx(0.0)

    def test_unit_shift(self):
        a = np.array([[0, 0], [10, 0]], float)
        b = a + np.array([1.0, 0.0])
        # symmetric mean NN distance for 2-point sets shifted by 1
        assert m.chamfer_distance(a, b) == pytest.approx(1.0)

    def test_diagonal_shift(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        assert m.chamfer_distance(a, b) == pytest.approx(5.0)

    def test_normalized(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[0.0, 10.0]])
        assert m.chamfer_distance(a, b, normalize_by=100.0) == pytest.approx(0.1)

    def test_bad_normalize_raises(self):
        with pytest.raises(ValueError):
            m.chamfer_distance(np.zeros((1, 2)), np.zeros((1, 2)), normalize_by=0)

    def test_masks_identical_zero(self):
        a = _rect_mask((100, 100), 10, 10, 60, 60)
        assert m.chamfer_distance_masks(a, a) == pytest.approx(0.0)

    def test_masks_empty_pair_zero(self):
        z = np.zeros((50, 50), np.uint8)
        assert m.chamfer_distance_masks(z, z) == 0.0

    def test_masks_one_empty_inf(self):
        a = _rect_mask((50, 50), 5, 5, 25, 25)
        assert m.chamfer_distance_masks(a, np.zeros((50, 50), np.uint8)) == float("inf")

    def test_empty_points_raise(self):
        with pytest.raises(ValueError):
            m.chamfer_distance(np.zeros((0, 2)), np.zeros((1, 2)))


# ---------------------------------------------------------------------------
# rasterization
# ---------------------------------------------------------------------------

class TestRasterize:
    def test_square_area(self):
        poly = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
        mask = m.rasterize_polygon(poly, (100, 100))
        # cv2.fillPoly includes the right/bottom edge: 25..75 px inclusive
        assert (mask > 0).sum() == 51 * 51

    def test_degenerate_polygon_empty(self):
        mask = m.rasterize_polygon(np.array([[0.5, 0.5], [0.6, 0.6]]), (100, 100))
        assert mask.sum() == 0

    def test_bad_size_raises(self):
        with pytest.raises(ValueError):
            m.rasterize_polygon(np.zeros((3, 2)), (0, 100))

    def test_paths_with_hole(self):
        outer = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        hole = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])
        mask = m.rasterize_paths([outer], (100, 100), holes=[hole])
        assert mask[50, 50] == 0          # hole centre carved out
        assert mask[20, 20] == 255        # outer ring filled
        assert m.hole_count(mask) == 1

    def test_polyline(self):
        line = np.array([[0.1, 0.5], [0.9, 0.5]])
        mask = m.polyline_to_mask(line, (100, 100), thickness_px=3)
        assert (mask > 0).sum() > 100


# ---------------------------------------------------------------------------
# coordinate helpers
# ---------------------------------------------------------------------------

class TestCoordinateHelpers:
    def test_round_trip_identity(self):
        to_crop = lambda x, y: (x * 0.5, y * 0.5)
        to_orig = lambda u, v: (u / 0.5, v / 0.5)
        pts = np.array([[3.0, 7.0], [100.0, 250.0]])
        errs = m.round_trip_errors(pts, to_crop, to_orig)
        assert errs.max() < 1e-9

    def test_round_trip_detects_bias(self):
        to_crop = lambda x, y: (x - 10.0, y - 10.0)
        to_orig = lambda u, v: (u + 9.0, v + 9.0)   # 1px systematic bias
        errs = m.round_trip_errors(np.array([[50.0, 50.0]]), to_crop, to_orig)
        assert errs[0] == pytest.approx(np.sqrt(2.0))

    def test_aspect_ratio_error_zero(self):
        assert m.aspect_ratio_error(16, 9, 1600, 900) == pytest.approx(0.0)

    def test_aspect_ratio_error_known(self):
        assert m.aspect_ratio_error(2.0, 1.0, 1.0, 1.0) == pytest.approx(0.5)

    def test_aspect_ratio_invalid(self):
        with pytest.raises(ValueError):
            m.aspect_ratio_error(0, 1, 1, 1)
