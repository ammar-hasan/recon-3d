"""Level B stage eval: construction plan validity (EVAL.md Eval 14).

Builds hand-crafted valid and invalid plans and checks that
recon3d.construction_plan.validate_plan accepts the valid ones and catches
each class of invalidity (schema, references, cycles, non-finite values,
impossible parameters).
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("recon3d.construction_plan")

from recon3d import schemas as S  # noqa: E402
from recon3d.construction_plan import validate_plan  # noqa: E402

OC = S.OperatorCategory


def _revolve_part(pid, parent=None):
    return S.PlanPart(
        id=pid, operator=OC.REVOLVE, parent=parent,
        axis={"origin": [0, 0, 0], "direction": [0, 1, 0]},
        profile={"type": "polyline",
                 "points": [[0.05, 0.0], [0.2, 0.1], [0.2, 0.4], [0.05, 0.5]],
                 "closed": True})


def _extrude_part(pid, parent=None):
    return S.PlanPart(
        id=pid, operator=OC.EXTRUDE, parent=parent,
        profile={"type": "polyline",
                 "points": [[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]],
                 "closed": True},
        depth=0.05)


def _valid_plan():
    return S.ConstructionPlan(
        object_id="wheel_like",
        parts=[_revolve_part("tyre"),
               _revolve_part("rim"),
               _extrude_part("spoke", parent="rim"),
               S.PlanPart(id="spoke_array", operator=OC.RADIAL_ARRAY,
                          source_part="spoke", count=5, angle_degrees=72.0)])


class TestValidPlans:
    def test_valid_plan_passes(self):
        assert validate_plan(_valid_plan()) == []

    def test_schema_round_trip_still_valid(self, tmp_path):
        plan = _valid_plan()
        p = S.SchemaIO.save_yaml(plan, tmp_path / "plan.yaml")
        loaded = S.SchemaIO.load_yaml(S.ConstructionPlan, p)
        assert validate_plan(loaded) == []


class TestReferenceIntegrity:
    def test_missing_parent_caught(self):
        plan = _valid_plan()
        plan.parts[2].parent = "does_not_exist"
        errors = validate_plan(plan)
        assert any("parent" in e for e in errors)

    def test_missing_array_source_caught(self):
        plan = _valid_plan()
        plan.parts[3].source_part = "ghost"
        errors = validate_plan(plan)
        assert any("source_part" in e for e in errors)

    def test_parent_cycle_caught(self):
        plan = _valid_plan()
        plan.parts[0].parent = "spoke"
        plan.parts[2].parent = "tyre"
        errors = validate_plan(plan)
        assert any("cycle" in e for e in errors)

    def test_duplicate_ids_caught(self):
        plan = _valid_plan()
        plan.parts.append(_revolve_part("tyre"))
        errors = validate_plan(plan)
        assert any("duplicate" in e for e in errors)


class TestImpossibleParameters:
    def test_non_finite_transform_caught(self):
        plan = _valid_plan()
        plan.parts[0].transform = {"location": [0.0, math.inf, 0.0]}
        errors = validate_plan(plan)
        assert any("transform" in e for e in errors)

    def test_zero_length_axis_caught(self):
        plan = _valid_plan()
        plan.parts[0].axis = {"origin": [0, 0, 0], "direction": [0, 0, 0]}
        errors = validate_plan(plan)
        assert any("zero-length axis" in e for e in errors)

    def test_negative_revolve_radius_caught(self):
        plan = _valid_plan()
        plan.parts[0].profile["points"][0][0] = -0.1
        errors = validate_plan(plan)
        assert any("negative radius" in e for e in errors)

    def test_extrude_without_depth_caught(self):
        part = _extrude_part("plate")
        part.depth = None
        plan = S.ConstructionPlan(object_id="p", parts=[part])
        errors = validate_plan(plan)
        assert any("extrude depth" in e for e in errors)

    def test_radial_array_bad_count_caught(self):
        plan = _valid_plan()
        plan.parts[3].count = 1
        errors = validate_plan(plan)
        assert any("radial_array count" in e for e in errors)

    def test_material_out_of_range_caught(self):
        plan = _valid_plan()
        plan.parts[0].material.roughness = 1.7
        errors = validate_plan(plan)
        assert any("material" in e for e in errors)
