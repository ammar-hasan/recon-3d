from evals.downstream.run_edit_tasks import apply_task
from recon3d.schemas import ConstructionPlan, OperatorCategory, PlanPart


def _extrude(part_id, parent=None):
    return PlanPart(
        id=part_id, parent=parent, operator=OperatorCategory.EXTRUDE,
        profile={"type": "polygon",
                 "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                 "closed": True}, depth=0.1)


def test_component_and_material_edits_are_declarative():
    plan = ConstructionPlan(object_id="box", parts=[_extrude("body")])
    target, before, after = apply_task(plan, "resize_component")
    assert target == "body"
    assert before["scale"] == [1.0, 1.0, 1.0]
    assert after["scale"] == [1.2, 1.0, 1.0]
    assert plan.parts[0].transform["scale"] == after["scale"]

    _, before, after = apply_task(plan, "replace_material")
    assert before["material"]["material_class"] == "plastic"
    assert after["material"]["material_class"] == "wood"


def test_repetition_and_joint_edits_change_named_parameters():
    prototype = _extrude("spoke")
    array = PlanPart(
        id="spokes", parent=None, operator=OperatorCategory.RADIAL_ARRAY,
        source_part="spoke", count=5, angle_degrees=360.0,
        axis={"origin": [0, 0, 0], "direction": [0, 0, 1]})
    plan = ConstructionPlan(object_id="wheel", parts=[prototype, array])
    _, before, after = apply_task(plan, "alter_repetition_count")
    assert before["count"] == 5 and after["count"] == 6

    chair = ConstructionPlan(
        object_id="chair", parts=[_extrude("seat"), _extrude("back", "seat")])
    target, _, after = apply_task(chair, "move_joint")
    assert target == "back"
    assert after["location"] == [0.12, 0.0, 0.0]


def test_profile_and_animation_edits_preserve_part_structure():
    body = PlanPart(
        id="part_wheel_0", operator=OperatorCategory.REVOLVE,
        profile={"type": "polyline", "points": [[0.2, -1], [0.4, 0], [0.2, 1]],
                 "closed": False},
        axis={"origin": [0, 0, 0], "direction": [0, 1, 0]})
    cap = body.model_copy(deep=True)
    cap.id = "part_cap_0"
    cap.parent = body.id
    plan = ConstructionPlan(object_id="wheel", parts=[body, cap])

    _, before, after = apply_task(plan, "change_profile")
    assert before["profile_points"] != after["profile_points"]
    _, _, rotation = apply_task(plan, "rotate_wheel")
    assert rotation["rotation_deg"] == [0.0, 0.0, 30.0]
    target, _, rotation = apply_task(plan, "open_lid")
    assert target == "part_cap_0"
    assert rotation["rotation_deg"] == [35.0, 0.0, 0.0]
