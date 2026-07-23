from evals.materials.run_material_eval import material_class, node_materials, summarize


def test_material_class_from_exported_names():
    assert material_class("DarkRubber") == "rubber"
    assert material_class("mat_metal_03") == "metal"
    assert material_class("GlassGreen") == "glass"


def test_node_materials_extracts_pbr_assignment():
    gltf = {
        "materials": [{"name": "Steel", "pbrMetallicRoughness": {
            "baseColorFactor": [0.4, 0.5, 0.6, 1.0],
            "metallicFactor": 0.8, "roughnessFactor": 0.2}}],
        "meshes": [{"primitives": [{"material": 0}]}],
        "nodes": [{"name": "rim", "mesh": 0}],
    }
    entry = node_materials(gltf)[0]
    assert entry["material_class"] == "metal"
    assert entry["metallic"] == 0.8


def test_summary_counts_missing_assignments_as_failures():
    expected = {"material_class": "metal"}
    predicted = {"material_class": "metal"}
    cases = [{"major_parts": [
        {"expected": expected, "predicted": predicted,
         "assignment_present": True, "class_correct": True,
         "color_delta_e76": 2.0, "metallic_absolute_error": 0.1,
         "roughness_absolute_error": 0.2},
        {"expected": expected, "predicted": None,
         "assignment_present": False, "class_correct": False,
         "color_delta_e76": None, "metallic_absolute_error": None,
         "roughness_absolute_error": None},
    ]}]
    result = summarize(cases)
    assert result["major_part_material_assignment_accuracy"] == 0.5
    assert result["material_class_accuracy"] == 0.5
