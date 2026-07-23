"""Compare generated major-part materials with synthetic reference GLBs."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from evals.e2e.run_e2e import _labels_match
from evals.metrics import color_delta_e76


def read_glb_json(path: Path) -> Dict:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("invalid GLB: %s" % path)
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A or 20 + json_length > len(data):
        raise ValueError("GLB JSON chunk invalid: %s" % path)
    return json.loads(data[20:20 + json_length].decode("utf-8"))


def material_class(name: str) -> str:
    value = name.lower()
    for token, result in (
            ("rubber", "rubber"), ("steel", "metal"), ("metal", "metal"),
            ("plastic", "plastic"), ("wood", "wood"),
            ("glass", "glass"), ("ceramic", "ceramic"),
            ("fabric", "fabric")):
        if token in value:
            return result
    return "unknown"


def node_materials(gltf: Dict) -> List[Dict]:
    materials = gltf.get("materials") or []
    meshes = gltf.get("meshes") or []
    result = []
    for node in gltf.get("nodes") or []:
        mesh_index = node.get("mesh")
        if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
            continue
        indices = [primitive.get("material")
                   for primitive in meshes[mesh_index].get("primitives", [])]
        indices = [index for index in indices
                   if isinstance(index, int) and 0 <= index < len(materials)]
        if not indices:
            continue
        material = materials[indices[0]]
        pbr = material.get("pbrMetallicRoughness") or {}
        color = list(pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0]))[:3]
        result.append({
            "node_name": str(node.get("name", "")),
            "material_name": str(material.get("name", "")),
            "material_class": material_class(str(material.get("name", ""))),
            "base_color": [float(value) for value in color],
            "metallic": float(pbr.get("metallicFactor", 1.0)),
            "roughness": float(pbr.get("roughnessFactor", 1.0)),
        })
    return result


def _match_material(entries: List[Dict], part: Dict) -> Optional[Dict]:
    labels = [str(part.get("label", "")), str(part.get("id", ""))]
    matches = [entry for entry in entries if any(
        _labels_match(label, entry["node_name"]) for label in labels if label)]
    return sorted(matches, key=lambda item: item["node_name"])[0] if matches else None


def evaluate_case(case_dir: Path, project_dir: Path) -> Dict:
    parts = json.loads((case_dir / "parts.json").read_text())
    reference = node_materials(read_glb_json(case_dir / "reference.glb"))
    generated = node_materials(read_glb_json(
        project_dir / "blender" / "model.glb"))
    rows = []
    for part in parts.get("parts", []):
        if not part.get("major"):
            continue
        expected = _match_material(reference, part)
        predicted = _match_material(generated, part)
        row = {
            "part_id": part.get("id"), "label": part.get("label"),
            "expected": expected, "predicted": predicted,
            "assignment_present": predicted is not None,
            "class_correct": False, "color_delta_e76": None,
            "metallic_absolute_error": None,
            "roughness_absolute_error": None,
        }
        if expected is not None and predicted is not None:
            row["class_correct"] = (
                expected["material_class"] == predicted["material_class"])
            row["color_delta_e76"] = color_delta_e76(
                expected["base_color"], predicted["base_color"])
            row["metallic_absolute_error"] = abs(
                expected["metallic"] - predicted["metallic"])
            row["roughness_absolute_error"] = abs(
                expected["roughness"] - predicted["roughness"])
        rows.append(row)
    return {"case_id": case_dir.name, "major_parts": rows}


def summarize(cases: List[Dict]) -> Dict:
    rows = [row for case in cases for row in case["major_parts"]]
    comparable = [row for row in rows if row["expected"] is not None]
    paired = [row for row in comparable if row["predicted"] is not None]

    def median(key: str) -> Optional[float]:
        values = [float(row[key]) for row in paired if row[key] is not None]
        return float(np.median(values)) if values else None

    return {
        "case_count": len(cases), "major_part_count": len(comparable),
        "major_part_material_assignment_accuracy": (
            len(paired) / len(comparable) if comparable else 0.0),
        "material_class_accuracy": (
            sum(row["class_correct"] for row in comparable) / len(comparable)
            if comparable else 0.0),
        "median_color_delta_e76": median("color_delta_e76"),
        "median_metallic_absolute_error": median("metallic_absolute_error"),
        "median_roughness_absolute_error": median("roughness_absolute_error"),
    }


def markdown(result: Dict) -> str:
    summary = result["aggregate"]
    lines = ["# Eval 21 Material Summary", "", "| Case | Assigned | Class correct | Major parts |",
             "| --- | ---: | ---: | ---: |"]
    for case in result["cases"]:
        rows = [row for row in case["major_parts"] if row["expected"] is not None]
        lines.append("| `%s` | %d | %d | %d |" % (
            case["case_id"], sum(row["assignment_present"] for row in rows),
            sum(row["class_correct"] for row in rows), len(rows)))
    lines += ["", "- major-part assignment accuracy: %.3f" % summary[
        "major_part_material_assignment_accuracy"],
        "- material-class accuracy: %.3f" % summary["material_class_accuracy"],
        "- median color Delta E76: %.3f" % summary["median_color_delta_e76"],
        "- median metallic absolute error: %.3f" % summary[
            "median_metallic_absolute_error"],
        "- median roughness absolute error: %.3f" % summary[
            "median_roughness_absolute_error"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/benchmark/dataset")
    parser.add_argument("--projects-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dataset, projects = Path(args.dataset), Path(args.projects_root)
    cases = [evaluate_case(case, projects / case.name)
             for case in sorted(dataset.iterdir())
             if case.is_dir() and (case / "parts.json").is_file()
             and (projects / case.name / "blender" / "model.glb").is_file()]
    result = {"cases": cases, "aggregate": summarize(cases)}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    output.with_suffix(".md").write_text(markdown(result))
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
