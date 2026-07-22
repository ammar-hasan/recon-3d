"""Level A unit tests for the e2e helper modules (GLB validator, safety
scanner, SVG rasteriser, part matching)."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from evals.e2e.glbcheck import validate_glb
from evals.e2e.run_e2e import (_case_label, _labels_match, major_part_recall,
                               rasterize_cleaned_paths,
                               rasterize_svg_silhouette, safety_scan,
                               score_blender, score_camera,
                               score_vectorization)
from evals import metrics as m


def _write_glb(path: Path, gltf: dict, magic=b"glTF", version=2,
               fudge_length=0) -> None:
    payload = json.dumps(gltf).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total = 12 + 8 + len(payload) + fudge_length
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", magic, version, total))
        f.write(struct.pack("<II", len(payload), 0x4E4F534A))
        f.write(payload)


class TestGlbValidator:
    def test_valid_glb(self, tmp_path):
        p = tmp_path / "ok.glb"
        _write_glb(p, {"asset": {"version": "2.0"},
                       "meshes": [{"primitives": []}],
                       "nodes": [{"mesh": 0}],
                       "accessors": [{}]})
        r = validate_glb(str(p))
        assert r["valid"] is True
        assert r["mesh_count"] == 1
        assert r["node_count"] == 1

    def test_bad_magic(self, tmp_path):
        p = tmp_path / "bad.glb"
        _write_glb(p, {"meshes": [{}], "nodes": [{}]}, magic=b"NOPE")
        r = validate_glb(str(p))
        assert r["valid"] is False
        assert any("magic" in e for e in r["errors"])

    def test_truncated(self, tmp_path):
        p = tmp_path / "trunc.glb"
        p.write_bytes(b"glTF\x02\x00")
        r = validate_glb(str(p))
        assert r["valid"] is False

    def test_no_meshes(self, tmp_path):
        p = tmp_path / "empty.glb"
        _write_glb(p, {"asset": {"version": "2.0"}, "meshes": [], "nodes": []})
        r = validate_glb(str(p))
        assert r["valid"] is False
        assert any("meshes" in e for e in r["errors"])

    def test_corrupt_json(self, tmp_path):
        p = tmp_path / "corrupt.glb"
        payload = b"{not json!!"
        payload += b" " * ((4 - len(payload) % 4) % 4)
        with open(p, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload)))
            f.write(struct.pack("<II", len(payload), 0x4E4F534A))
            f.write(payload)
        r = validate_glb(str(p))
        assert r["valid"] is False
        assert any("JSON" in e for e in r["errors"])

    def test_missing_file(self, tmp_path):
        r = validate_glb(str(tmp_path / "nope.glb"))
        assert r["valid"] is False


class TestSafetyScan:
    def test_clean_script(self, tmp_path):
        p = tmp_path / "build_model.py"
        p.write_text("import bpy\nbpy.ops.mesh.primitive_cube_add()\n")
        assert safety_scan(p, tmp_path) == []

    def test_subprocess_import_flagged(self, tmp_path):
        p = tmp_path / "evil.py"
        p.write_text("import subprocess\nsubprocess.run(['ls'])\n")
        v = safety_scan(p, tmp_path)
        assert any("subprocess" in x for x in v)

    def test_os_system_flagged(self, tmp_path):
        p = tmp_path / "evil.py"
        p.write_text("import os\nos.system('rm -rf /')\n")
        v = safety_scan(p, tmp_path)
        assert any("os.system" in x for x in v)

    def test_eval_flagged(self, tmp_path):
        p = tmp_path / "evil.py"
        p.write_text("eval('1+1')\n")
        assert any("eval" in x for x in safety_scan(p, tmp_path))

    def test_open_outside_project_flagged(self, tmp_path):
        p = tmp_path / "evil.py"
        p.write_text("open('/etc/passwd')\n")
        v = safety_scan(p, tmp_path)
        assert any("outside project" in x for x in v)

    def test_open_inside_project_allowed(self, tmp_path):
        p = tmp_path / "ok.py"
        p.write_text("open(%r)\n" % str(tmp_path / "out.txt"))
        assert safety_scan(p, tmp_path) == []

    def test_missing_script_is_not_a_safety_violation(self, tmp_path):
        result = score_blender(tmp_path, blender=None)
        assert result["safety_scan_available"] is False
        assert result["safety_violations"] == []
        assert result["execution_success"] is False


class TestPartMatching:
    def test_token_match(self):
        assert _labels_match("mug_body", "body mug_body")
        assert _labels_match("handle", "Handle_01")
        assert not _labels_match("tyre", "hub")

    def test_synonyms(self):
        assert _labels_match("tyre", "part_outer_ring")
        assert _labels_match("tyre", "tire")
        assert _labels_match("rim", "wheel_ring_0")
        assert _labels_match("hub", "center")
        assert _labels_match("hub", "centre_disk")
        assert _labels_match("spokes", "arms")
        assert _labels_match("spokes", "spoke_01")
        # synonyms must not bridge distinct parts
        assert not _labels_match("tyre", "rim")
        assert not _labels_match("rim", "hub")

    def test_major_part_recall(self):
        gt = {"parts": [
            {"id": "body", "label": "mug_body", "major": True},
            {"id": "handle", "label": "handle", "major": True},
            {"id": "foot", "label": "foot", "major": False}]}
        pred = [{"id": "body", "part_class": "mug_body"},
                {"id": "thing", "part_class": "handle"}]
        r = major_part_recall(gt, pred)
        assert r["recall"] == pytest.approx(1.0)
        r2 = major_part_recall(gt, [{"id": "body", "part_class": "mug_body"}])
        assert r2["recall"] == pytest.approx(0.5)
        assert r2["missing"] == ["handle"]


class TestCaseLabel:
    def test_strips_instance_number(self):
        case = {"case_id": "wheel_01", "parts": {"object_id": "wheel_01"}}
        assert _case_label(case) == "wheel"
        case = {"case_id": "pipe_elbow_01", "parts": {"object_id": "pipe_elbow_01"}}
        assert _case_label(case) == "pipe_elbow"

    def test_falls_back_to_case_id(self):
        assert _case_label({"case_id": "gear_02", "parts": {}}) == "gear"


class TestSvgRasterize:
    def test_cleaned_paths_drop_canvas_background_and_carve_hole(self, tmp_path):
        cleaned = tmp_path / "cleaned_silhouette.json"
        cleaned.write_text(json.dumps({"paths": [
            {"points": [[0, 0], [1, 0], [1, 1], [0, 1]],
             "is_hole": False},
            {"points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
             "is_hole": False},
            {"points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
             "is_hole": True},
        ]}))
        mask = rasterize_cleaned_paths(cleaned, (100, 100))
        assert mask is not None
        assert mask[20, 20] == 255
        assert mask[50, 50] == 0
        assert mask[5, 5] == 0

    def test_square_svg(self, tmp_path):
        svg = tmp_path / "sil.svg"
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
            '<path d="M 10 10 L 90 10 L 90 90 L 10 90 Z"/></svg>')
        mask = rasterize_svg_silhouette(svg, (100, 100))
        assert mask is not None
        ref = np.zeros((100, 100), np.uint8)
        ref[10:91, 10:91] = 255
        assert m.mask_iou(mask, ref) > 0.95

    def test_curves_svg(self, tmp_path):
        svg = tmp_path / "curved.svg"
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
            '<path d="M 20 50 C 20 20 80 20 80 50 C 80 80 20 80 20 50 Z"/>'
            "</svg>")
        mask = rasterize_svg_silhouette(svg, (100, 100))
        assert mask is not None
        assert (mask > 0).sum() > 500

    def test_garbage_returns_none(self, tmp_path):
        svg = tmp_path / "bad.svg"
        svg.write_text("this is not svg")
        assert rasterize_svg_silhouette(svg, (100, 100)) is None


class TestCameraScoring:
    def test_projection_and_focal_are_graded(self, tmp_path):
        geom = tmp_path / "geometry"
        geom.mkdir()
        (geom / "construction_plan.yaml").write_text(
            "camera:\n"
            "  projection: perspective\n"
            "  focal_length_px:\n"
            "    value: 900.0\n"
            "    confidence: 0.3\n"
            "  rotation_euler_deg:\n"
            "    value: [0.0, 0.0, 0.0]\n"
            "    confidence: 0.3\n")
        result = score_camera(tmp_path, {
            "projection": "perspective", "focal_length_px": 900.0})
        assert result["available"] is True
        assert result["projection_match"] is True
        assert result["focal_rel_error"] == pytest.approx(0.0)
        assert result["score"] == pytest.approx(1.0)


class TestVectorizationScoring:
    def test_uses_cleaned_paths_in_crop_space(self, tmp_path):
        seg = tmp_path / "segmentation"
        traces = tmp_path / "traces"
        seg.mkdir()
        traces.mkdir()
        # Original 100x100 mask occupying 20:80. The crop maps it to 10:90.
        (seg / "crop_metadata.json").write_text(json.dumps({
            "scale": 4.0 / 3.0,
            "offset": [12.5, 12.5],
            "output_size": [100, 100],
        }))
        (traces / "cleaned_silhouette.json").write_text(json.dumps({
            "paths": [{
                "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "is_hole": False,
            }]
        }))
        gt = np.zeros((100, 100), np.uint8)
        gt[20:81, 20:81] = 255
        result = score_vectorization(tmp_path, gt)
        assert result["available"] is True
        assert result["source"] == "cleaned_silhouette_json"
        assert result["reference_space"] == "crop"
        assert result["silhouette_svg_iou"] > 0.95
