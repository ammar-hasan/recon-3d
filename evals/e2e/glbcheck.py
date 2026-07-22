"""Minimal, self-contained GLB (binary glTF) structural validator.

No external dependencies. Checks:
- 12-byte header: magic 'glTF', version 2, declared length == file size
- chunk 0 is a JSON chunk that parses
- the JSON declares at least one mesh and one node
- chunk lengths stay inside the file

This is a structural check, not a full glTF spec validation; it catches
truncated, corrupt or empty exports (EVAL.md Eval 25).
"""
from __future__ import annotations

import json
import struct
from typing import Any, Dict, List

GLB_MAGIC = b"glTF"
CHUNK_JSON = 0x4E4F534A  # 'JSON'
CHUNK_BIN = 0x004E4942   # 'BIN\0'


def validate_glb(path: str) -> Dict[str, Any]:
    """Validate a GLB file; returns {valid, errors, mesh_count, node_count, ...}."""
    errors: List[str] = []
    result: Dict[str, Any] = {
        "valid": False, "errors": errors, "path": path,
        "mesh_count": 0, "node_count": 0, "material_count": 0,
        "file_size": 0,
    }
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        errors.append("cannot read file: %s" % exc)
        return result

    result["file_size"] = len(data)
    if len(data) < 20:
        errors.append("file too small for a GLB header + one chunk")
        return result

    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC:
        errors.append("bad magic %r (expected b'glTF')" % (magic,))
        return result
    if version != 2:
        errors.append("unsupported glTF version %d (expected 2)" % version)
    if total_length != len(data):
        errors.append("declared length %d != file size %d"
                      % (total_length, len(data)))

    # chunk 0 must be JSON
    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != CHUNK_JSON:
        errors.append("first chunk is not JSON (type 0x%08x)" % json_type)
        return result
    if 20 + json_len > len(data):
        errors.append("JSON chunk overruns file")
        return result
    try:
        gltf = json.loads(data[20:20 + json_len].decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        errors.append("JSON chunk does not parse: %s" % exc)
        return result

    meshes = gltf.get("meshes") or []
    nodes = gltf.get("nodes") or []
    materials = gltf.get("materials") or []
    result["mesh_count"] = len(meshes)
    result["node_count"] = len(nodes)
    result["material_count"] = len(materials)
    if not meshes:
        errors.append("no meshes declared")
    if not nodes:
        errors.append("no nodes declared")
    accessors = gltf.get("accessors") or []
    if meshes and not accessors:
        errors.append("meshes present but no accessors")

    # remaining chunks must stay inside the file
    offset = 20 + json_len
    while offset < len(data):
        if offset + 8 > len(data):
            errors.append("truncated chunk header at %d" % offset)
            break
        clen, ctype = struct.unpack_from("<II", data, offset)
        if offset + 8 + clen > len(data):
            errors.append("chunk at %d overruns file" % offset)
            break
        offset += 8 + clen

    result["valid"] = not errors
    return result
