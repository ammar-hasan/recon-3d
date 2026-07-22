"""Stage 15 execution: sandboxed Blender runner.

Runs a generated Blender script with:
- a static safety scan before execution (banned tokens, suspicious writes);
- a runtime sandbox (guard module monkeypatches os.system / subprocess /
  socket before the target script is exec'd);
- cwd=project_dir, proxy env scrubbed, timeout, output captured to
  <project_dir>/validation/blender_run.log;
- the blender_manifest.json written by the script parsed back into a
  BlenderManifest.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig
from .schemas import BlenderManifest


class ScriptSafetyError(ValueError):
    """Raised when a script fails the static safety scan."""


# Tokens that must never appear in a generated/target script.
BANNED_TOKENS = [
    "os.system",
    "os.popen",
    "subprocess",
    "socket",
    "urllib",
    "requests",
    "shutil.rmtree",
    "eval(",
    "exec(",
    "__import__",
    "ctypes",
    "pty.",
    "os.remove",
    "os.unlink",
]

# Absolute-path write heuristic: open("/abs/path", any write mode) where the
# path is not inside the project dir.
_OPEN_WRITE_RE = re.compile(
    r"""open\(\s*(['"])(?P<path>/[^'"]+)(['"])\s*,\s*(['"])(?P<mode>[wax+bt]*)""")

# Runtime guard: injected into Blender via --python <wrapper>. It patches
# dangerous calls to raise, then executes the target script. The wrapper is
# written by us and is NOT subject to the static scan.
_GUARD_SOURCE = '''\
"""recon3d runtime sandbox guard (written by recon3d.runner)."""
import os
import sys


def _blocked(name):
    def _raise(*args, **kwargs):
        raise RuntimeError("recon3d sandbox: %s is blocked" % name)
    return _raise


os.system = _blocked("os.system")
os.popen = _blocked("os.popen")
try:
    import subprocess as _sp
    _sp.Popen = _blocked("subprocess.Popen")
    _sp.run = _blocked("subprocess.run")
    _sp.call = _blocked("subprocess.call")
    _sp.check_call = _blocked("subprocess.check_call")
    _sp.check_output = _blocked("subprocess.check_output")
except Exception:
    pass
try:
    import socket as _so
    _so.socket = _blocked("socket.socket")
    _so.create_connection = _blocked("socket.create_connection")
except Exception:
    pass

_target = sys.argv[sys.argv.index("--") + 1]
sys.argv = [_target] + sys.argv[sys.argv.index("--") + 2:]
with open(_target, "r") as _fh:
    _code = compile(_fh.read(), _target, "exec")
exec(_code, {"__name__": "__main__", "__file__": _target})
'''


def scan_script_safety(script_text: str, project_dir: str) -> List[str]:
    """Return a list of safety violations found in the script text."""
    violations = []
    for token in BANNED_TOKENS:
        if token in script_text:
            violations.append("banned token: %s" % token)
    proj = os.path.abspath(project_dir)
    for m in _OPEN_WRITE_RE.finditer(script_text):
        path = os.path.abspath(m.group("path"))
        if not (path == proj or path.startswith(proj + os.sep)):
            violations.append("write outside project dir: %s" % m.group("path"))
    return violations


def _scrub_env(env: dict) -> dict:
    out = dict(env)
    for key in list(out):
        if "proxy" in key.lower():
            out.pop(key)
    # keep Blender from phoning home / loading user addons
    out["BLENDER_USER_CONFIG"] = os.devnull
    out["BLENDER_USER_SCRIPTS"] = os.devnull
    return out


def run_blender(script_path: str, project_dir: str, cfg: PipelineConfig,
                extra_args: Optional[List[str]] = None) -> BlenderManifest:
    """Run a Blender script in the sandbox and parse the manifest it writes."""
    project = Path(project_dir).resolve()
    script = Path(script_path).resolve()
    blender_dir = project / "blender"
    validation_dir = project / "validation"
    blender_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    # build_model.py writes the canonical blender_manifest.json; any other
    # script (e.g. render_validation.py) writes <stem>_manifest.json so the
    # build manifest is never clobbered.
    manifest_name = ("blender_manifest.json" if script.stem == "build_model"
                     else "%s_manifest.json" % script.stem)
    manifest_path = blender_dir / manifest_name
    blend_path = blender_dir / "scene.blend"
    glb_path = blender_dir / "model.glb"
    log_path = validation_dir / "blender_run.log"

    def _fail(message: str, log: str = "") -> BlenderManifest:
        return BlenderManifest(
            blend_path=str(blend_path),
            glb_path=str(glb_path) if glb_path.exists() else None,
            script_path=str(script),
            objects=[],
            collections=[],
            execution_log=log[-4000:],
            success=False,
            errors=[message],
        )

    # ---- static safety scan -------------------------------------------
    text = script.read_text()
    violations = scan_script_safety(text, str(project))
    if violations:
        raise ScriptSafetyError(
            "script %s rejected by safety scan: %s" % (script, "; ".join(violations)))

    # ---- runtime sandbox ----------------------------------------------
    guard_path = blender_dir / "_sandbox_guard.py"
    guard_path.write_text(_GUARD_SOURCE)

    cmd = [
        cfg.blender.blender_bin,
        "--background",
        "--factory-startup",
        "--python", str(guard_path),
        "--",
        str(script),
        str(project),
        str(blend_path),
        str(glb_path),
        str(manifest_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    stdout, stderr = "", ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project),
            env=_scrub_env(os.environ),
            capture_output=True,
            text=True,
            timeout=cfg.blender.timeout_seconds,
        )
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        returncode = None
        stderr += "\nrecon3d: blender run timed out after %ds" % cfg.blender.timeout_seconds

    with open(log_path, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n[stdout]\n" + stdout +
                 "\n[stderr]\n" + stderr)

    if returncode != 0:
        excerpt = (stderr or stdout)[-2000:]
        return _fail("blender exited with code %s: %s" % (returncode, excerpt),
                     stdout + "\n" + stderr)

    if not manifest_path.exists():
        return _fail("blender run finished but wrote no manifest at %s; "
                     "stderr tail: %s" % (manifest_path, stderr[-1500:]),
                     stdout + "\n" + stderr)

    try:
        data = json.loads(manifest_path.read_text())
    except Exception as exc:
        return _fail("manifest is not valid JSON: %s" % exc, stdout + "\n" + stderr)

    manifest = BlenderManifest(
        blend_path=data.get("blend_path", str(blend_path)),
        glb_path=data.get("glb_path"),
        script_path=str(script),
        objects=data.get("objects", []),
        collections=data.get("collections", []),
        execution_log=(stdout + "\n" + stderr)[-4000:],
        blender_version=data.get("blender_version", ""),
        success=bool(data.get("success", False)) and blend_path.exists(),
        errors=list(data.get("errors", [])),
    )
    if blend_path.exists() and not manifest.blend_path:
        manifest.blend_path = str(blend_path)
    if manifest.success and glb_path.exists() and not manifest.glb_path:
        manifest.glb_path = str(glb_path)
    return manifest
