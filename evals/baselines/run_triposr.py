"""Run the official TripoSR model as an explicit image-to-mesh baseline.

This adapter is intentionally executed with a separate TripoSR environment:

    /path/to/triposr-python evals/baselines/run_triposr.py \
      --triposr-root /path/to/TripoSR \
      --dataset evals/benchmark/dataset \
      --out projects/triposr_baseline

The project environment does not gain a Torch dependency. Reference masks can
be supplied for synthetic geometry comparisons so segmentation quality is not
silently conflated with reconstruction quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def prepare_reference_masked_image(
    image_path: Path, mask_path: Path, foreground_ratio: float = 0.85,
) -> Image.Image:
    """Match TripoSR's RGBA crop/pad and gray-background preprocessing."""
    if not 0.0 < foreground_ratio <= 1.0:
        raise ValueError("foreground ratio must be in (0, 1]")
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    alpha = np.asarray(Image.open(mask_path).convert("L"))
    if rgb.shape[:2] != alpha.shape:
        raise ValueError("image and mask dimensions differ")
    foreground = np.where(alpha > 0)
    if not len(foreground[0]):
        raise ValueError("reference mask is empty")
    y0, y1 = int(foreground[0].min()), int(foreground[0].max()) + 1
    x0, x1 = int(foreground[1].min()), int(foreground[1].max()) + 1
    rgba = np.dstack((rgb, alpha))[y0:y1, x0:x1]
    size = max(rgba.shape[:2])
    square = np.zeros((size, size, 4), dtype=np.uint8)
    top = (size - rgba.shape[0]) // 2
    left = (size - rgba.shape[1]) // 2
    square[top:top + rgba.shape[0], left:left + rgba.shape[1]] = rgba
    output_size = max(size, int(size / foreground_ratio))
    padded = np.zeros((output_size, output_size, 4), dtype=np.uint8)
    top = (output_size - size) // 2
    left = (output_size - size) // 2
    padded[top:top + size, left:left + size] = square
    values = padded.astype(np.float32) / 255.0
    composited = (values[:, :, :3] * values[:, :, 3:4]
                  + 0.5 * (1.0 - values[:, :, 3:4]))
    return Image.fromarray(np.rint(255.0 * composited).astype(np.uint8))


def _case_paths(dataset: Path, case_ids: Iterable[str]) -> list[Path]:
    requested = set(case_ids)
    cases = sorted(path for path in dataset.iterdir()
                   if path.is_dir() and (path / "input.png").is_file())
    if requested:
        cases = [path for path in cases if path.name in requested]
        missing = requested - {path.name for path in cases}
        if missing:
            raise ValueError("unknown cases: %s" % ", ".join(sorted(missing)))
    if not cases:
        raise ValueError("no benchmark cases selected")
    return cases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str | None:
    head = repository / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text().strip()
    if not value.startswith("ref: "):
        return value or None
    reference = value[5:]
    loose = repository / ".git" / reference
    if loose.is_file():
        return loose.read_text().strip() or None
    packed = repository / ".git" / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                revision, name = line.split(" ", 1)
                if name == reference:
                    return revision
    return None


def run(args: argparse.Namespace) -> dict:
    triposr_root = Path(args.triposr_root).resolve()
    if not (triposr_root / "tsr" / "system.py").is_file():
        raise FileNotFoundError("TripoSR source not found: %s" % triposr_root)
    sys.path.insert(0, str(triposr_root))
    import torch
    from tsr.system import TSR

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    cases = _case_paths(
        Path(args.dataset),
        () if args.cases == "all" else args.cases.split(","))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    load_started = time.monotonic()
    model = TSR.from_pretrained(
        args.model, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    load_seconds = time.monotonic() - load_started

    results = []
    for case in cases:
        started = time.monotonic()
        case_output = output / case.name
        case_output.mkdir(parents=True, exist_ok=True)
        prepared = prepare_reference_masked_image(
            case / "input.png", case / "mask.png", args.foreground_ratio)
        prepared_path = case_output / "prepared_input.png"
        prepared.save(prepared_path)
        with torch.no_grad():
            scene_codes = model([prepared], device=device)
            mesh = model.extract_mesh(
                scene_codes, has_vertex_color=True,
                resolution=args.mc_resolution)[0]
        mesh_path = case_output / "mesh.glb"
        mesh.export(mesh_path)
        result = {
            "case_id": case.name,
            "input_sha256": _sha256(case / "input.png"),
            "mask_sha256": _sha256(case / "mask.png"),
            "prepared_input": str(prepared_path),
            "mesh": str(mesh_path),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "seconds": time.monotonic() - started,
        }
        (case_output / "manifest.json").write_text(json.dumps(
            result, indent=2, sort_keys=True))
        results.append(result)

    summary = {
        "schema_version": 1,
        "method": "TripoSR",
        "model": args.model,
        "source": "official VAST-AI-Research/TripoSR implementation",
        "source_revision": _git_revision(triposr_root),
        "input_policy": "benchmark RGB plus reference foreground mask",
        "device": device,
        "chunk_size": args.chunk_size,
        "marching_cubes_resolution": args.mc_resolution,
        "foreground_ratio": args.foreground_ratio,
        "model_load_seconds": load_seconds,
        "case_count": len(results),
        "cases": results,
    }
    (output / "summary.json").write_text(json.dumps(
        summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triposr-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--model", default="stabilityai/TripoSR")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
