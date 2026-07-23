"""Stage 1: input loading and validation.

Loads PNG/JPEG/WEBP (plus BMP/TIFF), applies EXIF orientation, copies the
primary image untouched (pixels) into <output_dir>/input/original.png and
validates optional box / point / mask hints.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from .schemas import InputBundle, InputSpec, LoadedImage, sha256_file

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class InputError(Exception):
    """Raised for unreadable, corrupt, or unsupported input files/hints."""


def _load_image(path: Path) -> Tuple[Image.Image, bool]:
    """Open + fully decode an image, applying EXIF orientation."""
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        raise InputError(f"corrupt or unreadable image file: {path} ({exc})") from exc
    orientation = 1
    try:
        orientation = img.getexif().get(0x0112, 1)  # EXIF Orientation tag
    except Exception:
        orientation = 1
    transposed = ImageOps.exif_transpose(img)
    applied = transposed is not img and orientation not in (1, None)
    if applied:
        img = transposed
    return img, bool(applied)


def _normalise_mode(img: Image.Image) -> Image.Image:
    """Keep RGB/RGBA/L pixels untouched; convert exotic modes predictably."""
    if img.mode in ("RGB", "RGBA", "L"):
        return img
    if img.mode in ("LA", "PA", "P"):
        p = img.convert("RGBA")
        # fully-opaque palette images collapse to RGB
        if p.getextrema()[3][0] == 255:
            return p.convert("RGB")
        return p
    if img.mode in ("I;16", "I", "F"):
        arr = np.asarray(img)
        arr = (arr.astype(np.float64) - arr.min()) / max(arr.ptp(), 1e-9) * 255.0
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    if img.mode == "CMYK":
        return img.convert("RGB")
    return img.convert("RGB")


def _validate_box(
    box: Tuple[float, float, float, float], w: int, h: int, warnings: List[str]
) -> Optional[Tuple[float, float, float, float]]:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)
    if x1 - x0 < 2.0 or y1 - y0 < 2.0:
        warnings.append(
            f"invalid bounding box {box}: empty after clamping to image {w}x{h}; ignoring it"
        )
        return None
    return (x0, y0, x1, y1)


def _validate_point(
    point: Tuple[float, float], w: int, h: int, warnings: List[str]
) -> Optional[Tuple[float, float]]:
    x, y = float(point[0]), float(point[1])
    if not (0.0 <= x < w and 0.0 <= y < h):
        warnings.append(f"point {point} outside image {w}x{h}; ignoring it")
        return None
    return (x, y)


def _validate_mask(mask_path: str, w: int, h: int, warnings: List[str]) -> Optional[str]:
    p = Path(mask_path)
    if not p.is_file():
        raise InputError(f"mask file not found: {mask_path}")
    try:
        m = Image.open(p)
        m.load()
    except Exception as exc:
        raise InputError(f"corrupt or unreadable mask file: {mask_path} ({exc}") from exc
    if m.size != (w, h):
        raise InputError(
            f"mask size {m.size} does not match image size {(w, h)}: {mask_path}"
        )
    arr = np.asarray(m.convert("L"))
    if int(arr.max()) == 0:
        warnings.append(f"mask {mask_path} is entirely empty; ignoring it")
        return None
    return str(p)


def load_input(spec: InputSpec) -> InputBundle:
    """Validate + load images and hints; copy the primary image to the project.

    Raises InputError on corrupt/unsupported files, missing mask files, or
    mask/image size mismatch. Invalid boxes/points/empty masks degrade to a
    warning and are ignored.
    """
    if not spec.image_paths:
        raise InputError("InputSpec.image_paths is empty")
    if (spec.view_azimuths_deg is not None
            and len(spec.view_azimuths_deg) != len(spec.image_paths)):
        raise InputError(
            "view_azimuths_deg must contain one angle per image "
            "(%d images, %d angles)" % (
                len(spec.image_paths), len(spec.view_azimuths_deg)))
    if (spec.view_azimuths_deg is not None
            and not all(math.isfinite(v) for v in spec.view_azimuths_deg)):
        raise InputError("view_azimuths_deg values must be finite")
    input_dir = Path(spec.output_dir) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    images: List[LoadedImage] = []
    primary_size: Optional[Tuple[int, int]] = None

    for idx, raw in enumerate(spec.image_paths):
        src = Path(raw)
        if not src.is_file():
            raise InputError(f"image file not found: {raw}")
        ext = src.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise InputError(
                f"unsupported image extension '{ext}' for {raw}; "
                f"supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        img, exif_applied = _load_image(src)
        img = _normalise_mode(img)

        dest = input_dir / ("original.png" if idx == 0 else f"view_{idx:03d}.png")
        img.save(dest, format="PNG")  # re-save as PNG, pixels untouched
        w, h = img.size
        channels = len(img.getbands())
        images.append(
            LoadedImage(
                path=str(dest),
                width=w,
                height=h,
                sha256=sha256_file(src),
                exif_orientation_applied=exif_applied,
                channels=channels,
            )
        )
        if idx == 0:
            primary_size = (w, h)

    assert primary_size is not None
    w, h = primary_size

    updates = {}
    if spec.box is not None:
        box = _validate_box(spec.box, w, h, warnings)
        if box != spec.box:
            updates["box"] = box
    if spec.point is not None:
        point = _validate_point(spec.point, w, h, warnings)
        if point != spec.point:
            updates["point"] = point
    if spec.mask_path is not None:
        mask_path = _validate_mask(spec.mask_path, w, h, warnings)
        if mask_path != spec.mask_path:
            updates["mask_path"] = mask_path
    if updates:
        spec = spec.model_copy(update=updates)

    return InputBundle(spec=spec, images=images, warnings=warnings)
