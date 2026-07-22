"""Stage 3: padded square crop + exact coordinate normalisation.

The transform is a pure similarity: crop = (original - offset) * scale.
`offset` and `scale` are recorded in float64 so CropMetadata.to_crop /
to_original round-trips to floating-point precision.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from .config import PipelineConfig
from .schemas import CropMetadata, SchemaIO, SegmentationResult


def make_crop(
    seg: SegmentationResult, out_dir: str, cfg: PipelineConfig
) -> Tuple[CropMetadata, str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rgba = np.asarray(Image.open(seg.rgba_path).convert("RGBA"))
    mask = np.asarray(Image.open(seg.mask_path).convert("L"))
    h, w = mask.shape

    x0, y0, x1, y1 = seg.bbox
    pad = cfg.crop.padding_px
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        raise ValueError(f"degenerate segmentation bbox {seg.bbox}")

    canvas = cfg.crop.canvas_size
    side = max(bw, bh)
    scale = float(canvas) / float(side)

    # centre the padded bbox on the square canvas
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    offset_x = cx - canvas / (2.0 * scale)
    offset_y = cy - canvas / (2.0 * scale)

    m = np.array(
        [[scale, 0.0, -offset_x * scale], [0.0, scale, -offset_y * scale]],
        dtype=np.float64,
    )
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    crop_rgba = cv2.warpAffine(
        rgba, m, (canvas, canvas), flags=interp, borderValue=(0, 0, 0, 0)
    )
    crop_mask = cv2.warpAffine(mask, m, (canvas, canvas), flags=interp, borderValue=0)
    crop_mask = np.where(crop_mask >= 128, 255, 0).astype(np.uint8)
    # keep the RGBA alpha consistent with the resampled binary mask
    crop_rgba[:, :, 3] = crop_mask

    crop_rgba_path = out / "crop_rgba.png"
    crop_mask_path = out / "crop_mask.png"
    Image.fromarray(crop_rgba).save(crop_rgba_path)
    cv2.imwrite(str(crop_mask_path), crop_mask)

    meta = CropMetadata(
        source_image_size=(w, h),
        source_bbox=(x0, y0, x1, y1),
        padding=pad,
        output_size=(canvas, canvas),
        scale=scale,
        offset=(offset_x, offset_y),
    )
    SchemaIO.save_json(meta, out / "crop_metadata.json")
    return meta, str(crop_rgba_path), str(crop_mask_path)
