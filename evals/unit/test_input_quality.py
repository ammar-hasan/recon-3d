from pathlib import Path

import cv2
import numpy as np

from recon3d.input_quality import assess_input_quality
from recon3d.pipeline import _result_status
from recon3d.schemas import (EvidenceSource, InputBundle, InputSpec, LoadedImage,
                             SegmentationResult)


def _bundle(tmp_path: Path, images, description=None) -> InputBundle:
    loaded = []
    paths = []
    for index, image in enumerate(images):
        path = tmp_path / ("view_%d.png" % index)
        cv2.imwrite(str(path), image)
        paths.append(str(path))
        loaded.append(LoadedImage(
            path=str(path), width=image.shape[1], height=image.shape[0],
            sha256="0" * 64, channels=1 if image.ndim == 2 else image.shape[2]))
    return InputBundle(
        spec=InputSpec(image_paths=paths, description=description),
        images=loaded)


def _seg(bbox=(48, 48, 208, 208)) -> SegmentationResult:
    return SegmentationResult(
        mask_path="mask.png", rgba_path="rgba.png", original_path="input.png",
        confidence=0.9, backend="test", bbox=bbox, coverage=0.4,
        selection_source=EvidenceSource.DIRECTLY_OBSERVED)


def test_clean_sharp_input_stays_low_risk(tmp_path):
    image = np.full((256, 256, 3), 245, np.uint8)
    cv2.rectangle(image, (48, 48), (207, 207), (20, 80, 180), -1)
    cv2.line(image, (48, 48), (207, 207), (255, 255, 255), 4)
    result = assess_input_quality(_bundle(tmp_path, [image]), _seg())
    assert result.risk == "low"
    assert not result.unreliable_input_detected


def test_observable_and_explicit_hazards_are_high_risk(tmp_path):
    image = np.full((48, 48, 4), 180, np.uint8)
    image[:, :, 3] = 128
    bundle = _bundle(tmp_path, [image], "heavily occluded transparent object")
    result = assess_input_quality(bundle, _seg((8, 8, 38, 38)))
    codes = {signal.code for signal in result.signals}
    assert result.risk == "high"
    assert result.unreliable_input_detected
    assert {"transparent_object", "heavy_occlusion",
            "very_low_resolution", "target_smaller_than_64px"} <= codes
    assert result.recommendations
    preflight = assess_input_quality(bundle)
    assert preflight.risk == "high"
    assert {"transparent_object", "heavy_occlusion",
            "very_low_resolution"} <= {s.code for s in preflight.signals}


def test_grossly_conflicting_views_are_detected(tmp_path):
    first = np.full((256, 256, 3), (0, 0, 255), np.uint8)
    second = np.full((256, 256, 3), (0, 255, 0), np.uint8)
    result = assess_input_quality(_bundle(tmp_path, [first, second]), _seg())
    assert result.risk == "high"
    assert "conflicting_views" in {signal.code for signal in result.signals}


def test_high_input_risk_cannot_be_reported_as_success():
    assert _result_status(True, "high") == "partial_success"
    assert _result_status(False, "low") == "partial_success"
    assert _result_status(True, "low") == "success"
