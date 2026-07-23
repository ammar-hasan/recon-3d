import numpy as np
import pytest
from PIL import Image

from evals.baselines.evaluate_mesh import evaluate_mesh
from evals.baselines.evaluate_suite import aggregate_case_metrics
from evals.baselines.run_triposr import (
    _git_revision,
    prepare_reference_masked_image,
)


def test_reference_masked_preprocessing_centers_on_gray(tmp_path):
    image = np.zeros((20, 30, 3), np.uint8)
    image[5:15, 10:20] = (240, 20, 10)
    mask = np.zeros((20, 30), np.uint8)
    mask[5:15, 10:20] = 255
    image_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    prepared = np.asarray(prepare_reference_masked_image(
        image_path, mask_path, foreground_ratio=0.5))

    assert prepared.shape == (20, 20, 3)
    assert np.all(prepared[0, 0] == (128, 128, 128))
    assert np.all(prepared[10, 10] == (240, 20, 10))


def test_reference_masked_preprocessing_rejects_empty_mask(tmp_path):
    image_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(image_path)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(mask_path)

    with pytest.raises(ValueError, match="empty"):
        prepare_reference_masked_image(image_path, mask_path)


def test_mesh_evaluator_validates_sample_count_before_files(tmp_path):
    with pytest.raises(ValueError, match="at least 100"):
        evaluate_mesh(
            tmp_path / "generated.glb", tmp_path / "reference.glb",
            tmp_path / "out", sample_count=99)


def test_baseline_suite_aggregate_reports_surface_gate():
    aggregate = aggregate_case_metrics([
        {"normalized_surface_chamfer_distance": 0.04,
         "surface_normal_consistency": 0.8},
        {"normalized_surface_chamfer_distance": 0.08,
         "surface_normal_consistency": 0.6},
    ])

    assert aggregate["case_count"] == 2
    assert aggregate["median_normalized_surface_chamfer_distance"] == 0.06
    assert aggregate["median_surface_normal_consistency"] == 0.7
    assert aggregate["chamfer_pass_count_0_05"] == 1


def test_baseline_source_revision_reads_loose_git_ref(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("abc123\n")

    assert _git_revision(tmp_path) == "abc123"
