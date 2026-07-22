# Calibrated Multiview Evaluation

Date: 2026-07-23

## Scope

This evaluation reconstructs `box_01` from the primary view and the calibrated
`+45°` view. The `+90°` image and mask are held out until after reconstruction.
The final `.blend` is rendered at exactly `+90°` with camera scale and offset
fixed from the primary image; the evaluator does not search pose, scale, or
alignment against the held-out target.

## Result

| Measurement | Result |
| --- | ---: |
| Primary Blender silhouette IoU | **0.962** |
| Primary visual-hull reprojection IoU | **0.914** |
| `+45°` evidence-view reprojection IoU | **0.889** |
| Held-out `+90°` silhouette IoU | **0.819** |
| Held-out contour Chamfer distance | **0.0174** |
| Held-out IoU ≥ 0.75 | **pass** |

The exported mesh has 8,144 vertices and 16,284 faces in the symmetry-prior
run. The semantic source parts remain in the `.blend` as hidden editable
guides and are excluded from GLB/render output. `geometry/multiview.json`
records the hull, observed-view scores, source guide IDs, and the generated
symmetry view as `semantic_prior`; it also records that primary observed
geometry was not overwritten.

## Ablation

The maximal two-view visual hull is underconstrained along the unseen
direction. With the enclosure symmetry/compactness hypothesis disabled, the
held-out IoU is **0.532**. Enabling the explicit source-labelled hypothesis
raises it to **0.819** while preserving the primary score. This is a measured
gain of **+0.287 IoU**.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m evals.benchmark.generate_benchmark \
  --cases box_01 \
  --out projects/multiview_benchmark \
  --resolution 320 \
  --view-offsets 45,90 \
  --workers 1

PYTHONPATH=. .venv/bin/python -m recon3d.pipeline \
  --image projects/multiview_benchmark/box_01/input.png \
  --image projects/multiview_benchmark/box_01/views/view_001/input.png \
  --label box \
  --view-azimuth 0 \
  --view-azimuth 45 \
  --out projects/multiview_box_visual_hull

PYTHONPATH=. .venv/bin/python -m evals.multiview.run_multiview_eval \
  --case projects/multiview_benchmark/box_01 \
  --project projects/multiview_box_visual_hull \
  --heldout-view view_002
```

For the ablation, add
`--config evals/ablations/no_multiview_box_symmetry.yaml` to the reconstruction
command and use a different output directory.

## Interpretation and remaining scope

This closes the previous box-case gap where multiview metadata did not affect
3D geometry. It does not establish cross-family multiview generalization.
Additional held-out cases, full-intrinsics camera support, all EVAL ablations,
opaque baselines, failure-detection evaluation, and human edit-task evaluation
remain required by the repository's full success definition.
