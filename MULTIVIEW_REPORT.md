# Calibrated Multiview Evaluation

Date: 2026-07-23

## Scope

This evaluation reconstructs six manufactured-object families from the primary
view and the calibrated `+45°` view. The `+90°` image and mask are held out
until after reconstruction.
The final `.blend` is rendered at exactly `+90°` with camera scale and offset
fixed from the primary image; the evaluator does not search pose, scale, or
alignment against the held-out target.

## Result

| Case / family | Primary IoU | Held-out `+90°` IoU | ≥ 0.75 | Completion hypothesis |
| --- | ---: | ---: | :---: | --- |
| `box_01` / enclosure | 0.962 | **0.903** | yes | planar symmetry |
| `bottle_01` / revolution | 0.940 | **0.944** | yes | axial invariance |
| `gear_01` / radial extrusion | 0.951 | **0.950** | yes | axial invariance |
| `mug_01` / revolution + sweep | 0.942 | **0.847** | yes | planar symmetry |
| `chair_01` / primitive assembly | 0.905 | 0.283 | no | none; high-risk warning |
| `pipe_elbow_01` / sweep | 0.907 | 0.631 | no | planar symmetry |

Median held-out silhouette IoU is **0.875**, above Eval 20's `≥ 0.75`
median target. Four of six individual cases pass.

| Case | Normalized surface Chamfer | Normal consistency | Chamfer ≤ 0.05 |
| --- | ---: | ---: | :---: |
| `box_01` | 0.102 | 0.435 | no |
| `bottle_01` | 0.079 | 0.534 | no |
| `gear_01` | **0.048** | 0.699 | yes |
| `mug_01` | 0.092 | 0.443 | no |
| `chair_01` | 0.067 | 0.661 | no |
| `pipe_elbow_01` | 0.074 | 0.544 | no |

Median normalized surface Chamfer is **0.077**, so the Eval 20 surface target
is not met. The evaluator similarity-normalizes each 3,000-point surface
sample, searches proper axis rotations, performs rigid ICP alignment, and then
computes symmetric nearest-surface Chamfer. Volumetric IoU and partwise
held-out accuracy are not yet measured.

Semantic source parts remain in each `.blend` as hidden editable guides and
are excluded from GLB/render output. `geometry/multiview.json` records the
hull, observed-view scores, source guide IDs, angular evidence span, completion
confidence, unseen-view risk, and any generated completion view as
`semantic_prior`; it also records that primary observed geometry was not
overwritten. Hidden completion confidence is capped below 0.5.

## Ablation

The maximal two-view visual hull is underconstrained along the unseen
direction. With semantic completion disabled, held-out IoUs for
box/bottle/gear/mug/chair/pipe are respectively
`0.532 / 0.538 / 0.557 / 0.496 / 0.283 / 0.370`; median is **0.514**. With the
source-labelled priors enabled, median rises to **0.875** (**+0.361**). Chair
does not receive an unsupported prior and stays unchanged. Pipe improves to
0.631 but remains below threshold.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m evals.benchmark.generate_benchmark \
  --cases box_01,bottle_01,gear_01,mug_01,chair_01,pipe_elbow_01 \
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
`--config evals/ablations/no_multiview_semantic_completion.yaml` to the
reconstruction command and use a different output directory.

## Interpretation and remaining scope

This closes the previous gap where multiview metadata did not affect 3D
geometry and reaches the Eval 20 median silhouette target across six families.
The two individual failures are retained and labelled rather than hidden.
Full-intrinsics camera support, held-out volumetric/part metrics, harder cases,
all EVAL ablations, opaque baselines, formal confidence calibration, and human
edit-task evaluation remain required by the full success definition.
