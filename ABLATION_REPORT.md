# Ablation Evaluation Status

Date: 2026-07-23

The repository now exposes real stage bypasses for all eleven of `EVAL.md`'s
required ablations. Every config can be
passed to either `recon3d.pipeline --config ...` or the full E2E runner's new
`--config` option.

| Required ablation | Config | Implementation |
| --- | --- | --- |
| without background removal | `no_background_removal.yaml` | treats the full image as foreground and records a degraded low-confidence backend |
| without VTracer | `no_vtracer.yaml` | forces contour vectorizer |
| without SVG simplification | `no_svg_simplification.yaml` | disables filtering, dedupe tolerance, simplification, and smoothing |
| without primitive fitting | `no_primitive_fitting.yaml` | emits source-curve fallback primitives only |
| without constraint detection | `no_constraint_detection.yaml` | returns no geometric constraints |
| without semantic part reasoning | `no_semantic_part_reasoning.yaml` | groups all traced geometry into one anonymous object part |
| without camera estimation | `no_camera_estimation.yaml` | emits an explicit zero-confidence fallback camera |
| without depth | `no_depth.yaml` | retains normals but omits depth output and per-part estimates |
| without normals | `no_normals.yaml` | retains depth but omits normal output |
| without refinement | `no_refinement.yaml` | preserves initial validation and records zero iterations |
| without uncertainty tracking | `no_uncertainty_tracking.yaml` | forces all pre-decision confidence values to 1.0 and clears uncertainty summaries while preserving provenance |

`no_depth_normals.yaml` is retained as an additional combined ablation.

## Real Blender smoke evidence

On the unguided 320 px `box_01` input:

| Ablation | Pipeline status | Silhouette IoU | Refinement iterations | Artifact behavior |
| --- | --- | ---: | ---: | --- |
| no refinement | `partial_success` | 0.487 | 0 | valid `.blend`/GLB and audit log retained |
| no semantic reasoning | `partial_success` | 0.286 | 11 | anonymous part fallback remains executable |
| no uncertainty tracking | `partial_success` | 0.558 | 11 | primitive/part confidences all 1.0; two hypotheses accepted; uncertainty summary empty |

## Matched 11-way matrix: `box_01`

The resumable matrix runner executed a fresh full baseline plus all eleven
ablations with the same supplied mask, label, benchmark case, current code,
Blender reopen probe, and scoring logic. All 12 pipelines produced Blender
artifacts; the matrix is 11/11 complete for this case.

Full baseline: silhouette IoU **0.922**, part recall **1.000**, editability
**1.000**, Blender/GLB success **1.000**, runtime **84 s**.

| Ablation | Silhouette Δ | Part recall Δ | Editability Δ | Runtime Δ | MVP |
| --- | ---: | ---: | ---: | ---: | :---: |
| no background removal | +0.078 | 0.000 | 0.000 | +1 s | no (segmentation gate) |
| no VTracer | -0.006 | 0.000 | 0.000 | -1 s | yes |
| no SVG simplification | -0.000 | 0.000 | 0.000 | -3 s | yes |
| no primitive fitting | -0.006 | 0.000 | 0.000 | -8 s | yes |
| no constraint detection | 0.000 | 0.000 | 0.000 | -8 s | yes |
| no semantic part reasoning | **-0.553** | **-1.000** | **-1.000** | +332 s | no |
| no camera estimation | 0.000 | 0.000 | 0.000 | -9 s | yes |
| no depth | **-0.202** | 0.000 | 0.000 | +162 s | no |
| no normals | 0.000 | 0.000 | 0.000 | -13 s | yes |
| no refinement | 0.000 | 0.000 | 0.000 | -51 s | yes |
| no uncertainty tracking | 0.000 | 0.000 | 0.000 | -32 s | yes |

All ablations retained Blender execution and GLB validity on this case. The
background-removal silhouette score of 1.0 is deliberately not a success: its
full-frame prediction fails the segmentation gate. Semantic reasoning and
depth are the only controls with large downstream geometry losses on this
case; broader cases are required before judging the other modules redundant.

These are execution smokes, not contribution estimates against a matched full
run. The full cross-case matrix still needs to measure accuracy, part recall,
editability, reliability, runtime, and human preference deltas.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python evals/e2e/run_e2e.py \
  --cases box_01,bottle_01,gear_01 \
  --dataset evals/benchmark/dataset \
  --config evals/ablations/no_constraint_detection.yaml \
  --projects-root projects/ablation_no_constraints \
  --out evals/results_ablation_no_constraints \
  --python .venv/bin/python
```

Run or resume the complete orchestrated matrix with:

```bash
PYTHONPATH=. .venv/bin/python -m evals.ablations.run_matrix \
  --cases box_01,bottle_01,gear_01 \
  --ablations all \
  --projects-root projects/ablation_matrix \
  --results-root evals/results_ablation_matrix \
  --workers 2 \
  --python .venv/bin/python \
  --resume
```

Use distinct project/result roots for every ablation.

## Remaining evaluation work

All required controls and automated comparison dimensions now work, and one
case has a complete matched matrix. The matrix must still be expanded across
construction families. Human preference remains explicitly `not_measured`
because it cannot be inferred from automated metrics.
