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

Use distinct project/result roots for every ablation.

## Remaining evaluation work

All required controls exist, but the full matched cross-case matrix has not
been run. It must still measure accuracy, part recall, editability, execution
reliability, runtime, and human preference for every ablation before the
required 11-way ablation evaluation is complete.
