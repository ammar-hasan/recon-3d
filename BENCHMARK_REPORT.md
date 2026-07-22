# Final Automated Benchmark Report

- Run date: 2026-07-23
- Code under test: `d5d864e`
- Dataset: 18 synthetic cases from `evals/benchmark/dataset`
- Blender: 5.2.0, background execution plus independent reopen probe

## Result

The structured reconstruction pipeline passes all automated MVP hard gates on
all 18 benchmark cases.

| Metric | Result |
|---|---:|
| MVP cases passed | 18 / 18 |
| Mean overall score | 0.946 |
| Mean segmentation IoU | 1.000 |
| Mean rasterized trace IoU | 0.989 |
| Mean control-point reduction | 0.951 |
| Mean primitive accuracy | 1.000 |
| Mean major-part recall | 1.000 |
| Mean camera score | 0.500 |
| Mean final silhouette IoU | 0.910 |
| Mean contour error | 0.004 |
| Blender execution rate | 1.000 |
| Independent `.blend` reopen rate | 1.000 |
| GLB validity rate | 1.000 |
| Safety violations | 0 |

## Per-case results

| Case | Difficulty | MVP | Overall | Silhouette IoU | SVG baseline IoU |
|---|---|---:|---:|---:|---:|
| bottle_01 | easy | PASS | 0.954 | 0.960 | 0.959 |
| bottle_02 | hard | PASS | 0.955 | 0.973 | 0.949 |
| box_01 | easy | PASS | 0.948 | 0.922 | 0.887 |
| bracket_01 | medium | PASS | 0.947 | 0.910 | 0.900 |
| chair_01 | medium | PASS | 0.939 | 0.860 | 0.801 |
| crate_01 | medium | PASS | 0.943 | 0.900 | 0.845 |
| gear_01 | easy | PASS | 0.947 | 0.915 | 0.885 |
| gear_02 | hard | PASS | 0.951 | 0.944 | 0.844 |
| knob_01 | easy | PASS | 0.947 | 0.913 | 0.929 |
| lamp_01 | medium | PASS | 0.945 | 0.904 | 0.883 |
| mug_01 | easy | PASS | 0.945 | 0.901 | 0.868 |
| mug_02 | medium | PASS | 0.945 | 0.904 | 0.853 |
| pipe_elbow_01 | medium | PASS | 0.931 | 0.800 | 0.921 |
| sign_01 | easy | PASS | 0.947 | 0.911 | 0.915 |
| table_01 | easy | PASS | 0.940 | 0.869 | 0.780 |
| vase_01 | medium | PASS | 0.950 | 0.934 | 0.945 |
| wheel_01 | easy | PASS | 0.945 | 0.920 | 0.931 |
| wheel_02 | medium | PASS | 0.948 | 0.942 | 0.928 |

All construction plans were valid; Blender execution, reopen, GLB validity,
major visible-part recall, editability, and safety gates passed for every case.

## Baselines and ablations

| Variant | Cases | Gate passes | Mean silhouette IoU | Interpretation |
|---|---:|---:|---:|---|
| Full structured pipeline | 18 | 18 | 0.910 | Final system |
| No refinement | 18 | 15 silhouette gates | 0.880 | Refinement adds +0.030 mean IoU and recovers pipe plus both wheels |
| Blender-rendered SVG extrusion | 18 | not structurally eligible | 0.890 | Strong single-view silhouette, but no semantic parts or meaningful editability |
| Full pipeline, selected families | 3 | 3 | 0.892 | Bottle, gear, and pipe comparison subset |
| No depth/normals, selected families | 3 | 2 | 0.906 | Mixed result; pipe regresses below its hard gate |

The three no-depth/normals per-case deltas relative to the full pipeline are:

| Case | Family | Full | No depth/normals | Delta |
|---|---|---:|---:|---:|
| bottle_01 | revolve | 0.960 | 0.960 | +0.000 |
| gear_01 | extrude | 0.915 | 0.962 | +0.047 |
| pipe_elbow_01 | sweep | 0.800 | 0.796 | −0.004 |

This sample does not establish a mean accuracy benefit for depth/normals. It
does show that disabling them can change operator/refinement outcomes and makes
the boundary sweep case fail. A broader isolated ablation is still required
before making stronger causal claims.

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests evals/unit evals/stage

PYTHONPATH=. .venv/bin/python evals/e2e/run_e2e.py \
  --cases all \
  --projects-root projects/e2e_final \
  --out evals/results_final \
  --workers 1 \
  --python .venv/bin/python
```

The full regression run returned `236 passed, 1 skipped`. Generated benchmark
projects and dashboards are ignored because they contain large Blender/render
artifacts; this report preserves the checked result in version control.

## Scope caveats

- The benchmark is synthetic and the main reconstruction path is mask- and
  label-guided.
- The benchmark cases are single-view; multiview is covered separately by
  tests and a two-image smoke run.
- Camera calibration remains intentionally low-confidence from a single view.
- Human edit-task studies, real-photo stress tests, opaque image-to-mesh/VLM
  baselines, and the complete ablation matrix in `EVAL.md` were not run.
