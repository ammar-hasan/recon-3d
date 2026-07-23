# Eval 27 — Runtime and Resource Efficiency

Date: 2026-07-23

## Existing 18-case cohort

The completed single-view E2E cohort was summarized from manifest wall-clock
timestamps. Timestamps have one-second resolution, so these are operational
runtime measurements rather than microbenchmarks.

| Difficulty | Cases | Mean | Median | P90 |
| --- | ---: | ---: | ---: | ---: |
| easy | 8 | 84.0 s | 70.5 s | 131.9 s |
| medium | 8 | 102.9 s | 103.0 s | 150.8 s |
| hard | 2 | 53.5 s | 53.5 s | 57.9 s |
| **all** | **18** | **89.0 s** | **81.5 s** | **148.3 s** |

The two hard cases happen to require less refinement than several easy/medium
objects; this cohort is too small to infer that hard inputs are generally
faster.

## Prospective instrumentation

New manifests record monotonic grouped stage timings and peak process RSS. A
fresh 48×48 low-resolution smoke run produced:

| Group | Seconds |
| --- | ---: |
| input, segmentation, crop | 0.077 |
| preprocess, vectorize, cleanup | 28.239 |
| primitives, constraints, semantics | 4.305 |
| camera, depth, multiview, planning | 0.313 |
| Blender build and multiview geometry | 0.789 |
| validation and refinement | 22.282 |
| **total** | **56.006** |

Peak process RSS was **441.3 MiB**. The run made zero model/API calls. GPU
memory is explicitly `not measured`; the current smoke used CPU/classical
paths plus background Blender.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m evals.runtime.summarize \
  --projects-root projects/e2e_final \
  --dataset evals/benchmark/dataset \
  --out evals/results_runtime_summary
```

The summarizer accepts both new instrumented manifests and legacy manifests
that only have start/finish timestamps. Full-cohort stage distributions,
per-process Blender peak memory, GPU memory, and energy/cost accounting remain
unmeasured until the 18-case suite is rerun with the new instrumentation.
