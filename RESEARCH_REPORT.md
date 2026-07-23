# Research Follow-up

Date: 2026-07-24

## Decisions

### Sparse-view geometry

Classic silhouette intersection produces a maximal visual hull: it encloses
the object but cannot recover concavities or choose the correct hidden surface
from sparse views. Recent reconstruction work improves that inverse problem by
optimizing a differentiable projection objective together with a shape or
surface prior, rather than adding unconstrained silhouette cones alone:

- [Shape Reconstruction Using Differentiable Projections and Deep Priors
  (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gadelha_Shape_Reconstruction_Using_Differentiable_Projections_and_Deep_Priors_ICCV_2019_paper.pdf)
  jointly optimizes projection agreement and an implicit shape prior.
- [Critical Regularizations for Neural Surface Reconstruction in the Wild
  (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Critical_Regularizations_for_Neural_Surface_Reconstruction_in_the_Wild_CVPR_2022_paper.html)
  identifies smoothness and minimal-surface regularization as important for
  compact completion of missing geometry.

The actionable conclusion for this repository is to keep the calibrated hull
as a silhouette-consistent outer support, but move the next geometry effort to
a canonical parametric or differentiable surface prior. Increasing voxel
resolution and repeating semantic silhouettes are not substitutes for that
prior.

A controlled full-orbit axial-invariance experiment on `bottle_01` confirmed
this. Six generated supports improved held-out silhouette IoU from 0.897 to
0.917, but worsened surface Chamfer from 0.080 to 0.090 and normal consistency
from 0.604 to 0.352. The production implementation was left unchanged.

### Small-cohort calibration

[Beta calibration (AISTATS
2017)](https://proceedings.mlr.press/v54/kull17a.html) specifically warns that
non-parametric isotonic calibration can overfit small datasets and proposes a
richer parametric map than ordinary logistic calibration. [Generalized
Venn-Abers calibration (ICML
2025)](https://proceedings.mlr.press/v267/van-der-laan25a.html) provides a
finite-sample, set-valued direction that also exposes epistemic uncertainty.

Nested out-of-fold prototypes on the 18-case cohort did not produce an honest
improvement over the current model. Strong shrinkage could force 5-bin ECE
below 0.08 only by collapsing predictions near the cohort pass rate; Brier
score rose from 0.136 to roughly 0.24 and useful ranking disappeared. That is a
metric artifact, so it was rejected. The repository continues to report the
0.169 ECE failure. The defensible next steps are a larger disjoint calibration
cohort and interval-valued Venn-Abers evidence, not a constant predictor.

### Direct image-to-mesh baseline

[TripoSR](https://github.com/VAST-AI-Research/TripoSR) was selected as the
first direct image-to-mesh comparator because its official implementation and
weights are ungated and MIT-licensed. The repository now contains an adapter
that:

- runs TripoSR in an isolated environment, keeping Torch out of the main
  pipeline;
- uses the benchmark reference mask to isolate geometry reconstruction from
  segmentation quality;
- records input hashes, model settings, source revision, runtime, and mesh
  size;
- scores generated GLBs with the same similarity-normalized, 3,000-point
  surface metric used by Eval 20.

The full 18-case official 256³ run completed on CPU with a median 18.1-second
per-case inference/extraction time. TripoSR's median normalized surface
Chamfer is **0.0823**, 2/18 cases pass the 0.05 threshold, and median normal
consistency is **0.4569**. The current calibrated structured pipeline measures
**0.0724**, 3/18, and **0.5777** respectively, and has lower Chamfer on 15/18
cases. TripoSR is better on `vase_01` and both wheel cases.

This is a useful external geometry comparator, but not an input-matched causal
comparison: TripoSR accepts one masked primary image, while the calibrated
pipeline result uses the primary and supplied `+45°` view. Both methods are
scored against the same untouched reference GLB using the same alignment and
surface sampler.

[Stable Fast
3D](https://github.com/Stability-AI/stable-fast-3d) remains a useful second
comparator, particularly for materials and mesh cleanup, but its official
weights are gated and its Apple Silicon path is explicitly experimental. It
is deferred until access and a matched environment are available.

## Reproduction

After creating a separate TripoSR environment:

```bash
projects/external_baselines/TripoSR/.baseline-venv/bin/python \
  evals/baselines/run_triposr.py \
  --triposr-root projects/external_baselines/TripoSR \
  --dataset projects/multiview_benchmark_full \
  --out projects/triposr_baseline_full \
  --mc-resolution 256

PYTHONPATH=. .venv/bin/python -m evals.baselines.evaluate_suite \
  --dataset projects/multiview_benchmark_full \
  --baseline projects/triposr_baseline_full \
  --out projects/triposr_baseline_full_eval
```
