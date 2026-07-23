# Eval 29 Blind Human Evaluation

Date: 2026-07-23

## Ready protocol

The repository now builds deterministic blind pairwise packets. Each case gets
an opaque item ID, a reference image, randomized A/B method assets, an
eight-dimension 1–5 rating form, and an A/B/tie preference. The public manifest
contains no case or method identity; the answer key is stored separately under
`private/`. A scorer validates every rating, restores method identities only
after collection, reports per-method medians, and computes decisive preference
rates.

The eight dimensions are reference resemblance, shape plausibility, part
correctness, editability, topology cleanliness, material plausibility,
starting-asset usefulness, and uncertainty trustworthiness. Evaluator IDs and
expertise are required by the scorer.

A complete 18-case pipeline-versus-SVG-extrusion packet was generated locally
from the final shaded renders and baseline silhouette renders. Reproduce it:

```bash
PYTHONPATH=. .venv/bin/python -m evals.human.build_packet \
  --method pipeline=projects/e2e_final/{case_id}/validation/render_shaded.png \
  --method svg_extrusion=evals/results_final/baseline/{case_id}/baseline_mask.png \
  --out projects/human_eval29_svg
```

Score returned forms only after collection:

```bash
PYTHONPATH=. .venv/bin/python -m evals.human.score_packet \
  --ratings projects/human_eval29_svg/public/ratings.csv \
  --answer-key projects/human_eval29_svg/private/answer_key.json \
  --out projects/human_eval29_svg/results.json
```

## Direct image-to-mesh baseline

The official TripoSR implementation is now integrated as an isolated,
provenance-recorded baseline. A complete 18-case 256³ GLB set was generated
locally from each primary benchmark image plus its reference foreground mask.
The mask policy isolates 3D reconstruction quality from segmentation quality.
Model settings, source revision, input hashes, runtime, and mesh size are
recorded per case.

Using the same 3,000-point, similarity-normalized surface evaluator as Eval 20,
TripoSR has median Chamfer **0.0823**, 2/18 cases at or below 0.05, and median
normal consistency **0.4569**. The calibrated structured pipeline measures
0.0724, 3/18, and 0.5777, with lower Chamfer on 15/18 cases. This geometry
comparison is not input-matched: TripoSR is single-view, while the calibrated
pipeline uses the primary and `+45°` views.

An additional input-matched, blind 18-case GLB packet was generated locally
from the single-view `e2e_final` pipeline meshes and the TripoSR meshes:

```bash
PYTHONPATH=. .venv/bin/python -m evals.human.build_packet \
  --dataset projects/multiview_benchmark_full \
  --method pipeline=projects/e2e_final/{case_id}/blender/model.glb \
  --method triposr=projects/triposr_baseline_full/{case_id}/mesh.glb \
  --out projects/human_eval29_triposr
```

Reproduce baseline generation and objective scoring:

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

## Current status

No human ratings have been collected, so none of Eval 29's median-score or 65%
preference targets is claimed. Direct image-to-mesh GLBs are now present in the
local ignored workspace and an input-matched blind GLB packet is ready. Matched
comparison renders and one-shot Blender-agent assets are not yet present. The
same packet builder accepts those methods as soon as matched per-case assets
exist. Human recruitment remains external work that an automated repository
run cannot fabricate.
