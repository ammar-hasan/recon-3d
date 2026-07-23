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

## Current status

No human ratings have been collected, so none of Eval 29's median-score or 65%
preference targets is claimed. Direct image-to-mesh and one-shot Blender-agent
assets are also not present in the workspace; the same packet builder accepts
those methods as soon as their per-case renders exist. Human recruitment and
third-party baseline generation are external work, not measurements an
automated repository run can fabricate.
