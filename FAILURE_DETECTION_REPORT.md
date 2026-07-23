# Eval 28 — Failure Detection and Graceful Degradation

Date: 2026-07-23

## Result

A deterministic dedicated suite covers all ten difficult-input categories in
`EVAL.md` plus 20 clean controls.

| Metric | Result | Target | Outcome |
| --- | ---: | ---: | --- |
| Impossible-case detection | **1.000** | ≥ 0.90 | pass |
| False failure on easy controls | **0.000** | ≤ 0.05 | pass |
| Graceful partial output | **1.000** | ≥ 0.90 | pass |
| Misleading success claims | **0.000** | 0.00 | pass |

The ten difficult cases are transparent alpha, mirror/reflective, heavy
occlusion, very low resolution, target below 64 pixels, severe blur,
overlapping identical instances, unknown target, texture without a geometric
boundary, and conflicting views. Each detected case writes a structured
`input/quality_assessment.json`, records observed evidence and confidence,
recommends additional evidence, and is prevented from returning an
unqualified `success` status.

Twenty varied sharp 256×256 controls produce no high-risk decisions. The
same assessment also labelled all 18 existing synthetic benchmark primary
inputs low risk, providing an independent regression check outside the
generated control set. The
suite is reproducible with:

```bash
PYTHONPATH=. .venv/bin/python -m evals.failure_detection.run_suite \
  --out evals/results_failure_detection
```

## Pipeline smoke evidence

The 48×48 case was also run through the complete reconstruction pipeline with
a supplied mask. It preserved all normal intermediates, `.blend`, and GLB;
recorded `very_low_resolution` and `target_smaller_than_64px` at confidence
1.0; recommended a tighter, higher-resolution image; and returned
`partial_success` even though it produced a valid model. The report explicitly
states that no unqualified reconstruction success is claimed.

## Detection basis and scope

Resolution, target extent, partial alpha, blur, weak geometric boundaries,
and gross cross-view color conflict are measured directly. Mirror, heavy
occlusion, overlapping instances, and unknown-target ambiguity are detected
when the optional user label/description supplies that evidence. This is
deliberately conservative: the system does not claim reliable pixel-only
recognition of those semantic hazards from arbitrary real photographs.

The controlled Eval 28 gate is therefore closed, but real-photo
generalization remains open. Future work should add labelled natural images
and measure semantic-hazard detection without descriptive hints.
