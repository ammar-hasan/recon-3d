# Calibrated Multiview Evaluation

Date: 2026-07-23

## Scope

This evaluation reconstructs all 18 benchmark objects from the primary view
and the calibrated `+45°` view. The `+90°` image and mask are held out until
after reconstruction.
The final `.blend` is rendered at exactly `+90°` with the primary camera's
exact focal length and sensor width. Camera distance and offset are derived
only from the primary mask; the evaluator does not search pose, scale, or
alignment against the held-out target.

## Result

| Case | Primary IoU | Held-out `+90°` IoU | Chamfer | Risk |
| --- | ---: | ---: | ---: | --- |
| `bottle_01` | 0.944 | **0.869** | 0.079 | low |
| `bottle_02` | 0.940 | **0.841** | 0.077 | low |
| `box_01` | 0.961 | **0.796** | 0.102 | medium |
| `bracket_01` | 0.943 | 0.435 | 0.054 | high |
| `chair_01` | 0.897 | 0.332 | 0.066 | high |
| `crate_01` | 0.925 | 0.452 | 0.072 | high |
| `gear_01` | 0.947 | **0.754** | **0.049** | low |
| `gear_02` | 0.945 | 0.162 | **0.039** | high (hull rejected) |
| `knob_01` | 0.751 | 0.471 | 0.087 | low |
| `lamp_01` | 0.507 | 0.151 | 0.056 | high (hull rejected) |
| `mug_01` | 0.944 | **0.760** | 0.093 | medium |
| `mug_02` | 0.945 | 0.711 | 0.092 | medium |
| `pipe_elbow_01` | 0.907 | 0.652 | 0.076 | high |
| `sign_01` | 0.936 | 0.166 | **0.044** | high (hull rejected) |
| `table_01` | 0.900 | 0.292 | 0.064 | high |
| `vase_01` | 0.936 | **0.841** | 0.086 | low |
| `wheel_01` | 0.681 | **0.864** | 0.067 | low |
| `wheel_02` | 0.730 | 0.686 | 0.078 | low |

Median held-out silhouette IoU is **0.669**, below Eval 20's `≥ 0.75`
target; 7/18 individual cases pass. Median normalized surface Chamfer is
**0.074**, above the `≤ 0.05` target; 3/18 pass that metric and only
`gear_01` passes both measured targets. The earlier six-family discovery
subset had a 0.875 median and therefore overestimated broader performance.

The evaluator similarity-normalizes each 3,000-point surface
sample, searches proper axis rotations, performs rigid ICP alignment, and then
computes symmetric nearest-surface Chamfer. Volumetric IoU and partwise
held-out accuracy are not yet measured.

## Uncertainty and failure detection

The suite aggregator reads completion confidence and unseen-view risk from the
reconstruction artifacts before comparing them with held-out outcomes.

| Diagnostic | Result | Target / interpretation |
| --- | ---: | --- |
| High-risk silhouette-failure detection | **0.727** | 8/11 failures detected; target 0.90 not met |
| High-risk false-failure rate on passing cases | **0.000** | no passing case labelled high risk |
| Hidden completion confidence > 0.5 | **0 cases** | generated completion stays capped |
| Stratified out-of-fold silhouette ECE | **0.203** | fails Eval 24 target ≤ 0.08 |
| Stratified out-of-fold Brier / ROC AUC | **0.157 / 0.818** | useful ranking, poor probability calibration |
| Strict hash-split external-test ECE | **0.441** | 12 calibration / 6 test objects; prevalence shifted |

Axial completions are `low` risk, planar completions are normally `medium`, and
unprioritized hulls are `high`. A rejected calibrated hull is now treated as
high operational risk because its artifact already warns that observed-view
reprojection is inadequate. This catches gear_02, lamp, and sign failures in
addition to the explicitly underconstrained hulls, but knob, mug_02, and
wheel_02 remain missed. This is still not Eval 28's dedicated difficult-input
suite.

The earlier report's ECE of 0.241 was invalid: it treated
`completion_confidence` (confidence in hidden surfaces, deliberately capped at
0.5) as the probability that held-out silhouette would pass. The aggregator
now keeps that field only for hidden-geometry overconfidence. Silhouette ECE is
`not_measured` unless a serialized calibrator trained on a separate cohort is
provided. The reported 0.203 value is deterministic three-fold out-of-fold
evidence from primary-view IoU and operational risk; every prediction excludes
its case from training. It fails the target and is not presented as calibrated.

Semantic source parts remain in each `.blend` as hidden editable guides and
are excluded from GLB/render output. `geometry/multiview.json` records the
hull, observed-view scores, source guide IDs, angular evidence span, completion
confidence, unseen-view risk, and any generated completion view as
`semantic_prior`; it also records that primary observed geometry was not
overwritten. Hidden completion confidence is capped below 0.5.

## Ablation

The maximal two-view visual hull is underconstrained along the unseen
direction. A fresh matched run disables semantic completion while retaining
the same primary and `+45°` observations, current code, exact-intrinsics
evaluator, and surface scorer.

| Case | Full IoU | No semantic completion | IoU gain | Full Chamfer | Ablated Chamfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `box_01` | 0.796 | 0.477 | **+0.319** | 0.102 | 0.087 |
| `bottle_01` | 0.869 | 0.514 | **+0.354** | 0.079 | 0.072 |
| `gear_01` | 0.754 | 0.514 | **+0.241** | 0.049 | 0.047 |
| `mug_01` | 0.760 | 0.494 | **+0.266** | 0.093 | 0.083 |
| `chair_01` | 0.332 | 0.332 | 0.000 | 0.066 | 0.066 |
| `pipe_elbow_01` | 0.652 | 0.367 | **+0.285** | 0.076 | 0.058 |

Median held-out IoU rises from **0.485** to **0.757** (**+0.272**) and 5/6
cases improve. Chair receives no unsupported completion and remains unchanged.
The surface result moves in the opposite direction: every case receiving a
semantic prior has worse Chamfer. The priors therefore provide measurable
silhouette value but are not yet accurate 3D completion; they cannot close
Eval 20's surface target.

### Visual hull versus parametric construction

The `no_multiview_visual_hull` control now disables only the hull and keeps
joint multiview refinement enabled. This compares the hull against the
editable parametric construction without conflating the refinement stage.

| Case | Full IoU | No hull IoU | Full Chamfer | No hull Chamfer |
| --- | ---: | ---: | ---: | ---: |
| `box_01` | 0.796 | 0.184 | 0.102 | **0.045** |
| `bottle_01` | 0.869 | **0.973** | **0.079** | 0.087 |
| `gear_01` | 0.754 | 0.090 | 0.049 | **0.044** |
| `mug_01` | 0.760 | 0.168 | 0.093 | **0.037** |
| `chair_01` | 0.332 | 0.094 | 0.066 | **0.041** |
| `pipe_elbow_01` | 0.652 | 0.409 | **0.076** | 0.086 |

The hull raises median held-out IoU from **0.176** to **0.757**, while the
parametric construction lowers median Chamfer from **0.078** to **0.044** and
passes the surface target on this subset. Joint refinement executed for the
no-hull runs but retained no depth change: its unconstrained 3–4× depth
solutions improved secondary silhouettes while violating the primary-view
preservation gate. The next geometry problem is therefore view-consistent
pose/depth for the editable parametric construction, not simply more hull
resolution.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m evals.benchmark.generate_benchmark \
  --cases all \
  --out projects/multiview_benchmark_full \
  --resolution 320 \
  --view-offsets 45,90 \
  --workers 1

PYTHONPATH=. .venv/bin/python -m recon3d.pipeline \
  --image projects/multiview_benchmark_full/box_01/input.png \
  --image projects/multiview_benchmark_full/box_01/views/view_001/input.png \
  --mask projects/multiview_benchmark_full/box_01/mask.png \
  --label box \
  --view-azimuth 0 \
  --view-azimuth 45 \
  --out projects/multiview_box_visual_hull

PYTHONPATH=. .venv/bin/python -m evals.multiview.run_multiview_eval \
  --case projects/multiview_benchmark_full/box_01 \
  --project projects/multiview_box_visual_hull \
  --heldout-view view_002
```

Aggregate completed project metrics with
`python -m evals.multiview.summarize_suite`; pass one
`--entry case_id=project_dir` argument per case and an `--out` JSON path. The
tool writes both JSON and Markdown summaries. Train a prospective calibrator
and emit out-of-fold evidence with `python -m evals.multiview.calibration
--suite SUITE.json --out MODEL.json --cross-validation-out CV.json`. Supply the
model to a disjoint suite with `summarize_suite --calibration-model MODEL.json`.

For the ablation, add
`--config evals/ablations/no_multiview_semantic_completion.yaml` to the
reconstruction command and use a different output directory.

## Interpretation and remaining scope

This closes the previous gap where multiview metadata did not affect 3D
geometry, but the complete benchmark disproves the earlier subset-level Eval
20 pass. All failures are retained and labelled rather than hidden.
Full-intrinsics camera support, held-out volumetric/part metrics, improved
geometry and risk prediction, opaque baselines, target-level confidence
calibration, and human edit-task evaluation remain required by the full
success definition.
