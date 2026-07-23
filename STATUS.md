# STATUS — Image-to-Editable-3D Reconstruction Pipeline

Last updated: 2026-07-23 after the full-camera 18-case multiview evaluation.

## Current outcome

The structured single-view Level C benchmark is complete and passes its
automated acceptance suite. The repository's broader `GOAL.md` success
definition is still in progress because `EVAL.md` requires more than the MVP.

- Fresh Level C benchmark: **18/18 MVP hard-gate passes**.
- Mean final silhouette IoU: **0.910**.
- Mean Blender-rendered SVG-extrusion baseline IoU: **0.890**.
- Mean no-refinement IoU: **0.880**; only 15/18 cases meet the 0.80
  silhouette gate without refinement.
- Blender execution, independent `.blend` reopen, and GLB validity: **18/18**.
- Mean major visible-part recall: **1.000**.
- Safety violations: **0**.
- Regression suite: **281 passed, 1 skipped**, including real Blender build,
  validation, and refinement tests.
- Full-camera perspective carving is implemented and measured across all 18
  calibrated cases. Median held-out IoU improves from **0.669** to **0.723**
  and passes rise from 7/18 to 8/18, but Eval 20's ≥ 0.75 median target remains
  open. Rejected exact hulls safely fall back to the supplied azimuth model.
- The full-camera suite's median normalized surface Chamfer improves from
  **0.074** to **0.072**, above Eval 20's ≤ 0.05 target; 3/18 pass Chamfer and
  only `gear_01` passes both measured targets.
- High operational risk detects 8/10 held-out silhouette failures with no
  high-risk false alarms. Deterministic out-of-fold confidence ECE is **0.169**,
  so Eval 24's ≤ 0.08 target remains open.
- Dedicated Eval 28 suite: **10/10** difficult inputs detected, **0/20** easy
  controls falsely rejected, 100% graceful partial artifacts, and zero
  misleading success claims. Semantic hazards such as mirrors and occlusion
  use explicit label/description evidence; real-photo generalization remains
  open.
- Eval 22 reference-view cohort: mean shaded SSIM **0.926**, mean color-region
  agreement **0.761**, mean clay silhouette IoU **0.906**, and 0/18 cases where
  a high shaded score conceals clay silhouette IoU below 0.80. The validation
  renderer now produces the previously missing material-ID pass.
- Automated Eval 30 subset: **7/7** editing, variant, articulation, and GLB
  tasks complete; every edited `.blend` reopens, every GLB passes structural
  validation, and no manual fixes or broken dependencies were recorded.
- Controlled print-task subset: **3/3** box/bottle/gear assets convert to
  watertight voxel meshes, pass a 0.030 normalized thickness probe, export to
  STL, and re-import with zero non-manifold edges. Physical units remain
  unknown, so this is not a millimetre manufacturing guarantee.
- Eval 27 existing-cohort runtime: mean **89.0 s**, median **81.5 s**, and p90
  **148.3 s** across 18 cases. New prospective manifests add grouped monotonic
  timings and peak RSS; an instrumented smoke run used 441.3 MiB peak RSS.
- Exact-intrinsics semantic-completion ablation: six-family median held-out IoU
  improves from 0.485 to 0.757 (**+0.272**), but surface Chamfer worsens on
  every case receiving a prior.
- Clean no-visual-hull ablation: editable parametric geometry improves
  six-family median Chamfer from 0.078 to **0.044**, but median held-out IoU
  collapses from 0.757 to **0.176** because view-consistent pose/depth remains
  unresolved.

The reproducible results and per-case table are in
[`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md). Generated projects and raw result
directories remain gitignored.

## Implemented system

- All 18 stages described by `GOAL.md`: input, target selection,
  segmentation, crop normalization, preprocessing, vectorization, cleanup,
  primitive fitting, constraints, sketch graph, semantic decomposition,
  camera estimation, depth/normals, operator selection, construction plan,
  Blender generation, sandboxed execution, render validation, refinement,
  reporting, and export.
- Manufactured-object operator families: revolve, extrude, sweep, primitive
  assembly, Boolean, loft/freeform fallback, displacement, mirror, and radial
  array.
- Evidence provenance and confidence are retained across observed geometry,
  geometric/semantic inference, and generated hypotheses.
- Phase 6 multiview support independently processes secondary views, builds
  source-labelled cross-view part matches, estimates relative pose and scale
  consensus, and preserves primary observed primitives. Calibrated views now
  alter shared 3D geometry through voxel-carved visual hulls; original semantic
  parts remain hidden editable guides.
- Phase 7 hidden-geometry support proposes and audits revolve
  cross-sections, hidden-side continuations, mirror completions, and partial
  occlusion completions. Hypothesis confidence is capped at 0.5 and every
  candidate is accepted or rejected explicitly.
- Corrupt, missing, and unsupported source files now produce the distinct
  `unsupported_input` outcome, retain a partial project, record a
  machine-readable reason, and explicitly avoid a reconstruction success claim
  in the report.
- Multiview calibration now separates capped hidden-surface confidence from
  held-out silhouette-pass probability, supports serialized external
  calibrators, and emits deterministic out-of-fold evidence without fitting on
  the case being scored.
- Refinement now preserves confidence-1.0 user-supplied object pose instead of
  silently replacing it with a lower-confidence silhouette hypothesis.
- Held-out surface metrics are sampled in canonical object pose, independent
  of the yaw used to render the held-out silhouette; visual-hull geometry also
  ignores inapplicable parametric base rotations.
- Input-quality preflight records structured, source-labelled hazards and
  evidence recommendations. High-risk inputs can return only
  `partial_success`, even when a technically valid model is produced.
- Blender manifests now expose local location, Euler rotation, and scale so
  downstream edits can be verified after a real rebuild instead of assumed
  from plan changes.
- Run manifests now expose grouped stage timings, total monotonic runtime,
  peak process RSS, model-call count, and an explicit unmeasured GPU-memory
  field; the runtime summarizer remains compatible with legacy timestamps.
- Per-project Blender MCP configuration is tracked in `.codex/config.toml`.
- All eleven required stage ablations now have executable configs; smoke
  evidence and complete matched 11-way `box_01`, `bottle_01`, and `gear_01`
  matrices are documented in
  [`ABLATION_REPORT.md`](ABLATION_REPORT.md).

## Verification commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests evals/unit evals/stage

PYTHONPATH=. .venv/bin/python evals/e2e/run_e2e.py \
  --cases all \
  --projects-root projects/e2e_final \
  --out evals/results_final \
  --workers 1 \
  --python .venv/bin/python
```

The latest test run produced `281 passed, 1 skipped`. The single-view E2E run produced
`18/18 passed MVP | silhouette IoU mean 0.910 | baseline IoU mean 0.890`.

The calibrated multiview commands, exact-camera held-out result, and ablation
are recorded in [`MULTIVIEW_REPORT.md`](MULTIVIEW_REPORT.md).
Controlled failure detection and downstream edit-task evidence are in
[`FAILURE_DETECTION_REPORT.md`](FAILURE_DETECTION_REPORT.md) and
[`DOWNSTREAM_REPORT.md`](DOWNSTREAM_REPORT.md).
Runtime evidence is in [`RUNTIME_REPORT.md`](RUNTIME_REPORT.md).
Perceptual render evidence is in [`PERCEPTUAL_REPORT.md`](PERCEPTUAL_REPORT.md).

## Additional ablation evidence

- No refinement, all 18 cases: mean IoU falls from 0.910 to 0.880; pipe and
  both wheels fall below the 0.80 silhouette gate.
- No depth/normals, three-family sample (`bottle_01`, `gear_01`,
  `pipe_elbow_01`): 2/3 hard-gate passes and mean IoU 0.906. The effect is
  mixed; the pipe loses 0.0044 IoU and falls below its gate.
- Blender-rendered direct SVG extrusion, all 18 cases: mean IoU 0.890 but zero
  semantic part recall and zero meaningful editability by definition.

The depth/normals ablation config is tracked at
`evals/ablations/no_depth_normals.yaml`.

## Known limits and next research work

These do not block the synthetic single-view benchmark result, but they do
block the full `GOAL.md` success definition:

1. The 18-case benchmark is synthetic and uses a supplied mask and label for
   the reconstruction path. Unguided segmentation is scored where available,
   but real-photo coverage remains limited.
2. Full camera calibration and exact held-out views now cover all 18 benchmark
   objects, but the improved suite still misses both Eval 20 median targets.
3. Mean camera score is 0.500 because single-view focal length/physical scale
   remain weakly observable; the system reports that uncertainty rather than
   inventing calibration. An absolute-orbit experiment was rejected because
   observed extrusion profiles are still camera-frame rather than canonical;
   applying the orbit double-counted foreshortening and reduced `box_01`
   primary IoU to 0.538.
4. `pipe_elbow_01` passes narrowly at 0.80008. `chair_01`, `pipe_elbow_01`,
   and `table_01` retain internal `partial_success` status because the
   pipeline's preferred refinement target is 0.90 even though every MVP hard
   gate passes.
5. The 11-way matrix is complete for `box_01`, `bottle_01`, and `gear_01`, but
   broader geometrically sensitive coverage is still required;
   opaque image-to-mesh/VLM baselines, human quality evaluation, runtime engine
   import and known-scale manufacturing validation from
   `EVAL.md` remain research evaluation work. Eval 28 now passes its controlled
   suite, but natural-image semantic-hazard detection is not yet established.

## Repository checkpoints

- `175c018` — project-level Blender MCP configuration.
- `39bd7fc` — semantic planning and benchmark hardening.
- `d5d864e` — multiview, hypotheses, targeted geometry fixes, and refinement
  hardening.

`main` is pushed to `origin/main` after each completed implementation chunk.
