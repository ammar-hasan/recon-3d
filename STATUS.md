# STATUS — Image-to-Editable-3D Reconstruction Pipeline

Last updated: 2026-07-23 after calibrated multiview held-out evaluation.

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
- Regression suite: **265 passed, 1 skipped**, including real Blender build,
  validation, and refinement tests.
- Calibrated two-evidence-view `box_01`: primary IoU **0.961** and genuinely
  held-out `+90°` IoU **0.796** with exact primary intrinsics (target ≥ 0.75).
- Full 18-case calibrated suite: median held-out IoU **0.669**, below Eval 20's
  silhouette target ≥ 0.75; 7/18 individual cases pass.
- The same suite's median normalized surface Chamfer is **0.074**, above Eval
  20's ≤ 0.05 target; 3/18 pass Chamfer and only `gear_01` passes both
  measured targets.
- High operational risk detects 8/11 held-out silhouette failures with no
  high-risk false alarms. Deterministic out-of-fold confidence ECE is **0.203**,
  so Eval 24's ≤ 0.08 target remains open.
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

The latest test run produced `265 passed, 1 skipped`. The single-view E2E run produced
`18/18 passed MVP | silhouette IoU mean 0.910 | baseline IoU mean 0.890`.

The calibrated multiview commands, exact-camera held-out result, and ablation
are recorded in [`MULTIVIEW_REPORT.md`](MULTIVIEW_REPORT.md).

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
2. Calibrated multiview ground truth and exact held-out views now cover all 18
   benchmark objects, but the suite misses both Eval 20 median targets.
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
   opaque image-to-mesh/VLM baselines,
   failure-detection evaluation, and human edit-task evaluation from
   `EVAL.md` remain research evaluation work. Only the concrete ablations
   listed above have been run.

## Repository checkpoints

- `175c018` — project-level Blender MCP configuration.
- `39bd7fc` — semantic planning and benchmark hardening.
- `d5d864e` — multiview, hypotheses, targeted geometry fixes, and refinement
  hardening.

`main` is pushed to `origin/main` after each completed implementation chunk.
