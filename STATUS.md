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
- Regression suite: **253 passed, 1 skipped**, including real Blender build,
  validation, and refinement tests.
- Calibrated two-evidence-view `box_01`: primary IoU **0.962** and genuinely
  held-out `+90°` IoU **0.903** (target ≥ 0.75).
- Six-family calibrated suite: median held-out IoU **0.875** (Eval 20
  silhouette target ≥ 0.75), with 4/6 individual cases passing.
- The same suite's median normalized surface Chamfer is **0.077**, so Eval
  20's surface target (≤ 0.05) remains open; only `gear_01` passes it.
- Semantic unseen-view risk detects both held-out silhouette failures with no
  high-risk false alarms in this six-case suite. Confidence ECE is **0.241**,
  so Eval 24 calibration (target ≤ 0.08) remains open.
- Two-view maximal-hull ablation: held-out IoU **0.532**; the explicit
  source-labelled enclosure symmetry hypothesis contributes **+0.287 IoU**.

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
- Per-project Blender MCP configuration is tracked in `.codex/config.toml`.
- All eleven required stage ablations now have executable configs; smoke
  evidence and the still-unrun matched matrix are documented in
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

The latest test run produced `253 passed, 1 skipped`. The single-view E2E run produced
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
2. Calibrated multiview ground truth and a held-out view now exist for
   `box_01`, but cross-family calibrated coverage is not yet established.
3. Mean camera score is 0.500 because single-view focal length/physical scale
   remain weakly observable; the system reports that uncertainty rather than
   inventing calibration.
4. `pipe_elbow_01` passes narrowly at 0.80008. `chair_01`, `pipe_elbow_01`,
   and `table_01` retain internal `partial_success` status because the
   pipeline's preferred refinement target is 0.90 even though every MVP hard
   gate passes.
5. The full 11-way ablation matrix, opaque image-to-mesh/VLM baselines,
   failure-detection evaluation, and human edit-task evaluation from
   `EVAL.md` remain research evaluation work. Only the concrete ablations
   listed above have been run.

## Repository checkpoints

- `175c018` — project-level Blender MCP configuration.
- `39bd7fc` — semantic planning and benchmark hardening.
- `d5d864e` — multiview, hypotheses, targeted geometry fixes, and refinement
  hardening.

`main` is pushed to `origin/main` after each completed implementation chunk.
