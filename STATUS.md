# STATUS — Image-to-Editable-3D Reconstruction Pipeline

Last updated: after integration hardening (commit `a7ece4d`).
Resume instructions are at the bottom.

## What exists and works

- **Full 18-stage pipeline** in `recon3d/` per GOAL.md + CONTRACTS.md:
  input → segmentation (rembg/grabcut/classical) → crop → preprocessing
  (5 evidence layers) → vectorize (vtracer) → SVG cleanup → primitive
  fitting → constraint detection → sketch graph → semantic parts → camera →
  depth/normals → operator classification → construction plan → Blender
  codegen → sandboxed runner → render validation → refinement loop →
  report.md + manifest.json per run.
- **Eval framework** in `evals/`: metrics lib, Level A unit tests, Level B
  stage tests, Level C e2e runner (`evals/e2e/run_e2e.py` with MVP hard
  gates, baselines, dashboard), umbrella CLI `evals/run_evals.py`.
- **Synthetic benchmark**: 18 cases in `evals/benchmark/dataset/` (gitignored,
  regenerable via `evals/benchmark/generate_benchmark.py`), covering
  revolution/extruded/radial/mirrored/assembly/sweep classes with full GT.
- **Test status**: `pytest tests/ evals/unit evals/stage -q` → 203 passed,
  1 skipped (includes real Blender build/validation/refinement tests).
- **Verified e2e runs**:
  - `projects/smoke_wheel_01`: status=success, silhouette IoU 0.720 →
    **0.912** after refinement (clay 0.903, chamfer 0.008, SSIM 0.906).
    Plan: wheel/tyre/rim/hub revolve parts. Spokes NOT yet reconstructed
    (no clean count-5 repetition cluster found in traces).
  - `projects/smoke_bracket_01`: partial_success, valid plan (extrude +
    booleans), but silhouette IoU 0.338 — flat plate modeled face-on
    (orientation of extruded parts is wrong for this view).

## Environment notes

- Python 3.9.6 venv at `.venv/` (no match statements, no PEP 604 unions).
- Blender 5.2.0 at `/Applications/Blender.app/Contents/MacOS/Blender`.
  Eevee id is `BLENDER_EEVEE`; Principled input is `Transmission Weight`;
  no capsule primitive; `matrix_world` stale until `view_layer.update()`.
- rembg model cached at `~/.u2net/isnet-general-use.onnx`.
- Blender MCP is connected (interactive session) — prefer background mode
  for pipeline work to avoid touching the open user project.
- Codex CLI available at `~/.local/bin/codex` for occasional second opinions.
- User instruction: **commit + push (origin/main) after every big chunk.**

## Remaining work (priority order)

1. **Run full e2e benchmark** (all 18 cases):
   `.venv/bin/python -m evals.run_evals --level e2e`
   (or `evals/e2e/run_e2e.py --cases all --out evals/results/`).
   Inspect `dashboard.json/dashboard.md`, record per-case hard-gate status.
2. **Fix extruded-part orientation** (bracket_01 IoU 0.338): extrude depth
   axis must align with camera view axis for flat plates, not face-on
   default. Likely affects sign_01, gear_01/02, crate/box cases too.
3. **Spoke/radial repetition recovery**: wheel spokes not reconstructed —
   improve repetition clustering on spoke traces (structural_edges layer)
   so radial_array(count=5) fires for wheels/gears.
4. **Iterate per-case failures** from the dashboard until MVP hard gates
   pass on easy+medium cases (segmentation ≥0.85, plan valid, blender ok,
   blend reopens, glb valid, silhouette IoU ≥0.80, part recall ≥0.85,
   editability, 0 safety violations).
5. **Phase 5 polish**: depth/normals exist (silhouette+shading); verify
   depth-aware profiles help and don't regress (Eval 12).
6. **Phase 6 — multiview**: multiple images accepted by InputSpec but no
   cross-view reasoning yet. Need: per-view segmentation/traces, shared
   part graph, relative camera pose solving, consistent scale, joint
   silhouette optimization. (Biggest missing feature.)
7. **Phase 7 — generative hypotheses**: module to propose hidden-side
   completion / cross-sections / orthographic views as
   `generated_hypothesis` (low confidence), validated against observed
   evidence, rejected when inconsistent. Procedural mirror/revolve
   hypotheses are acceptable (no external gen model required).
8. **Ablations + regression suite** runs for the report (EVAL.md requires
   ablation table: no-refinement, no-primitive-fitting, etc.).
9. **Final report + README refresh**, final commit + push.

## How to resume

1. `git pull` (everything is pushed; latest commit on `main`).
2. Read this file + TODO state above; item 1 is the next action.
3. Run tests to confirm a green baseline:
   `.venv/bin/python -m pytest tests/ evals/unit evals/stage -q`
4. Continue with remaining work item 1.
