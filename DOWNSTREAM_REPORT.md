# Eval 30 — Downstream Task Evaluation

Date: 2026-07-23

## Result

Seven concrete tasks were applied to reconstruction plans from the completed
18-case E2E project set. Every edited plan was rebuilt through the production
Blender code generator, reopened independently in background Blender, and
exported to a structurally valid GLB.

| Task | Source family | Outcome |
| --- | --- | --- |
| Resize a component | box | pass |
| Change radial repetition 5→6 | wheel | pass |
| Widen a revolve profile | bottle | pass |
| Move a child/joint | chair | pass |
| Replace a material | box | pass |
| Rotate a wheel | wheel | pass |
| Open/articulate a cap | bottle | pass |

```yaml
task_completion_rate: 1.0
median_elapsed_seconds: 1.41
manual_fixes_total: 0
broken_dependencies_total: 0
blend_reopen_rate: 1.0
glb_structural_validation_rate: 1.0
model_rebuild_required_rate: 1.0
```

The evaluator verifies that each changed parameter survives into the Blender
manifest: local scale, radial duplicate count, nonempty modified profile,
child location, named material, or local Euler rotation as appropriate.
Blender manifests now retain local location, rotation, and scale explicitly,
which makes these checks auditable rather than inferred from build success.

## Reproduction

First generate the normal E2E projects, then run:

```bash
PYTHONPATH=. .venv/bin/python -m evals.downstream.run_edit_tasks \
  --projects-root projects/e2e_final \
  --out evals/results_downstream
```

## Scope boundary

This closes automated editing, variant-generation, basic articulation, and
structural game-export checks. The GLB check validates container structure,
meshes, nodes, materials, and accessors; it is not an actual Unity/Unreal
runtime import. 3D-printing tasks (watertight conversion, minimum thickness,
and STL export) and human completion-time studies are not yet measured and are
not claimed as passes.
