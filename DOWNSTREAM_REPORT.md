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
runtime import. Human completion-time studies are not yet measured.

## Controlled 3D-print conversion

Box, bottle, and gear reconstructions were joined and voxel-remeshed at 0.015
normalized units, checked for manifold/boundary edges, probed inward for a
minimum thickness of 0.030 normalized units, exported to STL, and re-imported.

```yaml
watertight_conversion_rate: 1.0
minimum_thickness_pass_rate: 1.0
stl_export_reimport_rate: 1.0
task_completion_rate: 1.0
median_elapsed_seconds: 0.73
```

The measured minimum ray thicknesses were 0.124 (box), 0.310 (bottle), and
0.121 (gear), with zero non-manifold or boundary edges before export and after
STL re-import. Reproduce with:

```bash
PYTHONPATH=. .venv/bin/python -m evals.downstream.run_print_tasks \
  --projects-root projects/e2e_final \
  --out evals/results_printing \
  --cases box_01,bottle_01,gear_01
```

This is a relative-geometry check. Most source reconstructions have unknown
physical scale, so 0.030 normalized units cannot be presented as a millimetre
manufacturing guarantee. Voxel remeshing also trades fine detail for
watertightness; dimensional tolerance against a known-scale reference remains
open.
