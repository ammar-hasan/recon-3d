# Eval 21 Material Accuracy

Date: 2026-07-23

## Method

The 18 synthetic reference GLBs retain authoritative material names and PBR
parameters. The evaluator maps each ground-truth major part to its reference
node and material, maps the same semantic label to the generated GLB, and
compares material class, base color (Delta E76), metallic, and roughness.
Missing generated assignments count as failures. Material classes are derived
only from explicit exported names such as `DarkRubber`, `Steel`, `Plastic*`,
`Wood`, `Glass*`, and `Ceramic`.

## Results

| Case | Assigned | Class correct | Major parts |
| --- | ---: | ---: | ---: |
| `bottle_01` | 2 | 1 | 2 |
| `bottle_02` | 2 | 1 | 2 |
| `box_01` | 2 | 0 | 2 |
| `bracket_01` | 3 | 0 | 3 |
| `chair_01` | 3 | 0 | 3 |
| `crate_01` | 1 | 0 | 3 |
| `gear_01` | 1 | 1 | 2 |
| `gear_02` | 1 | 1 | 2 |
| `knob_01` | 1 | 0 | 1 |
| `lamp_01` | 4 | 3 | 4 |
| `mug_01` | 2 | 0 | 2 |
| `mug_02` | 2 | 1 | 2 |
| `pipe_elbow_01` | 3 | 3 | 3 |
| `sign_01` | 2 | 0 | 2 |
| `table_01` | 2 | 0 | 2 |
| `vase_01` | 1 | 0 | 1 |
| `wheel_01` | 4 | 2 | 4 |
| `wheel_02` | 4 | 2 | 4 |

- Major-part material-assignment accuracy: **0.909**, meeting the ≥0.90
  target.
- Material-class accuracy: **0.341**, failing the ≥0.80 target.
- Median base-color Delta E76: **22.795**. `EVAL.md` intentionally leaves the
  acceptable color threshold class-specific, so no global pass is claimed.
- Median metallic absolute error: **0.050**.
- Median roughness absolute error: **0.200**.

The existing controlled highlight/shadow regression confirms that extreme
white and black patches are trimmed before base-color estimation. A broad
highlight-baking rate requires additional relit ground-truth images and is not
claimed from this single-lighting cohort.

Reproduce with:

```bash
PYTHONPATH=. .venv/bin/python -m evals.materials.run_material_eval \
  --projects-root projects/e2e_final \
  --out projects/material_eval21.json
```
