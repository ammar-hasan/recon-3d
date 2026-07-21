# soul-3d — Image-to-Editable-3D Reconstruction

Converts reference images of manufactured / hard-surface objects into
editable, semantically structured Blender models, via progressive structured
representations (mask → crop → vector traces → primitives → constraints →
semantic sketch graph → construction plan → Blender scene → render-validated
refinement). See `GOAL.md` for the full spec and `EVAL.md` for the
evaluation criteria. Module interfaces are fixed in `CONTRACTS.md`; shared
data models live in `recon3d/schemas.py`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Requires Blender 4.x+ at `/Applications/Blender.app/Contents/MacOS/Blender`
(configurable via `PipelineConfig.blender.blender_bin`).

## Run

```bash
.venv/bin/python -m recon3d.pipeline --image my_object.png --label wheel --out projects/wheel_01
```

Outputs land in the project directory (input/, segmentation/, traces/,
geometry/, blender/, validation/, report.md, manifest.json).

## Evaluate

```bash
.venv/bin/python -m pytest evals/unit -q          # Level A unit evals
.venv/bin/python -m evals.run_evals --level stage  # Level B stage evals
.venv/bin/python -m evals.run_evals --level e2e    # Level C end-to-end on benchmark
```
