# Eval 22 Perceptual Render Evaluation

Date: 2026-07-23

## Scope and method

This evaluation uses the final refined reference-view projects from the
18-case Level C cohort. `recon3d.validation` renders silhouette, neutral clay,
world-space normal, depth, part-ID, material-ID, full shaded, and turntable
passes from the same scene and camera. The host compares the full shaded pass
with the isolated reference crop using masked grayscale SSIM and mean color
agreement. Clay silhouette IoU is reported beside shaded SSIM so a strong
material render cannot conceal weak geometry.

A geometry-compensation flag is raised when shaded SSIM is at least 0.80 but
clay silhouette IoU is below 0.80. `EVAL.md` defines no numeric pass target for
Eval 22, so these results are measurements rather than a fabricated pass gate.

## Results

| Case | Shaded SSIM | Color agreement | Clay silhouette IoU | Geometry-compensation flag |
| --- | ---: | ---: | ---: | --- |
| `bottle_01` | 0.970 | 0.777 | 0.955 | no |
| `bottle_02` | 0.962 | 0.744 | 0.967 | no |
| `box_01` | 0.926 | 0.749 | 0.918 | no |
| `bracket_01` | 0.967 | 0.769 | 0.905 | no |
| `chair_01` | 0.960 | 0.772 | 0.853 | no |
| `crate_01` | 0.881 | 0.729 | 0.896 | no |
| `gear_01` | 0.951 | 0.701 | 0.909 | no |
| `gear_02` | 0.934 | 0.769 | 0.939 | no |
| `knob_01` | 0.931 | 0.804 | 0.915 | no |
| `lamp_01` | 0.960 | 0.726 | 0.902 | no |
| `mug_01` | 0.909 | 0.752 | 0.897 | no |
| `mug_02` | 0.916 | 0.751 | 0.900 | no |
| `pipe_elbow_01` | 0.919 | 0.793 | 0.805 | no |
| `sign_01` | 0.937 | 0.926 | 0.907 | no |
| `table_01` | 0.953 | 0.812 | 0.863 | no |
| `vase_01` | 0.968 | 0.936 | 0.927 | no |
| `wheel_01` | 0.815 | 0.512 | 0.911 | no |
| `wheel_02` | 0.814 | 0.678 | 0.941 | no |

- Mean / median shaded SSIM: **0.926 / 0.936**.
- Mean / median color-region agreement: **0.761 / 0.761**.
- Mean / median clay silhouette IoU: **0.906 / 0.908**.
- Geometry-compensation flags: **0/18**.
- Mean depth correlation: **0.201**; median **0.052**. Monocular depth remains
  weak and the shaded scores must not be interpreted as full 3D accuracy.

## Render-pass coverage

The historical 18-case cohort contains silhouette, clay, normal, depth,
part-ID, and shaded passes for every case. Material-ID was not emitted when
that cohort was generated. The standard validation renderer now emits
`render_materialid.png`; a real Blender regression builds the pass and verifies
the file. Future cohorts therefore cover all seven required passes without
retroactively claiming that the legacy artifacts contain it.

Reproduce the aggregate from an existing project cohort with:

```bash
PYTHONPATH=. .venv/bin/python -m evals.perceptual.summarize \
  --projects-root projects/e2e_final \
  --out projects/perceptual_eval22.json
```
