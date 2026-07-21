# recon3d — Module Contracts

Every stage module implements exactly the public functions below.
Stages communicate ONLY through `recon3d.schemas` models and files in the
project directory. Coordinates:

- **pixels**: original image pixel coordinates.
- **crop pixels**: pixels in the normalised square canvas.
- **normalised**: 0..1 over the crop canvas, origin top-left (2D stages).
- **object units** (3D): object width = 1.0, origin at object centre,
  x right, y up, z toward camera.

Project directory layout (produced by `pipeline.py`):

```text
project/
├── input/original.png
├── segmentation/{object_mask.png, object_rgba.png, crop_metadata.json}
├── traces/{silhouette.svg, color_regions.svg, structural_edges.svg, details.svg}
├── geometry/{fitted_primitives.json, sketch_graph.json, depth.png, normals.png, construction_plan.yaml}
├── blender/{build_model.py, scene.blend, model.glb}
├── validation/{reference_overlay.png, silhouette_comparison.png, depth_comparison.png, turntable.mp4, metrics.json}
├── manifest.json
└── report.md
```

## Stage modules

```python
# input_manager.py  (Stage 1)
def load_input(spec: InputSpec) -> InputBundle
    # Validates + loads images (PNG/JPEG/WEBP, alpha, grayscale, EXIF
    # orientation). Copies original into <output_dir>/input/original.png.
    # Raises InputError with a clear message on corrupt/unsupported files.

# segmentation.py  (Stage 2)
def segment(bundle: InputBundle, out_dir: str, cfg: PipelineConfig) -> SegmentationResult
    # Backends: user_mask > box/point-guided grabcut > rembg > classical.
    # Writes object_mask.png, object_rgba.png into out_dir. Never edits the
    # original. Honours spec.target_label/point/box/mask_path.

# crop.py  (Stage 3)
def make_crop(seg: SegmentationResult, out_dir: str, cfg: PipelineConfig
              ) -> tuple[CropMetadata, str, str]
    # Returns (metadata, crop_rgba_path, crop_mask_path). Padded square crop,
    # aspect preserved, exact transform recorded. Writes crop_metadata.json.

# preprocess.py  (Stage 4)
def preprocess(crop_rgba_path: str, crop_mask_path: str, out_dir: str,
               cfg: PipelineConfig) -> PreprocessLayers
    # Writes silhouette.png, color_quantized.png, structural_edges.png,
    # details.png, lighting_normalized.png into out_dir. Deterministic.

# vectorize.py  (Stage 5)
def vectorize(layers: PreprocessLayers, out_dir: str, cfg: PipelineConfig) -> list[TraceLayer]
    # vtracer backend with OpenCV-contour fallback. Writes the 4 SVGs.

# svg_cleanup.py  (Stage 6)
def cleanup_layers(layers: list[TraceLayer], out_dir: str,
                   cfg: PipelineConfig) -> list[TraceLayer]
    # Removes tiny/duplicate paths, simplifies (RDP), smooths, fixes closure,
    # detects holes + nesting, normalises coordinates to 0..1, keeps svg_d.

# primitives.py  (Stage 7)
def fit_primitives(layers: list[TraceLayer], cfg: PipelineConfig) -> list[GeometricPrimitive]
    # Fits line/polyline/circle/arc/ellipse/rectangle/polygon/spline/region.
    # Always keeps fallback_points. fit_error in normalised units.

# constraints.py  (Stage 8)
def detect_constraints(primitives: list[GeometricPrimitive],
                       cfg: PipelineConfig) -> list[GeometricConstraint]
    # All ConstraintType relations. Precision over recall: only report
    # constraints with confidence >= 0.6.

# sketch_graph.py  (Stage 9)
def build_sketch_graph(primitives: list[GeometricPrimitive],
                       constraints: list[GeometricConstraint]) -> SketchGraph

# semantic_parts.py  (Stage 10)
def decompose_parts(graph: SketchGraph, crop_rgba_path: str, spec: InputSpec,
                    cfg: PipelineConfig) -> SketchGraph
    # Groups primitives into SemanticParts (labels, hierarchy, appearance,
    # visibility, confidence). Heuristic + optional VLM hook (pluggable).
    # Never overwrites observed geometry; marks inferred geometry.

# camera.py  (Stage 11)
def estimate_camera(graph: SketchGraph, seg: SegmentationResult,
                    crop_meta: CropMetadata, spec: InputSpec,
                    cfg: PipelineConfig) -> CameraEstimate
    # Ellipse->circle unprojection, vanishing hints, ortho/persp decision.
    # Scale stays UNKNOWN unless spec.known_dimension given.

# depth.py  (Stage 12)
def estimate_depth(crop_rgba_path: str, crop_mask_path: str,
                   graph: SketchGraph, out_dir: str,
                   cfg: PipelineConfig) -> DepthEvidence
    # Optional backends; shading/shape-from-silhouette fallback. Writes
    # depth.png + normals.png when enabled. Kept separate from vector evidence.

# operators.py  (Stage 13)
def classify_operators(graph: SketchGraph, depth: DepthEvidence,
                       cfg: PipelineConfig) -> SketchGraph
    # Fills construction_candidates + selected_operator per part.

# construction_plan.py  (Stage 14)
def build_plan(graph: SketchGraph, camera: CameraEstimate, depth: DepthEvidence,
               spec: InputSpec, cfg: PipelineConfig) -> ConstructionPlan
def validate_plan(plan: ConstructionPlan) -> list[str]   # list of errors, [] = valid

# materials.py  (Stage 16 support)
def estimate_materials(graph: SketchGraph, crop_rgba_path: str,
                       cfg: PipelineConfig) -> dict[str, MaterialSpec]
    # Per-part PBR estimates from quantized colours; strips highlights/shadows.

# blender_codegen.py  (Stage 15)
def generate_blender_script(plan: ConstructionPlan, out_dir: str,
                            cfg: PipelineConfig) -> str   # returns script path
    # Pure function of the plan -> deterministic bpy script building named,
    # hierarchical, editable scene + materials + exports scene.blend/model.glb.
    # The generated script must contain NO file/network/subprocess access
    # outside the project dir.

# runner.py  (Stage 15 execution, sandboxed)
def run_blender(script_path: str, project_dir: str, cfg: PipelineConfig,
                extra_args: list[str] | None = None) -> BlenderManifest
    # Runs blender --background --python script in a constrained env:
    # cwd=project_dir, no network env, timeout, output capture, manifest
    # JSON written by the script itself and parsed back.

# validation.py  (Stage 17)
def validate_reconstruction(manifest: BlenderManifest, plan: ConstructionPlan,
                            seg: SegmentationResult, crop_meta: CropMetadata,
                            project_dir: str, cfg: PipelineConfig) -> ValidationResult
    # Generates a render script (estimated camera), renders silhouette/clay/
    # depth/normal/part-id passes, computes ValidationMetrics, writes overlays
    # + turntable.mp4 into validation/.

# refinement.py  (Stage 18)
def refine(plan: ConstructionPlan, seg: SegmentationResult, crop_meta: CropMetadata,
           project_dir: str, cfg: PipelineConfig
           ) -> tuple[ConstructionPlan, BlenderManifest, ValidationResult, RefinementLog]
    # generate -> render -> compare -> tweak responsible parameters -> repeat.
    # Keeps best valid model, rolls back regressions, full audit trail.
```

## Rules for all modules

- Deterministic: seed every RNG from `cfg.seed`.
- Never silently mix evidence categories; use `EvidencedValue`.
- Every stage writes its schema outputs into the project dir AND returns them.
- No network access at runtime except the rembg model download (cached) and
  optional VLM/depth backends, which must degrade gracefully when absent.
- Errors: raise clear exceptions; the orchestrator converts them into a
  `partial_success` / `unsupported_input` outcome, never a silent bad mesh.
