# Goal: Build an Image-to-Editable-3D Reconstruction Pipeline

## Mission

Build a modular system that converts one or more reference images of an object into an editable, semantically structured 3D model in Blender.

The system must not rely on a single opaque image-to-mesh generation step. It should progressively transform visual evidence into increasingly structured representations:

```text
reference image
→ object isolation
→ normalized crop
→ vector traces
→ simplified geometric primitives
→ constrained parametric sketch
→ inferred 3D construction plan
→ Blender geometry
→ render-based validation and refinement
```

The final result should be a Blender scene whose geometry is understandable, editable, reusable, and organized into meaningful parts.

---

## Primary Objective

Given a reference image containing a target object, automatically produce:

1. A clean foreground mask.
2. A normalized object crop.
3. One or more vector traces of the object.
4. A simplified geometric representation of those traces.
5. A semantic decomposition of the object into parts.
6. A parametric construction plan for each part.
7. A Blender model generated from that plan.
8. A rendered comparison against the reference.
9. A refinement loop that improves geometric agreement.
10. Exportable Blender and interchange-format assets.

The first version should prioritize manufactured and hard-surface objects with clear silhouettes, including:

* wheels and rims;
* containers;
* appliances;
* furniture;
* brackets and mechanical parts;
* logos and reliefs;
* basic product designs;
* stylized game props;
* architectural details.

---

## Core Design Principle

The system must distinguish between:

* **observed evidence**, directly extracted from the image;
* **geometric inference**, estimated from visual evidence;
* **semantic inference**, based on object knowledge;
* **generated hypotheses**, invented to explain missing or occluded geometry.

These categories must never be silently mixed.

Every inferred property should include its source and confidence where practical.

Example:

```yaml
wheel_radius:
  value: 0.126
  unit: normalized_object_width
  source: fitted_outer_ellipse
  confidence: 0.91

tyre_depth:
  value: 0.042
  unit: normalized_object_width
  source: monocular_depth_and_object_prior
  confidence: 0.48

rear_side_geometry:
  source: generated_hypothesis
  confidence: 0.31
```

---

# System Outputs

For each reconstruction, produce a project directory similar to:

```text
project/
├── input/
│   └── original.png
├── segmentation/
│   ├── object_mask.png
│   ├── object_rgba.png
│   └── crop_metadata.json
├── traces/
│   ├── silhouette.svg
│   ├── color_regions.svg
│   ├── structural_edges.svg
│   └── details.svg
├── geometry/
│   ├── fitted_primitives.json
│   ├── sketch_graph.json
│   ├── depth.png
│   ├── normals.png
│   └── construction_plan.yaml
├── blender/
│   ├── build_model.py
│   ├── scene.blend
│   └── model.glb
├── validation/
│   ├── reference_overlay.png
│   ├── silhouette_comparison.png
│   ├── depth_comparison.png
│   ├── turntable.mp4
│   └── metrics.json
└── report.md
```

---

# Required Pipeline

## Stage 1: Input and Object Selection

Accept:

* a single image;
* multiple images of the same object;
* an optional text description;
* an optional target-object label;
* an optional point, box, or mask;
* an optional known physical dimension.

The system should support both automatic and user-guided object selection.

Examples:

```text
Select the front-left wheel.
```

```text
Select the coffee machine.
```

```text
Use the object inside this bounding box.
```

---

## Stage 2: Foreground Segmentation

Isolate the target object from the background.

Potential tools may include:

* rembg;
* SAM-family models;
* Grounded SAM;
* interactive segmentation;
* custom masks.

Generate:

* a binary mask;
* a transparent foreground image;
* the original untouched image;
* segmentation confidence or diagnostics.

The original image must always be preserved because the background may contain useful information about:

* contact shadows;
* scale;
* ground plane;
* perspective;
* reflections;
* lighting;
* occlusion;
* camera orientation.

---

## Stage 3: Crop and Coordinate Normalization

Create a padded crop around the object.

Requirements:

* preserve aspect ratio;
* do not stretch the object;
* add configurable padding;
* center the target;
* normalize to a standard canvas;
* record the exact crop transformation.

Example metadata:

```json
{
  "source_image_size": [1920, 1080],
  "source_bbox": [410, 160, 1480, 1010],
  "padding": 72,
  "output_size": [1024, 1024],
  "scale": 0.71,
  "offset": [-214, -48]
}
```

Every later coordinate must be transformable between:

* normalized crop coordinates;
* original pixel coordinates;
* estimated camera coordinates;
* Blender world coordinates.

---

## Stage 4: Image Preprocessing

Generate several purpose-specific image variants.

At minimum:

### Foreground silhouette

A binary image representing the outer object boundary and internal holes.

### Color-quantized image

Reduce the object to a configurable number of color regions while preserving meaningful material boundaries.

### Structural-edge image

Extract visible geometric boundaries such as:

* panel seams;
* holes;
* ridges;
* spokes;
* creases;
* sharp transitions;
* profile edges.

### Detail image

Preserve small visual features such as:

* tread;
* vents;
* logos;
* engravings;
* fasteners;
* decorative patterns.

### Lighting-normalized image

Reduce shadows, highlights, reflections, and uneven illumination where possible without altering geometry.

Each preprocessing result should remain a separate evidence layer.

---

## Stage 5: Vectorization

Use VTracer or a comparable raster-to-vector system to generate separate SVG representations.

Required SVG layers:

```text
silhouette.svg
color_regions.svg
structural_edges.svg
details.svg
```

Do not immediately merge these SVGs.

Each path should retain metadata describing its origin:

```yaml
path_id: edge_014
source_layer: structural_edges
closed: false
confidence: 0.74
```

---

## Stage 6: SVG Cleanup and Simplification

Convert noisy traced paths into compact geometry.

Perform:

* removal of tiny isolated paths;
* removal of near-duplicate paths;
* control-point reduction;
* path smoothing;
* closure correction;
* self-intersection repair;
* winding normalization;
* path hierarchy detection;
* hole detection;
* contour ordering.

Simplification must preserve meaningful shape while reducing unnecessary complexity.

The goal is not merely fewer Bézier points. The goal is a representation suitable for geometric reasoning.

---

## Stage 7: Geometric Primitive Fitting

Attempt to replace arbitrary SVG paths with recognizable primitives.

Supported primitive types should include:

* point;
* line segment;
* polyline;
* circle;
* circular arc;
* ellipse;
* elliptical arc;
* rectangle;
* rounded rectangle;
* regular polygon;
* symmetric spline;
* arbitrary Bézier curve;
* closed region.

Example:

```yaml
id: tyre_outer
type: ellipse
center: [0.713, 0.684]
radii: [0.128, 0.109]
rotation_degrees: -2.4
fit_error: 0.008
source_path: silhouette_003
confidence: 0.96
```

Retain both:

* the fitted primitive;
* the original path.

The system must be able to fall back to the original curve when primitive fitting is unreliable.

---

## Stage 8: Constraint and Relationship Detection

Detect relationships between primitives.

Supported constraints should include:

* concentric;
* tangent;
* coincident;
* collinear;
* parallel;
* perpendicular;
* equal length;
* equal radius;
* equal spacing;
* mirror symmetry;
* radial symmetry;
* rotational repetition;
* containment;
* adjacency;
* intersection;
* alignment;
* shared center;
* shared axis.

Example:

```yaml
constraints:
  - type: concentric
    entities:
      - tyre_outer
      - rim_outer
      - hub

  - type: radial_repetition
    prototype: spoke_01
    count: 5
    center: hub_center

  - type: mirror_symmetry
    entities:
      - body_left
      - body_right
    axis: object_vertical_axis
```

---

## Stage 9: Parametric Sketch Graph

Convert fitted primitives and constraints into a machine-readable sketch graph.

The sketch graph should contain:

* coordinate systems;
* paths and primitives;
* constraints;
* semantic labels;
* part membership;
* source evidence;
* confidence;
* uncertainty;
* visibility;
* occlusion state.

Example:

```yaml
coordinate_system:
  type: normalized_image
  origin: top_left
  width: 1.0
  height: 1.0

parts:
  - id: front_wheel
    class: wheel
    visibility: partial

    observed_geometry:
      tyre_outer:
        type: ellipse
        center: [0.713, 0.684]
        radii: [0.128, 0.109]
        rotation_degrees: -2.4

      rim_outer:
        type: ellipse
        center: [0.713, 0.683]
        radii: [0.082, 0.070]

      hub:
        type: ellipse
        center: [0.712, 0.682]
        radii: [0.021, 0.018]

    constraints:
      - rim_outer concentric_with tyre_outer
      - hub concentric_with rim_outer

    appearance:
      tyre:
        estimated_color_srgb: [28, 27, 26]
        material_class: rubber

      rim:
        estimated_color_srgb: [112, 117, 121]
        material_class: metal

    uncertainty:
      physical_scale: unknown
      rear_profile: unobserved
```

---

## Stage 10: Semantic Part Decomposition

Use a vision-language model or other semantic system to group primitives and regions into meaningful components.

For example:

```text
wheel
├── tyre
├── rim
│   ├── outer lip
│   ├── barrel
│   ├── spokes
│   └── hub
└── axle connection
```

Each part should receive:

* a stable identifier;
* a semantic class;
* parent-child relationships;
* visible geometry;
* inferred hidden geometry;
* appearance information;
* likely construction method;
* confidence.

The semantic model must not directly overwrite observed geometry. It may label, group, constrain, or propose hypotheses.

---

## Stage 11: Camera Estimation

Estimate the camera model before converting projected measurements into 3D geometry.

Estimate where possible:

* perspective or orthographic projection;
* focal length;
* principal point;
* camera rotation;
* object rotation;
* lens distortion;
* vanishing points;
* ground plane;
* relative scale.

The system should jointly reason about:

```text
camera parameters
+
object pose
+
object geometry
```

A projected ellipse must not automatically be treated as an elliptical 3D object. It may be the perspective projection of a circle.

Unknown physical scale must remain explicitly unknown unless the system receives a reliable scale reference.

---

## Stage 12: Depth and Surface Orientation

Optionally estimate:

* relative depth;
* metric depth where possible;
* surface normals;
* occlusion boundaries;
* front-to-back ordering.

Depth and normal predictions must remain separate from observed vector evidence.

For each region, the system should be able to record information such as:

```yaml
region: rim_center
relative_depth: -0.041
normal_camera_space: [0.04, -0.12, 0.99]
depth_confidence: 0.62
```

These predictions should guide geometry but should not be treated as exact measurements.

---

## Stage 13: Construction-Method Classification

Assign each part one or more possible Blender construction operators.

Supported operator categories should include:

### Extrusion

For:

* plates;
* signs;
* logos;
* brackets;
* flat-profile components.

### Revolution

For:

* wheels;
* tyres;
* bottles;
* bowls;
* knobs;
* cylindrical housings.

### Sweep

For:

* pipes;
* handles;
* cables;
* rails;
* tubular structures.

### Primitive assembly

For:

* boxes;
* cylinders;
* spheres;
* cones;
* toruses;
* capsules.

### Boolean construction

For:

* holes;
* cutouts;
* vents;
* sockets;
* recesses.

### Surface lofting

For transitions between multiple profiles.

### Freeform surface fitting

For irregular shells, ergonomic shapes, or body panels.

### Displacement or texture-based detail

For:

* tread;
* engraving;
* embossing;
* small grooves;
* surface noise.

Each proposed operator should include a confidence score.

Example:

```yaml
part: tyre
construction_candidates:
  - operator: revolve
    confidence: 0.89
  - operator: torus
    confidence: 0.56

selected_operator: revolve
```

---

## Stage 14: Parametric 3D Construction Plan

Generate a declarative construction plan before generating Blender code.

Example:

```yaml
object:
  id: wheel_assembly
  units: normalized

parts:
  - id: tyre
    operator: revolve

    axis:
      origin: [0.0, 0.0, 0.0]
      direction: [1.0, 0.0, 0.0]

    profile:
      type: bezier
      points:
        - [0.00, -0.093]
        - [0.03, -0.098]
        - [0.08, -0.076]
        - [0.108, 0.000]
        - [0.08, 0.076]
        - [0.03, 0.098]
        - [0.00, 0.093]

    material:
      class: rubber
      base_color: [0.025, 0.023, 0.022]
      roughness: 0.78

  - id: rim
    operator: revolve

    profile:
      source: inferred_rim_cross_section

  - id: spoke
    operator: extrude
    source_curve: spoke_profile_01

  - id: spoke_array
    operator: radial_array
    source_part: spoke
    count: 5
    angle_degrees: 72

  - id: hub
    operator: revolve
```

The construction plan must be editable independently of Blender Python.

---

## Stage 15: Blender Generation

Translate the construction plan into Blender Python.

The generated Blender scene should:

* use named objects;
* preserve semantic part boundaries;
* use meaningful collections;
* use modifiers where appropriate;
* use non-destructive operations where practical;
* retain source curves;
* retain construction parameters;
* assign materials;
* establish hierarchy;
* create correct pivots;
* create UVs where needed;
* generate clean normals;
* avoid unnecessary mesh density.

Example hierarchy:

```text
Wheel_Assembly
├── Tyre
├── Rim
│   ├── Barrel
│   ├── Spokes
│   └── Hub
└── Axle_Mount
```

Avoid generating a single fused mesh unless explicitly requested.

---

## Stage 16: Material Reconstruction

Estimate and assign basic physically based materials.

At minimum estimate:

* base color;
* roughness;
* metallic;
* opacity;
* transmission;
* normal intensity.

Separate:

* geometric boundaries;
* material boundaries;
* lighting effects;
* shadows;
* highlights;
* reflections.

Do not convert highlights or shadows into permanent material colors.

---

## Stage 17: Render-Based Validation

Render the generated model using the estimated reference camera.

Compare the rendering against the original image using multiple signals.

Required comparison signals:

* silhouette overlap;
* contour distance;
* fitted-feature alignment;
* part-mask overlap;
* depth agreement;
* normal agreement;
* keypoint alignment;
* color-region agreement;
* perceptual image similarity.

Suggested metrics:

```yaml
metrics:
  silhouette_iou: 0.91
  contour_chamfer_distance: 0.018
  feature_alignment_error_px: 4.8
  depth_correlation: 0.76
  part_mask_iou: 0.83
  perceptual_similarity: 0.88
```

No single metric should determine success.

---

## Stage 18: Iterative Refinement

Improve the reconstruction by adjusting parameters rather than rewriting the entire model whenever possible.

Candidate optimization parameters include:

* dimensions;
* radii;
* thickness;
* profile control points;
* bevel size;
* part placement;
* part rotation;
* repetition count;
* camera pose;
* focal length;
* material values;
* depth offsets.

Recommended loop:

```text
generate model
→ render
→ compare
→ identify largest mismatch
→ modify responsible parameters
→ render again
→ stop when converged or budget exhausted
```

The refinement system should produce an audit trail.

Example:

```yaml
iteration: 4
observed_problem: front tyre silhouette too narrow
modified_parameters:
  tyre_width:
    previous: 0.081
    new: 0.094
metric_change:
  silhouette_iou:
    previous: 0.86
    new: 0.91
```

---

# Role of Generative 2D Models

Generative 2D models may be used only as a secondary hypothesis source.

Permitted uses:

* proposing orthographic views;
* removing distracting lighting;
* suggesting hidden contours;
* completing partially occluded regions;
* proposing front, side, rear, or top views;
* estimating likely material segmentation;
* suggesting cross-sections.

Generated imagery must never replace the original evidence branch.

Maintain separate branches:

```text
observed evidence branch
generated hypothesis branch
```

Every generated hypothesis must be:

* marked as generated;
* assigned lower default confidence;
* validated against observed views;
* rejected when inconsistent with the source image.

---

# Functional Requirements

The system must:

1. Accept common raster image formats.
2. Support automatic and guided object segmentation.
3. Preserve all coordinate transformations.
4. Generate multiple trace layers.
5. Convert traces into SVG.
6. Simplify SVG paths.
7. Fit geometric primitives.
8. infer geometric constraints.
9. Build a parametric sketch graph.
10. Group geometry into semantic parts.
11. estimate camera parameters.
12. optionally estimate depth and normals.
13. classify parts by construction method.
14. generate a declarative construction plan.
15. produce Blender Python.
16. create an editable Blender scene.
17. render from the estimated camera.
18. compare the render with the source.
19. iteratively improve the model.
20. export results and diagnostics.

---

# Non-Functional Requirements

## Modularity

Each pipeline stage must have clear inputs and outputs.

Components should be replaceable without rewriting the entire system.

Examples:

* replace VTracer with another vectorizer;
* replace the segmentation model;
* replace the depth model;
* replace the vision-language model;
* replace the Blender code generator.

## Reproducibility

Every run should record:

* model versions;
* parameters;
* random seeds;
* input hashes;
* transformations;
* intermediate files;
* software versions;
* prompts;
* generated code;
* validation metrics.

## Inspectability

Every major inference should be visible in generated reports or debug overlays.

The system should not hide errors behind a final mesh.

## Determinism

Deterministic processing should be used where possible for:

* geometry fitting;
* coordinate conversion;
* constraint solving;
* Blender execution;
* render comparison.

## Safety

Generated Blender Python must execute in a constrained environment.

Prevent:

* unrestricted filesystem access;
* arbitrary network access;
* shell execution;
* deletion outside the project directory;
* loading untrusted Blender scripts;
* uncontrolled subprocess creation.

---

# Initial MVP Scope

The MVP should focus on single-image reconstruction of relatively simple hard-surface objects.

## MVP Inputs

* one image;
* optional target description;
* optional bounding box or point;
* optional known dimension.

## MVP Outputs

* object mask;
* normalized crop;
* silhouette SVG;
* color-region SVG;
* simplified primitive graph;
* semantic part labels;
* basic construction plan;
* Blender Python;
* editable `.blend`;
* `.glb` export;
* reference-camera render;
* silhouette comparison report.

## MVP Supported Construction Methods

* extrusion;
* revolution;
* sweep;
* primitive assembly;
* radial arrays;
* mirror symmetry;
* simple Booleans;
* bevels;
* basic materials.

## MVP Supported Objects

Prioritize:

* wheels;
* bottles;
* cups;
* lamps;
* simple chairs;
* tables;
* boxes and enclosures;
* signs and logos;
* brackets;
* basic appliances;
* simple vehicle or machine components.

---

# MVP Acceptance Criteria

A reconstruction is considered successful when:

1. The correct object is isolated.
2. The crop preserves aspect ratio and coordinate mappings.
3. The silhouette SVG accurately follows the object boundary.
4. The simplified geometry uses substantially fewer entities than the raw SVG.
5. Major circles, ellipses, lines, and symmetry relationships are correctly detected.
6. The object is decomposed into meaningful named parts.
7. Each major part has an explicit construction method.
8. Blender code executes without manual repair.
9. The resulting Blender model remains editable.
10. The reference-view render achieves acceptable silhouette agreement.
11. Exported files open correctly in Blender and a glTF viewer.
12. The system produces a diagnostic report explaining its assumptions and uncertainties.

Suggested MVP metric targets:

```yaml
targets:
  segmentation_iou: ">= 0.90 on curated test images"
  silhouette_vector_error: "<= 2% of object diagonal"
  primitive_reduction: ">= 70% fewer control points than raw SVG"
  blender_execution_success: ">= 95%"
  silhouette_render_iou: ">= 0.85"
  valid_glb_export: "100% of successful runs"
```

---

# Development Phases

## Phase 1: Deterministic 2D Pipeline

Implement:

* segmentation;
* padded crop;
* image normalization;
* VTracer integration;
* SVG parsing;
* path cleanup;
* primitive fitting;
* debug overlays.

Deliverable:

```text
image → clean parametric 2D sketch
```

## Phase 2: Semantic Sketch Graph

Implement:

* part labeling;
* path grouping;
* symmetry detection;
* repetition detection;
* geometric constraints;
* uncertainty tracking.

Deliverable:

```text
parametric 2D sketch → semantic sketch graph
```

## Phase 3: Blender Construction

Implement:

* construction-plan schema;
* operator selection;
* Blender Python generation;
* editable hierarchy;
* material assignment;
* `.blend` and `.glb` export.

Deliverable:

```text
semantic sketch graph → editable Blender model
```

## Phase 4: Camera and Validation

Implement:

* reference-camera estimation;
* render overlays;
* silhouette comparison;
* feature alignment;
* automated parameter correction.

Deliverable:

```text
Blender model → validated reference-view reconstruction
```

## Phase 5: Depth and Surface Reasoning

Add:

* monocular depth;
* surface normals;
* depth-aware profile estimation;
* improved part ordering;
* improved recessed and protruding geometry.

## Phase 6: Multiview Reconstruction

Add support for:

* multiple images;
* cross-view feature matching;
* shared part graph;
* camera pose solving;
* consistent scale;
* joint optimization across views.

## Phase 7: Generative Hypotheses

Add optional:

* orthographic-view generation;
* hidden-side completion;
* proposed cross-sections;
* occlusion completion;
* hypothesis scoring and rejection.

---

# Recommended Architecture

```text
Input Manager
    │
    ▼
Object Selector
    │
    ▼
Segmentation Module
    │
    ▼
Crop and Coordinate Manager
    │
    ├── Silhouette Processor
    ├── Color Quantizer
    ├── Structural Edge Extractor
    ├── Detail Extractor
    ├── Depth Estimator
    └── Normal Estimator
            │
            ▼
       Vectorization Layer
            │
            ▼
      SVG Geometry Parser
            │
            ▼
       Primitive Fitter
            │
            ▼
      Constraint Detector
            │
            ▼
     Parametric Sketch Graph
            │
            ▼
       Semantic Part Agent
            │
            ▼
      Camera Estimation Agent
            │
            ▼
  Construction Planning Agent
            │
            ▼
      Blender Code Generator
            │
            ▼
      Sandboxed Blender Runner
            │
            ▼
     Render Comparison System
            │
            ▼
       Refinement Controller
```

---

# Suggested Intermediate Schemas

Use machine-readable schemas for:

```text
CropMetadata
TraceLayer
VectorPath
GeometricPrimitive
GeometricConstraint
SemanticPart
CameraEstimate
DepthEvidence
ConstructionOperator
ConstructionPlan
BlenderObjectManifest
ValidationResult
RefinementAction
```

Prefer JSON Schema, Pydantic models, typed Python dataclasses, or an equivalent strongly validated format.

Do not rely on unconstrained natural-language communication between stages.

---

# Evaluation Dataset

Create a curated benchmark containing:

* clean product images;
* cluttered backgrounds;
* multiple viewing angles;
* clear and difficult silhouettes;
* symmetric and asymmetric objects;
* reflective and matte surfaces;
* partially occluded objects;
* known dimensions where possible;
* ground-truth masks;
* manually labeled primitives;
* manually labeled part structure;
* reference 3D models where available.

Begin with approximately 20–50 carefully selected objects rather than a large uncurated dataset.

Include separate test groups for:

```text
revolution objects
extruded-profile objects
primitive assemblies
repeated radial structures
mirrored structures
freeform objects
known failure cases
```

---

# Known Limitations

The system must explicitly acknowledge that a single RGB image generally cannot uniquely determine:

* absolute physical scale;
* hidden surfaces;
* exact depth;
* internal structure;
* rear geometry;
* true cross-sections;
* material composition;
* lens parameters;
* whether some lines are geometry, texture, shadow, or reflection.

The system should output the simplest plausible model consistent with the available evidence, not claim exact reconstruction when the evidence is insufficient.

---

# Final Success Definition

The project succeeds when it can take a reference image of a suitable manufactured object and produce an editable Blender model whose:

* silhouette matches the reference;
* major parts are correctly identified;
* geometric relationships are preserved;
* construction history is understandable;
* parameters can be adjusted;
* materials are approximately correct;
* hidden geometry is clearly marked as inferred;
* output can be exported and reused;
* reconstruction process can be inspected and reproduced.

The defining product principle is:

> Convert images into structured geometric evidence first, then use that evidence to construct and refine an editable 3D model.
