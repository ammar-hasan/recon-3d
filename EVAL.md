# Evaluation Specification: Image-to-Editable-3D Reconstruction System

## Purpose

This document defines how to determine whether the image-to-editable-3D reconstruction system is functioning correctly.

The system must not be evaluated only by asking whether the final render “looks good.” It must be evaluated at multiple levels:

1. Input handling.
2. Object segmentation.
3. Coordinate preservation.
4. Vector tracing.
5. Geometric simplification.
6. Primitive and constraint detection.
7. Semantic part decomposition.
8. Camera estimation.
9. 3D construction planning.
10. Blender execution.
11. Geometric correctness.
12. Editability and structure.
13. Visual agreement.
14. Export validity.
15. Uncertainty calibration.
16. End-to-end usability.

The evaluation framework must distinguish between:

* deterministic correctness;
* geometric accuracy;
* perceptual similarity;
* semantic correctness;
* structural editability;
* plausibility of inferred geometry;
* confidence calibration;
* operational reliability.

---

# Evaluation Principles

## 1. Evaluate Every Stage Independently

A strong final render can hide serious pipeline errors.

For example:

* an incorrect camera can compensate for incorrect geometry;
* texture can hide silhouette errors;
* a fused mesh can look correct but be unusable;
* generated details can appear plausible while contradicting the source;
* an inaccurate model can match one view but fail from all others.

Every major stage must therefore produce independently testable outputs.

---

## 2. Separate Observed Geometry From Inferred Geometry

The system must be evaluated more strictly on geometry directly visible in the reference image.

Use three evaluation categories:

```yaml
evaluation_regions:
  observed:
    definition: Directly visible in the input image
    strictness: high

  partially_observed:
    definition: Visible but occluded, foreshortened, reflective, or ambiguous
    strictness: medium

  unobserved:
    definition: Hidden rear, underside, interior, or occluded geometry
    strictness: plausibility_only
```

The system should not be penalized heavily for choosing one plausible hidden shape among several valid possibilities.

It should be penalized for:

* presenting hidden geometry as observed fact;
* contradicting visible evidence;
* assigning unjustified high confidence;
* producing physically impossible hidden geometry.

---

## 3. Prefer Objective Metrics, But Do Not Use One Metric Alone

No single metric can determine reconstruction quality.

For example:

* silhouette IoU ignores internal geometry;
* perceptual similarity can reward texture over geometry;
* Chamfer distance may not reflect semantic part structure;
* human ratings may be inconsistent;
* code execution success says nothing about model quality.

Use metric groups and acceptance gates.

---

## 4. Compare Against Explicit Baselines

At minimum, compare the system against:

1. Direct SVG extrusion.
2. Direct image-to-mesh generation.
3. A one-shot vision-language-model-to-Blender-script workflow.
4. The proposed full structured pipeline.
5. An ablated version without refinement.
6. An ablated version without depth or normals.
7. An ablated version without primitive fitting.

This establishes whether each component materially improves results.

---

# Evaluation Levels

The evaluation system should operate at four levels.

## Level A: Unit Evals

Test individual functions and modules using synthetic or manually constructed inputs.

Examples:

* crop transforms;
* circle fitting;
* ellipse fitting;
* SVG simplification;
* symmetry detection;
* Blender operator generation;
* file export.

## Level B: Stage Evals

Evaluate each pipeline stage on curated reference data.

Examples:

* segmentation accuracy;
* primitive classification;
* part grouping;
* camera estimation;
* construction-method selection.

## Level C: End-to-End Evals

Evaluate complete reconstructions from image input to exported 3D asset.

## Level D: Human and Task-Based Evals

Evaluate whether artists, designers, developers, or downstream agents can actually use and edit the resulting models.

---

# Evaluation Dataset

## Dataset Composition

Build a curated benchmark with known ground truth wherever possible.

Include at least the following object classes:

```text
rotational objects
extruded-profile objects
box-like products
primitive assemblies
wheels and rims
containers and bottles
furniture
mechanical parts
appliances
radially repeated structures
bilaterally symmetric objects
asymmetric products
logos and reliefs
simple stylized props
```

Include controlled difficulty levels.

### Easy

* isolated object;
* clean background;
* diffuse lighting;
* high resolution;
* minimal occlusion;
* canonical camera angle;
* matte material.

### Medium

* mild clutter;
* noncanonical angle;
* moderate perspective;
* partial occlusion;
* mixed materials;
* soft shadows;
* some reflections.

### Hard

* cluttered background;
* severe perspective;
* reflective surfaces;
* low contrast;
* heavy occlusion;
* transparent components;
* motion blur;
* unusual shape;
* small image size;
* ambiguous scale.

---

## Required Ground Truth

Where possible, each benchmark object should include:

```text
source image
foreground mask
object bounding box
camera calibration
known object dimensions
reference 3D model
reference part hierarchy
reference materials
reference semantic labels
reference visible contours
reference construction family
reference rendered views
```

For images rendered from known 3D assets, retain exact:

* camera intrinsics;
* camera extrinsics;
* depth;
* normals;
* object transforms;
* masks;
* object IDs;
* material IDs.

Synthetic renderings are especially useful because they provide perfect ground truth.

Real photographs should also be included because synthetic-only evaluation can overestimate performance.

---

# Eval 1: Input Handling

## Objective

Verify that the system safely and correctly loads supported inputs.

## Test Cases

* PNG with alpha.
* PNG without alpha.
* JPEG.
* WEBP.
* grayscale image.
* large image.
* small image.
* rotated EXIF image.
* image with transparency.
* corrupt file.
* unsupported file.
* multiple views.
* missing optional metadata.
* invalid bounding box.
* empty mask.

## Pass Conditions

```yaml
input_handling:
  valid_file_success_rate: 1.0
  corrupt_file_graceful_failure_rate: 1.0
  unsupported_file_clear_error_rate: 1.0
  exif_orientation_correctness: 1.0
```

The system must fail clearly rather than silently modifying or misreading the input.

---

# Eval 2: Object Selection and Segmentation

## Objective

Determine whether the correct target object is selected and isolated.

## Metrics

### Mask Intersection over Union

```text
IoU = intersection(predicted_mask, reference_mask)
      / union(predicted_mask, reference_mask)
```

### Boundary F-score

Measures agreement between predicted and reference mask boundaries.

### Target Selection Accuracy

Whether the selected mask corresponds to the requested object.

### Hole Preservation Accuracy

Whether meaningful negative spaces are preserved.

Examples:

* chair gap;
* mug handle hole;
* wheel centre;
* bracket opening.

## Acceptance Targets

```yaml
segmentation:
  easy_mask_iou: ">= 0.95"
  medium_mask_iou: ">= 0.90"
  hard_mask_iou: ">= 0.80"
  boundary_f_score: ">= 0.90"
  target_selection_accuracy: ">= 0.98"
  meaningful_hole_recall: ">= 0.95"
```

## Failure Cases

* nearby object included;
* thin parts removed;
* reflective edge lost;
* shadow included as geometry;
* holes filled;
* object fragmented;
* wrong object selected.

---

# Eval 3: Crop and Coordinate Integrity

## Objective

Verify that cropping and normalization preserve all spatial relationships.

## Required Tests

Select known points in the original image, transform them into normalized crop coordinates, and transform them back.

## Metrics

### Round-Trip Coordinate Error

```text
original point
→ normalized point
→ reconstructed original point
```

Measure pixel error.

### Aspect Ratio Preservation

Verify that circles are not stretched due to resizing.

### Crop Coverage

Measure how much of the target object lies inside the normalized crop.

## Acceptance Targets

```yaml
coordinate_integrity:
  mean_round_trip_error_px: "< 0.1"
  max_round_trip_error_px: "< 0.5"
  aspect_ratio_error: "< 0.001"
  object_crop_coverage: "100%"
  recorded_transform_reproducibility: "100%"
```

Any unexplained coordinate drift is a blocking failure.

---

# Eval 4: Preprocessing Quality

## Objective

Verify that preprocessing improves traceability without altering true object geometry.

## Tests

Evaluate:

* silhouette image;
* color quantization;
* structural edges;
* lighting normalization;
* detail isolation.

## Metrics

### Silhouette Preservation

Compare the preprocessed silhouette with the ground-truth mask.

### Structural Edge Precision and Recall

Compare extracted edges with manually labeled geometric edges.

### False Geometry Rate

Measure edges created from:

* shadows;
* highlights;
* reflections;
* texture;
* background leakage.

### Color Region Stability

Measure whether repeated runs produce consistent material-region boundaries.

## Acceptance Targets

```yaml
preprocessing:
  silhouette_iou: ">= 0.97"
  structural_edge_precision: ">= 0.85"
  structural_edge_recall: ">= 0.80"
  false_geometry_edge_rate: "<= 0.15"
  deterministic_repeatability: ">= 0.99"
```

---

# Eval 5: Vectorization Quality

## Objective

Determine whether SVG traces faithfully represent the raster evidence.

## Metrics

### Rasterized SVG IoU

Rasterize the SVG and compare it with the source mask or edge map.

### Boundary Distance

Measure the average distance between source and traced contours.

### Path Complexity

Count:

* paths;
* control points;
* segments;
* self-intersections.

### Topology Preservation

Verify:

* hole count;
* contour nesting;
* connected component count;
* winding order.

## Acceptance Targets

```yaml
vectorization:
  silhouette_svg_iou: ">= 0.97"
  mean_boundary_error: "<= 1% of object diagonal"
  p95_boundary_error: "<= 2% of object diagonal"
  meaningful_hole_recall: ">= 0.95"
  invalid_self_intersection_rate: "<= 0.01"
```

---

# Eval 6: SVG Simplification

## Objective

Reduce complexity without materially changing shape.

## Metrics

### Control-Point Reduction

```text
1 - simplified_points / original_points
```

### Shape Deviation

Measure maximum and mean distance between the original and simplified paths.

### Topology Preservation

Verify that simplification does not:

* close open paths incorrectly;
* remove holes;
* merge separate components;
* introduce intersections;
* reverse containment.

## Acceptance Targets

```yaml
svg_simplification:
  control_point_reduction: ">= 70%"
  mean_shape_deviation: "<= 0.5% of object diagonal"
  max_shape_deviation: "<= 2% of object diagonal"
  topology_preservation_rate: ">= 0.99"
  meaningful_feature_preservation: ">= 0.95"
```

The system should report a failure rather than oversimplify geometry beyond the permitted tolerance.

---

# Eval 7: Geometric Primitive Fitting

## Objective

Determine whether traced paths are correctly converted into circles, ellipses, lines, arcs, rectangles, splines, and other primitives.

## Test Dataset

Use synthetic contours with known:

* noise;
* occlusion;
* rotation;
* scale;
* perspective;
* partial arcs;
* imperfect tracing.

## Metrics

### Primitive Classification Accuracy

Was the correct primitive family selected?

### Parameter Error

For circles:

* centre error;
* radius error.

For ellipses:

* centre error;
* major-axis error;
* minor-axis error;
* rotation error.

For lines:

* endpoint error;
* angular error.

### Fit Residual

Distance between the fitted primitive and source contour.

### Confidence Calibration

Compare predicted confidence with actual fit quality.

## Acceptance Targets

```yaml
primitive_fitting:
  primitive_type_accuracy: ">= 0.92"
  circle_radius_relative_error: "<= 0.02"
  ellipse_axis_relative_error: "<= 0.03"
  ellipse_rotation_error_deg: "<= 2.0"
  line_angle_error_deg: "<= 1.0"
  incorrect_high_confidence_rate: "<= 0.03"
```

---

# Eval 8: Constraint Detection

## Objective

Determine whether geometric relationships are correctly identified.

## Constraints to Evaluate

* concentric;
* parallel;
* perpendicular;
* tangent;
* collinear;
* equal radius;
* equal spacing;
* mirror symmetry;
* rotational symmetry;
* repeated pattern;
* containment;
* shared axis;
* alignment.

## Metrics

For each constraint type:

* precision;
* recall;
* F1 score.

## Acceptance Targets

```yaml
constraint_detection:
  overall_precision: ">= 0.90"
  overall_recall: ">= 0.85"
  concentric_f1: ">= 0.95"
  symmetry_f1: ">= 0.90"
  repetition_count_accuracy: ">= 0.95"
  false_constraint_rate: "<= 0.08"
```

False constraints are especially harmful because they can force incorrect geometry.

Precision should generally be prioritized over recall.

---

# Eval 9: Semantic Part Decomposition

## Objective

Verify that geometry is grouped into meaningful object parts.

## Metrics

### Part Detection Precision and Recall

Compare predicted semantic parts against ground-truth part labels.

### Part Boundary Accuracy

Measure whether each predicted part corresponds to the correct geometry.

### Hierarchy Accuracy

Compare parent-child relationships.

### Stable Naming

Verify that repeated runs produce consistent identifiers or canonical labels.

### Duplicate and Missing Parts

Count:

* missing required components;
* duplicate components;
* hallucinated components.

## Acceptance Targets

```yaml
semantic_parts:
  part_detection_precision: ">= 0.90"
  part_detection_recall: ">= 0.85"
  part_boundary_iou: ">= 0.80"
  hierarchy_edge_accuracy: ">= 0.90"
  hallucinated_major_part_rate: "<= 0.05"
  missing_major_part_rate: "<= 0.08"
```

Major functional parts should be weighted more heavily than decorative details.

---

# Eval 10: Observed-vs-Inferred Evidence Tracking

## Objective

Verify that the system accurately labels the origin of each property.

## Property Sources

```text
directly_observed
fitted_from_observation
estimated_from_depth
estimated_from_camera
semantic_prior
generated_hypothesis
user_supplied
unknown
```

## Tests

Create examples where:

* the rear surface is hidden;
* a component is partially occluded;
* scale is unknown;
* reflection resembles a groove;
* an object has asymmetric hidden geometry.

## Metrics

### Source Attribution Accuracy

Was each property assigned the correct evidence category?

### Unsupported Certainty Rate

How often does the system assign high confidence to unobserved details?

### Unknown Recognition Rate

How often does the system correctly state that information is unavailable?

## Acceptance Targets

```yaml
evidence_tracking:
  source_attribution_accuracy: ">= 0.90"
  unsupported_high_confidence_rate: "<= 0.03"
  hidden_geometry_marked_inferred: ">= 0.98"
  unknown_scale_recognition: "100%"
```

This is a critical trustworthiness eval.

---

# Eval 11: Camera Estimation

## Objective

Determine whether the camera parameters and object pose are estimated correctly.

## Metrics

### Focal Length Error

Compare estimated and reference focal lengths.

### Camera Rotation Error

Measure angular difference.

### Camera Translation Error

Use normalized or metric error depending on available scale.

### Reprojection Error

Project known 3D points into the image using the predicted camera and measure pixel error.

### Projection-Type Accuracy

Determine whether the system correctly identifies perspective versus orthographic-like imagery.

## Acceptance Targets

```yaml
camera_estimation:
  median_rotation_error_deg: "<= 5"
  median_reprojection_error_px: "<= 5"
  focal_length_relative_error: "<= 0.15"
  projection_type_accuracy: ">= 0.95"
```

Camera and geometry should also be evaluated jointly because multiple camera-geometry combinations can explain the same image.

---

# Eval 12: Depth and Normal Estimation

## Objective

Measure whether estimated depth and normals provide useful geometric evidence.

## Metrics

### Depth

* absolute relative error;
* scale-invariant logarithmic error;
* rank-order accuracy;
* edge preservation;
* depth correlation.

### Normals

* mean angular error;
* median angular error;
* percentage below 11.25°;
* percentage below 22.5°;
* percentage below 30°.

## Acceptance Targets

Targets depend heavily on the selected depth and normal model.

The system should primarily evaluate whether adding depth and normals improves downstream reconstruction.

```yaml
depth_normals:
  downstream_silhouette_improvement: ">= 0"
  downstream_depth_improvement: "> 0"
  regression_rate_after_enabling: "<= 0.10"
```

If depth or normals make the reconstruction worse, the system should lower their influence or disable them.

---

# Eval 13: Construction-Method Selection

## Objective

Determine whether the system chooses an appropriate modeling operator for each part.

## Supported Classes

* extrusion;
* revolution;
* sweep;
* primitive;
* Boolean;
* loft;
* freeform surface;
* displacement;
* texture-only detail.

## Metrics

### Top-1 Operator Accuracy

Was the selected operator correct?

### Top-3 Operator Recall

Was an acceptable operator included among the top candidates?

### Modeling Efficiency

Does the selected operator produce an editable model with reasonable complexity?

## Acceptance Targets

```yaml
operator_selection:
  top_1_accuracy: ">= 0.80"
  top_3_recall: ">= 0.95"
  unusable_operator_rate: "<= 0.05"
```

Multiple operators may be valid. The reference labels should therefore allow a set of acceptable construction methods.

---

# Eval 14: Construction Plan Validity

## Objective

Verify that the declarative construction plan is complete, consistent, and executable.

## Validation Checks

* valid schema;
* all referenced parts exist;
* no circular parenting;
* no missing source curves;
* valid units;
* valid transforms;
* valid material references;
* valid operator parameters;
* finite numeric values;
* no impossible dimensions;
* no invalid array counts;
* no zero-length axes;
* no conflicting constraints.

## Acceptance Targets

```yaml
construction_plan:
  schema_validity: "100%"
  reference_integrity: "100%"
  impossible_parameter_rate: "0%"
  executable_plan_rate: ">= 98%"
```

The plan should fail validation before Blender execution if it is inconsistent.

---

# Eval 15: Blender Script Execution

## Objective

Verify that generated Blender code executes reliably and safely.

## Tests

* fresh Blender environment;
* repeated execution;
* headless execution;
* different supported Blender versions;
* missing optional dependency;
* malformed plan;
* interrupted run;
* timeout;
* invalid material;
* export attempt.

## Metrics

### Execution Success Rate

Percentage of valid plans that produce a completed scene.

### Runtime

Measure total and per-stage runtime.

### Determinism

Run the same input multiple times and compare outputs.

### Safety Violations

Detect attempts to:

* access files outside the project directory;
* execute shell commands;
* use unrestricted network access;
* delete unrelated files;
* create uncontrolled subprocesses.

## Acceptance Targets

```yaml
blender_execution:
  valid_plan_success_rate: ">= 0.98"
  crash_rate: "<= 0.01"
  deterministic_scene_rate: ">= 0.95"
  unauthorized_filesystem_access: "0"
  shell_execution_attempts: "0"
  uncontrolled_network_requests: "0"
```

---

# Eval 16: Scene Structure and Editability

## Objective

Determine whether the resulting Blender scene is meaningfully editable.

## Structural Checks

* major parts are separate objects;
* objects have meaningful names;
* collections correspond to semantic groups;
* modifiers remain editable;
* pivots are logically placed;
* parent-child relationships are correct;
* repeated elements use arrays or instances where appropriate;
* source curves are retained;
* materials are not unnecessarily duplicated;
* object transforms are valid;
* geometry is not arbitrarily fused.

## Quantitative Metrics

```yaml
editability:
  major_part_separation_accuracy: ">= 0.90"
  meaningful_object_name_rate: ">= 0.95"
  hierarchy_accuracy: ">= 0.90"
  correct_pivot_rate: ">= 0.85"
  reusable_modifier_rate: ">= 0.80"
  duplicate_material_rate: "<= 0.10"
```

## Edit Task Tests

Ask an independent agent or human to perform edits such as:

* increase tyre radius by 10%;
* change spoke count from 5 to 6;
* widen the handle;
* remove one component;
* change the rim material;
* mirror a part;
* adjust the bottle neck height.

Measure:

* whether the edit is possible;
* number of actions required;
* whether unrelated geometry breaks;
* time to completion;
* whether the output remains valid.

## Editability Success Target

At least 85% of benchmark edit tasks should be completed without rebuilding the object from scratch.

---

# Eval 17: Mesh and Geometry Validity

## Objective

Verify that generated geometry is technically valid.

## Checks

* manifoldness;
* watertightness where required;
* non-manifold edges;
* degenerate faces;
* zero-area faces;
* duplicate vertices;
* inverted normals;
* inconsistent normals;
* self-intersections;
* disconnected fragments;
* extreme aspect-ratio triangles;
* unapplied destructive transforms;
* invalid UVs;
* overlapping UVs where prohibited.

## Acceptance Targets

Targets may vary by asset type.

```yaml
mesh_validity:
  invalid_object_rate: "<= 0.02"
  non_manifold_major_part_rate: "<= 0.03"
  degenerate_face_rate: "<= 0.001"
  inverted_normal_rate: "<= 0.001"
  unexpected_disconnected_component_rate: "<= 0.02"
```

For 3D-printable mode, watertightness should be mandatory.

---

# Eval 18: Silhouette Agreement

## Objective

Measure whether the generated model matches the reference outline from the reference camera.

## Metrics

### Silhouette IoU

Compare rendered and reference masks.

### Contour Chamfer Distance

Measure bidirectional distance between contours.

### Landmark Error

Compare important points such as:

* corners;
* wheel centres;
* handle endpoints;
* symmetry centres;
* contact points.

## Acceptance Targets

```yaml
silhouette:
  easy_iou: ">= 0.92"
  medium_iou: ">= 0.85"
  hard_iou: ">= 0.75"
  contour_error: "<= 2% of object diagonal"
  landmark_error: "<= 3% of object diagonal"
```

Silhouette should be the primary geometric metric for single-view MVP evaluation.

---

# Eval 19: Internal Feature Alignment

## Objective

Verify that visible internal structures align with the reference.

## Features

* holes;
* spokes;
* seams;
* panel boundaries;
* rim boundaries;
* vents;
* handles;
* ridges;
* recesses;
* logos;
* fasteners.

## Metrics

* keypoint error;
* feature-curve distance;
* region overlap;
* repetition count accuracy;
* concentricity error;
* spacing error.

## Acceptance Targets

```yaml
internal_features:
  major_feature_recall: ">= 0.85"
  major_feature_precision: ">= 0.90"
  repeated_element_count_accuracy: ">= 0.95"
  normalized_feature_alignment_error: "<= 0.03"
```

---

# Eval 20: Multiview Geometric Accuracy

## Objective

Prevent overfitting to one reference view.

Where a reference 3D model or additional images exist, render the generated model from unseen views.

## Metrics

### Unseen-View Silhouette IoU

Render from cameras not used during reconstruction.

### Surface Chamfer Distance

Compare generated and reference surfaces after alignment.

### Normal Consistency

Compare surface orientation.

### Volumetric IoU

Compare occupied volumes.

### Partwise Geometric Error

Evaluate major components separately.

## Acceptance Targets

Initial targets should be class-specific.

```yaml
multiview_geometry:
  median_unseen_view_silhouette_iou: ">= 0.75"
  normalized_chamfer_distance: "<= 0.05"
  major_part_presence_accuracy: ">= 0.90"
```

A strong reference-view match with poor unseen-view performance indicates camera-view overfitting.

---

# Eval 21: Material Accuracy

## Objective

Determine whether material assignments approximately match the object.

## Evaluate

* material class;
* base color;
* metallic;
* roughness;
* transmission;
* opacity;
* emission;
* part-material assignment.

## Metrics

### Material Classification Accuracy

Examples:

* rubber;
* metal;
* plastic;
* glass;
* wood;
* fabric.

### Color Difference

Use a perceptually meaningful color-space difference.

### Highlight-Invariance Test

Test whether the system incorrectly bakes highlights or shadows into material color.

## Acceptance Targets

```yaml
materials:
  material_class_accuracy: ">= 0.80"
  major_part_material_assignment: ">= 0.90"
  median_color_difference: "< acceptable class-specific threshold"
  highlight_baked_as_color_rate: "<= 0.10"
```

Material evaluation should be secondary to geometric evaluation during the MVP.

---

# Eval 22: Perceptual Render Similarity

## Objective

Measure overall reference-view appearance.

## Metrics

Potential metrics include:

* structural similarity;
* perceptual feature distance;
* masked perceptual similarity;
* edge-weighted similarity;
* color-region similarity.

Render similarity should be evaluated both:

* with materials;
* with neutral clay materials.

The clay render reveals geometry without texture compensation.

## Required Render Passes

```text
silhouette
clay
normal
depth
part-ID
material-ID
full shaded render
```

## Acceptance Rule

A high full-render score with a low clay-render score must not count as geometric success.

---

# Eval 23: Refinement Loop Effectiveness

## Objective

Determine whether iterative refinement improves reconstruction quality.

## Metrics

### Improvement Rate

Percentage of cases where final metrics exceed initial metrics.

### Mean Metric Improvement

For example:

```text
final silhouette IoU - initial silhouette IoU
```

### Regression Rate

Percentage of cases made worse.

### Iteration Efficiency

Improvement per iteration or compute unit.

### Convergence Stability

Whether parameter values stabilize.

## Acceptance Targets

```yaml
refinement:
  improved_case_rate: ">= 0.75"
  regression_rate: "<= 0.10"
  average_silhouette_iou_gain: ">= 0.03"
  invalid_model_after_refinement_rate: "<= 0.02"
```

The loop should keep the best-known valid model and roll back harmful changes.

---

# Eval 24: Uncertainty Calibration

## Objective

Determine whether confidence scores reflect actual correctness.

## Metrics

### Expected Calibration Error

Group predictions by confidence and compare confidence with empirical accuracy.

### Brier Score

Evaluate probabilistic predictions.

### Selective Accuracy

Measure accuracy when the system accepts only predictions above a confidence threshold.

### Overconfidence Rate

Measure high-confidence errors.

## Acceptance Targets

```yaml
uncertainty:
  expected_calibration_error: "<= 0.08"
  high_confidence_error_rate: "<= 0.05"
  hidden_geometry_overconfidence_rate: "<= 0.03"
```

The system should be rewarded for correctly saying:

```text
unknown
ambiguous
not visible
requires another view
requires scale reference
```

---

# Eval 25: Export Validity

## Objective

Verify that exported assets work in downstream applications.

## Required Formats

* `.blend`;
* `.glb`;
* optional `.obj`;
* optional `.fbx`;
* optional `.stl`.

## Tests

* reopen exported `.blend`;
* load `.glb` in at least one independent glTF validator or viewer;
* verify hierarchy;
* verify transforms;
* verify materials;
* verify normals;
* verify animations if present;
* verify units;
* verify no missing textures;
* verify bounding box.

## Acceptance Targets

```yaml
exports:
  blend_reopen_success: "100%"
  glb_validation_success: "100%"
  missing_texture_rate: "0%"
  hierarchy_preservation_rate: ">= 0.98"
  transform_preservation_rate: ">= 0.99"
```

---

# Eval 26: Reproducibility

## Objective

Determine whether runs can be reproduced.

## Required Records

* input hash;
* configuration;
* model versions;
* prompts;
* random seeds;
* code version;
* Blender version;
* dependency versions;
* generated construction plan;
* generated Blender script;
* all intermediate artifacts.

## Metrics

### Exact Reproducibility

For deterministic stages, outputs should match exactly.

### Metric Reproducibility

For stochastic stages, outputs should remain within defined tolerances.

## Acceptance Targets

```yaml
reproducibility:
  deterministic_stage_exact_match: "100%"
  stochastic_metric_variance: "<= defined threshold"
  missing_run_metadata_rate: "0%"
```

---

# Eval 27: Runtime and Resource Efficiency

## Objective

Measure practical operational performance.

## Metrics

* segmentation time;
* vectorization time;
* planning time;
* Blender execution time;
* refinement time;
* peak memory;
* GPU memory;
* total cost;
* number of model calls;
* number of Blender renders.

## Report By Difficulty

```yaml
runtime:
  easy:
    median_total_time: ...
  medium:
    median_total_time: ...
  hard:
    median_total_time: ...
```

Do not optimize runtime at the expense of validity or editability.

---

# Eval 28: Failure Detection and Graceful Degradation

## Objective

Verify that the system knows when it is unlikely to succeed.

## Difficult Inputs

* transparent object;
* mirror;
* heavy occlusion;
* very low resolution;
* object smaller than 64 pixels;
* severe blur;
* multiple identical overlapping objects;
* unknown target;
* texture with no geometric boundary;
* impossible or conflicting views.

## Desired Behavior

The system should:

* lower confidence;
* explain the limitation;
* preserve intermediate outputs;
* request or recommend additional evidence;
* avoid presenting a speculative model as exact;
* return the best valid partial result where possible.

## Metrics

```yaml
failure_detection:
  impossible_case_detection_rate: ">= 0.90"
  false_failure_rate_on_easy_cases: "<= 0.05"
  graceful_partial_output_rate: ">= 0.90"
  misleading_success_claim_rate: "0%"
```

---

# Eval 29: Human Quality Evaluation

## Objective

Capture qualities not fully represented by automated metrics.

## Evaluators

Use a mixture of:

* 3D artists;
* Blender users;
* CAD users;
* game-asset developers;
* nonexpert users.

## Blind Evaluation Dimensions

Rate each result from 1 to 5 on:

1. Reference-view resemblance.
2. Shape plausibility.
3. Part correctness.
4. Editability.
5. Cleanliness of geometry.
6. Material plausibility.
7. Usefulness as a starting asset.
8. Trustworthiness of uncertainty reporting.

## Pairwise Comparisons

Ask evaluators to choose between:

* full pipeline;
* direct image-to-mesh;
* one-shot Blender agent;
* SVG extrusion baseline.

Pairwise evaluation is generally more reliable than absolute scoring.

## Acceptance Targets

```yaml
human_evaluation:
  median_editability_score: ">= 4/5"
  median_reference_similarity_score: ">= 4/5"
  preferred_over_direct_image_to_mesh: ">= 65%"
  preferred_over_one_shot_agent: ">= 65%"
```

---

# Eval 30: Downstream Task Evaluation

## Objective

Determine whether the generated asset is useful for actual work.

## Tasks

### Editing

* resize a component;
* alter repetition count;
* change a profile;
* move a joint;
* replace a material.

### Animation

* rotate a wheel;
* open a lid;
* articulate a hinge;
* move a handle.

### Game Export

* export to glTF;
* import into an engine;
* verify hierarchy and materials.

### 3D Printing

* convert to watertight geometry;
* verify minimum thickness;
* export STL.

### Variant Generation

* create a wider version;
* create a six-spoke version;
* create a taller version;
* create a different material variant.

## Metrics

* task completion rate;
* time to complete;
* number of manual fixes;
* number of broken dependencies;
* model rebuild requirement;
* downstream import success.

---

# End-to-End Success Score

The system should compute a weighted score, but it must also enforce hard gates.

## Example Weighted Score

```yaml
weights:
  segmentation: 0.08
  vectorization: 0.07
  primitive_fitting: 0.08
  constraints: 0.05
  semantic_parts: 0.10
  camera: 0.07
  construction_plan: 0.08
  blender_execution: 0.07
  editability: 0.12
  silhouette: 0.12
  internal_features: 0.06
  multiview_geometry: 0.05
  materials: 0.02
  uncertainty: 0.03
```

Weights should vary by target use case.

For example:

* product visualization emphasizes appearance;
* CAD reconstruction emphasizes dimensions and editability;
* game assets emphasize topology and export;
* 3D printing emphasizes manifoldness and scale.

---

# Hard Acceptance Gates

A result must not be marked successful if any of the following are true:

```yaml
blocking_failures:
  - wrong target object selected
  - crop transform is invalid
  - construction plan fails schema validation
  - Blender script fails to execute
  - final scene cannot be reopened
  - GLB export is invalid
  - major visible part is missing
  - hidden geometry is presented as directly observed
  - unauthorized code execution occurs
  - final model is not meaningfully editable
  - reference silhouette is below minimum threshold
```

Suggested MVP hard gates:

```yaml
mvp_hard_gates:
  target_selection_correct: true
  segmentation_iou: ">= 0.85"
  construction_plan_valid: true
  blender_execution_success: true
  blend_reopens: true
  glb_valid: true
  reference_silhouette_iou: ">= 0.80"
  major_visible_part_recall: ">= 0.85"
  meaningful_editability: true
  safety_violations: 0
```

---

# Per-Case Evaluation Report

Each run should produce a report similar to:

```yaml
case_id: wheel_017

status: partial_success

input:
  difficulty: medium
  source_views: 1
  known_scale: false

stage_results:
  segmentation:
    mask_iou: 0.94
    passed: true

  vectorization:
    silhouette_iou: 0.98
    control_points_raw: 742
    control_points_simplified: 113
    passed: true

  primitive_fitting:
    detected:
      circles: 1
      ellipses: 4
      lines: 10
    primitive_accuracy: 0.91
    passed: true

  semantic_parts:
    expected_major_parts: 4
    detected_major_parts: 4
    hallucinated_major_parts: 0
    passed: true

  blender:
    execution_success: true
    valid_objects: 12
    invalid_objects: 0

  visual:
    silhouette_iou_initial: 0.81
    silhouette_iou_final: 0.90
    contour_error: 0.014

  editability:
    part_separation: true
    spoke_count_editable: true
    tyre_profile_editable: true
    material_editable: true

uncertainty:
  physical_scale: unknown
  rear_profile: inferred
  hub_depth_confidence: 0.46

blocking_failures: []

final_result:
  passed_mvp: true
  overall_score: 0.88
```

---

# Regression Evaluation

Every code change should run against a fixed regression suite.

## Required Regression Groups

```text
basic segmentation
crop transforms
simple circles
partial ellipses
SVG holes
symmetry
radial repetition
wheel reconstruction
bottle reconstruction
chair reconstruction
Blender execution
GLB export
edit tasks
uncertainty labeling
known failure cases
```

## Regression Rules

A change must be flagged when:

* any hard gate starts failing;
* a major metric falls beyond tolerance;
* runtime increases substantially;
* editability decreases;
* confidence becomes less calibrated;
* a new safety violation appears.

Example:

```yaml
regression_thresholds:
  silhouette_iou_drop: "> 0.02"
  blender_success_drop: "> 0.01"
  part_recall_drop: "> 0.03"
  runtime_increase: "> 25%"
  edit_task_success_drop: "> 0.03"
```

---

# Ablation Evals

Run the full system with selected components disabled.

## Required Ablations

```text
without background removal
without VTracer
without SVG simplification
without primitive fitting
without constraint detection
without semantic part reasoning
without camera estimation
without depth
without normals
without refinement
without uncertainty tracking
```

For each ablation, measure:

* change in silhouette accuracy;
* change in part accuracy;
* change in editability;
* change in execution reliability;
* change in runtime;
* change in human preference.

A module should remain in the system only if it provides measurable value or important operational safeguards.

---

# Agent Self-Evaluation Procedure

The reconstruction agent should not merely produce a model. It should execute a defined self-check before declaring success.

## Required Self-Checks

### Evidence Check

* Did I preserve the original image?
* Did I record which properties were observed versus inferred?
* Did I avoid inventing unjustified dimensions?
* Did I identify ambiguous or hidden geometry?

### Geometry Check

* Does the rendered silhouette align with the target mask?
* Are major internal features aligned?
* Are repeated components counted correctly?
* Are symmetry relationships preserved?
* Are major proportions plausible?

### Scene Check

* Are major parts separate and named?
* Are pivots correctly placed?
* Are modifiers and source curves retained?
* Can the intended parameters be edited?
* Are there accidental duplicate objects?

### Technical Check

* Does Blender open the scene without errors?
* Are normals valid?
* Is the geometry manifold where required?
* Are exports valid?
* Are textures and materials present?
* Does the GLB open in an independent viewer?

### Refinement Check

* Did the final iteration improve measured quality?
* Did any metric regress?
* Was the best valid model retained?
* Did refinement damage editability?

### Confidence Check

* Are low-confidence fields marked clearly?
* Is hidden geometry labeled as inferred?
* Is physical scale marked unknown when no scale evidence exists?
* Did I make any unsupported high-confidence claims?

---

# Agent Completion Policy

The agent may declare one of four outcomes.

## Success

Use when:

* all hard gates pass;
* major visible geometry is reconstructed;
* the model is editable;
* exports are valid;
* uncertainty is correctly reported.

## Partial Success

Use when:

* the model is usable;
* some uncertain or hidden geometry remains approximate;
* no blocking technical failure exists;
* limitations are clearly reported.

## Failed Validation

Use when:

* the model was generated but fails a hard gate;
* geometry is invalid;
* export fails;
* major parts are missing;
* reference agreement is below threshold.

## Unsupported Input

Use when:

* the image lacks sufficient evidence;
* the object cannot be isolated;
* transparency, reflection, blur, or occlusion prevents reliable reconstruction;
* the requested precision cannot be achieved from the available views.

The agent must not label a result as successful solely because Blender produced a render.

---

# MVP Evaluation Dashboard

The MVP dashboard should prominently display:

```text
target selection
segmentation IoU
SVG trace error
control-point reduction
primitive accuracy
part detection
Blender execution status
silhouette IoU
contour error
editability checks
GLB validation
uncertainty warnings
final pass/fail status
```

Recommended top-level summary:

```yaml
summary:
  reconstruction_status: pass
  geometry_score: 0.87
  editability_score: 0.91
  visual_score: 0.85
  technical_validity: 1.00
  uncertainty_calibration: 0.82
  export_validity: 1.00
```

---

# Final Evaluation Standard

The system should be considered successful only when it demonstrates that it can:

1. Correctly isolate the target object.
2. Preserve image and coordinate evidence.
3. Produce faithful vector traces.
4. Convert noisy paths into compact geometric primitives.
5. Infer useful constraints and semantic parts.
6. Generate a valid and editable construction plan.
7. Execute reliably in Blender.
8. Match visible reference geometry.
9. Remain plausible from unseen viewpoints.
10. Preserve editability and hierarchy.
11. Export valid reusable assets.
12. Improve through refinement.
13. Correctly express uncertainty.
14. Detect unsupported or unreliable cases.
15. Outperform simpler baseline approaches.

The central evaluation principle is:

> A reconstruction is correct only when it is visually supported, geometrically coherent, structurally editable, technically valid, and honest about uncertainty.
