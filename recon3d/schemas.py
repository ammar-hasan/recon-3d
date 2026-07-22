"""Strongly-typed intermediate schemas shared by every pipeline stage.

Design rules (GOAL.md):
- Every inferred property carries its evidence source and confidence.
- Observed evidence, geometric inference, semantic inference and generated
  hypotheses must never be silently mixed.
- Stages communicate only through these schemas + files on disk, never
  through unconstrained natural language.

All models are Pydantic v2 models and serialise to JSON/YAML.
"""
from __future__ import annotations

import enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evidence tracking
# ---------------------------------------------------------------------------

class EvidenceSource(str, enum.Enum):
    DIRECTLY_OBSERVED = "directly_observed"
    FITTED_FROM_OBSERVATION = "fitted_from_observation"
    ESTIMATED_FROM_DEPTH = "estimated_from_depth"
    ESTIMATED_FROM_CAMERA = "estimated_from_camera"
    SEMANTIC_PRIOR = "semantic_prior"
    GENERATED_HYPOTHESIS = "generated_hypothesis"
    USER_SUPPLIED = "user_supplied"
    UNKNOWN = "unknown"


class Visibility(str, enum.Enum):
    VISIBLE = "visible"
    PARTIAL = "partial"        # occluded / foreshortened / ambiguous
    UNOBSERVED = "unobserved"  # hidden rear, underside, interior


class EvidencedValue(BaseModel):
    """A value annotated with where it came from and how much to trust it."""
    value: Any = None
    unit: Optional[str] = None
    source: EvidenceSource = EvidenceSource.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 1: input
# ---------------------------------------------------------------------------

class InputSpec(BaseModel):
    """User-facing description of a reconstruction job."""
    image_paths: List[str]
    description: Optional[str] = None
    target_label: Optional[str] = None
    point: Optional[Tuple[float, float]] = None          # pixel xy in first image
    box: Optional[Tuple[float, float, float, float]] = None  # x0,y0,x1,y1 pixels
    mask_path: Optional[str] = None
    known_dimension: Optional[float] = None              # physical units
    known_dimension_axis: Optional[str] = None           # "width"|"height"|"depth"|"diameter"
    view_azimuths_deg: Optional[List[float]] = None       # one calibrated orbit angle per image
    output_dir: str = "projects/run"


class LoadedImage(BaseModel):
    path: str
    width: int
    height: int
    sha256: str
    exif_orientation_applied: bool = False
    channels: int = 3


class InputBundle(BaseModel):
    spec: InputSpec
    images: List[LoadedImage]
    warnings: List[str] = []


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Stage 2: segmentation
# ---------------------------------------------------------------------------

class SegmentationResult(BaseModel):
    mask_path: str                 # binary PNG, 255=foreground
    rgba_path: str                 # transparent foreground PNG
    original_path: str             # untouched copy of the input
    confidence: float = Field(ge=0.0, le=1.0)
    backend: str                   # "rembg" | "grabcut" | "threshold" | "user_mask" | ...
    bbox: Tuple[int, int, int, int]  # x0,y0,x1,y1 tight foreground bbox
    coverage: float                # foreground fraction of image
    diagnostics: Dict[str, Any] = {}
    selection_source: EvidenceSource = EvidenceSource.UNKNOWN
    warnings: List[str] = []


# ---------------------------------------------------------------------------
# Stage 3: crop / coordinate normalisation
# ---------------------------------------------------------------------------

class CropMetadata(BaseModel):
    """Records the exact transform between original pixels and the normalised
    square crop canvas.

    crop = (original - offset) * scale        (applied to x,y points)
    original = crop / scale + offset
    """
    source_image_size: Tuple[int, int]       # w,h
    source_bbox: Tuple[int, int, int, int]   # padded bbox actually cropped
    padding: int
    output_size: Tuple[int, int]             # w,h of canvas
    scale: float
    offset: Tuple[float, float]              # x,y subtracted before scaling

    def to_crop(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self.offset[0]) * self.scale, (y - self.offset[1]) * self.scale)

    def to_original(self, u: float, v: float) -> Tuple[float, float]:
        return (u / self.scale + self.offset[0], v / self.scale + self.offset[1])

    def to_crop_norm(self, x: float, y: float) -> Tuple[float, float]:
        u, v = self.to_crop(x, y)
        return (u / self.output_size[0], v / self.output_size[1])

    def norm_to_original(self, u: float, v: float) -> Tuple[float, float]:
        return self.to_original(u * self.output_size[0], v * self.output_size[1])


# ---------------------------------------------------------------------------
# Stage 4: preprocessing
# ---------------------------------------------------------------------------

class PreprocessLayers(BaseModel):
    silhouette_path: str           # binary image, object boundary + holes
    color_quantized_path: str      # reduced colour regions
    structural_edges_path: str     # geometric boundaries
    details_path: str              # small features preserved
    lighting_normalized_path: str  # shadows/highlights reduced
    params: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Stage 5/6: vectorisation + cleanup
# ---------------------------------------------------------------------------

class TraceLayerName(str, enum.Enum):
    SILHOUETTE = "silhouette"
    COLOR_REGIONS = "color_regions"
    STRUCTURAL_EDGES = "structural_edges"
    DETAILS = "details"


class VectorPath(BaseModel):
    """One path extracted from an SVG trace. Points are in normalised crop
    coordinates (0..1, origin top-left) after cleanup."""
    path_id: str
    source_layer: TraceLayerName
    closed: bool
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    svg_d: str = ""                       # original SVG path data
    points: List[Tuple[float, float]] = []  # polyline sampling (normalised)
    is_hole: bool = False
    parent_path_id: Optional[str] = None
    area: float = 0.0                     # normalised area (signed ok pre-normalisation)
    source: EvidenceSource = EvidenceSource.DIRECTLY_OBSERVED


class TraceLayer(BaseModel):
    name: TraceLayerName
    svg_path: str                         # file on disk
    paths: List[VectorPath] = []
    image_size: Tuple[int, int] = (0, 0)  # pixel size the SVG was traced from
    stats: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Stage 7: primitive fitting
# ---------------------------------------------------------------------------

class PrimitiveType(str, enum.Enum):
    POINT = "point"
    LINE = "line"
    POLYLINE = "polyline"
    CIRCLE = "circle"
    CIRCULAR_ARC = "circular_arc"
    ELLIPSE = "ellipse"
    ELLIPTICAL_ARC = "elliptical_arc"
    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    REGULAR_POLYGON = "regular_polygon"
    SYMMETRIC_SPLINE = "symmetric_spline"
    BEZIER = "bezier"
    CLOSED_REGION = "closed_region"


class GeometricPrimitive(BaseModel):
    """A fitted primitive. Coordinates normalised (0..1 crop space)."""
    id: str
    type: PrimitiveType
    params: Dict[str, Any] = {}      # e.g. center, radii, rotation_degrees, points...
    fit_error: float = 0.0           # mean normalised distance to source path
    source_path: str                 # VectorPath.path_id
    source_layer: TraceLayerName
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    fallback_points: List[Tuple[float, float]] = []  # original curve, always kept
    source: EvidenceSource = EvidenceSource.FITTED_FROM_OBSERVATION


# ---------------------------------------------------------------------------
# Stage 8: constraints
# ---------------------------------------------------------------------------

class ConstraintType(str, enum.Enum):
    CONCENTRIC = "concentric"
    TANGENT = "tangent"
    COINCIDENT = "coincident"
    COLLINEAR = "collinear"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    EQUAL_LENGTH = "equal_length"
    EQUAL_RADIUS = "equal_radius"
    EQUAL_SPACING = "equal_spacing"
    MIRROR_SYMMETRY = "mirror_symmetry"
    RADIAL_SYMMETRY = "radial_symmetry"
    ROTATIONAL_REPETITION = "rotational_repetition"
    CONTAINMENT = "containment"
    ADJACENCY = "adjacency"
    INTERSECTION = "intersection"
    ALIGNMENT = "alignment"
    SHARED_CENTER = "shared_center"
    SHARED_AXIS = "shared_axis"


class GeometricConstraint(BaseModel):
    type: ConstraintType
    entities: List[str]              # primitive ids
    params: Dict[str, Any] = {}      # e.g. count, axis, center, angle_degrees
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: EvidenceSource = EvidenceSource.FITTED_FROM_OBSERVATION


# ---------------------------------------------------------------------------
# Stage 9/10: sketch graph + semantic parts
# ---------------------------------------------------------------------------

class AppearanceEstimate(BaseModel):
    estimated_color_srgb: Optional[Tuple[int, int, int]] = None
    material_class: Optional[str] = None       # rubber|metal|plastic|glass|wood|fabric|...
    roughness: Optional[float] = None
    metallic: Optional[float] = None
    source: EvidenceSource = EvidenceSource.FITTED_FROM_OBSERVATION
    confidence: float = 0.5


class SemanticPart(BaseModel):
    id: str
    part_class: str                            # e.g. wheel, tyre, rim, handle, body
    parent_id: Optional[str] = None
    child_ids: List[str] = []
    primitive_ids: List[str] = []
    visibility: Visibility = Visibility.VISIBLE
    construction_candidates: List["OperatorCandidate"] = []
    selected_operator: Optional[str] = None
    appearance: Optional[AppearanceEstimate] = None
    inferred_geometry: Dict[str, EvidencedValue] = {}   # hidden/hypothesised props
    confidence: float = 0.5
    notes: List[str] = []


class SketchGraph(BaseModel):
    coordinate_system: Dict[str, Any] = {
        "type": "normalized_image", "origin": "top_left", "width": 1.0, "height": 1.0
    }
    primitives: List[GeometricPrimitive] = []
    constraints: List[GeometricConstraint] = []
    parts: List[SemanticPart] = []
    uncertainty: Dict[str, Any] = {}           # e.g. {"physical_scale": "unknown"}
    stats: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Stage 11/12: camera + depth
# ---------------------------------------------------------------------------

class ProjectionType(str, enum.Enum):
    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"


class CameraEstimate(BaseModel):
    projection: ProjectionType = ProjectionType.PERSPECTIVE
    focal_length_px: EvidencedValue = Field(default_factory=EvidencedValue)
    principal_point: Tuple[float, float] = (0.5, 0.5)   # normalised
    rotation_euler_deg: EvidencedValue = Field(default_factory=EvidencedValue)  # [rx,ry,rz]
    translation: EvidencedValue = Field(default_factory=EvidencedValue)          # [tx,ty,tz]
    object_rotation_euler_deg: EvidencedValue = Field(default_factory=EvidencedValue)
    ground_plane: Optional[Dict[str, Any]] = None
    scale: EvidencedValue = Field(default_factory=EvidencedValue)  # units per normalised width
    notes: List[str] = []


class DepthEvidence(BaseModel):
    depth_path: Optional[str] = None           # 16-bit or float PNG/npy
    normals_path: Optional[str] = None
    backend: str = "none"
    region_estimates: Dict[str, EvidencedValue] = {}   # per-part relative depth etc.
    confidence: float = 0.0
    notes: List[str] = []


class MultiViewObservation(BaseModel):
    view_id: str
    image_path: str
    graph_path: Optional[str] = None
    mask_path: Optional[str] = None
    crop_metadata_path: Optional[str] = None
    object_bbox: Optional[Tuple[int, int, int, int]] = None
    segmentation_confidence: float = 0.0
    camera: Optional[CameraEstimate] = None
    scale_to_primary: EvidencedValue = Field(default_factory=EvidencedValue)
    status: str = "success"
    warnings: List[str] = []


class CrossViewPartMatch(BaseModel):
    primary_part_id: str
    secondary_part_id: str
    view_id: str
    part_class: str
    geometric_cost: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source: EvidenceSource = EvidenceSource.FITTED_FROM_OBSERVATION


class MultiViewResult(BaseModel):
    enabled: bool = False
    primary_view_id: str = "view_000"
    observations: List[MultiViewObservation] = []
    matches: List[CrossViewPartMatch] = []
    shared_part_graph: Dict[str, List[Dict[str, Any]]] = {}
    relative_camera_poses: Dict[str, EvidencedValue] = {}
    consistent_scale: EvidencedValue = Field(default_factory=EvidencedValue)
    joint_optimization: Dict[str, Any] = {}
    warnings: List[str] = []


class HypothesisCandidate(BaseModel):
    id: str
    part_id: str
    hypothesis_type: str
    proposal: Dict[str, Any] = {}
    source: EvidenceSource = EvidenceSource.GENERATED_HYPOTHESIS
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=0.5)
    accepted: bool = False
    supporting_evidence: List[str] = []
    rejection_reasons: List[str] = []


class HypothesisReport(BaseModel):
    candidates: List[HypothesisCandidate] = []
    accepted_ids: List[str] = []
    rejected_ids: List[str] = []
    scoring_policy: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Stage 13/14: operators + construction plan
# ---------------------------------------------------------------------------

class OperatorCategory(str, enum.Enum):
    EXTRUDE = "extrude"
    REVOLVE = "revolve"
    SWEEP = "sweep"
    PRIMITIVE = "primitive"            # box/cylinder/sphere/cone/torus/capsule
    BOOLEAN = "boolean"
    LOFT = "loft"
    FREEFORM = "freeform"
    DISPLACEMENT = "displacement"
    TEXTURE_ONLY = "texture_only"
    RADIAL_ARRAY = "radial_array"
    MIRROR = "mirror"


class OperatorCandidate(BaseModel):
    operator: OperatorCategory
    confidence: float = Field(ge=0.0, le=1.0)


class MaterialSpec(BaseModel):
    material_class: str = "plastic"
    base_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)  # linear-ish 0..1
    roughness: float = 0.5
    metallic: float = 0.0
    opacity: float = 1.0
    transmission: float = 0.0
    normal_intensity: float = 1.0
    source: EvidenceSource = EvidenceSource.FITTED_FROM_OBSERVATION


class PlanPart(BaseModel):
    """One part in the declarative construction plan.

    Coordinates are in normalised object units: object width = 1.0,
    origin at object centre, x right, y up, z toward camera.
    """
    id: str
    operator: OperatorCategory
    parent: Optional[str] = None
    primitive_shape: Optional[str] = None    # for PRIMITIVE: cube|cylinder|sphere|cone|torus|capsule
    axis: Optional[Dict[str, Any]] = None    # {origin:[...], direction:[...]} for revolve/sweep
    profile: Optional[Dict[str, Any]] = None # {type, points:[[r,h]...] or [[x,y]...], closed}
    source_curve: Optional[str] = None       # sketch-graph primitive/part id
    depth: Optional[float] = None            # extrusion depth (normalised units)
    count: Optional[int] = None              # for radial_array
    angle_degrees: Optional[float] = None
    source_part: Optional[str] = None        # for arrays/mirrors
    boolean_target: Optional[str] = None     # for boolean cutters
    boolean_operation: Optional[str] = None  # difference|union|intersect
    transform: Dict[str, Any] = {}           # {location, rotation_deg, scale}
    material: MaterialSpec = Field(default_factory=MaterialSpec)
    visibility: Visibility = Visibility.VISIBLE
    render_visible: bool = True
    evidence: EvidencedValue = Field(default_factory=EvidencedValue)


class ConstructionPlan(BaseModel):
    object_id: str
    units: str = "normalized"                # or "meters" when scale known
    physical_width: Optional[float] = None   # set only with reliable scale reference
    parts: List[PlanPart] = []
    camera: Optional[CameraEstimate] = None
    uncertainty: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Stage 15-18: blender manifest, validation, refinement
# ---------------------------------------------------------------------------

class BlenderObjectInfo(BaseModel):
    name: str
    type: str                      # MESH|CURVE|EMPTY
    collection: str
    parent: Optional[str] = None
    part_id: Optional[str] = None
    modifiers: List[str] = []
    materials: List[str] = []
    has_uv: bool = False
    vertex_count: int = 0
    face_count: int = 0
    is_manifold: Optional[bool] = None
    pivot: Optional[Tuple[float, float, float]] = None


class BlenderManifest(BaseModel):
    blend_path: str
    glb_path: Optional[str] = None
    script_path: str
    objects: List[BlenderObjectInfo] = []
    collections: List[str] = []
    execution_log: str = ""
    blender_version: str = ""
    success: bool = False
    errors: List[str] = []


class ValidationMetrics(BaseModel):
    silhouette_iou: Optional[float] = None
    contour_chamfer_distance: Optional[float] = None
    feature_alignment_error_px: Optional[float] = None
    depth_correlation: Optional[float] = None
    part_mask_iou: Optional[float] = None
    perceptual_similarity: Optional[float] = None
    color_region_agreement: Optional[float] = None
    clay_silhouette_iou: Optional[float] = None
    extra: Dict[str, float] = {}


class ValidationResult(BaseModel):
    metrics: ValidationMetrics = Field(default_factory=ValidationMetrics)
    overlay_path: Optional[str] = None
    silhouette_comparison_path: Optional[str] = None
    depth_comparison_path: Optional[str] = None
    turntable_path: Optional[str] = None
    passed: bool = False
    notes: List[str] = []


class RefinementAction(BaseModel):
    iteration: int
    observed_problem: str
    modified_parameters: Dict[str, Dict[str, Any]] = {}  # name -> {previous, new}
    metric_change: Dict[str, Dict[str, Optional[float]]] = {}  # metric -> {previous, new}
    kept: bool = True                 # False if rolled back


class RefinementLog(BaseModel):
    actions: List[RefinementAction] = []
    initial_metrics: Dict[str, Optional[float]] = {}
    final_metrics: Dict[str, Optional[float]] = {}
    converged: bool = False
    iterations: int = 0


# ---------------------------------------------------------------------------
# Run manifest (reproducibility)
# ---------------------------------------------------------------------------

class RunManifest(BaseModel):
    run_id: str
    input_hashes: Dict[str, str] = {}
    config: Dict[str, Any] = {}
    software: Dict[str, str] = {}    # python, blender, deps
    seeds: Dict[str, int] = {}
    stage_outputs: Dict[str, str] = {}
    started_at: str = ""
    finished_at: str = ""
    status: str = "running"          # running|success|partial_success|failed_validation|unsupported_input


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

class SchemaIO:
    """Dump/load any of the models above as JSON or YAML."""

    @staticmethod
    def save_json(model: BaseModel, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(model.model_dump_json(indent=2))
        return str(p)

    @staticmethod
    def save_yaml(model: BaseModel, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(json.loads(model.model_dump_json()), sort_keys=False))
        return str(p)

    @staticmethod
    def load_json(cls, path: str | Path):
        return cls.model_validate(json.loads(Path(path).read_text()))

    @staticmethod
    def load_yaml(cls, path: str | Path):
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
