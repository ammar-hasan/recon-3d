"""Pipeline configuration: every knob in one validated model."""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class SegmentationConfig(BaseModel):
    background_removal_enabled: bool = True
    backend: str = "auto"            # auto|rembg|grabcut|threshold|user_mask
    rembg_model: str = "isnet-general-use"
    grabcut_iterations: int = 8
    min_coverage: float = 0.005      # warn if object smaller than this fraction
    max_coverage: float = 0.98


class CropConfig(BaseModel):
    padding_px: int = 48
    canvas_size: int = 1024


class PreprocessConfig(BaseModel):
    color_regions: int = 8
    edge_low_threshold: int = 60
    edge_high_threshold: int = 160
    detail_kernel: int = 3


class VectorizeConfig(BaseModel):
    backend: str = "auto"            # auto|vtracer|contour
    color_precision: int = 6
    filter_speckle_px: int = 8
    corner_threshold_deg: float = 60.0


class CleanupConfig(BaseModel):
    min_path_area_norm: float = 1e-5
    dedupe_distance_norm: float = 0.002
    simplify_tolerance_norm: float = 0.0015
    smooth: bool = True


class PrimitiveConfig(BaseModel):
    enabled: bool = True
    max_fit_error_norm: float = 0.01      # above this keep fallback curve
    min_arc_coverage: float = 0.15        # fraction of full circle to accept arc
    ransac_iterations: int = 200
    seed: int = 1337


class CameraConfig(BaseModel):
    enabled: bool = True
    default_focal_px: float = 1200.0      # for 1024 canvas, ~50mm-ish
    assume_projection: str = "auto"       # auto|perspective|orthographic


class DepthConfig(BaseModel):
    backend: str = "auto"                 # auto|midas|shading|none
    enabled: bool = True
    depth_enabled: bool = True
    normals_enabled: bool = True


class BlenderConfig(BaseModel):
    blender_bin: str = "/Applications/Blender.app/Contents/MacOS/Blender"
    timeout_seconds: int = 600
    render_engine: str = "BLENDER_EEVEE_NEXT"
    render_samples: int = 32
    render_resolution: int = 1024
    sandbox: bool = True


class RefinementConfig(BaseModel):
    enabled: bool = True
    max_iterations: int = 6
    target_silhouette_iou: float = 0.90
    min_iou_gain: float = 0.003           # stop when improvement below this
    max_renders: int = 12


class MultiviewConfig(BaseModel):
    enabled: bool = True
    max_views: int = Field(default=8, ge=2, le=32)
    max_part_match_cost: float = Field(default=0.85, gt=0.0, le=2.0)
    min_pose_baseline_deg: float = Field(default=5.0, ge=0.0, le=180.0)
    joint_refinement_enabled: bool = True
    yaw_step_deg: int = Field(default=15, ge=5, le=90)
    depth_scale_candidates: List[float] = Field(
        default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
    joint_render_resolution: int = Field(default=256, ge=64, le=512)
    min_joint_iou_gain: float = Field(default=0.005, ge=0.0, le=1.0)
    max_primary_iou_drop: float = Field(default=0.005, ge=0.0, le=0.1)
    visual_hull_enabled: bool = True
    visual_hull_grid_size: int = Field(default=48, ge=24, le=128)
    visual_hull_depth_extent: float = Field(default=1.5, gt=0.1, le=4.0)
    visual_hull_mask_dilation_px: int = Field(default=1, ge=0, le=8)
    visual_hull_min_primary_iou: float = Field(default=0.80, ge=0.0, le=1.0)
    visual_hull_min_secondary_iou: float = Field(default=0.70, ge=0.0, le=1.0)
    visual_hull_box_symmetry_prior_enabled: bool = True
    visual_hull_semantic_completion_enabled: bool = True


class HypothesisConfig(BaseModel):
    enabled: bool = True
    acceptance_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    max_confidence: float = Field(default=0.5, ge=0.0, le=0.5)


class ConstraintConfig(BaseModel):
    enabled: bool = True


class SemanticConfig(BaseModel):
    enabled: bool = True


class PipelineConfig(BaseModel):
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    vectorize: VectorizeConfig = Field(default_factory=VectorizeConfig)
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)
    primitives: PrimitiveConfig = Field(default_factory=PrimitiveConfig)
    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    semantics: SemanticConfig = Field(default_factory=SemanticConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    blender: BlenderConfig = Field(default_factory=BlenderConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    multiview: MultiviewConfig = Field(default_factory=MultiviewConfig)
    hypotheses: HypothesisConfig = Field(default_factory=HypothesisConfig)
    seed: int = 1337
    keep_intermediates: bool = True

    @staticmethod
    def from_yaml(path: str | Path) -> "PipelineConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return PipelineConfig.model_validate(data)

    def to_yaml(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        import json
        p.write_text(yaml.safe_dump(json.loads(self.model_dump_json()), sort_keys=False))
        return str(p)


def software_versions() -> Dict[str, str]:
    import sys
    versions: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("numpy", "cv2", "skimage", "PIL", "yaml", "pydantic", "scipy"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "missing"
    try:
        import importlib.metadata as md
        versions["vtracer"] = md.version("vtracer")
    except Exception:
        versions["vtracer"] = "missing"
    try:
        import rembg  # noqa: F401
        versions["rembg"] = "installed"
    except Exception:
        versions["rembg"] = "missing"
    return versions
