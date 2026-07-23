"""Uncertainty-tracking controls used by the required ablation evaluation."""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)


def _uniform_confidence(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if ((key == "confidence" or key.endswith("_confidence"))
                    and isinstance(child, (int, float))):
                out[key] = 1.0
            elif key == "uncertainty" and isinstance(child, dict):
                out[key] = {}
            else:
                out[key] = _uniform_confidence(child)
        return out
    if isinstance(value, list):
        return [_uniform_confidence(child) for child in value]
    return value


def disable_tracking(model: ModelT) -> ModelT:
    """Return a copy with every confidence fixed to one and summaries empty.

    Evidence sources are deliberately preserved: provenance and uncertainty
    are different safety properties. Uniform confidence removes the system's
    ability to distinguish weak from strong evidence and therefore changes
    downstream confidence-weighted decisions instead of merely hiding fields.
    """
    data = _uniform_confidence(model.model_dump(mode="json"))
    return model.__class__.model_validate(data)
