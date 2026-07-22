"""Stage 9: assemble the parametric sketch graph.

Pure assembly of fitted primitives + detected constraints into a
``SketchGraph`` with an explicit coordinate system and uncertainty record.
Semantic parts are added later by Stage 10.
"""
from __future__ import annotations

from typing import List

from .schemas import GeometricConstraint, GeometricPrimitive, SketchGraph


def build_sketch_graph(primitives: List[GeometricPrimitive],
                       constraints: List[GeometricConstraint]) -> SketchGraph:
    prim_counts: dict = {}
    for p in primitives:
        prim_counts[p.type.value] = prim_counts.get(p.type.value, 0) + 1
    con_counts: dict = {}
    for c in constraints:
        con_counts[c.type.value] = con_counts.get(c.type.value, 0) + 1

    return SketchGraph(
        coordinate_system={
            "type": "normalized_image",
            "origin": "top_left",
            "width": 1.0,
            "height": 1.0,
        },
        primitives=list(primitives),
        constraints=list(constraints),
        parts=[],
        uncertainty={"physical_scale": "unknown"},
        stats={
            "primitive_count": len(primitives),
            "primitives_by_type": prim_counts,
            "constraint_count": len(constraints),
            "constraints_by_type": con_counts,
        },
    )
