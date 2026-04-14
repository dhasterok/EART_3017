"""
polygon.py
----------
Base class for a closed 2-D polygon.

``Polygon`` stores the common attributes shared by any polygon regardless
of application domain (vertices, display name, colour, visibility) and
provides pure geometry operations.  Application-specific subclasses should
inherit from ``Polygon`` and add domain attributes:

    class DensityBody(Polygon):
        def __init__(self, ..., density=0.0, **kw):
            super().__init__(**kw)
            self.density = density

        def clone(self):
            b = super().clone()
            b.density = self.density
            return b

Conventions
-----------
    x  horizontal axis (any consistent unit — km, m, …)
    z  vertical axis   (positive downward for depth sections)
    Vertices are stored as a mutable list of [x, z] pairs so that
    individual coordinates can be updated in-place.
"""

import math
from typing import List, Optional, Tuple

import numpy as np


# Default colour cycle used when no colour is supplied
DEFAULT_COLORS: List[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


class Polygon:
    """
    A closed 2-D polygon with display attributes and geometry helpers.

    Attributes
    ----------
    vertices : list of [x, z]
        Mutable list of vertex coordinate pairs.  The polygon is implicitly
        closed (last vertex connects back to the first).
    name     : str   — display name (defaults to ``"Polygon"``; subclasses
                        typically supply a numbered name)
    color    : str   — hex colour string, e.g. ``"#4C72B0"``
    visible  : bool  — whether the polygon is shown / included in models
    """

    def __init__(
        self,
        vertices: Optional[List[List[float]]] = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
        visible: bool = True,
    ):
        self.vertices: List[List[float]] = vertices if vertices is not None else []
        self.name:    str  = name  or "Polygon"
        self.color:   str  = color or DEFAULT_COLORS[0]
        self.visible: bool = visible

    # ── completeness ──────────────────────────────────────────────────────

    def is_complete(self) -> bool:
        """Return True when the polygon has at least three vertices."""
        return len(self.vertices) >= 3

    # ── array view ────────────────────────────────────────────────────────

    def vertex_array(self) -> np.ndarray:
        """Return vertices as an (N, 2) float64 array, or empty (0, 2)."""
        return (np.array(self.vertices, dtype=float)
                if self.vertices else np.empty((0, 2)))

    # ── point-in-polygon ─────────────────────────────────────────────────

    def contains_point(self, x: float, z: float) -> bool:
        """
        Ray-casting test: return True if (x, z) is inside the polygon.

        Always returns False for polygons with fewer than 3 vertices.
        """
        verts = self.vertices
        if len(verts) < 3:
            return False
        inside = False
        xj, zj = verts[-1]
        for xi, zi in verts:
            if ((zi > z) != (zj > z)) and \
               (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
                inside = not inside
            xj, zj = xi, zi
        return inside

    # ── nearest geometry ──────────────────────────────────────────────────

    def nearest_vertex(self, x: float, z: float) -> Tuple[int, float]:
        """
        Return ``(index, distance)`` of the nearest vertex.

        Returns ``(-1, 1e9)`` when the vertex list is empty.
        """
        best_i, best_d = -1, 1e9
        for i, (vx, vz) in enumerate(self.vertices):
            d = math.hypot(vx - x, vz - z)
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d

    def nearest_edge_point(
        self, x: float, z: float
    ) -> Tuple[int, float, float, float]:
        """
        Return the closest point on any edge to (x, z).

        Returns
        -------
        (edge_i, t, px, pz)
            edge_i : start-vertex index of the edge  (-1 if no valid edge)
            t      : parameter along the edge  (0 = start, 1 = end)
            px, pz : coordinates of the closest point
        """
        best_i, best_d, best_t = -1, 1e9, 0.0
        best_px, best_pz = x, z
        n = len(self.vertices)
        for i in range(n):
            ax, az = self.vertices[i]
            bx, bz = self.vertices[(i + 1) % n]
            dx, dz = bx - ax, bz - az
            len2 = dx * dx + dz * dz
            if len2 < 1e-12:
                continue
            t = max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / len2))
            px = ax + t * dx
            pz = az + t * dz
            d = math.hypot(px - x, pz - z)
            if d < best_d:
                best_d, best_t, best_i = d, t, i
                best_px, best_pz = px, pz
        return best_i, best_t, best_px, best_pz

    # ── copying ───────────────────────────────────────────────────────────

    def clone(self) -> "Polygon":
        """
        Return a deep copy of the polygon's common attributes.

        Uses ``__class__.__new__`` so the returned object is always the
        same concrete type as ``self``.  Subclasses should call
        ``super().clone()`` and then copy their own attributes::

            def clone(self):
                p = super().clone()
                p.density = self.density
                return p
        """
        p = self.__class__.__new__(self.__class__)
        p.vertices = [v[:] for v in self.vertices]
        p.name     = self.name
        p.color    = self.color
        p.visible  = self.visible
        return p


# ---------------------------------------------------------------------------
# Shared geometry utilities
# ---------------------------------------------------------------------------

def estimate_strike_half_length(vertices) -> float:
    """
    Estimate a reasonable along-strike half-length (km) for a 2.5-D extrusion.

    Uses 5× the larger of the polygon's horizontal width or mean depth, which
    produces an approximately 2-D response for typical crustal geometries.

    Parameters
    ----------
    vertices : list of [x, z]
        Polygon vertices in km (z positive downward).

    Returns
    -------
    float  -- strike half-length in km
    """
    import numpy as np
    x = np.array([v[0] for v in vertices])
    z = np.array([v[1] for v in vertices])
    width = x.max() - x.min()
    depth = z.mean()
    return 5.0 * max(width, depth)
