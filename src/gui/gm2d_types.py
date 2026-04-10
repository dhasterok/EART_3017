"""
gm2d_types.py
-------------
Shared enumerations and data model for the 2-D gravity + magnetic modeller.

Imported by gm2d_canvas, gm2d_control_dock, gm2d_polygon_dock, and
gravmag2d_gui so that all modules work with the same objects.
"""

from enum import Enum, auto
from pathlib import Path
import sys

_HERE   = Path(__file__).resolve().parent
_COURSE = _HERE.parent.parent
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from src.utils.polygon import Polygon, DEFAULT_COLORS


# ── display / component enums ─────────────────────────────────────────────────

class DisplayMode(Enum):
    NONE      = "none"
    BOTH      = "both"
    GRAVITY   = "gravity"
    MAGNETICS = "magnetics"


class MagComponent(Enum):
    TMI = "Total field (ΔT)"   # solid line
    BX  = "Horizontal (Bx)"   # dashed line
    BZ  = "Vertical (Bz)"     # solid line


# ── interaction mode ──────────────────────────────────────────────────────────

class Mode(Enum):
    NONE          = auto()   # no editing tool active; navigation only
    DRAW          = auto()
    SELECT        = auto()
    ADD_VERTEX    = auto()
    DELETE        = auto()
    MASK          = auto()
    MOVE_BODY     = auto()
    DELETE_VERTEX = auto()


# ── data model ────────────────────────────────────────────────────────────────

class PolygonBody(Polygon):
    """
    A single 2-D density/susceptibility body.

    Extends ``Polygon`` (vertices, name, color, visible + geometry methods)
    with geophysics-specific attributes: density, susceptibility and
    remanent magnetisation.
    """

    _counter = 0

    def __init__(self, vertices=None, density=2370.0,
                 susceptibility=0.001,
                 remanence_Am=0.0, remanence_inc_deg=0.0, remanence_dec_deg=0.0,
                 color=None, name=None, visible=True):
        PolygonBody._counter += 1
        super().__init__(
            vertices = vertices or [],
            name     = name or f"Body {PolygonBody._counter}",
            color    = color,
            visible  = visible,
        )
        self.density            = float(density)
        self.susceptibility     = float(susceptibility)
        self.remanence_Am       = float(remanence_Am)
        self.remanence_inc_deg  = float(remanence_inc_deg)
        self.remanence_dec_deg  = float(remanence_dec_deg)

    def clone(self) -> "PolygonBody":
        b = super().clone()   # copies vertices, name, color, visible
        b.density            = self.density
        b.susceptibility     = self.susceptibility
        b.remanence_Am       = self.remanence_Am
        b.remanence_inc_deg  = self.remanence_inc_deg
        b.remanence_dec_deg  = self.remanence_dec_deg
        return b
