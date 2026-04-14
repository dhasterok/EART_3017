"""
model_io.py
-----------
Save and load 2-D gravity model state as human-readable JSON.

Saved fields
------------
  version          : file-format version (int)
  bg_density       : background density in kg/m³
  profile          : x_min, x_max, n_pts  (km)
  bodies           : list of polygon bodies
  observed_data    : loaded gravity stations (optional)
  computed_gravity : forward-model result at save time
  misfit           : residuals and RMS against unmasked stations (optional)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).resolve().parent   # src/physics/gravity/
_COURSE = _HERE.parent.parent.parent       # project root
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from src.physics.gravity.talwani_model import compute_gz

if TYPE_CHECKING:
    from src.gui.gravity2d_gui import GravityCanvas
    from src.physics.gravity.data_loader import ObservedData


# ─────────────────────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────────────────────

def save_model(path: str | Path,
               canvas: "GravityCanvas",
               bg_density: float,
               obs_data: "ObservedData | None" = None) -> None:
    """Write the current model to *path* as indented JSON."""

    # ── polygon bodies ────────────────────────────────────────────────────
    bodies_list = []
    for b in canvas.bodies:
        bodies_list.append({
            "name":     b.name,
            "density":  b.density,
            "color":    b.color,
            "visible":  b.visible,
            "vertices": [list(v) for v in b.vertices],
        })

    # ── computed gravity ──────────────────────────────────────────────────
    xp  = np.linspace(canvas.x_min, canvas.x_max, canvas.n_pts)
    gz  = np.zeros(len(xp))
    for b in canvas.bodies:
        if b.visible and len(b.vertices) >= 3:
            gz += compute_gz(xp, b.vertices, b.density - bg_density)

    data = {
        "version":   1,
        "bg_density": bg_density,
        "profile": {
            "x_min": canvas.x_min,
            "x_max": canvas.x_max,
            "n_pts": canvas.n_pts,
        },
        "bodies": bodies_list,
        "computed_gravity": {
            "x_km": [round(v, 6) for v in xp.tolist()],
            "gz_mGal": [round(v, 6) for v in gz.tolist()],
        },
    }

    # ── observed data ─────────────────────────────────────────────────────
    if obs_data is not None:
        od: dict = {
            "source_file":  obs_data.source_file,
            "x_col":        obs_data.x_col,
            "y_col":        obs_data.y_col,
            "gz_col":       obs_data.gz_col,
            "gz_unc_col":   obs_data.gz_unc_col,
            "x_scale_km":   obs_data.x_scale_km,
            "x_km":         [round(v, 6) for v in obs_data.x.tolist()],
            "gz_mGal":      [round(v, 6) for v in obs_data.gz.tolist()],
        }
        if obs_data.gz_unc is not None:
            od["gz_unc_mGal"] = [round(v, 6) for v in obs_data.gz_unc.tolist()]
        if obs_data.masked is not None:
            od["masked"] = obs_data.masked.tolist()
        if obs_data.profile_start is not None:
            od["profile_start"] = list(obs_data.profile_start)
            od["profile_end"]   = list(obs_data.profile_end)
            od["swath_half_width"] = obs_data.swath_half_width
        data["observed_data"] = od

        # ── misfit ────────────────────────────────────────────────────────
        unmasked = (~obs_data.masked) if obs_data.masked is not None \
                   else np.ones(len(obs_data.x), dtype=bool)
        if unmasked.any():
            x_um  = obs_data.x[unmasked]
            gz_um = obs_data.gz[unmasked]
            gz_interp = np.interp(x_um, xp, gz)
            residuals = gz_um - gz_interp
            rms = float(np.sqrt(np.mean(residuals ** 2)))
            data["misfit"] = {
                "x_km":       [round(v, 6) for v in x_um.tolist()],
                "residuals_mGal": [round(v, 6) for v in residuals.tolist()],
                "rms_mGal":   round(rms, 4),
            }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  Load
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str | Path) -> dict:
    """Read a JSON model file and return the raw dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_model(data: dict, canvas: "GravityCanvas") -> tuple:
    """
    Restore canvas state from a loaded model dict.

    Returns
    -------
    bg_density : float
    obs_data   : ObservedData or None
    """
    from src.gui.gravity2d_gui import PolygonBody
    from src.physics.gravity.data_loader import ObservedData

    # ── profile ───────────────────────────────────────────────────────────
    prof = data.get("profile", {})
    canvas.x_min = float(prof.get("x_min", -50.0))
    canvas.x_max = float(prof.get("x_max",  50.0))
    canvas.n_pts = int(prof.get("n_pts",    201))

    # ── background density ────────────────────────────────────────────────
    bg_density = float(data.get("bg_density", 2670.0))
    canvas.bg_density = bg_density

    # ── polygon bodies ────────────────────────────────────────────────────
    canvas.bodies.clear()
    PolygonBody._counter = 0
    for bd in data.get("bodies", []):
        b = PolygonBody(
            vertices = [list(v) for v in bd["vertices"]],
            density  = float(bd["density"]),
            color    = bd.get("color", "#4C72B0"),
            name     = bd.get("name", "Body"),
            visible  = bool(bd.get("visible", True)),
        )
        b.name = bd.get("name", b.name)   # override auto name
        canvas.bodies.append(b)

    # ── observed data (optional) ──────────────────────────────────────────
    obs_data = None
    od = data.get("observed_data")
    if od is not None:
        x_arr  = np.array(od["x_km"],   dtype=float)
        gz_arr = np.array(od["gz_mGal"], dtype=float)
        gz_unc = (np.array(od["gz_unc_mGal"], dtype=float)
                  if "gz_unc_mGal" in od else None)
        masked = (np.array(od["masked"], dtype=bool)
                  if "masked" in od else np.zeros(len(x_arr), dtype=bool))
        p_start = tuple(od["profile_start"]) if "profile_start" in od else None
        p_end   = tuple(od["profile_end"])   if "profile_end"   in od else None

        obs_data = ObservedData(
            x             = x_arr,
            gz            = gz_arr,
            gz_unc        = gz_unc,
            masked        = masked,
            profile_start = p_start,
            profile_end   = p_end,
            swath_half_width = float(od.get("swath_half_width", 5.0)),
            x_col         = od.get("x_col",       ""),
            y_col         = od.get("y_col",        ""),
            gz_col        = od.get("gz_col",       ""),
            gz_unc_col    = od.get("gz_unc_col",   ""),
            x_scale_km    = float(od.get("x_scale_km", 1.0)),
            source_file   = od.get("source_file",  ""),
        )

    return bg_density, obs_data
