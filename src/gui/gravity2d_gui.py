#!/usr/bin/env python3
"""
gravity2d_gui.py
----------------
Interactive 2-D gravity modelling GUI.

  • Draw and edit polygonal density bodies with the mouse.
  • Talwani (1959) forward model updates in real time.
  • Controls in a left QDockWidget; polygon table in a bottom QDockWidget.

Usage
-----
    python gravity2d_gui.py


To do:
    Add delete vertex
    Add a checkbox to prevent snapping when vertices are very close together (e.g. on a vertical edge)
    Add undo/redo stack
    Add ability to change density contrast, not just density
    Add ability to drag a vertex without changing the plot scale
    Add tooltips
    When changing the vertex z location to a lower value it will sometimes result in flipping the sign of the anomaly
"""

import sys
import math
from enum import Enum, auto
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np

# ── locate the course root (new_version/) and add it to sys.path so that
#    src.utils.CustomWidgets (and its transitive deps) can be imported
_HERE      = Path(__file__).resolve().parent          # src/gui/
_COURSE    = _HERE.parent.parent                       # new_version/
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from src.common.gui.CustomWidgets import ToggleSwitch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QPushButton, QColorDialog, QButtonGroup,
    QGroupBox, QFormLayout, QHeaderView, QAbstractItemView,
    QSizePolicy, QStatusBar, QMessageBox, QSplitter, QFileDialog,
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QAction, QKeySequence

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from src.physics.gravity.talwani_model import compute_gz
from src.physics.gravity.data_loader import load_csv_data, ObservedData
from src.physics.gravity.model_io import save_model, load_model, apply_model


# ─────────────────────────────────────────────────────────────────────────────
#  Application constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

VERTEX_RADIUS    = 6     # display radius of vertex circles (points)
HIT_RADIUS_PX    = 10    # pixels -- vertex / edge pick tolerance
SNAP_RADIUS_PX   = 12    # pixels -- snap-to-vertex activation radius
SNAP_INDICATOR_PX = 14   # pixels -- radius of the snap ring drawn on canvas
_LINK_EPS        = 1e-9  # km -- two vertices "at same location" threshold


# ─────────────────────────────────────────────────────────────────────────────
#  Interaction mode
# ─────────────────────────────────────────────────────────────────────────────

class Mode(Enum):
    DRAW       = auto()   # clicking adds vertices; close = right-click / Enter
    SELECT     = auto()   # click selects; drag moves vertex or whole body
    ADD_VERTEX = auto()   # click on edge inserts a new vertex
    DELETE     = auto()   # click on a polygon deletes it
    MASK       = auto()   # click in gravity axes toggles station mask


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

class PolygonBody:
    """A single 2-D density body defined by a closed polygon."""

    _counter = 0

    def __init__(self, vertices=None, density=300.0,
                 color=None, name=None, visible=True):
        PolygonBody._counter += 1
        self.name     = name or f"Body {PolygonBody._counter}"
        self.vertices = vertices or []   # list of [x, z] in km
        self.density  = float(density)   # kg/m³  (contrast)
        self.color    = color or DEFAULT_COLORS[
            (PolygonBody._counter - 1) % len(DEFAULT_COLORS)]
        self.visible  = visible

    def is_complete(self):
        return len(self.vertices) >= 3

    def clone(self):
        b = PolygonBody.__new__(PolygonBody)
        b.name     = self.name
        b.vertices = [v[:] for v in self.vertices]
        b.density  = self.density
        b.color    = self.color
        b.visible  = self.visible
        return b

    def vertex_array(self):
        """Return vertices as (N, 2) numpy array."""
        return np.array(self.vertices, dtype=float) if self.vertices else np.empty((0, 2))

    def contains_point(self, x, z):
        """Ray-casting point-in-polygon test."""
        verts = self.vertices
        inside = False
        xj, zj = verts[-1]
        for xi, zi in verts:
            if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
                inside = not inside
            xj, zj = xi, zi
        return inside

    def nearest_vertex(self, x, z) -> Tuple[int, float]:
        """Return (index, distance_km) of nearest vertex."""
        best_i, best_d = -1, 1e9
        for i, (vx, vz) in enumerate(self.vertices):
            d = math.hypot(vx - x, vz - z)
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d

    def nearest_edge_point(self, x, z) -> Tuple[int, float, float, float]:
        """
        Return (edge_start_index, t, px, pz) for the nearest edge point.
        t is the parameter along the edge [0, 1], (px, pz) the closest point.
        """
        best_i, best_d, best_t, best_px, best_pz = -1, 1e9, 0, x, z
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


# ─────────────────────────────────────────────────────────────────────────────
#  Matplotlib canvas
# ─────────────────────────────────────────────────────────────────────────────

class GravityCanvas(FigureCanvas):
    """
    Two-panel matplotlib canvas:
      • Top  (ax_grav)  -- gravity profile (calculated ± observed)
      • Bottom (ax_model) -- model cross-section; polygon editing
    """

    bodies_changed = pyqtSignal()      # emit whenever polygons are modified
    rms_updated    = pyqtSignal(float) # emit current RMS misfit (mGal) after each forward solve

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 8), tight_layout=False)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

        # ── application state (must precede _setup_axes) ──────────────────
        self.bodies: List[PolygonBody] = []
        self.mode   = Mode.SELECT

        # profile  (always stored in km internally)
        self.x_min    = -50.0
        self.x_max    =  50.0
        self.n_pts    =  201

        # display units: True = km, False = m
        self.use_km   = True

        # background density (kg/m³); forward model uses body.density - bg_density
        self.bg_density = 2670.0

        # ── axes ──────────────────────────────────────────────────────────
        gs = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[1, 2],
            hspace=0.08,
        )
        self.ax_grav  = self.fig.add_subplot(gs[0])
        self.ax_model = self.fig.add_subplot(gs[1], sharex=self.ax_grav)

        self._setup_axes()

        # selection
        self.selected_body:   Optional[PolygonBody] = None
        self.selected_vertex: int = -1      # index within selected_body
        self._drag_active     = False
        self._drag_prev       = (None, None)

        # in-progress drawing
        self._draw_verts: List[List[float]] = []
        self._draw_line  = None
        self._draw_cursor_line = None
        self._cursor_pos = (0.0, 0.0)

        # matplotlib handles for polygons
        self._body_patches: dict = {}    # body → Polygon patch
        self._vertex_scatter = None
        self._grav_line = None
        self._obs_line  = None
        self._obs_errbar = None    # error-bar container (ErrorbarContainer)

        # snap indicator artist (ring drawn when cursor is near a snap target)
        self._snap_ring = None

        # observed data (optional)
        self.obs_data: Optional[ObservedData] = None

        # ── event connections ─────────────────────────────────────────────
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("key_press_event",      self._on_key)

        self._full_redraw()

    # ── axes appearance ───────────────────────────────────────────────────

    def _setup_axes(self):
        self.ax_grav.set_ylabel("Gravity anomaly (mGal)")
        self.ax_grav.axhline(0, color="gray", lw=0.6, ls="--")
        self.ax_grav.grid(True, alpha=0.3)
        self.ax_grav.tick_params(labelbottom=False)

        self.ax_model.set_xlabel("Distance (km)")
        self.ax_model.set_ylabel("Depth (km)")
        self.ax_model.invert_yaxis()
        self.ax_model.set_xlim(self.x_min, self.x_max)
        self.ax_model.set_ylim(20, -0.5)
        self.ax_model.axhline(0, color="gray", lw=1.0, ls="-")
        self.ax_model.grid(True, alpha=0.2)

        self.fig.canvas.draw_idle()

    def _apply_unit_formatters(self):
        """
        Axes data are always in km.  When use_km=False, tick labels are
        multiplied by 1000 so the user sees metres, while the underlying
        coordinate system (and all vertex data) stays in km.
        """
        import matplotlib.ticker as ticker

        if self.use_km:
            fmt = ticker.ScalarFormatter()
            x_label, z_label = "Distance (km)", "Depth (km)"
        else:
            scale = 1000.0
            fmt_x = ticker.FuncFormatter(lambda v, _: f"{v * scale:g}")
            fmt_z = ticker.FuncFormatter(lambda v, _: f"{v * scale:g}")
            self.ax_model.xaxis.set_major_formatter(fmt_x)
            self.ax_model.yaxis.set_major_formatter(fmt_z)
            self.ax_grav.xaxis.set_major_formatter(fmt_x)
            self.ax_model.set_xlabel("Distance (m)")
            self.ax_model.set_ylabel("Depth (m)")
            self.fig.canvas.draw_idle()
            return

        self.ax_model.xaxis.set_major_formatter(fmt)
        self.ax_model.yaxis.set_major_formatter(ticker.ScalarFormatter())
        self.ax_grav.xaxis.set_major_formatter(fmt)
        self.ax_model.set_xlabel(x_label)
        self.ax_model.set_ylabel(z_label)
        self.fig.canvas.draw_idle()

    # ── public profile / unit controls ────────────────────────────────────

    def set_units(self, use_km: bool):
        """Switch display units between km (True) and m (False).
        Vertex data and axis limits are always stored in km."""
        self.use_km = use_km
        self._apply_unit_formatters()
        self.fig.canvas.draw_idle()

    def set_profile(self, x_min, x_max, n_pts):
        """x_min / x_max are passed in display units; stored internally as km."""
        scale = 1.0 if self.use_km else 1e-3
        self.x_min = x_min * scale
        self.x_max = x_max * scale
        self.n_pts = max(3, int(n_pts))
        self.ax_model.set_xlim(self.x_min, self.x_max)
        self.ax_grav.set_xlim(self.x_min, self.x_max)
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def set_depth_range(self, z_max):
        """z_max is in display units; stored/used internally as km."""
        scale = 1.0 if self.use_km else 1e-3
        z_max_km = max(z_max * scale, 1.0)
        self.ax_model.set_ylim(z_max_km, -0.5)
        self.fig.canvas.draw_idle()

    def set_mode(self, mode: Mode):
        self.mode = mode
        if mode != Mode.DRAW:
            self._draw_verts.clear()
            self._remove_draw_artists()
        self._hide_snap_ring()
        self.fig.canvas.draw_idle()

    # ── pixel ↔ km conversion ─────────────────────────────────────────────

    def _px_to_km(self, n_px: float) -> float:
        """Convert a screen distance (pixels) to km in the model axes."""
        inv = self.ax_model.transData.inverted()
        p0 = inv.transform((0.0, 0.0))
        p1 = inv.transform((n_px, 0.0))
        return abs(float(p1[0]) - float(p0[0]))

    # ── snapping helpers ──────────────────────────────────────────────────

    def _find_snap_target(self, x: float, z: float,
                          exclude_body=None,
                          exclude_vi: int = -1) -> Optional[Tuple[float, float]]:
        """
        Return (sx, sz) of the nearest existing vertex within SNAP_RADIUS_PX
        screen pixels, or None.  Also considers the in-progress draw vertices
        (so you can close back to the first point precisely).
        """
        snap_km = self._px_to_km(SNAP_RADIUS_PX)
        best_d, best_pos = snap_km, None

        # Committed polygon vertices
        for body in self.bodies:
            for vi, (vx, vz) in enumerate(body.vertices):
                if body is exclude_body and vi == exclude_vi:
                    continue
                d = math.hypot(vx - x, vz - z)
                if d < best_d:
                    best_d, best_pos = d, (vx, vz)

        # In-progress draw vertices (skip the last one to avoid self-snap
        # when adding the very point we're about to place)
        for vi, (vx, vz) in enumerate(self._draw_verts[:-1]
                                       if self._draw_verts else []):
            d = math.hypot(vx - x, vz - z)
            if d < best_d:
                best_d, best_pos = d, (vx, vz)

        return best_pos

    def _find_linked_vertices(self, body: PolygonBody,
                              vi: int) -> List[Tuple['PolygonBody', int]]:
        """
        Return all (body, vertex_index) pairs across every polygon whose
        coordinates match body.vertices[vi] within _LINK_EPS km.
        These are vertices that should move together.
        """
        x, z = body.vertices[vi]
        linked = []
        for b in self.bodies:
            for i, (vx, vz) in enumerate(b.vertices):
                if b is body and i == vi:
                    continue
                if math.hypot(vx - x, vz - z) < _LINK_EPS:
                    linked.append((b, i))
        return linked

    # ── snap indicator ────────────────────────────────────────────────────

    def _show_snap_ring(self, sx: float, sz: float):
        """Draw (or move) the yellow snap ring at (sx, sz)."""
        r = self._px_to_km(SNAP_INDICATOR_PX)
        if self._snap_ring is not None:
            try:
                self._snap_ring.remove()
            except Exception:
                pass
        self._snap_ring = mpatches.Circle(
            (sx, sz), r,
            facecolor="none", edgecolor="#FFD700",
            linewidth=2.0, zorder=10,
        )
        self.ax_model.add_patch(self._snap_ring)

    def _hide_snap_ring(self):
        if self._snap_ring is not None:
            try:
                self._snap_ring.remove()
            except Exception:
                pass
            self._snap_ring = None

    # ── gravity computation ───────────────────────────────────────────────

    def _profile_x(self):
        return np.linspace(self.x_min, self.x_max, self.n_pts)

    def _update_gravity(self):
        xp = self._profile_x()

        # Forward model: use density contrast = body.density - bg_density
        gz = np.zeros(len(xp))
        for body in self.bodies:
            if body.visible and body.is_complete():
                contrast = body.density - self.bg_density
                gz += compute_gz(xp, body.vertices, contrast)

        if self._grav_line is not None:
            self._grav_line.remove()
        self._grav_line, = self.ax_grav.plot(
            xp, gz, color="navy", lw=1.5, label="Calculated", zorder=3)

        # Remove old observed artists
        if self._obs_line is not None:
            try:
                self._obs_line.remove()
            except Exception:
                pass
            self._obs_line = None
        if self._obs_errbar is not None:
            try:
                self._obs_errbar.remove()
            except Exception:
                pass
            self._obs_errbar = None

        # Plot observed data
        if self.obs_data is not None:
            od = self.obs_data
            mask = od.masked if od.masked is not None \
                   else np.zeros(len(od.x), dtype=bool)

            if self.mode == Mode.MASK:
                # Show all stations; masked = grey, unmasked = red
                unmasked = ~mask
                masked   =  mask
                if unmasked.any():
                    self.ax_grav.plot(
                        od.x[unmasked], od.gz[unmasked],
                        "ro", ms=4, zorder=5, label="Observed")
                if masked.any():
                    self.ax_grav.plot(
                        od.x[masked], od.gz[masked],
                        "o", color="grey", ms=4, zorder=4, label="Masked")
                # Store a dummy handle so we can remove later
                self._obs_line = None
            else:
                # Normal mode: show only unmasked stations
                show = ~mask
                if show.any():
                    x_s  = od.x[show]
                    gz_s = od.gz[show]
                    unc_s = od.gz_unc[show] if od.gz_unc is not None else None
                    if unc_s is not None:
                        self._obs_errbar = self.ax_grav.errorbar(
                            x_s, gz_s, yerr=unc_s,
                            fmt="ro", ms=3, elinewidth=0.8, capsize=2,
                            label="Observed", zorder=4)
                    else:
                        self._obs_line, = self.ax_grav.plot(
                            x_s, gz_s, "ro", ms=3,
                            label="Observed", zorder=4)

        # Compute and emit RMS misfit against unmasked observed stations
        if self.obs_data is not None:
            od   = self.obs_data
            umsk = (~od.masked) if od.masked is not None \
                   else np.ones(len(od.x), dtype=bool)
            if umsk.any():
                gz_interp = np.interp(od.x[umsk], xp, gz)
                rms = float(np.sqrt(np.mean((od.gz[umsk] - gz_interp) ** 2)))
                self.rms_updated.emit(rms)

        self.ax_grav.relim()
        self.ax_grav.autoscale_view(scalex=False, scaley=True)

    def set_bg_density(self, bg: float):
        """Update background density and recompute forward model."""
        self.bg_density = bg
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def load_observed(self, obs_data: ObservedData):
        """Load an ObservedData object and refresh display."""
        self.obs_data = obs_data
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def _toggle_station_mask(self, x_click: float):
        """Toggle the mask of the station nearest to x_click (in km)."""
        if self.obs_data is None:
            return
        dists = np.abs(self.obs_data.x - x_click)
        idx   = int(np.argmin(dists))
        self.obs_data.masked[idx] = not self.obs_data.masked[idx]
        self._update_gravity()
        self.fig.canvas.draw_idle()

    # ── full redraw ───────────────────────────────────────────────────────

    def _full_redraw(self):
        """Redraw all polygon patches then update gravity."""
        # remove stale patches
        for patch in list(self._body_patches.values()):
            try:
                patch.remove()
            except Exception:
                pass
        self._body_patches.clear()

        if self._vertex_scatter is not None:
            try:
                self._vertex_scatter.remove()
            except Exception:
                pass
            self._vertex_scatter = None

        for body in self.bodies:
            self._add_patch(body)

        self._draw_selected_vertices()
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def _add_patch(self, body: PolygonBody):
        if not body.is_complete():
            return
        verts = body.vertex_array()
        color = mcolors.to_rgba(body.color, alpha=0.45)
        patch = mpatches.Polygon(
            verts, closed=True,
            facecolor=color,
            edgecolor=body.color,
            linewidth=1.5,
            zorder=2,
            visible=body.visible,
            picker=True,
        )
        self.ax_model.add_patch(patch)
        self._body_patches[id(body)] = patch

    def _refresh_patch(self, body: PolygonBody):
        """Update (or create) the patch for one body."""
        bid = id(body)
        if bid in self._body_patches:
            self._body_patches[bid].remove()
            del self._body_patches[bid]
        self._add_patch(body)
        self._draw_selected_vertices()
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def _draw_selected_vertices(self):
        if self._vertex_scatter is not None:
            try:
                self._vertex_scatter.remove()
            except Exception:
                pass
            self._vertex_scatter = None

        if self.selected_body is None or not self.selected_body.vertices:
            return

        verts = self.selected_body.vertex_array()
        self._vertex_scatter = self.ax_model.scatter(
            verts[:, 0], verts[:, 1],
            s=VERTEX_RADIUS ** 2,
            c="white", edgecolors="black",
            linewidths=1.2, zorder=5,
        )

    # ── in-progress draw artists ──────────────────────────────────────────

    def _remove_draw_artists(self):
        for attr in ("_draw_line", "_draw_cursor_line"):
            artist = getattr(self, attr)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _update_draw_preview(self, cx=None, cz=None):
        """Redraw in-progress polygon preview."""
        self._remove_draw_artists()
        if not self._draw_verts:
            return
        vx = [v[0] for v in self._draw_verts]
        vz = [v[1] for v in self._draw_verts]

        self._draw_line, = self.ax_model.plot(
            vx + [vx[0]], vz + [vz[0]],
            "k--", lw=1.2, zorder=6)

        if cx is not None and len(self._draw_verts) >= 1:
            self._draw_cursor_line, = self.ax_model.plot(
                [vx[-1], cx], [vz[-1], cz],
                color="gray", lw=0.9, ls=":", zorder=6)

        self.fig.canvas.draw_idle()

    # ── mouse event handlers ──────────────────────────────────────────────

    def _on_press(self, event):
        # MASK mode: click in gravity axes to toggle station
        if self.mode == Mode.MASK:
            if event.inaxes is self.ax_grav and event.button == 1:
                if event.xdata is not None:
                    self._toggle_station_mask(event.xdata)
            return

        if event.inaxes is not self.ax_model:
            return
        x, z = event.xdata, event.ydata
        if x is None or z is None:
            return

        btn = event.button   # 1=left, 2=middle, 3=right

        if self.mode == Mode.DRAW:
            self._handle_draw_press(x, z, btn)

        elif self.mode == Mode.SELECT:
            self._handle_select_press(x, z)

        elif self.mode == Mode.ADD_VERTEX:
            self._handle_add_vertex_press(x, z)

        elif self.mode == Mode.DELETE:
            self._handle_delete_press(x, z)

    def _on_release(self, _):
        if self._drag_active:
            self._drag_active = False
            self._hide_snap_ring()
            self.fig.canvas.draw_idle()
            self.bodies_changed.emit()

    def _on_motion(self, event):
        if event.inaxes is not self.ax_model:
            self._hide_snap_ring()
            self.fig.canvas.draw_idle()
            return
        x, z = event.xdata, event.ydata
        if x is None or z is None:
            return
        self._cursor_pos = (x, z)

        if self.mode == Mode.DRAW:
            snap = self._find_snap_target(x, z)
            if snap:
                self._show_snap_ring(*snap)
                cx, cz = snap
            else:
                self._hide_snap_ring()
                cx, cz = x, z
            self._update_draw_preview(cx, cz)

        elif self.mode == Mode.SELECT and self._drag_active:
            snap = self._find_snap_target(
                x, z,
                exclude_body=self.selected_body,
                exclude_vi=self.selected_vertex,
            )
            if snap:
                self._show_snap_ring(*snap)
            else:
                self._hide_snap_ring()
            self._drag_vertex(x, z)

        else:
            self._hide_snap_ring()
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if self.mode == Mode.DRAW:
            if event.key in ("enter", "return"):
                self._close_polygon()
            elif event.key == "escape":
                self._draw_verts.clear()
                self._remove_draw_artists()
                self.fig.canvas.draw_idle()
        elif self.mode == Mode.SELECT:
            if event.key == "delete" and self.selected_body is not None:
                self._delete_selected_body()
            elif event.key == "escape":
                self._deselect()

    # ── draw mode ─────────────────────────────────────────────────────────

    def _handle_draw_press(self, x, z, btn):
        if btn == 1:          # left → add vertex
            snap = self._find_snap_target(x, z)
            if snap:
                x, z = snap
            self._draw_verts.append([x, z])
            self._update_draw_preview(x, z)
        elif btn == 3:        # right → close polygon
            self._close_polygon()

    def _close_polygon(self):
        if len(self._draw_verts) < 3:
            self._draw_verts.clear()
            self._remove_draw_artists()
            self.fig.canvas.draw_idle()
            return
        body = PolygonBody(vertices=[v[:] for v in self._draw_verts])
        self.bodies.append(body)
        self._draw_verts.clear()
        self._remove_draw_artists()
        self._add_patch(body)
        self.selected_body = body
        self._draw_selected_vertices()
        self._update_gravity()
        self.fig.canvas.draw_idle()
        self.bodies_changed.emit()

    # ── select mode ───────────────────────────────────────────────────────

    def _handle_select_press(self, x, z):
        hit_km = self._px_to_km(HIT_RADIUS_PX)
        # Check if we're near a vertex of any body (selected first, then others)
        candidates = (
            [self.selected_body] if self.selected_body is not None else []
        ) + [b for b in self.bodies if b is not self.selected_body]

        for body in candidates:
            if body is None:
                continue
            vi, vd = body.nearest_vertex(x, z)
            if vd < hit_km:
                if body is not self.selected_body:
                    self._deselect()
                    self.selected_body = body
                    self._draw_selected_vertices()
                self.selected_vertex = vi
                self._drag_active = True
                self._drag_prev = (x, z)
                return

        # Otherwise, pick a body by containment / edge proximity
        hit = self._pick_body(x, z)
        self._deselect()
        if hit is not None:
            self.selected_body = hit
            self._draw_selected_vertices()
            self.fig.canvas.draw_idle()
            self.bodies_changed.emit()

    def _drag_vertex(self, x, z):
        if self.selected_body is None or self.selected_vertex < 0:
            return

        # Snap to a nearby vertex on a *different* body / different vertex
        snap = self._find_snap_target(
            x, z,
            exclude_body=self.selected_body,
            exclude_vi=self.selected_vertex,
        )
        nx, nz = snap if snap else (x, z)

        # Collect all linked vertices (same location) and move them together
        linked = self._find_linked_vertices(self.selected_body,
                                            self.selected_vertex)
        new_pos = [nx, nz]
        self.selected_body.vertices[self.selected_vertex] = new_pos
        for b, i in linked:
            b.vertices[i] = new_pos[:]
            if b is not self.selected_body:
                self._refresh_patch(b)

        self._refresh_patch(self.selected_body)
        self._draw_selected_vertices()
        self.fig.canvas.draw_idle()

    def _deselect(self):
        self.selected_body   = None
        self.selected_vertex = -1
        self._drag_active    = False
        if self._vertex_scatter is not None:
            try:
                self._vertex_scatter.remove()
            except Exception:
                pass
            self._vertex_scatter = None
        self.fig.canvas.draw_idle()

    # ── add-vertex mode ───────────────────────────────────────────────────

    def _handle_add_vertex_press(self, x, z):
        hit = self._pick_body(x, z)
        if hit is None:
            return
        ei, _, px, pz = hit.nearest_edge_point(x, z)
        if ei < 0:
            return
        # Insert new vertex after edge start
        hit.vertices.insert(ei + 1, [px, pz])
        self.selected_body = hit
        self._refresh_patch(hit)
        self.bodies_changed.emit()

    # ── delete mode ───────────────────────────────────────────────────────

    def _handle_delete_press(self, x, z):
        hit = self._pick_body(x, z)
        if hit is not None:
            self.bodies.remove(hit)
            bid = id(hit)
            if bid in self._body_patches:
                self._body_patches[bid].remove()
                del self._body_patches[bid]
            if self.selected_body is hit:
                self._deselect()
            self._update_gravity()
            self.fig.canvas.draw_idle()
            self.bodies_changed.emit()

    def _delete_selected_body(self):
        if self.selected_body is not None:
            self._handle_delete_press(*self.selected_body.vertices[0])

    # ── hit testing ───────────────────────────────────────────────────────

    def _pick_body(self, x, z) -> Optional[PolygonBody]:
        """Return the topmost body containing (x, z), or None."""
        for body in reversed(self.bodies):
            if body.is_complete() and body.visible and body.contains_point(x, z):
                return body
        # Fallback: nearest body by edge distance within HIT_RADIUS_PX pixels
        hit_km = self._px_to_km(HIT_RADIUS_PX) * 2
        best, best_d = None, hit_km
        for body in self.bodies:
            if not body.is_complete() or not body.visible:
                continue
            _, _, px, pz = body.nearest_edge_point(x, z)
            d = math.hypot(px - x, pz - z)
            if d < best_d:
                best_d, best = d, body
        return best

    # ── serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "x_min":      self.x_min,
            "x_max":      self.x_max,
            "n_pts":      self.n_pts,
            "bg_density": self.bg_density,
            "bodies": [
                {"name": b.name, "density": b.density,
                 "color": b.color, "visible": b.visible,
                 "vertices": [v[:] for v in b.vertices]}
                for b in self.bodies
            ],
        }

    def from_dict(self, d: dict):
        self.x_min      = float(d.get("x_min",      -50.0))
        self.x_max      = float(d.get("x_max",       50.0))
        self.n_pts      = int(d.get("n_pts",          201))
        self.bg_density = float(d.get("bg_density", 2670.0))
        self.bodies.clear()
        PolygonBody._counter = 0
        for bd in d.get("bodies", []):
            b = PolygonBody(
                vertices=[list(v) for v in bd["vertices"]],
                density =float(bd["density"]),
                color   =bd.get("color", "#4C72B0"),
                name    =bd.get("name", "Body"),
                visible =bool(bd.get("visible", True)),
            )
            b.name = bd.get("name", b.name)
            self.bodies.append(b)
        self.ax_model.set_xlim(self.x_min, self.x_max)
        self.ax_grav.set_xlim(self.x_min, self.x_max)
        self._full_redraw()
        self.bodies_changed.emit()

    # ── external body management ──────────────────────────────────────────

    def add_body(self, body: PolygonBody):
        self.bodies.append(body)
        self._full_redraw()
        self.bodies_changed.emit()

    def remove_body(self, body: PolygonBody):
        if body in self.bodies:
            self.bodies.remove(body)
            self._full_redraw()
            self.bodies_changed.emit()

    def update_body_display(self, body: PolygonBody):
        """Call after external changes to body properties (density, color…)."""
        self._refresh_patch(body)
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def select_body_external(self, body: Optional[PolygonBody]):
        self.selected_body   = body
        self.selected_vertex = -1
        self._draw_selected_vertices()
        self.fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
#  Controls dock
# ─────────────────────────────────────────────────────────────────────────────

class ControlsDock(QDockWidget):
    mode_changed      = pyqtSignal(Mode)
    profile_changed   = pyqtSignal(float, float, int)
    depth_changed     = pyqtSignal(float)
    units_changed     = pyqtSignal(bool)    # True = km, False = m
    bg_density_changed = pyqtSignal(float)  # kg/m³

    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                             Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        w = QWidget()
        self.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._use_km = True    # current unit state

        # ── mode buttons ──────────────────────────────────────────────────
        mode_group = QGroupBox("Editing Mode")
        mode_layout = QVBoxLayout(mode_group)
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        modes = [
            ("Draw Polygon",  Mode.DRAW,       "LMB: add vertex\nRMB / Enter: close"),
            ("Select / Move", Mode.SELECT,      "LMB: select body\nDrag vertex to move"),
            ("Add Vertex",    Mode.ADD_VERTEX,  "LMB: insert vertex\non an edge"),
            ("Delete Body",   Mode.DELETE,      "LMB: remove body"),
            ("Mask Stations", Mode.MASK,        "LMB in gravity panel: toggle station mask"),
        ]
        self._mode_buttons = {}
        for label, mode, tip in modes:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setMinimumHeight(30)
            self._btn_group.addButton(btn)
            self._mode_buttons[mode] = btn
            mode_layout.addWidget(btn)
            btn.clicked.connect(lambda _, m=mode: self.mode_changed.emit(m))

        self._mode_buttons[Mode.SELECT].setChecked(True)
        layout.addWidget(mode_group)

        # ── distance units ────────────────────────────────────────────────
        units_group = QGroupBox("Distance Units")
        units_layout = QHBoxLayout(units_group)
        units_layout.addWidget(QLabel("km"))
        self._unit_toggle = ToggleSwitch(height=20)   # off=km, on=m
        self._unit_toggle.stateChanged.connect(self._on_unit_toggled)
        units_layout.addWidget(self._unit_toggle)
        units_layout.addWidget(QLabel("m"))
        units_layout.addStretch()
        layout.addWidget(units_group)

        # ── background density ────────────────────────────────────────────
        bg_group = QGroupBox("Background Density")
        bg_form  = QFormLayout(bg_group)
        self.spin_bg_density = QDoubleSpinBox()
        self.spin_bg_density.setRange(0.0, 1e5)
        self.spin_bg_density.setValue(2670.0)
        self.spin_bg_density.setSuffix(" kg/m³")
        self.spin_bg_density.setSingleStep(10.0)
        self.spin_bg_density.setDecimals(1)
        self.spin_bg_density.setToolTip(
            "Reference density.  Forward model uses: contrast = body density − background")
        self.spin_bg_density.valueChanged.connect(
            lambda v: self.bg_density_changed.emit(v))
        bg_form.addRow("ρ background:", self.spin_bg_density)
        layout.addWidget(bg_group)

        # ── profile ───────────────────────────────────────────────────────
        profile_group = QGroupBox("Gravity Profile")
        pform = QFormLayout(profile_group)

        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setRange(-1e6, 0)
        self.spin_xmin.setValue(-50.0)
        self.spin_xmin.setSuffix(" km")
        self.spin_xmin.setSingleStep(10.0)
        self.spin_xmin.setDecimals(1)

        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setRange(0, 1e6)
        self.spin_xmax.setValue(50.0)
        self.spin_xmax.setSuffix(" km")
        self.spin_xmax.setSingleStep(10.0)
        self.spin_xmax.setDecimals(1)

        self.spin_npts = QSpinBox()
        self.spin_npts.setRange(10, 2000)
        self.spin_npts.setValue(201)
        self.spin_npts.setSingleStep(10)

        pform.addRow("x min:", self.spin_xmin)
        pform.addRow("x max:", self.spin_xmax)
        pform.addRow("Points:", self.spin_npts)

        apply_btn = QPushButton("Apply Profile")
        apply_btn.clicked.connect(self._emit_profile)
        pform.addRow(apply_btn)

        layout.addWidget(profile_group)

        # ── depth range ───────────────────────────────────────────────────
        depth_group = QGroupBox("Model Depth")
        dform = QFormLayout(depth_group)
        self.spin_zmax = QDoubleSpinBox()
        self.spin_zmax.setRange(0.1, 1e6)
        self.spin_zmax.setValue(20.0)
        self.spin_zmax.setSuffix(" km")
        self.spin_zmax.setSingleStep(5.0)
        self.spin_zmax.setDecimals(1)
        self.spin_zmax.valueChanged.connect(
            lambda v: self.depth_changed.emit(v))
        dform.addRow("Max depth:", self.spin_zmax)
        layout.addWidget(depth_group)

        layout.addStretch()

    # ── unit toggle ───────────────────────────────────────────────────────

    def _on_unit_toggled(self, is_m: bool):
        """Fired when the ToggleSwitch changes.  is_m=True means metres."""
        use_km = not is_m
        if use_km == self._use_km:
            return
        factor = 1e-3 if use_km else 1e3   # current → new display unit
        self._use_km = use_km
        suffix = " km" if use_km else " m"
        step   = 10.0 if use_km else 10_000.0

        for spin in (self.spin_xmin, self.spin_xmax, self.spin_zmax):
            spin.blockSignals(True)
            spin.setValue(spin.value() * factor)
            spin.setSuffix(suffix)
            spin.setSingleStep(step)
            spin.blockSignals(False)

        self.units_changed.emit(use_km)

    def _emit_profile(self):
        self.profile_changed.emit(
            self.spin_xmin.value(),
            self.spin_xmax.value(),
            self.spin_npts.value(),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Polygon table dock
# ─────────────────────────────────────────────────────────────────────────────

COL_VIS      = 0
COL_COLOR    = 1
COL_NAME     = 2
COL_DENSITY  = 3
COL_CONTRAST = 4   # read-only: body.density − bg_density
COL_NVERTS   = 5
N_COLS       = 6

VCOL_IDX = 0
VCOL_X   = 1
VCOL_Z   = 2


class PolygonTableDock(QDockWidget):
    """
    Bottom dock with two side-by-side tables:
      Left  -- one row per PolygonBody (Vis | Color | Name | Density | Contrast | #Verts)
      Right -- vertices of the selected body (#, X, Z; editable)
    """

    body_selected       = pyqtSignal(object)   # PolygonBody or None
    body_changed        = pyqtSignal(object)   # PolygonBody that changed
    body_vertex_changed = pyqtSignal(object)   # PolygonBody whose vertices changed

    def __init__(self, parent=None):
        super().__init__("Polygon Bodies", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea |
                             Qt.DockWidgetArea.TopDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── left: polygon table ───────────────────────────────────────────
        left_w  = QWidget()
        left_vb = QVBoxLayout(left_w)
        left_vb.setContentsMargins(0, 0, 0, 0)
        left_vb.addWidget(QLabel("Polygon Bodies"))

        self.table = QTableWidget(0, N_COLS)
        self.table.setHorizontalHeaderLabels(
            ["Vis", "Color", "Name", "Density\n(kg/m³)", "Contrast\n(kg/m³)", "Verts"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_NAME,     QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(COL_DENSITY,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_CONTRAST, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_NVERTS,   QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_VIS,      QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_COLOR,    QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(120)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        left_vb.addWidget(self.table)
        splitter.addWidget(left_w)

        # ── right: vertex table ───────────────────────────────────────────
        right_w  = QWidget()
        right_vb = QVBoxLayout(right_w)
        right_vb.setContentsMargins(0, 0, 0, 0)
        self._vert_label = QLabel("Vertices")
        right_vb.addWidget(self._vert_label)

        self.vert_table = QTableWidget(0, 3)
        self.vert_table.setHorizontalHeaderLabels(["#", "X (km)", "Z (km)"])
        vh = self.vert_table.horizontalHeader()
        vh.setSectionResizeMode(VCOL_IDX, QHeaderView.ResizeMode.ResizeToContents)
        vh.setSectionResizeMode(VCOL_X,   QHeaderView.ResizeMode.Stretch)
        vh.setSectionResizeMode(VCOL_Z,   QHeaderView.ResizeMode.Stretch)
        self.vert_table.setMinimumHeight(120)
        self.vert_table.setAlternatingRowColors(True)
        self.vert_table.itemChanged.connect(self._on_vert_item_changed)
        right_vb.addWidget(self.vert_table)
        splitter.addWidget(right_w)

        splitter.setSizes([600, 300])
        outer.addWidget(splitter)
        self.setWidget(container)

        self._bodies:  List[PolygonBody]   = []
        self._sel_body: Optional[PolygonBody] = None
        self._updating = False      # guard against recursive signals
        self._bg_density: float = 2670.0
        self._use_km:     bool  = True

    # ── synchronise with canvas ───────────────────────────────────────────

    def sync(self, bodies: List[PolygonBody]):
        """Rebuild table from current bodies list."""
        self._updating = True
        self._bodies = bodies
        self.table.setRowCount(len(bodies))
        for row, body in enumerate(bodies):
            self._populate_row(row, body)
        self._updating = False

    def _populate_row(self, row: int, body: PolygonBody):
        # Visible checkbox
        vis_item = QTableWidgetItem()
        vis_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable |
                          Qt.ItemFlag.ItemIsEnabled)
        vis_item.setCheckState(
            Qt.CheckState.Checked if body.visible else Qt.CheckState.Unchecked)
        self.table.setItem(row, COL_VIS, vis_item)

        # Color button
        color_btn = QPushButton()
        color_btn.setFixedSize(28, 22)
        self._set_btn_color(color_btn, body.color)
        color_btn.clicked.connect(lambda _, b=body, btn=color_btn:
                                  self._pick_color(b, btn))
        self.table.setCellWidget(row, COL_COLOR, color_btn)

        # Name
        name_item = QTableWidgetItem(body.name)
        self.table.setItem(row, COL_NAME, name_item)

        # Density (editable)
        dens_item = QTableWidgetItem(f"{body.density:.1f}")
        dens_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_DENSITY, dens_item)

        # Contrast = density − bg_density  (read-only)
        contrast = body.density - self._bg_density
        cont_item = QTableWidgetItem(f"{contrast:.1f}")
        cont_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        cont_item.setFlags(Qt.ItemFlag.ItemIsEnabled |
                           Qt.ItemFlag.ItemIsSelectable)
        self.table.setItem(row, COL_CONTRAST, cont_item)

        # #Vertices (read-only)
        nv_item = QTableWidgetItem(str(len(body.vertices)))
        nv_item.setFlags(Qt.ItemFlag.ItemIsEnabled |
                         Qt.ItemFlag.ItemIsSelectable)
        self.table.setItem(row, COL_NVERTS, nv_item)

    def update_row(self, body: PolygonBody):
        """Refresh a single row after external changes."""
        self._updating = True
        try:
            row = self._bodies.index(body)
        except ValueError:
            self._updating = False
            return
        nv = self.table.item(row, COL_NVERTS)
        if nv:
            nv.setText(str(len(body.vertices)))
        ct = self.table.item(row, COL_CONTRAST)
        if ct:
            ct.setText(f"{body.density - self._bg_density:.1f}")
        self._updating = False
        # Refresh vertex table if this is the selected body
        if body is self._sel_body:
            self._populate_vert_table(body)

    def set_bg_density(self, bg: float):
        """Update background density and refresh all contrast cells."""
        self._bg_density = bg
        self._updating = True
        for row, body in enumerate(self._bodies):
            ct = self.table.item(row, COL_CONTRAST)
            if ct:
                ct.setText(f"{body.density - bg:.1f}")
        self._updating = False

    def set_units(self, use_km: bool):
        """Update vertex table header when units change."""
        self._use_km = use_km
        unit = "km" if use_km else "m"
        self.vert_table.setHorizontalHeaderLabels(
            ["#", f"X ({unit})", f"Z ({unit})"])
        if self._sel_body is not None:
            self._populate_vert_table(self._sel_body)

    # ── vertex table ──────────────────────────────────────────────────────

    def show_vertices(self, body: Optional[PolygonBody]):
        """Show vertices of *body* in the right-hand table."""
        self._sel_body = body
        self._populate_vert_table(body)

    def _populate_vert_table(self, body: Optional[PolygonBody]):
        self._updating = True
        self.vert_table.setRowCount(0)
        if body is None or not body.vertices:
            self._vert_label.setText("Vertices")
            self._updating = False
            return

        self._vert_label.setText(f"Vertices -- {body.name}")
        scale = 1.0 if self._use_km else 1000.0
        self.vert_table.setRowCount(len(body.vertices))
        for i, (vx, vz) in enumerate(body.vertices):
            idx_item = QTableWidgetItem(str(i))
            idx_item.setFlags(Qt.ItemFlag.ItemIsEnabled |
                              Qt.ItemFlag.ItemIsSelectable)
            self.vert_table.setItem(i, VCOL_IDX, idx_item)

            x_item = QTableWidgetItem(f"{vx * scale:.4f}")
            x_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.vert_table.setItem(i, VCOL_X, x_item)

            z_item = QTableWidgetItem(f"{vz * scale:.4f}")
            z_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.vert_table.setItem(i, VCOL_Z, z_item)

        self._updating = False

    def _on_vert_item_changed(self, item: QTableWidgetItem):
        if self._updating or self._sel_body is None:
            return
        row = item.row()
        col = item.column()
        if col not in (VCOL_X, VCOL_Z):
            return
        if row < 0 or row >= len(self._sel_body.vertices):
            return
        scale = 1.0 if self._use_km else 1e-3   # display → km
        try:
            val_km = float(item.text()) * scale
        except ValueError:
            # Restore old value
            self._populate_vert_table(self._sel_body)
            return
        coord = 0 if col == VCOL_X else 1
        self._sel_body.vertices[row][coord] = val_km
        self.body_vertex_changed.emit(self._sel_body)

    def _set_btn_color(self, btn: QPushButton, color: str):
        c = QColor(color)
        btn.setStyleSheet(
            f"background-color: {c.name()}; border: 1px solid #555;")

    # ── color picker ──────────────────────────────────────────────────────

    def _pick_color(self, body: PolygonBody, btn: QPushButton):
        c = QColorDialog.getColor(QColor(body.color), self, "Choose color")
        if c.isValid():
            body.color = c.name()
            self._set_btn_color(btn, body.color)
            self.body_changed.emit(body)

    # ── table item changed ────────────────────────────────────────────────

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._updating:
            return
        row = item.row()
        if row < 0 or row >= len(self._bodies):
            return
        body = self._bodies[row]
        col  = item.column()

        if col == COL_VIS:
            body.visible = (item.checkState() == Qt.CheckState.Checked)
            self.body_changed.emit(body)

        elif col == COL_NAME:
            body.name = item.text().strip() or body.name

        elif col == COL_DENSITY:
            try:
                body.density = float(item.text())
                # update contrast cell
                self._updating = True
                ct = self.table.item(row, COL_CONTRAST)
                if ct:
                    ct.setText(f"{body.density - self._bg_density:.1f}")
                self._updating = False
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.density:.1f}")
                self._updating = False

    # ── row selection ─────────────────────────────────────────────────────

    def _on_selection_changed(self):
        if self._updating:
            return
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        if rows:
            row = min(rows)
            if 0 <= row < len(self._bodies):
                body = self._bodies[row]
                self._sel_body = body
                self._populate_vert_table(body)
                self.body_selected.emit(body)
                return
        self._sel_body = None
        self._populate_vert_table(None)
        self.body_selected.emit(None)

    def select_body(self, body: Optional[PolygonBody]):
        self._updating = True
        self.table.clearSelection()
        if body is not None and body in self._bodies:
            row = self._bodies.index(body)
            self.table.selectRow(row)
        self._updating = False


# ─────────────────────────────────────────────────────────────────────────────
#  Inversion dock
# ─────────────────────────────────────────────────────────────────────────────

class InversionDock(QDockWidget):
    """
    Right-side dock for controlling the Gauss-Newton / LM inversion.

    Signals
    -------
    run_requested(InversionConfig)  -- user clicked Run
    stop_requested()                -- user clicked Stop
    """

    run_requested    = pyqtSignal(object)   # carries InversionConfig
    stop_requested   = pyqtSignal()
    revert_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Inversion", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                             Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)

        w = QWidget()
        self.setWidget(w)
        vb = QVBoxLayout(w)
        vb.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── what to invert ────────────────────────────────────────────────
        inv_group = QGroupBox("Invert")
        inv_vb    = QVBoxLayout(inv_group)
        self.chk_vertices  = QCheckBox("Vertex positions")
        self.chk_vertices.setChecked(True)
        self.chk_densities = QCheckBox("Density contrasts")
        self.chk_densities.setChecked(True)
        self.chk_weights   = QCheckBox("Use data weights (1/σ²)")
        self.chk_weights.setChecked(False)
        self.chk_weights.setToolTip(
            "Weight residuals by 1/σ² when uncertainty column is loaded")
        inv_vb.addWidget(self.chk_vertices)
        inv_vb.addWidget(self.chk_densities)
        inv_vb.addWidget(self.chk_weights)
        vb.addWidget(inv_group)

        # ── solver parameters ─────────────────────────────────────────────
        param_group = QGroupBox("Solver Parameters")
        pform       = QFormLayout(param_group)

        self.spin_damping = QDoubleSpinBox()
        self.spin_damping.setRange(1e-8, 1e4)
        self.spin_damping.setValue(1e-2)
        self.spin_damping.setDecimals(6)
        self.spin_damping.setSingleStep(0.001)
        self.spin_damping.setToolTip(
            "Tikhonov/LM damping λ  (larger = more conservative steps)")
        pform.addRow("Damping λ:", self.spin_damping)

        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(1, 500)
        self.spin_max_iter.setValue(20)
        pform.addRow("Max iterations:", self.spin_max_iter)

        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(1e-10, 1.0)
        self.spin_tol.setValue(1e-4)
        self.spin_tol.setDecimals(8)
        self.spin_tol.setSingleStep(1e-5)
        self.spin_tol.setToolTip("Convergence tolerance on ||Δp||")
        pform.addRow("Tolerance:", self.spin_tol)

        vb.addWidget(param_group)

        # ── run / stop buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_run  = QPushButton("Run")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        vb.addLayout(btn_row)

        # ── revert button ─────────────────────────────────────────────────
        self.btn_revert = QPushButton("Revert to Starting Model")
        self.btn_revert.setEnabled(False)
        self.btn_revert.setToolTip(
            "Restore the model to the state it was in before inversion was run")
        self.btn_revert.clicked.connect(self.revert_requested.emit)
        vb.addWidget(self.btn_revert)

        # ── progress label ────────────────────────────────────────────────
        self.lbl_progress = QLabel("Ready")
        self.lbl_progress.setWordWrap(True)
        vb.addWidget(self.lbl_progress)

        vb.addStretch()

    def _on_run(self):
        from src.inversion.gravity_inversion import InversionConfig
        cfg = InversionConfig(
            invert_vertices  = self.chk_vertices.isChecked(),
            invert_densities = self.chk_densities.isChecked(),
            use_weights      = self.chk_weights.isChecked(),
            damping          = self.spin_damping.value(),
            max_iter         = self.spin_max_iter.value(),
            tol              = self.spin_tol.value(),
        )
        self.run_requested.emit(cfg)

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def update_progress(self, it: int, rms: float):
        self.lbl_progress.setText(
            f"Iter: {it} / {self.spin_max_iter.value()}  |  RMS: {rms:.4f} mGal")

    def show_converged(self, rms: float):
        self.lbl_progress.setText(f"Converged  |  RMS: {rms:.4f} mGal")
        self.set_running(False)
        self.btn_revert.setEnabled(True)

    def show_stopped(self):
        self.lbl_progress.setText("Stopped by user")
        self.set_running(False)
        self.btn_revert.setEnabled(True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2-D Gravity Modeller")
        self.resize(1200, 780)

        # ── central canvas ────────────────────────────────────────────────
        self.canvas = GravityCanvas(self)
        nav_toolbar = NavigationToolbar2QT(self.canvas, self)

        central = QWidget()
        cv_layout = QVBoxLayout(central)
        cv_layout.setContentsMargins(0, 0, 0, 0)
        cv_layout.addWidget(nav_toolbar)
        cv_layout.addWidget(self.canvas)
        self.setCentralWidget(central)

        # ── controls dock (left) ──────────────────────────────────────────
        self.controls_dock = ControlsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self.controls_dock)
        self.controls_dock.mode_changed.connect(self._on_mode_changed)
        self.controls_dock.profile_changed.connect(
            lambda xn, xx, np_: self.canvas.set_profile(xn, xx, np_))
        self.controls_dock.depth_changed.connect(self.canvas.set_depth_range)
        self.controls_dock.units_changed.connect(self._on_units_changed)
        self.controls_dock.bg_density_changed.connect(self._on_bg_density_changed)

        # ── polygon table dock (bottom) ───────────────────────────────────
        self.table_dock = PolygonTableDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,
                           self.table_dock)
        self.table_dock.setMinimumHeight(160)
        self.table_dock.body_selected.connect(self._on_table_body_selected)
        self.table_dock.body_changed.connect(self._on_table_body_changed)
        self.table_dock.body_vertex_changed.connect(self._on_vertex_edited)

        # canvas → table synchronisation
        self.canvas.bodies_changed.connect(self._sync_table)

        # ── inversion dock (right) ────────────────────────────────────────
        self.inv_dock = InversionDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self.inv_dock)
        self.inv_dock.run_requested.connect(self._run_inversion)
        self.inv_dock.stop_requested.connect(self._stop_inversion)
        self.inv_dock.revert_requested.connect(self._revert_inversion)
        self._inv_worker    = None   # active InversionWorker or None
        self._inv_snapshot  = None   # body dicts saved before each run

        # ── status bar ────────────────────────────────────────────────────
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._rms_label = QLabel("RMS: --")
        self.status.addPermanentWidget(self._rms_label)
        self.canvas.rms_updated.connect(
            lambda rms: self._rms_label.setText(f"RMS: {rms:.3f} mGal"))
        self.status.showMessage(
            "Select mode: click body to select | Draw mode: LMB add vertex, RMB close")

        # ── menu ──────────────────────────────────────────────────────────
        self._build_menu()

        # ── keyboard shortcuts ────────────────────────────────────────────
        for key, mode in [("D", Mode.DRAW), ("S", Mode.SELECT),
                          ("A", Mode.ADD_VERTEX), ("X", Mode.DELETE)]:
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _, m=mode: self._on_mode_changed(m))
            self.addAction(act)

        self._sync_table()

    # ── menu ──────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")

        act_new = QAction("&New Model", self)
        act_new.setShortcut(QKeySequence("Ctrl+N"))
        act_new.triggered.connect(self._new_model)
        file_menu.addAction(act_new)

        file_menu.addSeparator()

        act_open_data = QAction("&Open Data…", self)
        act_open_data.setShortcut(QKeySequence("Ctrl+O"))
        act_open_data.triggered.connect(self._open_data)
        file_menu.addAction(act_open_data)

        file_menu.addSeparator()

        act_save = QAction("&Save Model…", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self._save_model)
        file_menu.addAction(act_save)

        act_load = QAction("&Load Model…", self)
        act_load.setShortcut(QKeySequence("Ctrl+L"))
        act_load.triggered.connect(self._load_model)
        file_menu.addAction(act_load)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Edit
        edit_menu = mb.addMenu("&Edit")
        act_del = QAction("&Delete Selected Body", self)
        act_del.setShortcut(QKeySequence("Delete"))
        act_del.triggered.connect(self._delete_selected)
        edit_menu.addAction(act_del)

        # Add demo body
        demo_menu = mb.addMenu("&Demo")
        act_demo = QAction("Add Rectangular Body", self)
        act_demo.triggered.connect(self._add_demo_body)
        demo_menu.addAction(act_demo)

        # Inversion
        inv_menu = mb.addMenu("&Inversion")
        act_inv_run = QAction("&Run Inversion", self)
        act_inv_run.setShortcut(QKeySequence("Ctrl+I"))
        act_inv_run.triggered.connect(lambda: self.inv_dock.btn_run.click())
        inv_menu.addAction(act_inv_run)
        act_inv_stop = QAction("&Stop Inversion", self)
        act_inv_stop.triggered.connect(self._stop_inversion)
        inv_menu.addAction(act_inv_stop)

        # Help
        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ── menu actions ──────────────────────────────────────────────────────

    def _new_model(self):
        reply = QMessageBox.question(
            self, "New Model",
            "Clear all bodies and start a new model?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.canvas.bodies.clear()
            self.canvas._full_redraw()
            self._sync_table()

    def _delete_selected(self):
        body = self.canvas.selected_body
        if body is not None:
            self.canvas.remove_body(body)
            self._sync_table()

    def _add_demo_body(self):
        """Add a 4 × 5 km rectangular body as an example."""
        body = PolygonBody(
            vertices=[[-2, 2], [2, 2], [2, 7], [-2, 7]],
            density=300.0,
            name="Demo Body",
        )
        self.canvas.add_body(body)
        self._sync_table()

    def _show_about(self):
        QMessageBox.about(
            self, "About",
            "<b>2-D Gravity Modeller</b><br><br>"
            "Forward model: Talwani et al. (1959)<br>"
            "Interface: PyQt6 + Matplotlib<br><br>"
            "Keyboard shortcuts:<br>"
            "&nbsp;&nbsp;<b>D</b> -- Draw mode<br>"
            "&nbsp;&nbsp;<b>S</b> -- Select/Move mode<br>"
            "&nbsp;&nbsp;<b>A</b> -- Add Vertex mode<br>"
            "&nbsp;&nbsp;<b>X</b> -- Delete Body mode<br>"
            "&nbsp;&nbsp;<b>Enter</b> -- Close polygon (in draw mode)<br>"
            "&nbsp;&nbsp;<b>Esc</b> -- Cancel / deselect",
        )

    # ── open data / save / load ───────────────────────────────────────────

    def _open_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Gravity Data", "",
            "CSV files (*.csv);;All files (*)")
        if not path:
            return
        obs = load_csv_data(path, self)
        if obs is None:
            return
        self.canvas.load_observed(obs)
        n = len(obs.x)
        self.status.showMessage(
            f"Loaded {n} stations from {Path(path).name}  |  "
            f"x range: {obs.x.min():.1f}--{obs.x.max():.1f} km")

    def _save_model(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "",
            "JSON model (*.json);;All files (*)")
        if not path:
            return
        try:
            bg = self.controls_dock.spin_bg_density.value()
            save_model(path, self.canvas, bg, self.canvas.obs_data)
            self.status.showMessage(f"Model saved to {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "",
            "JSON model (*.json);;All files (*)")
        if not path:
            return
        try:
            data = load_model(path)
            bg, obs = apply_model(data, self.canvas)
            # Restore controls
            self.controls_dock.spin_bg_density.blockSignals(True)
            self.controls_dock.spin_bg_density.setValue(bg)
            self.controls_dock.spin_bg_density.blockSignals(False)
            self.canvas.bg_density = bg
            if obs is not None:
                self.canvas.load_observed(obs)
            self.canvas.ax_model.set_xlim(self.canvas.x_min, self.canvas.x_max)
            self.canvas.ax_grav.set_xlim(self.canvas.x_min, self.canvas.x_max)
            self.canvas._full_redraw()
            self._sync_table()
            self.table_dock.set_bg_density(bg)
            self.status.showMessage(f"Model loaded from {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    # ── mode change ───────────────────────────────────────────────────────

    def _on_mode_changed(self, mode: Mode):
        self.canvas.set_mode(mode)
        btn = self.controls_dock._mode_buttons.get(mode)
        if btn:
            btn.setChecked(True)
        labels = {
            Mode.DRAW:       "Draw: LMB add vertex | RMB / Enter close polygon | Esc cancel",
            Mode.SELECT:     "Select: LMB pick body | drag vertex | Del remove",
            Mode.ADD_VERTEX: "Add Vertex: LMB click on an edge to insert vertex",
            Mode.DELETE:     "Delete: LMB click on a body to remove it",
            Mode.MASK:       "Mask: LMB click on a station in the gravity panel to toggle mask",
        }
        self.status.showMessage(labels.get(mode, ""))

    # ── units / bg density ────────────────────────────────────────────────

    def _on_units_changed(self, use_km: bool):
        self.canvas.set_units(use_km)
        self.table_dock.set_units(use_km)

    def _on_bg_density_changed(self, bg: float):
        self.canvas.set_bg_density(bg)
        self.table_dock.set_bg_density(bg)

    # ── table ↔ canvas ────────────────────────────────────────────────────

    def _sync_table(self):
        self.table_dock.sync(self.canvas.bodies)
        if self.canvas.selected_body is not None:
            self.table_dock.select_body(self.canvas.selected_body)
            self.table_dock.show_vertices(self.canvas.selected_body)

    def _on_table_body_selected(self, body):
        self.canvas.select_body_external(body)
        self.table_dock.show_vertices(body)

    def _on_table_body_changed(self, body: PolygonBody):
        self.canvas.update_body_display(body)
        self.table_dock.update_row(body)

    def _on_vertex_edited(self, body: PolygonBody):
        """Vertex edited in the table — refresh patch and gravity."""
        self.canvas.update_body_display(body)
        self.table_dock.update_row(body)

    # ── inversion ─────────────────────────────────────────────────────────

    def _run_inversion(self, cfg):
        from src.inversion.gravity_inversion import InversionWorker
        if self.canvas.obs_data is None:
            QMessageBox.warning(self, "No Data",
                                "Load observed gravity data before running inversion.")
            return
        if not self.canvas.bodies:
            QMessageBox.warning(self, "No Model",
                                "Draw at least one polygon body before running inversion.")
            return
        if self._inv_worker is not None and self._inv_worker.isRunning():
            return

        # Snapshot the current model so the user can revert after inversion
        self._inv_snapshot = [
            {"name": b.name, "density": b.density, "color": b.color,
             "visible": b.visible, "vertices": [v[:] for v in b.vertices]}
            for b in self.canvas.bodies
        ]
        self.inv_dock.btn_revert.setEnabled(False)

        self.inv_dock.set_running(True)
        self.inv_dock.lbl_progress.setText("Starting…")

        self._inv_worker = InversionWorker(
            self.canvas.bodies,
            self.canvas.bg_density,
            self.canvas.obs_data,
            cfg,
            parent=self,
        )
        self._inv_worker.iteration_done.connect(self._on_inv_iteration)
        self._inv_worker.converged.connect(self._on_inv_converged)
        self._inv_worker.stopped.connect(self.inv_dock.show_stopped)
        self._inv_worker.start()

    def _stop_inversion(self):
        if self._inv_worker is not None and self._inv_worker.isRunning():
            self._inv_worker.request_stop()

    def _on_inv_iteration(self, it: int, rms: float, body_dicts: list):
        """Apply body updates from the worker to the canvas (main thread)."""
        for body, bd in zip(self.canvas.bodies, body_dicts):
            body.density  = bd["density"]
            body.vertices = [v[:] for v in bd["vertices"]]
        self.canvas._full_redraw()
        self._sync_table()
        self.inv_dock.update_progress(it, rms)

    def _on_inv_converged(self, rms: float):
        self.canvas._full_redraw()
        self._sync_table()
        self.inv_dock.show_converged(rms)
        self.status.showMessage(f"Inversion converged  |  Final RMS: {rms:.4f} mGal")

    def _revert_inversion(self):
        """Restore the model to the snapshot taken before the last inversion run."""
        if self._inv_snapshot is None:
            return
        for body, bd in zip(self.canvas.bodies, self._inv_snapshot):
            body.density  = bd["density"]
            body.vertices = [v[:] for v in bd["vertices"]]
        self.canvas._full_redraw()
        self._sync_table()
        self.inv_dock.btn_revert.setEnabled(False)
        self.inv_dock.lbl_progress.setText("Reverted to starting model")
        self.status.showMessage("Model reverted to pre-inversion state")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
