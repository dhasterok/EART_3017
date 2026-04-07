#!/usr/bin/env python3
"""
gravmag2d_gui.py
----------------
Interactive 2-D gravity + magnetic modelling GUI.

  • Draw and edit polygonal density/susceptibility bodies with the mouse.
  • Talwani (1959) gravity and Blakely (1995) magnetic forward models update
    in real time.
  • Gravity (mGal) and total-field magnetic anomaly (nT) share the profile
    panel: gravity on the left y-axis, magnetics on the right y-axis.
  • Controls in a left QDockWidget; polygon table in a bottom QDockWidget.

Usage
-----
    python gravmag2d_gui.py
"""

import sys
import math
from enum import Enum, auto
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np

_HERE   = Path(__file__).resolve().parent          # src/gui/
_COURSE = _HERE.parent.parent                       # project root
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from src.utils.CustomWidgets import ToggleSwitch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QPushButton, QColorDialog, QButtonGroup,
    QGroupBox, QFormLayout, QHeaderView, QAbstractItemView,
    QSizePolicy, QStatusBar, QMessageBox, QSplitter, QFileDialog,
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QAction, QKeySequence, QActionGroup, QIcon

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from src.gravity.talwani_model import compute_gz
from src.gravity.mag2d_model   import compute_bt, compute_bx_bz
from src.gravity.data_loader   import load_csv_data, ObservedData


# ─────────────────────────────────────────────────────────────────────────────
#  Application constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

VERTEX_RADIUS    = 6
HIT_RADIUS_PX    = 10
SNAP_RADIUS_PX   = 12
SNAP_INDICATOR_PX = 14
_LINK_EPS        = 1e-9

# ★ Colours for the two profile curves
_GRAV_COLOR = "navy"
_MAG_COLOR  = "#8B0000"   # dark red


# ─────────────────────────────────────────────────────────────────────────────
#  Interaction mode
# ─────────────────────────────────────────────────────────────────────────────

class DisplayMode(Enum):
    BOTH      = "both"
    GRAVITY   = "gravity"
    MAGNETICS = "magnetics"


class MagComponent(Enum):
    TMI = "Total field (ΔT)"
    BX  = "Horizontal (Bx)"
    BZ  = "Vertical (Bz)"


class Mode(Enum):
    DRAW       = auto()
    SELECT     = auto()
    ADD_VERTEX = auto()
    DELETE     = auto()
    MASK       = auto()


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

class PolygonBody:
    """A single 2-D density/susceptibility body defined by a closed polygon."""

    _counter = 0

    def __init__(self, vertices=None, density=300.0,
                 susceptibility=0.001,
                 remanence_Am=0.0, remanence_inc_deg=0.0,
                 color=None, name=None, visible=True):
        PolygonBody._counter += 1
        self.name               = name or f"Body {PolygonBody._counter}"
        self.vertices           = vertices or []
        self.density            = float(density)
        self.susceptibility     = float(susceptibility)
        self.remanence_Am       = float(remanence_Am)       # A/m
        self.remanence_inc_deg  = float(remanence_inc_deg)  # degrees
        self.color              = color or DEFAULT_COLORS[
            (PolygonBody._counter - 1) % len(DEFAULT_COLORS)]
        self.visible            = visible

    def is_complete(self):
        return len(self.vertices) >= 3

    def clone(self):
        b = PolygonBody.__new__(PolygonBody)
        b.name              = self.name
        b.vertices          = [v[:] for v in self.vertices]
        b.density           = self.density
        b.susceptibility    = self.susceptibility
        b.remanence_Am      = self.remanence_Am
        b.remanence_inc_deg = self.remanence_inc_deg
        b.color             = self.color
        b.visible           = self.visible
        return b

    def vertex_array(self):
        return np.array(self.vertices, dtype=float) if self.vertices else np.empty((0, 2))

    def contains_point(self, x, z):
        verts = self.vertices
        inside = False
        xj, zj = verts[-1]
        for xi, zi in verts:
            if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
                inside = not inside
            xj, zj = xi, zi
        return inside

    def nearest_vertex(self, x, z) -> Tuple[int, float]:
        best_i, best_d = -1, 1e9
        for i, (vx, vz) in enumerate(self.vertices):
            d = math.hypot(vx - x, vz - z)
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d

    def nearest_edge_point(self, x, z) -> Tuple[int, float, float, float]:
        best_i, best_d, best_t, best_px, best_pz = -1, 1e9, 0, x, z
        n = len(self.vertices)
        for i in range(n):
            ax, az = self.vertices[i]
            bx, bz = self.vertices[(i + 1) % n]
            dx, dz = bx - ax, bz - az
            len2 = dx*dx + dz*dz
            if len2 < 1e-12:
                continue
            t = max(0.0, min(1.0, ((x - ax)*dx + (z - az)*dz) / len2))
            px = ax + t*dx
            pz = az + t*dz
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
      • Top  (ax_grav / ax_mag)  -- gravity profile (left axis, mGal)
                                    + total-field magnetic profile (right axis, nT)
      • Bottom (ax_model)        -- model cross-section; polygon editing
    """

    bodies_changed    = pyqtSignal()
    rms_updated       = pyqtSignal(float)
    delete_key_pressed = pyqtSignal()   # Delete key on selected body — main window confirms

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 8), tight_layout=False)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

        self.bodies: List[PolygonBody] = []
        self.mode   = Mode.SELECT

        self.x_min      = -50.0
        self.x_max      =  50.0
        self.n_pts      =  201
        self.use_km     = True
        self.bg_density = 2670.0
        self.snap_enabled = True     # snapping on by default

        # ★ Earth-field parameters
        self.earth_field_nT = 50000.0   # nT
        self.earth_inc_deg  =  60.0     # degrees

        self.display_mode  = DisplayMode.BOTH
        self.mag_component = MagComponent.TMI

        gs = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[1, 2],
            hspace=0.08,
        )
        self.ax_grav  = self.fig.add_subplot(gs[0])
        self.ax_model = self.fig.add_subplot(gs[1], sharex=self.ax_grav)
        self.ax_mag   = self.ax_grav.twinx()   # ★ right axis for magnetics

        self._setup_axes()

        self.selected_body:   Optional[PolygonBody] = None
        self.selected_vertex: int = -1
        self._drag_active     = False
        self._drag_prev       = (None, None)

        self._draw_verts: List[List[float]] = []
        self._draw_line        = None
        self._draw_cursor_line = None
        self._cursor_pos = (0.0, 0.0)

        self._body_patches: dict = {}
        self._vertex_scatter = None
        self._grav_line  = None
        self._mag_line   = None                # ★
        self._obs_line   = None
        self._obs_errbar = None

        self._snap_ring = None

        self.obs_data: Optional[ObservedData] = None

        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("key_press_event",      self._on_key)

        self._full_redraw()

    # ── axes appearance ───────────────────────────────────────────────────

    def _setup_axes(self):
        # Left gravity axis
        self.ax_grav.set_ylabel("Gravity anomaly (mGal)", color=_GRAV_COLOR)
        self.ax_grav.tick_params(axis='y', labelcolor=_GRAV_COLOR)
        self.ax_grav.axhline(0, color="gray", lw=0.6, ls="--")
        self.ax_grav.grid(True, alpha=0.3)
        self.ax_grav.tick_params(labelbottom=False)

        # ★ Right magnetic axis
        self.ax_mag.set_ylabel("Total-field anomaly (nT)", color=_MAG_COLOR)
        self.ax_mag.tick_params(axis='y', labelcolor=_MAG_COLOR)
        self.ax_mag.tick_params(labelbottom=False)

        # Model panel
        self.ax_model.set_xlabel("Distance (km)")
        self.ax_model.set_ylabel("Depth (km)")
        self.ax_model.invert_yaxis()
        self.ax_model.set_xlim(self.x_min, self.x_max)
        self.ax_model.set_ylim(20, -0.5)
        self.ax_model.axhline(0, color="gray", lw=1.0, ls="-")
        self.ax_model.grid(True, alpha=0.2)

        self.fig.canvas.draw_idle()

    def _apply_unit_formatters(self):
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

    # ── public profile / unit / earth-field controls ──────────────────────

    def set_units(self, use_km: bool):
        self.use_km = use_km
        self._apply_unit_formatters()
        self.fig.canvas.draw_idle()

    def set_profile(self, x_min, x_max, n_pts):
        scale = 1.0 if self.use_km else 1e-3
        self.x_min = x_min * scale
        self.x_max = x_max * scale
        self.n_pts = max(3, int(n_pts))
        self.ax_model.set_xlim(self.x_min, self.x_max)
        self.ax_grav.set_xlim(self.x_min, self.x_max)
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def set_depth_range(self, z_max):
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

    def set_bg_density(self, bg: float):
        self.bg_density = bg
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def set_earth_field(self, F_nT: float, IE_deg: float):   # ★
        """Update Earth-field parameters and recompute magnetic anomaly."""
        self.earth_field_nT = F_nT
        self.earth_inc_deg  = IE_deg
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def load_observed(self, obs_data: ObservedData):
        self.obs_data = obs_data
        self._update_gravity()
        self.fig.canvas.draw_idle()

    # ── pixel ↔ km conversion ─────────────────────────────────────────────

    def _px_to_km(self, n_px: float) -> float:
        inv = self.ax_model.transData.inverted()
        p0 = inv.transform((0.0, 0.0))
        p1 = inv.transform((n_px, 0.0))
        return abs(float(p1[0]) - float(p0[0]))

    # ── snapping helpers ──────────────────────────────────────────────────

    def set_snap_enabled(self, enabled: bool):
        self.snap_enabled = enabled
        self._hide_snap_ring()
        self.fig.canvas.draw_idle()

    def _find_snap_target(self, x, z, exclude_body=None,
                          exclude_vi=-1) -> Optional[Tuple[float, float]]:
        if not self.snap_enabled:
            return None
        snap_km = self._px_to_km(SNAP_RADIUS_PX)
        best_d, best_pos = snap_km, None
        for body in self.bodies:
            for vi, (vx, vz) in enumerate(body.vertices):
                if body is exclude_body and vi == exclude_vi:
                    continue
                d = math.hypot(vx - x, vz - z)
                if d < best_d:
                    best_d, best_pos = d, (vx, vz)
        for vi, (vx, vz) in enumerate(self._draw_verts[:-1]
                                       if self._draw_verts else []):
            d = math.hypot(vx - x, vz - z)
            if d < best_d:
                best_d, best_pos = d, (vx, vz)
        return best_pos

    def _find_linked_vertices(self, body, vi):
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

    def _show_snap_ring(self, sx, sz):
        r = self._px_to_km(SNAP_INDICATOR_PX)
        if self._snap_ring is not None:
            try:
                self._snap_ring.remove()
            except Exception:
                pass
        self._snap_ring = mpatches.Circle(
            (sx, sz), r, facecolor="none", edgecolor="#FFD700",
            linewidth=2.0, zorder=10)
        self.ax_model.add_patch(self._snap_ring)

    def _hide_snap_ring(self):
        if self._snap_ring is not None:
            try:
                self._snap_ring.remove()
            except Exception:
                pass
            self._snap_ring = None

    # ── display mode ──────────────────────────────────────────────────────

    def set_display_mode(self, mode: DisplayMode):
        self.display_mode = mode
        show_grav = mode in (DisplayMode.BOTH, DisplayMode.GRAVITY)
        show_mag  = mode in (DisplayMode.BOTH, DisplayMode.MAGNETICS)
        self.ax_grav.set_visible(show_grav)
        self.ax_mag.set_visible(show_mag)
        self.ax_mag.set_ylabel(self.mag_component.value + " (nT)", color=_MAG_COLOR)
        # Swap which axis carries the x-tick labels (the visible top axis)
        self.ax_grav.tick_params(labelbottom=False)
        self.ax_mag.tick_params(labelbottom=False)
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def set_mag_component(self, component: MagComponent):
        self.mag_component = component
        self.ax_mag.set_ylabel(component.value + " (nT)", color=_MAG_COLOR)
        self._update_gravity()
        self.fig.canvas.draw_idle()

    # ── gravity + magnetic computation ────────────────────────────────────

    def _profile_x(self):
        return np.linspace(self.x_min, self.x_max, self.n_pts)

    def _update_gravity(self):
        xp = self._profile_x()

        # ── gravity ──────────────────────────────────────────────────────
        gz = np.zeros(len(xp))
        if self.display_mode in (DisplayMode.BOTH, DisplayMode.GRAVITY):
            for body in self.bodies:
                if body.visible and body.is_complete():
                    contrast = body.density - self.bg_density
                    gz += compute_gz(xp, body.vertices, contrast)

        if self._grav_line is not None:
            self._grav_line.remove()
        if self.display_mode in (DisplayMode.BOTH, DisplayMode.GRAVITY):
            self._grav_line, = self.ax_grav.plot(
                xp, gz, color=_GRAV_COLOR, lw=1.5, label="Gravity (calc.)", zorder=3)
        else:
            self._grav_line = None

        # ── observed data ────────────────────────────────────────────────
        for attr in ("_obs_line", "_obs_errbar"):
            artist = getattr(self, attr, None)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        if self.obs_data is not None:
            od   = self.obs_data
            mask = od.masked if od.masked is not None \
                   else np.zeros(len(od.x), dtype=bool)
            if self.mode == Mode.MASK:
                unmasked = ~mask
                if unmasked.any():
                    self.ax_grav.plot(od.x[unmasked], od.gz[unmasked],
                                      "ro", ms=4, zorder=5, label="Observed")
                if mask.any():
                    self.ax_grav.plot(od.x[mask], od.gz[mask],
                                      "o", color="grey", ms=4, zorder=4, label="Masked")
            else:
                show = ~mask
                if show.any():
                    x_s  = od.x[show]
                    gz_s = od.gz[show]
                    unc_s = od.gz_unc[show] if od.gz_unc is not None else None
                    if unc_s is not None:
                        self._obs_errbar = self.ax_grav.errorbar(
                            x_s, gz_s, yerr=unc_s, fmt="ro", ms=3,
                            elinewidth=0.8, capsize=2, label="Observed", zorder=4)
                    else:
                        self._obs_line, = self.ax_grav.plot(
                            x_s, gz_s, "ro", ms=3, label="Observed", zorder=4)

        # RMS misfit
        if self.obs_data is not None:
            od   = self.obs_data
            umsk = (~od.masked) if od.masked is not None \
                   else np.ones(len(od.x), dtype=bool)
            if umsk.any():
                gz_interp = np.interp(od.x[umsk], xp, gz)
                rms = float(np.sqrt(np.mean((od.gz[umsk] - gz_interp)**2)))
                self.rms_updated.emit(rms)

        self.ax_grav.relim()
        self.ax_grav.autoscale_view(scalex=False, scaley=True)

        # ── ★ magnetic ───────────────────────────────────────────────────
        if self.display_mode in (DisplayMode.BOTH, DisplayMode.MAGNETICS):
            self._update_magnetic(xp)
        elif self._mag_line is not None:
            try:
                self._mag_line.remove()
            except Exception:
                pass
            self._mag_line = None

        # ── combined legend ──────────────────────────────────────────────
        lines, labs = [], []
        if self.display_mode in (DisplayMode.BOTH, DisplayMode.GRAVITY):
            l1, lb1 = self.ax_grav.get_legend_handles_labels()
            lines += l1; labs += lb1
        if self.display_mode in (DisplayMode.BOTH, DisplayMode.MAGNETICS):
            l2, lb2 = self.ax_mag.get_legend_handles_labels()
            lines += l2; labs += lb2
        if lines:
            self.ax_grav.legend(lines, labs,
                                loc="upper right", fontsize=7, framealpha=0.7)

    def _update_magnetic(self, xp):
        """Compute and plot the selected magnetic component on ax_mag."""
        signal = np.zeros(len(xp))
        for body in self.bodies:
            if not (body.visible and body.is_complete()):
                continue
            if body.susceptibility == 0.0 and body.remanence_Am == 0.0:
                continue
            args = (xp, body.vertices, body.susceptibility,
                    self.earth_field_nT, self.earth_inc_deg,
                    body.remanence_Am, body.remanence_inc_deg)
            if self.mag_component == MagComponent.TMI:
                signal += compute_bt(*args)
            elif self.mag_component == MagComponent.BX:
                bx, _ = compute_bx_bz(*args)
                signal += bx
            else:  # BZ
                _, bz = compute_bx_bz(*args)
                signal += bz

        if self._mag_line is not None:
            try:
                self._mag_line.remove()
            except Exception:
                pass
            self._mag_line = None

        self._mag_line, = self.ax_mag.plot(
            xp, signal, color=_MAG_COLOR, lw=1.5, ls="--",
            label=f"{self.mag_component.value} (calc.)", zorder=3)

        self.ax_mag.relim()
        self.ax_mag.autoscale_view(scalex=False, scaley=True)

    def _toggle_station_mask(self, x_click):
        if self.obs_data is None:
            return
        dists = np.abs(self.obs_data.x - x_click)
        idx   = int(np.argmin(dists))
        self.obs_data.masked[idx] = not self.obs_data.masked[idx]
        self._update_gravity()
        self.fig.canvas.draw_idle()

    # ── full redraw ───────────────────────────────────────────────────────

    def _full_redraw(self):
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
            verts, closed=True, facecolor=color,
            edgecolor=body.color, linewidth=1.5,
            zorder=2, visible=body.visible, picker=True)
        self.ax_model.add_patch(patch)
        self._body_patches[id(body)] = patch

    def _refresh_patch(self, body: PolygonBody):
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
            s=VERTEX_RADIUS**2, c="white", edgecolors="black",
            linewidths=1.2, zorder=5)

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
        self._remove_draw_artists()
        if not self._draw_verts:
            return
        vx = [v[0] for v in self._draw_verts]
        vz = [v[1] for v in self._draw_verts]
        self._draw_line, = self.ax_model.plot(
            vx + [vx[0]], vz + [vz[0]], "k--", lw=1.2, zorder=6)
        if cx is not None and len(self._draw_verts) >= 1:
            self._draw_cursor_line, = self.ax_model.plot(
                [vx[-1], cx], [vz[-1], cz],
                color="gray", lw=0.9, ls=":", zorder=6)
        self.fig.canvas.draw_idle()

    # ── mouse event handlers ──────────────────────────────────────────────

    def _on_press(self, event):
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
        btn = event.button
        if   self.mode == Mode.DRAW:       self._handle_draw_press(x, z, btn)
        elif self.mode == Mode.SELECT:     self._handle_select_press(x, z)
        elif self.mode == Mode.ADD_VERTEX: self._handle_add_vertex_press(x, z)
        elif self.mode == Mode.DELETE:     self._handle_delete_press(x, z)

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
            snap = self._find_snap_target(x, z,
                                          exclude_body=self.selected_body,
                                          exclude_vi=self.selected_vertex)
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
                self.delete_key_pressed.emit()
            elif event.key == "escape":
                self._deselect()

    # ── draw mode ─────────────────────────────────────────────────────────

    def _handle_draw_press(self, x, z, btn):
        if btn == 1:
            snap = self._find_snap_target(x, z)
            if snap:
                x, z = snap
            self._draw_verts.append([x, z])
            self._update_draw_preview(x, z)
        elif btn == 3:
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
                self._drag_prev   = (x, z)
                return
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
        snap = self._find_snap_target(x, z,
                                      exclude_body=self.selected_body,
                                      exclude_vi=self.selected_vertex)
        nx, nz = snap if snap else (x, z)
        linked  = self._find_linked_vertices(self.selected_body, self.selected_vertex)
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

    # ── add-vertex / delete mode ──────────────────────────────────────────

    def _handle_add_vertex_press(self, x, z):
        hit = self._pick_body(x, z)
        if hit is None:
            return
        ei, _, px, pz = hit.nearest_edge_point(x, z)
        if ei < 0:
            return
        hit.vertices.insert(ei + 1, [px, pz])
        self.selected_body = hit
        self._refresh_patch(hit)
        self.bodies_changed.emit()

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
        for body in reversed(self.bodies):
            if body.is_complete() and body.visible and body.contains_point(x, z):
                return body
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
        # z_max is the first element of ylim on the (inverted) model axis
        z_max_km = float(self.ax_model.get_ylim()[0])
        return {
            "version":        2,
            # ── profile / domain ──────────────────────────────────────────
            "x_min":          self.x_min,
            "x_max":          self.x_max,
            "n_pts":          self.n_pts,
            "z_max_km":       z_max_km,
            "use_km":         self.use_km,
            # ── physical parameters ───────────────────────────────────────
            "bg_density":     self.bg_density,
            "earth_field_nT": self.earth_field_nT,
            "earth_inc_deg":  self.earth_inc_deg,
            # ── bodies ────────────────────────────────────────────────────
            "bodies": [
                {"name":               b.name,
                 "density":            b.density,
                 "susceptibility":     b.susceptibility,
                 "remanence_Am":       b.remanence_Am,
                 "remanence_inc_deg":  b.remanence_inc_deg,
                 "color":              b.color,
                 "visible":            b.visible,
                 "vertices":           [v[:] for v in b.vertices]}
                for b in self.bodies
            ],
        }

    def from_dict(self, d: dict):
        # ── profile / domain ──────────────────────────────────────────────
        self.x_min          = float(d.get("x_min",          -50.0))
        self.x_max          = float(d.get("x_max",           50.0))
        self.n_pts          = int(d.get("n_pts",              201))
        z_max_km            = float(d.get("z_max_km",         20.0))
        self.use_km         = bool(d.get("use_km",             True))
        # ── physical parameters ───────────────────────────────────────────
        self.bg_density     = float(d.get("bg_density",     2670.0))
        self.earth_field_nT = float(d.get("earth_field_nT", 50000.0))
        self.earth_inc_deg  = float(d.get("earth_inc_deg",     60.0))
        # ── bodies ────────────────────────────────────────────────────────
        self.bodies.clear()
        PolygonBody._counter = 0
        for bd in d.get("bodies", []):
            b = PolygonBody(
                vertices          = [list(v) for v in bd["vertices"]],
                density           = float(bd["density"]),
                susceptibility    = float(bd.get("susceptibility",    0.001)),
                remanence_Am      = float(bd.get("remanence_Am",      0.0)),
                remanence_inc_deg = float(bd.get("remanence_inc_deg", 0.0)),
                color             = bd.get("color",   "#4C72B0"),
                name              = bd.get("name",    "Body"),
                visible           = bool(bd.get("visible", True)),
            )
            b.name = bd.get("name", b.name)
            self.bodies.append(b)
        # ── restore axes ──────────────────────────────────────────────────
        self.ax_model.set_xlim(self.x_min, self.x_max)
        self.ax_grav.set_xlim(self.x_min, self.x_max)
        self.ax_model.set_ylim(z_max_km, -0.5)
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
    profile_changed      = pyqtSignal(float, float, int)
    depth_changed        = pyqtSignal(float)
    units_changed        = pyqtSignal(bool)
    bg_density_changed   = pyqtSignal(float)
    earth_field_changed  = pyqtSignal(float, float)   # ★ F_nT, IE_deg
    snap_changed         = pyqtSignal(bool)
    display_mode_changed = pyqtSignal(DisplayMode)

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

        self._use_km = True

        # ── snapping ──────────────────────────────────────────────────────
        self.chk_snap = QCheckBox("Enable vertex snapping")
        self.chk_snap.setChecked(True)
        self.chk_snap.setToolTip(
            "When enabled, nearby vertices snap together while drawing or dragging")
        self.chk_snap.stateChanged.connect(
            lambda s: self.snap_changed.emit(s == Qt.CheckState.Checked.value))
        layout.addWidget(self.chk_snap)

        # ── display mode ──────────────────────────────────────────────────
        disp_group  = QGroupBox("Display Mode")
        disp_layout = QHBoxLayout(disp_group)
        self._disp_btn_group = QButtonGroup(self)
        self._disp_btn_group.setExclusive(True)
        self._disp_buttons = {}
        for label, dm in [("Gravity", DisplayMode.GRAVITY),
                          ("Gravity && Magnetics",    DisplayMode.BOTH),
                          ("Magnetics", DisplayMode.MAGNETICS)]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(26)
            self._disp_btn_group.addButton(btn)
            self._disp_buttons[dm] = btn
            disp_layout.addWidget(btn)
            btn.clicked.connect(lambda _, d=dm: self.display_mode_changed.emit(d))
        self._disp_buttons[DisplayMode.BOTH].setChecked(True)
        layout.addWidget(disp_group)

        # ── distance units ────────────────────────────────────────────────
        units_group  = QGroupBox("Distance Units")
        units_layout = QHBoxLayout(units_group)
        units_layout.addWidget(QLabel("km"))
        self._unit_toggle = ToggleSwitch(height=20)
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

        # ★ ── Earth field ─────────────────────────────────────────────────
        ef_group = QGroupBox("Earth Field (magnetics)")
        ef_form  = QFormLayout(ef_group)

        self.spin_field_F = QDoubleSpinBox()
        self.spin_field_F.setRange(0.0, 100_000.0)
        self.spin_field_F.setValue(50_000.0)
        self.spin_field_F.setSuffix(" nT")
        self.spin_field_F.setSingleStep(1000.0)
        self.spin_field_F.setDecimals(0)
        self.spin_field_F.setToolTip("Total intensity of Earth's field (nT)")
        ef_form.addRow("F (intensity):", self.spin_field_F)

        self.spin_field_IE = QDoubleSpinBox()
        self.spin_field_IE.setRange(-90.0, 90.0)
        self.spin_field_IE.setValue(60.0)
        self.spin_field_IE.setSuffix(" °")
        self.spin_field_IE.setSingleStep(5.0)
        self.spin_field_IE.setDecimals(1)
        self.spin_field_IE.setToolTip(
            "Inclination of Earth's field projected onto the profile (degrees, + downward)")
        ef_form.addRow("Inclination IE:", self.spin_field_IE)

        self.spin_field_F.valueChanged.connect(self._emit_earth_field)
        self.spin_field_IE.valueChanged.connect(self._emit_earth_field)
        layout.addWidget(ef_group)
        # ★ ───────────────────────────────────────────────────────────────

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
        self.spin_zmax.valueChanged.connect(lambda v: self.depth_changed.emit(v))
        dform.addRow("Max depth:", self.spin_zmax)
        layout.addWidget(depth_group)

        layout.addStretch()

    # ── unit toggle ───────────────────────────────────────────────────────

    def _on_unit_toggled(self, is_m: bool):
        use_km = not is_m
        if use_km == self._use_km:
            return
        factor = 1e-3 if use_km else 1e3
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
            self.spin_xmin.value(), self.spin_xmax.value(), self.spin_npts.value())

    def _emit_earth_field(self):                              # ★
        self.earth_field_changed.emit(
            self.spin_field_F.value(), self.spin_field_IE.value())


# ─────────────────────────────────────────────────────────────────────────────
#  Polygon table dock
# ─────────────────────────────────────────────────────────────────────────────

COL_VIS      = 0
COL_COLOR    = 1
COL_NAME     = 2
COL_DENSITY  = 3
COL_CONTRAST = 4
COL_SUSCEPT  = 5
COL_REM_J    = 6   # remanence intensity (A/m)
COL_REM_INC  = 7   # remanence inclination (degrees)
COL_NVERTS   = 8
N_COLS       = 9

VCOL_IDX = 0
VCOL_X   = 1
VCOL_Z   = 2


class PolygonTableDock(QDockWidget):
    """
    Bottom dock – polygon table (left) and vertex table (right).
    Columns: Vis | Color | Name | Density | Contrast | χ (SI) | Verts
    """

    body_selected        = pyqtSignal(object)
    body_changed         = pyqtSignal(object)
    body_vertex_changed  = pyqtSignal(object)
    body_vertex_deleted  = pyqtSignal(object)   # emitted after a vertex is removed
    body_delete_requested = pyqtSignal(list)    # emitted when user confirms polygon deletion

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
            ["Vis", "Color", "Name",
             "Density\n(kg/m³)", "Contrast\n(kg/m³)",
             "χ (SI)",
             "J_rem\n(A/m)", "Inc_rem\n(°)",
             "Verts"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_NAME,     QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(COL_DENSITY,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_CONTRAST, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_SUSCEPT,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_REM_J,    QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_REM_INC,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_NVERTS,   QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_VIS,      QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_COLOR,    QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(120)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.installEventFilter(self)
        left_vb.addWidget(self.table)

        self.btn_del_poly = QPushButton("Delete Selected Polygon(s)")
        self.btn_del_poly.setToolTip("Remove the selected polygon(s) from the model")
        self.btn_del_poly.setEnabled(False)
        self.btn_del_poly.clicked.connect(self._confirm_delete_polygons)
        left_vb.addWidget(self.btn_del_poly)

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
        self.vert_table.itemSelectionChanged.connect(self._on_vert_selection_changed)
        self.vert_table.installEventFilter(self)
        right_vb.addWidget(self.vert_table)

        self.btn_del_vert = QPushButton("Delete Selected Vertex")
        self.btn_del_vert.setToolTip(
            "Remove the selected vertex (body must keep at least 3 vertices)")
        self.btn_del_vert.setEnabled(False)
        self.btn_del_vert.clicked.connect(self._delete_selected_vertex)
        right_vb.addWidget(self.btn_del_vert)

        splitter.addWidget(right_w)

        splitter.setSizes([600, 300])
        outer.addWidget(splitter)
        self.setWidget(container)

        self._bodies:   List[PolygonBody]    = []
        self._sel_body: Optional[PolygonBody] = None
        self._updating  = False
        self._bg_density: float = 2670.0
        self._use_km:     bool  = True

    # ── synchronise with canvas ───────────────────────────────────────────

    def sync(self, bodies: List[PolygonBody]):
        self._updating = True
        self._bodies   = bodies
        self.table.setRowCount(len(bodies))
        for row, body in enumerate(bodies):
            self._populate_row(row, body)
        self._updating = False

    def _populate_row(self, row: int, body: PolygonBody):
        # Visible
        vis_item = QTableWidgetItem()
        vis_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
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
        self.table.setItem(row, COL_NAME, QTableWidgetItem(body.name))

        # Density
        dens_item = QTableWidgetItem(f"{body.density:.1f}")
        dens_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_DENSITY, dens_item)

        # Contrast (editable; editing back-calculates density)
        cont_item = QTableWidgetItem(f"{body.density - self._bg_density:.1f}")
        cont_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_CONTRAST, cont_item)

        # Susceptibility
        susc_item = QTableWidgetItem(f"{body.susceptibility:.5f}")
        susc_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_SUSCEPT, susc_item)

        # Remanence intensity (A/m)
        rem_j_item = QTableWidgetItem(f"{body.remanence_Am:.4f}")
        rem_j_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_REM_J, rem_j_item)

        # Remanence inclination (degrees)
        rem_inc_item = QTableWidgetItem(f"{body.remanence_inc_deg:.1f}")
        rem_inc_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_REM_INC, rem_inc_item)

        # #Vertices (read-only)
        nv_item = QTableWidgetItem(str(len(body.vertices)))
        nv_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.table.setItem(row, COL_NVERTS, nv_item)

    def update_row(self, body: PolygonBody):
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
        if body is self._sel_body:
            self._populate_vert_table(body)

    def set_bg_density(self, bg: float):
        self._bg_density = bg
        self._updating   = True
        for row, body in enumerate(self._bodies):
            ct = self.table.item(row, COL_CONTRAST)
            if ct:
                ct.setText(f"{body.density - bg:.1f}")
        self._updating = False

    def set_units(self, use_km: bool):
        self._use_km = use_km
        unit = "km" if use_km else "m"
        self.vert_table.setHorizontalHeaderLabels(
            ["#", f"X ({unit})", f"Z ({unit})"])
        if self._sel_body is not None:
            self._populate_vert_table(self._sel_body)

    # ── vertex table ──────────────────────────────────────────────────────

    def show_vertices(self, body: Optional[PolygonBody]):
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
            idx_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.vert_table.setItem(i, VCOL_IDX, idx_item)

            x_item = QTableWidgetItem(f"{vx * scale:.4f}")
            x_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.vert_table.setItem(i, VCOL_X, x_item)

            z_item = QTableWidgetItem(f"{vz * scale:.4f}")
            z_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.vert_table.setItem(i, VCOL_Z, z_item)
        self._updating = False

    def _on_vert_selection_changed(self):
        """Enable the delete button only when a row is selected and body has > 3 verts."""
        rows = {idx.row() for idx in self.vert_table.selectedIndexes()}
        can_delete = (
            bool(rows)
            and self._sel_body is not None
            and len(self._sel_body.vertices) > 3
        )
        self.btn_del_vert.setEnabled(can_delete)

    def _delete_selected_vertex(self):
        if self._sel_body is None:
            return
        rows = sorted({idx.row() for idx in self.vert_table.selectedIndexes()})
        if not rows:
            return
        if len(self._sel_body.vertices) - len(rows) < 3:
            return   # would leave fewer than 3 vertices — refuse silently
        # When multiple vertices share the same coordinates, only remove the
        # one at the selected index (not all duplicates).  Removing in reverse
        # order keeps earlier indices valid.
        for row in reversed(rows):
            if 0 <= row < len(self._sel_body.vertices):
                del self._sel_body.vertices[row]
        self._populate_vert_table(self._sel_body)
        self.body_vertex_deleted.emit(self._sel_body)

    def eventFilter(self, source, event):
        """Intercept Delete / Backspace on both tables."""
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.KeyPress and event.key() in (
                Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if source is self.table:
                if self.btn_del_poly.isEnabled():
                    self._confirm_delete_polygons()
                return True
            if source is self.vert_table:
                if self.btn_del_vert.isEnabled():
                    self._delete_selected_vertex()
                return True
        return super().eventFilter(source, event)

    def _on_vert_item_changed(self, item: QTableWidgetItem):
        if self._updating or self._sel_body is None:
            return
        row, col = item.row(), item.column()
        if col not in (VCOL_X, VCOL_Z):
            return
        if row < 0 or row >= len(self._sel_body.vertices):
            return
        scale = 1.0 if self._use_km else 1e-3
        try:
            val_km = float(item.text()) * scale
        except ValueError:
            self._populate_vert_table(self._sel_body)
            return
        coord = 0 if col == VCOL_X else 1
        self._sel_body.vertices[row][coord] = val_km
        self.body_vertex_changed.emit(self._sel_body)

    def _set_btn_color(self, btn: QPushButton, color: str):
        c = QColor(color)
        btn.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #555;")

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

        elif col == COL_CONTRAST:
            try:
                contrast = float(item.text())
                body.density = contrast + self._bg_density
                self._updating = True
                dn = self.table.item(row, COL_DENSITY)
                if dn:
                    dn.setText(f"{body.density:.1f}")
                self._updating = False
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.density - self._bg_density:.1f}")
                self._updating = False

        elif col == COL_SUSCEPT:
            try:
                body.susceptibility = float(item.text())
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.susceptibility:.5f}")
                self._updating = False

        elif col == COL_REM_J:
            try:
                body.remanence_Am = float(item.text())
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.remanence_Am:.4f}")
                self._updating = False

        elif col == COL_REM_INC:
            try:
                body.remanence_inc_deg = float(item.text())
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.remanence_inc_deg:.1f}")
                self._updating = False

    # ── row selection ─────────────────────────────────────────────────────

    def _on_selection_changed(self):
        if self._updating:
            return
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        self.btn_del_poly.setEnabled(bool(rows))
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

    def _confirm_delete_polygons(self):
        """Show a confirmation dialog and emit body_delete_requested if confirmed."""
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        bodies = [self._bodies[r] for r in rows if 0 <= r < len(self._bodies)]
        if not bodies:
            return
        names = ", ".join(f'"{b.name}"' for b in bodies)
        reply = QMessageBox.question(
            self, "Delete Polygon(s)",
            f"Delete selected polygon(s)?\n{names}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.body_delete_requested.emit(bodies)

    def select_body(self, body: Optional[PolygonBody]):
        self._updating = True
        self.table.clearSelection()
        if body is not None and body in self._bodies:
            row = self._bodies.index(body)
            self.table.selectRow(row)
        self._updating = False


# ─────────────────────────────────────────────────────────────────────────────
#  Inversion dock  (unchanged from gravity2d_gui.py)
# ─────────────────────────────────────────────────────────────────────────────

class InversionDock(QDockWidget):
    run_requested    = pyqtSignal(object)
    stop_requested   = pyqtSignal()
    revert_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Inversion", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                             Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        w  = QWidget()
        self.setWidget(w)
        vb = QVBoxLayout(w)
        vb.setAlignment(Qt.AlignmentFlag.AlignTop)

        inv_group = QGroupBox("Invert")
        inv_vb    = QVBoxLayout(inv_group)
        self.chk_vertices  = QCheckBox("Vertex positions");  self.chk_vertices.setChecked(True)
        self.chk_densities = QCheckBox("Density contrasts"); self.chk_densities.setChecked(True)
        self.chk_weights   = QCheckBox("Use data weights (1/σ²)")
        self.chk_weights.setToolTip("Weight residuals by 1/σ² when uncertainty column is loaded")
        inv_vb.addWidget(self.chk_vertices)
        inv_vb.addWidget(self.chk_densities)
        inv_vb.addWidget(self.chk_weights)
        vb.addWidget(inv_group)

        param_group = QGroupBox("Solver Parameters")
        pform       = QFormLayout(param_group)
        self.spin_damping = QDoubleSpinBox()
        self.spin_damping.setRange(1e-8, 1e4); self.spin_damping.setValue(1e-2)
        self.spin_damping.setDecimals(6); self.spin_damping.setSingleStep(0.001)
        self.spin_damping.setToolTip("Tikhonov/LM damping λ")
        pform.addRow("Damping λ:", self.spin_damping)
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(1, 500); self.spin_max_iter.setValue(20)
        pform.addRow("Max iterations:", self.spin_max_iter)
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(1e-10, 1.0); self.spin_tol.setValue(1e-4)
        self.spin_tol.setDecimals(8); self.spin_tol.setSingleStep(1e-5)
        self.spin_tol.setToolTip("Convergence tolerance on ||Δp||")
        pform.addRow("Tolerance:", self.spin_tol)
        vb.addWidget(param_group)

        btn_row = QHBoxLayout()
        self.btn_run  = QPushButton("Run");  self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        btn_row.addWidget(self.btn_run); btn_row.addWidget(self.btn_stop)
        vb.addLayout(btn_row)

        self.btn_revert = QPushButton("Revert to Starting Model")
        self.btn_revert.setEnabled(False)
        self.btn_revert.setToolTip(
            "Restore the model to the state before inversion was run")
        self.btn_revert.clicked.connect(self.revert_requested.emit)
        vb.addWidget(self.btn_revert)

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
        self.setWindowTitle("2-D Gravity + Magnetic Modeller")   # ★
        self.resize(1200, 780)

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
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.controls_dock)
        self.controls_dock.profile_changed.connect(
            lambda xn, xx, np_: self.canvas.set_profile(xn, xx, np_))
        self.controls_dock.depth_changed.connect(self.canvas.set_depth_range)
        self.controls_dock.units_changed.connect(self._on_units_changed)
        self.controls_dock.bg_density_changed.connect(self._on_bg_density_changed)
        self.controls_dock.earth_field_changed.connect(self.canvas.set_earth_field)
        self.controls_dock.snap_changed.connect(self.canvas.set_snap_enabled)
        self.controls_dock.display_mode_changed.connect(self.canvas.set_display_mode)

        # ── editing mode toolbar ──────────────────────────────────────────
        self._build_toolbars()

        # ── polygon table dock (bottom) ───────────────────────────────────
        self.table_dock = PolygonTableDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.table_dock)
        self.table_dock.setMinimumHeight(160)
        self.table_dock.body_selected.connect(self._on_table_body_selected)
        self.table_dock.body_changed.connect(self._on_table_body_changed)
        self.table_dock.body_vertex_changed.connect(self._on_vertex_edited)
        self.table_dock.body_vertex_deleted.connect(self._on_vertex_edited)
        self.table_dock.body_delete_requested.connect(self._delete_bodies)
        self.canvas.bodies_changed.connect(self._sync_table)
        self.canvas.delete_key_pressed.connect(self._confirm_delete_selected)

        # ── inversion dock (right) ────────────────────────────────────────
        self.inv_dock = InversionDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inv_dock)
        self.inv_dock.run_requested.connect(self._run_inversion)
        self.inv_dock.stop_requested.connect(self._stop_inversion)
        self.inv_dock.revert_requested.connect(self._revert_inversion)
        self._inv_worker   = None
        self._inv_snapshot = None

        # ── status bar ────────────────────────────────────────────────────
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._rms_label = QLabel("RMS: --")
        self.status.addPermanentWidget(self._rms_label)
        self.canvas.rms_updated.connect(
            lambda rms: self._rms_label.setText(f"RMS: {rms:.3f} mGal"))
        self.status.showMessage(
            "Select mode: click body to select | Draw mode: LMB add vertex, RMB close")

        self._build_menu()
        self._sync_table()

    # ── toolbars ──────────────────────────────────────────────────────────

    def _build_toolbars(self):
        from PyQt6.QtWidgets import QToolBar
        from PyQt6.QtCore import QSize

        # ── editing mode toolbar ──────────────────────────────────────────
        mode_tb = QToolBar("Editing Mode", self)
        mode_tb.setObjectName("mode_toolbar")
        mode_tb.setIconSize(QSize(24, 24))
        mode_tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, mode_tb)

        mode_ag = QActionGroup(self)
        mode_ag.setExclusive(True)
        self._mode_actions: dict[Mode, QAction] = {}

        mode_defs = [
            ("Draw a new polygon",  Mode.DRAW,       "D",
             "LMB: add vertex  |  RMB / Enter: close  |  Esc: cancel",
             "src/resources/icons/icon-polygon-new-64.svg", "New"),
            ("Select / Move", Mode.SELECT,      "S",
             "LMB: select body  |  drag vertex to move",
             "src/resources/icons/icon-polygon-select-64.svg",
             "Move"),
            ("Add Vertex",    Mode.ADD_VERTEX,  "A",
             "LMB: insert vertex on an edge",
             "src/resources/icons/icon-add-point-64.svg",
             "Add"),
            ("Delete Body",   Mode.DELETE,      "X",
             "LMB: remove body",
             "src/resources/icons/icon-remove-point-64.svg",
             "Delete"),
            ("Mask Stations", Mode.MASK,        "M",
             "LMB in gravity panel: toggle station mask",
             "src/resources/icons/icon-mask-light-64.svg",
             "Mask"),
        ]
        for label, mode, key, tip, icon_path, icon_text in mode_defs:
            act = QAction(label, self)
            act.setIcon(QIcon(icon_path))
            act.setText(icon_text)
            act.setCheckable(True)
            act.setToolTip(f"{tip}  [{key}]")
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _, m=mode: self._on_mode_changed(m))
            mode_ag.addAction(act)
            mode_tb.addAction(act)
            self._mode_actions[mode] = act
        self._mode_actions[Mode.SELECT].setChecked(True)

        mode_tb.addSeparator()

        # ── magnetic component toolbar ────────────────────────────────────
        mag_ag = QActionGroup(self)
        mag_ag.setExclusive(True)
        self._mag_component_actions: dict[MagComponent, QAction] = {}

        for comp in (MagComponent.TMI, MagComponent.BX, MagComponent.BZ):
            act = QAction(comp.value, self)
            act.setCheckable(True)
            act.setToolTip(f"Display {comp.value} on the magnetic axis")
            act.triggered.connect(lambda _, c=comp: self.canvas.set_mag_component(c))
            mag_ag.addAction(act)
            mode_tb.addAction(act)
            self._mag_component_actions[comp] = act
        self._mag_component_actions[MagComponent.TMI].setChecked(True)

    # ── menu ──────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        act_new = QAction("&New Model", self); act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._new_model); file_menu.addAction(act_new)
        file_menu.addSeparator()
        act_open_data = QAction("&Open Data…", self); act_open_data.setShortcut("Ctrl+O")
        act_open_data.triggered.connect(self._open_data); file_menu.addAction(act_open_data)
        file_menu.addSeparator()
        act_save = QAction("&Save Model…", self); act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_model); file_menu.addAction(act_save)
        act_load = QAction("&Load Model…", self); act_load.setShortcut("Ctrl+L")
        act_load.triggered.connect(self._load_model); file_menu.addAction(act_load)
        file_menu.addSeparator()
        act_quit = QAction("&Quit", self); act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close); file_menu.addAction(act_quit)

        edit_menu = mb.addMenu("&Edit")
        act_del = QAction("&Delete Selected Body", self); act_del.setShortcut("Delete")
        act_del.triggered.connect(self._delete_selected); edit_menu.addAction(act_del)

        demo_menu = mb.addMenu("&Demo")
        act_demo = QAction("Add Rectangular Body", self)
        act_demo.triggered.connect(self._add_demo_body); demo_menu.addAction(act_demo)

        inv_menu = mb.addMenu("&Inversion")
        act_inv_run = QAction("&Run Inversion", self); act_inv_run.setShortcut("Ctrl+I")
        act_inv_run.triggered.connect(lambda: self.inv_dock.btn_run.click())
        inv_menu.addAction(act_inv_run)
        act_inv_stop = QAction("&Stop Inversion", self)
        act_inv_stop.triggered.connect(self._stop_inversion); inv_menu.addAction(act_inv_stop)

        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about); help_menu.addAction(act_about)

    # ── menu actions ──────────────────────────────────────────────────────

    def _new_model(self):
        reply = QMessageBox.question(
            self, "New Model", "Clear all bodies and start a new model?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.canvas.bodies.clear()
            self.canvas._full_redraw()
            self._sync_table()

    def _confirm_delete_selected(self):
        """Confirmation dialog triggered by Delete key on the canvas."""
        body = self.canvas.selected_body
        if body is None:
            return
        reply = QMessageBox.question(
            self, "Delete Polygon",
            f'Delete selected polygon(s)?\n"{body.name}"',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._delete_bodies([body])

    def _delete_bodies(self, bodies):
        """Remove a list of PolygonBody objects from the canvas and sync the table."""
        for body in bodies:
            self.canvas.remove_body(body)
        self._sync_table()

    def _delete_selected(self):
        body = self.canvas.selected_body
        if body is not None:
            self.canvas.remove_body(body)
            self._sync_table()

    def _add_demo_body(self):
        body = PolygonBody(
            vertices=[[-2, 2], [2, 2], [2, 7], [-2, 7]],
            density=300.0, susceptibility=0.01, name="Demo Body")
        self.canvas.add_body(body)
        self._sync_table()

    def _show_about(self):
        QMessageBox.about(
            self, "About",
            "<b>2-D Gravity + Magnetic Modeller</b><br><br>"
            "Gravity forward model: Talwani et al. (1959)<br>"
            "Magnetic forward model: Blakely (1995) / Won &amp; Bevis (1987)<br>"
            "Interface: PyQt6 + Matplotlib<br><br>"
            "Each polygon body has an independent density and susceptibility (SI).<br>"
            "The magnetic anomaly assumes induced magnetisation only (no remanence).<br><br>"
            "Keyboard shortcuts:<br>"
            "&nbsp;&nbsp;<b>D</b> – Draw mode<br>"
            "&nbsp;&nbsp;<b>S</b> – Select/Move mode<br>"
            "&nbsp;&nbsp;<b>A</b> – Add Vertex mode<br>"
            "&nbsp;&nbsp;<b>X</b> – Delete Body mode<br>"
            "&nbsp;&nbsp;<b>Enter</b> – Close polygon (draw mode)<br>"
            "&nbsp;&nbsp;<b>Esc</b> – Cancel / deselect",
        )

    # ── open data / save / load ───────────────────────────────────────────

    def _open_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Gravity Data", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        obs = load_csv_data(path, self)
        if obs is None:
            return
        self.canvas.load_observed(obs)
        n = len(obs.x)
        self.status.showMessage(
            f"Loaded {n} stations from {Path(path).name}  |  "
            f"x range: {obs.x.min():.1f}–{obs.x.max():.1f} km")

    def _save_model(self):
        import json
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "JSON model (*.json);;All files (*)")
        if not path:
            return
        try:
            data = self.canvas.to_dict()

            # ── observed data (optional) ──────────────────────────────────
            obs = self.canvas.obs_data
            if obs is not None:
                od: dict = {
                    "source_file": obs.source_file,
                    "x_km":        [round(v, 6) for v in obs.x.tolist()],
                    "gz_mGal":     [round(v, 6) for v in obs.gz.tolist()],
                }
                if obs.gz_unc is not None:
                    od["gz_unc_mGal"] = [round(v, 6) for v in obs.gz_unc.tolist()]
                if obs.masked is not None:
                    od["masked"] = obs.masked.tolist()
                data["observed_data"] = od

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            self.status.showMessage(f"Model saved to {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def _load_model(self):
        import json
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "JSON model (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # canvas.from_dict handles all defaults for missing keys
            self.canvas.from_dict(data)

            # ── sync controls dock ────────────────────────────────────────
            cd = self.controls_dock

            def _set(spin, value):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)

            _set(cd.spin_bg_density, self.canvas.bg_density)
            _set(cd.spin_field_F,    self.canvas.earth_field_nT)
            _set(cd.spin_field_IE,   self.canvas.earth_inc_deg)
            _set(cd.spin_xmin,       self.canvas.x_min)
            _set(cd.spin_xmax,       self.canvas.x_max)
            _set(cd.spin_npts,       self.canvas.n_pts)
            z_max_km = float(self.canvas.ax_model.get_ylim()[0])
            _set(cd.spin_zmax, z_max_km)

            # units toggle — update display without re-triggering conversion
            if self.canvas.use_km != cd._use_km:
                cd._unit_toggle.blockSignals(True)
                cd._unit_toggle.setChecked(not self.canvas.use_km)  # off=km, on=m
                cd._use_km = self.canvas.use_km
                suffix = " km" if self.canvas.use_km else " m"
                step   = 10.0 if self.canvas.use_km else 10_000.0
                for spin in (cd.spin_xmin, cd.spin_xmax, cd.spin_zmax):
                    spin.setSuffix(suffix)
                    spin.setSingleStep(step)
                cd._unit_toggle.blockSignals(False)
                self.table_dock.set_units(self.canvas.use_km)

            # ── observed data (optional) ──────────────────────────────────
            od = data.get("observed_data")
            if od is not None:
                from src.gravity.data_loader import ObservedData
                x_arr  = np.array(od["x_km"],   dtype=float)
                gz_arr = np.array(od["gz_mGal"], dtype=float)
                gz_unc = (np.array(od["gz_unc_mGal"], dtype=float)
                          if "gz_unc_mGal" in od else None)
                masked = (np.array(od["masked"], dtype=bool)
                          if "masked" in od
                          else np.zeros(len(x_arr), dtype=bool))
                obs = ObservedData(
                    x            = x_arr,
                    gz           = gz_arr,
                    gz_unc       = gz_unc,
                    masked       = masked,
                    source_file  = od.get("source_file", ""),
                    x_col="", y_col="", gz_col="", gz_unc_col="",
                    x_scale_km=1.0,
                )
                self.canvas.load_observed(obs)

            self._sync_table()
            self.table_dock.set_bg_density(self.canvas.bg_density)
            self.status.showMessage(f"Model loaded from {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    # ── mode change ───────────────────────────────────────────────────────

    def _on_mode_changed(self, mode: Mode):
        self.canvas.set_mode(mode)
        act = self._mode_actions.get(mode)
        if act and not act.isChecked():
            act.setChecked(True)
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
        self._inv_snapshot = [
            {"name": b.name, "density": b.density, "color": b.color,
             "visible": b.visible, "vertices": [v[:] for v in b.vertices]}
            for b in self.canvas.bodies
        ]
        self.inv_dock.btn_revert.setEnabled(False)
        self.inv_dock.set_running(True)
        self.inv_dock.lbl_progress.setText("Starting…")
        self._inv_worker = InversionWorker(
            self.canvas.bodies, self.canvas.bg_density,
            self.canvas.obs_data, cfg, parent=self)
        self._inv_worker.iteration_done.connect(self._on_inv_iteration)
        self._inv_worker.converged.connect(self._on_inv_converged)
        self._inv_worker.stopped.connect(self.inv_dock.show_stopped)
        self._inv_worker.start()

    def _stop_inversion(self):
        if self._inv_worker is not None and self._inv_worker.isRunning():
            self._inv_worker.request_stop()

    def _on_inv_iteration(self, it: int, rms: float, body_dicts: list):
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
