from typing import List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal

from src.gui.gm2d_types import DisplayMode, MagComponent, Mode, PolygonBody
from src.gravity.grav2_5d_model  import compute_gz     as _gz_25d
from src.gravity.talwani_model   import compute_gz     as _gz_2d
from src.magnetics.mag2_5d_model import compute_bt     as _bt_25d
from src.magnetics.mag2_5d_model import compute_bx_bz  as _bxbz_25d
from src.magnetics.mag2d_model   import compute_bt     as _bt_2d
from src.magnetics.mag2d_model   import compute_bx_bz  as _bxbz_2d
from src.gravity.data_loader   import ObservedData

# Colours for the two profile curves
_GRAV_COLOR = "navy"
_MAG_COLOR  = "#8B0000"   # dark red

VERTEX_RADIUS     = 6
HIT_RADIUS_PX     = 10
SNAP_RADIUS_PX    = 12
SNAP_INDICATOR_PX = 14
_LINK_EPS         = 1e-9

class MainCanvas(FigureCanvas):
    """
    Two-panel matplotlib canvas:
      - Top  (ax_grav / ax_mag)  -- gravity profile (left axis, mGal)
                                    + total-field magnetic profile (right axis, nT)
      - Bottom (ax_model)        -- model cross-section; polygon editing
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

        # Earth-field parameters
        self.earth_field_nT = 50000.0   # nT
        self.earth_inc_deg  =  60.0     # degrees
        self.earth_dec_deg  =   0.0     # degrees
        self.z_obs          =   0.0     # km (observation height)
        self.is_2_5d        = True      # True = 2.5D model, False = 2D model

        self.display_mode    = DisplayMode.BOTH
        self.mag_components: set = {MagComponent.TMI}  # active components

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
        self._move_body_active = False
        self._move_body_prev   = (None, None)

        self._draw_verts: List[List[float]] = []
        self._draw_line        = None
        self._draw_cursor_line = None
        self._cursor_pos = (0.0, 0.0)

        self._body_patches: dict = {}
        self._vertex_scatter = None
        self._grav_line  = None
        self._mag_lines: dict = {}   # MagComponent -> line artist
        self._obs_line   = None
        self._obs_errbar = None

        self._snap_ring = None

        self.obs_data: Optional[ObservedData] = None

        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("key_press_event",      self._on_key)

        self._full_redraw()

    # --- axes appearance ---

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

    # --- public profile / unit / earth-field controls ---

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
        if mode == Mode.MOVE_BODY:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()
        # Clear any in-progress drag if mode switches away
        if mode not in (Mode.SELECT, Mode.ADD_VERTEX):
            self._drag_active    = False
            self.selected_vertex = -1
        self.fig.canvas.draw_idle()

    def set_bg_density(self, bg: float):
        self.bg_density = bg
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def set_earth_field(self, F_nT: float, IE_deg: float, DE_deg: float = 0.0):
        """Update Earth-field parameters and recompute magnetic anomaly."""
        self.earth_field_nT = F_nT
        self.earth_inc_deg  = IE_deg
        self.earth_dec_deg  = DE_deg
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def set_z_obs(self, z_obs: float):
        """Update observation height (km) and recompute."""
        self.z_obs = z_obs
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def set_model_mode(self, is_2_5d: bool):
        """Switch between 2.5D and 2D forward models and recompute."""
        self.is_2_5d = is_2_5d
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def load_observed(self, obs_data: ObservedData):
        self.obs_data = obs_data
        self._update_gravity()
        self.fig.canvas.draw_idle()

    # --- pixel ↔ km conversion ---

    def _px_to_km(self, n_px: float) -> float:
        inv = self.ax_model.transData.inverted()
        p0 = inv.transform((0.0, 0.0))
        p1 = inv.transform((n_px, 0.0))
        return abs(float(p1[0]) - float(p0[0]))

    # --- snapping helpers ---

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
                d = np.hypot(vx - x, vz - z)
                if d < best_d:
                    best_d, best_pos = d, (vx, vz)
        for vi, (vx, vz) in enumerate(self._draw_verts[:-1]
                                       if self._draw_verts else []):
            d = np.hypot(vx - x, vz - z)
            if d < best_d:
                best_d, best_pos = d, (vx, vz)
        return best_pos

    def _find_snap_vertex(self, x, z) -> Optional[Tuple["PolygonBody", int]]:
        """Return (body, vertex_index) of the nearest vertex within snap radius, or None."""
        if not self.snap_enabled:
            return None
        snap_km = self._px_to_km(SNAP_RADIUS_PX)
        best_d, best_body, best_vi = snap_km, None, -1
        for body in self.bodies:
            for vi, (vx, vz) in enumerate(body.vertices):
                d = np.hypot(vx - x, vz - z)
                if d < best_d:
                    best_d, best_body, best_vi = d, body, vi
        return (best_body, best_vi) if best_body is not None else None

    def _find_linked_vertices(self, body, vi):
        x, z = body.vertices[vi]
        linked = []
        for b in self.bodies:
            for i, (vx, vz) in enumerate(b.vertices):
                if b is body and i == vi:
                    continue
                if np.hypot(vx - x, vz - z) < _LINK_EPS:
                    linked.append((b, i))
        return linked

    # --- snap indicator ---

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

    # --- display mode ---

    def set_display_mode(self, mode: DisplayMode):
        self.display_mode = mode
        show_grav = mode in (DisplayMode.BOTH, DisplayMode.GRAVITY)
        show_mag  = mode in (DisplayMode.BOTH, DisplayMode.MAGNETICS)
        # Keep the profile panel visible in all modes (just empty when NONE)
        self.ax_grav.set_visible(True)
        self.ax_mag.set_visible(show_mag)
        self.ax_grav.yaxis.set_visible(show_grav)
        self.ax_grav.tick_params(labelbottom=False)
        self.ax_mag.tick_params(labelbottom=False)
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def toggle_mag_component(self, component: MagComponent, active: bool):
        """Enable or disable a magnetic component line."""
        if active:
            self.mag_components.add(component)
        else:
            self.mag_components.discard(component)
        self._update_y_label()
        self._update_gravity()
        self.fig.canvas.draw_idle()

    def _update_y_label(self):
        parts = [c.value for c in
                 (MagComponent.TMI, MagComponent.BX, MagComponent.BZ)
                 if c in self.mag_components]
        self.ax_mag.set_ylabel(" / ".join(parts) + " (nT)" if parts else "(nT)",
                               color=_MAG_COLOR)

    # --- gravity + magnetic computation ---

    def _profile_x(self):
        return np.linspace(self.x_min, self.x_max, self.n_pts)

    def _update_gravity(self):
        xp = self._profile_x()

        # --- gravity ---
        gz = np.zeros(len(xp))
        if self.display_mode in (DisplayMode.BOTH, DisplayMode.GRAVITY):
            for body in self.bodies:
                if body.visible and body.is_complete():
                    contrast = body.density - self.bg_density
                    if self.is_2_5d:
                        gz += _gz_25d(xp, self.z_obs, body.vertices, contrast)
                    else:
                        gz += _gz_2d(xp, body.vertices, contrast)

        if self._grav_line is not None:
            self._grav_line.remove()
        if self.display_mode in (DisplayMode.BOTH, DisplayMode.GRAVITY):
            self._grav_line, = self.ax_grav.plot(
                xp, gz, color=_GRAV_COLOR, lw=1.5, label="Gravity (calc.)", zorder=3)
        else:
            self._grav_line = None

        # --- observed data ---
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

        # --- ★ magnetic ---
        if self.display_mode in (DisplayMode.BOTH, DisplayMode.MAGNETICS):
            self._update_magnetic(xp)
        else:
            for line in list(self._mag_lines.values()):
                try:
                    line.remove()
                except Exception:
                    pass
            self._mag_lines.clear()

        # --- combined legend ---
        lines, labs = [], []
        if self.display_mode in (DisplayMode.BOTH, DisplayMode.GRAVITY):
            l1, lb1 = self.ax_grav.get_legend_handles_labels()
            lines += l1; labs += lb1
        if self.display_mode in (DisplayMode.BOTH, DisplayMode.MAGNETICS) and self._mag_lines:
            l2, lb2 = self.ax_mag.get_legend_handles_labels()
            lines += l2; labs += lb2
        if lines:
            self.ax_grav.legend(lines, labs,
                                loc="upper right", fontsize=7, framealpha=0.7)

    def _update_magnetic(self, xp):
        """Compute and plot each active magnetic component on ax_mag."""
        # Pre-compute what's needed
        need_tmi = MagComponent.TMI in self.mag_components
        need_bx  = MagComponent.BX  in self.mag_components
        need_bz  = MagComponent.BZ  in self.mag_components

        tmi = np.zeros(len(xp))
        bx  = np.zeros(len(xp))
        bz  = np.zeros(len(xp))

        for body in self.bodies:
            if not (body.visible and body.is_complete()):
                continue
            if body.susceptibility == 0.0 and body.remanence_Am == 0.0:
                continue
            if self.is_2_5d:
                args = (xp, self.z_obs, body.vertices, body.susceptibility,
                        self.earth_field_nT, self.earth_inc_deg, self.earth_dec_deg,
                        body.remanence_Am, body.remanence_inc_deg, body.remanence_dec_deg)
                _bt, _bxbz = _bt_25d, _bxbz_25d
            else:
                args = (xp, body.vertices, body.susceptibility,
                        self.earth_field_nT, self.earth_inc_deg,
                        body.remanence_Am, body.remanence_inc_deg)
                _bt, _bxbz = _bt_2d, _bxbz_2d
            if need_tmi:
                tmi += _bt(*args)
            if need_bx or need_bz:
                bx_b, bz_b = _bxbz(*args)
                if need_bx: bx += bx_b
                if need_bz: bz += bz_b

        # Remove lines for components that are no longer active
        for comp in list(self._mag_lines):
            if comp not in self.mag_components:
                try:
                    self._mag_lines[comp].remove()
                except Exception:
                    pass
                del self._mag_lines[comp]

        # Plot / update each active component
        _styles = {
            MagComponent.TMI: ("-",  _MAG_COLOR),
            MagComponent.BX:  ("--", _MAG_COLOR),
            MagComponent.BZ:  ("-",  "#b05000"),   # dark orange to distinguish from TMI
        }
        _data = {MagComponent.TMI: tmi, MagComponent.BX: bx, MagComponent.BZ: bz}

        for comp in (MagComponent.TMI, MagComponent.BX, MagComponent.BZ):
            if comp not in self.mag_components:
                continue
            ls, col = _styles[comp]
            if comp in self._mag_lines:
                try:
                    self._mag_lines[comp].remove()
                except Exception:
                    pass
            line, = self.ax_mag.plot(
                xp, _data[comp], color=col, lw=1.5, ls=ls,
                label=f"{comp.value} (calc.)", zorder=3)
            self._mag_lines[comp] = line

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

    # --- full redraw ---─

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

    # --- in-progress draw artists ---

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

    # --- mouse event handlers ---

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
        if self.mode == Mode.DRAW: 
            self._handle_draw_press(x, z, btn)
        elif self.mode == Mode.SELECT: 
            self._handle_select_press(x, z)
        elif self.mode == Mode.ADD_VERTEX: 
            self._handle_add_vertex_press(x, z)
        elif self.mode == Mode.DELETE: 
            self._handle_delete_press(x, z)
        elif self.mode == Mode.MOVE_BODY: 
            self._handle_move_body_press(x, z)
        elif self.mode == Mode.DELETE_VERTEX: 
            self._handle_delete_vertex_press(x, z)

    def _on_release(self, _):
        if self._drag_active:
            self._drag_active = False
            self._hide_snap_ring()
            self.fig.canvas.draw_idle()
            self.bodies_changed.emit()

        if self._move_body_active:
            self._move_body_active = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
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
        elif self.mode in (Mode.SELECT, Mode.ADD_VERTEX) and self._drag_active:
            snap = self._find_snap_target(x, z,
                                          exclude_body=self.selected_body,
                                          exclude_vi=self.selected_vertex)
            if snap:
                self._show_snap_ring(*snap)
            else:
                self._hide_snap_ring()
            self._drag_vertex(x, z)
        elif self.mode == Mode.MOVE_BODY and self._move_body_active:
            self._drag_body(x, z)
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

    # --- draw mode ---

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

    # --- select mode ---─

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

    # --- add-vertex / delete mode ---

    def _handle_add_vertex_press(self, x, z):
        # If near an existing vertex, borrow move-vertex behaviour instead
        snap_vert = self._find_snap_vertex(x, z)
        if snap_vert is not None:
            body, vi = snap_vert
            self.selected_body   = body
            self.selected_vertex = vi
            self._drag_active    = True
            self._drag_prev      = (x, z)
            self._draw_selected_vertices()
            self.fig.canvas.draw_idle()
            return
        # Otherwise insert a new vertex on the nearest edge
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

    # --- move body mode ---

    def _handle_move_body_press(self, x, z):
        hit = self._pick_body(x, z)
        if hit is None:
            return
        self.selected_body     = hit
        self._move_body_active = True
        self._move_body_prev   = (x, z)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self._draw_selected_vertices()
        self.fig.canvas.draw_idle()

    def _drag_body(self, x, z):
        if self.selected_body is None or not self._move_body_active:
            return
        px, pz = self._move_body_prev
        dx, dz = x - px, z - pz
        for v in self.selected_body.vertices:
            v[0] += dx
            v[1] += dz
        self._move_body_prev = (x, z)
        self._refresh_patch(self.selected_body)
        self._draw_selected_vertices()
        self.fig.canvas.draw_idle()

    # --- delete vertex mode ---

    def _handle_delete_vertex_press(self, x, z):
        """Delete the nearest vertex on the clicked body (keep ≥ 3 vertices)."""
        hit = self._pick_body(x, z)
        if hit is None:
            # also try nearest vertex across all bodies if click missed interior
            hit_km = self._px_to_km(HIT_RADIUS_PX) * 3
            best, best_d = None, hit_km
            for body in self.bodies:
                if not body.is_complete() or not body.visible:
                    continue
                _, vd = body.nearest_vertex(x, z)
                if vd < best_d:
                    best_d, best = vd, body
            hit = best
        if hit is None or len(hit.vertices) <= 3:
            return
        vi, _ = hit.nearest_vertex(x, z)
        if vi < 0:
            return
        del hit.vertices[vi]
        self.selected_body = hit
        self._refresh_patch(hit)
        self._draw_selected_vertices()
        self.bodies_changed.emit()

    # --- hit testing ---─

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
            d = np.hypot(px - x, pz - z)
            if d < best_d:
                best_d, best = d, body
        return best

    # --- serialisation ---

    def to_dict(self) -> dict:
        # z_max is the first element of ylim on the (inverted) model axis
        z_max_km = float(self.ax_model.get_ylim()[0])
        return {
            "version":        2,
            # --- profile / domain ---
            "x_min":          self.x_min,
            "x_max":          self.x_max,
            "n_pts":          self.n_pts,
            "z_max_km":       z_max_km,
            "use_km":         self.use_km,
            # --- physical parameters ---
            "bg_density":     self.bg_density,
            "earth_field_nT": self.earth_field_nT,
            "earth_inc_deg":  self.earth_inc_deg,
            "earth_dec_deg":  self.earth_dec_deg,
            "z_obs":          self.z_obs,
            "is_2_5d":        self.is_2_5d,
            # --- bodies ---
            "bodies": [
                {"name":               b.name,
                 "density":            b.density,
                 "susceptibility":     b.susceptibility,
                 "remanence_Am":       b.remanence_Am,
                 "remanence_inc_deg":  b.remanence_inc_deg,
                 "remanence_dec_deg":  b.remanence_dec_deg,
                 "color":              b.color,
                 "visible":            b.visible,
                 "vertices":           [v[:] for v in b.vertices]}
                for b in self.bodies
            ],
        }

    def from_dict(self, d: dict):
        # --- profile / domain ---
        self.x_min          = float(d.get("x_min",          -50.0))
        self.x_max          = float(d.get("x_max",           50.0))
        self.n_pts          = int(d.get("n_pts",              201))
        z_max_km            = float(d.get("z_max_km",         20.0))
        self.use_km         = bool(d.get("use_km",             True))
        # --- physical parameters ---
        self.bg_density     = float(d.get("bg_density",     2670.0))
        self.earth_field_nT = float(d.get("earth_field_nT", 50000.0))
        self.earth_inc_deg  = float(d.get("earth_inc_deg",     60.0))
        self.earth_dec_deg  = float(d.get("earth_dec_deg",      0.0))
        self.z_obs          = float(d.get("z_obs",               0.0))
        self.is_2_5d        = bool(d.get("is_2_5d",              True))
        # --- bodies ---
        self.bodies.clear()
        PolygonBody._counter = 0
        for bd in d.get("bodies", []):
            b = PolygonBody(
                vertices          = [list(v) for v in bd["vertices"]],
                density           = float(bd["density"]),
                susceptibility    = float(bd.get("susceptibility",    0.001)),
                remanence_Am      = float(bd.get("remanence_Am",      0.0)),
                remanence_inc_deg = float(bd.get("remanence_inc_deg", 0.0)),
                remanence_dec_deg = float(bd.get("remanence_dec_deg", 0.0)),
                color             = bd.get("color",   "#4C72B0"),
                name              = bd.get("name",    "Body"),
                visible           = bool(bd.get("visible", True)),
            )
            b.name = bd.get("name", b.name)
            self.bodies.append(b)
        # --- restore axes ---
        self.ax_model.set_xlim(self.x_min, self.x_max)
        self.ax_grav.set_xlim(self.x_min, self.x_max)
        self.ax_model.set_ylim(z_max_km, -0.5)
        self._full_redraw()
        self.bodies_changed.emit()

    # --- external body management ---

    def add_body(self, body: PolygonBody):
        self.bodies.append(body)
        self._full_redraw()
        self.bodies_changed.emit()

    def remove_body(self, body: PolygonBody):
        if body in self.bodies:
            self.bodies.remove(body)
            if self.selected_body is body:
                self.selected_body   = None
                self.selected_vertex = -1
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