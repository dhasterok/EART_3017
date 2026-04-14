#!/usr/bin/env python3
"""
gravmag2d_gui.py
----------------
Interactive 2-D gravity + magnetic modelling GUI.

  - Draw and edit polygonal density/susceptibility bodies with the mouse.
  - Talwani (1959) gravity and Blakely (1995) magnetic forward models update
    in real time.
  - Gravity (mGal) and total-field magnetic anomaly (nT) share the profile
    panel: gravity on the left y-axis, magnetics on the right y-axis.
  - Controls in a left QDockWidget; polygon table in a bottom QDockWidget.

Usage
-----
    python gravmag2d_gui.py
"""

import sys
from pathlib import Path
import numpy as np

_HERE   = Path(__file__).resolve().parent          # src/apps/gravmag2d/
_COURSE = _HERE.parent.parent.parent               # project root
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from src.common.config import ICONPATH

from src.common.gui.polygon_editor_qt import EditMode, PolygonEditorActions
from src.apps.gravmag2d.gui.gm2d_types import DisplayMode, MagComponent, Mode, PolygonBody

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QStatusBar, QMessageBox, QFileDialog,
)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from src.apps.gravmag2d.gui.gm2d_control_dock import ControlsDock
from src.apps.gravmag2d.gui.gm2d_canvas import MainCanvas
from src.apps.gravmag2d.gui.gm2d_polygon_dock import PolygonDock
from src.apps.gravmag2d.gui.gm2d_inversion_dock import InversionDock

def _svg_icon(path: str, size: int = 64) -> QIcon:
    """
    Load an SVG file and return a QIcon, rasterising via QSvgRenderer.
    This works on Windows even without the Qt SVG image-format plugin.
    """
    renderer = QSvgRenderer(str(path))
    if not renderer.isValid():
        return QIcon()
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    from PyQt6.QtGui import QPainter
    painter = QPainter(pm)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return QIcon(pm)

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from src.physics.gravity.data_loader import load_csv_data, ObservedData


# DisplayMode, MagComponent, Mode, PolygonBody imported from gm2d_types
# DEFAULT_COLORS available via src.utils.polygon or src.gui.gm2d_types


# #--------------------
#  Main window
# #--------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2-D Gravity + Magnetic Modeller")   # ★
        self.resize(1200, 780)

        self.canvas = MainCanvas(self)
        nav_toolbar = NavigationToolbar2QT(self.canvas, self)

        central = QWidget()
        cv_layout = QVBoxLayout(central)
        cv_layout.setContentsMargins(0, 0, 0, 0)
        cv_layout.addWidget(nav_toolbar)
        cv_layout.addWidget(self.canvas)
        self.setCentralWidget(central)

        # --- controls dock (left) ---
        self.controls_dock = ControlsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.controls_dock)
        self.controls_dock.profile_changed.connect(
            lambda xn, xx, np_: self.canvas.set_profile(xn, xx, np_))
        self.controls_dock.depth_changed.connect(self.canvas.set_depth_range)
        self.controls_dock.units_changed.connect(self._on_units_changed)
        self.controls_dock.bg_density_changed.connect(self._on_bg_density_changed)
        self.controls_dock.earth_field_changed.connect(
            lambda F, IE, DE: self.canvas.set_earth_field(F, IE, DE))
        self.controls_dock.z_obs_changed.connect(self.canvas.set_z_obs)
        self.controls_dock.model_mode_changed.connect(self._on_model_mode_changed)

        # --- editing mode toolbar ---
        self._build_toolbars()

        # --- polygon table dock (bottom) ---
        self.table_dock = PolygonDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.table_dock)
        self.table_dock.setMinimumHeight(160)
        self.table_dock.body_selected.connect(self._on_table_body_selected)
        self.table_dock.body_changed.connect(self._on_table_body_changed)
        self.table_dock.body_vertex_changed.connect(self._on_vertex_edited)
        self.table_dock.body_vertex_deleted.connect(self._on_vertex_edited)
        self.table_dock.body_delete_requested.connect(self._delete_bodies)
        self.canvas.bodies_changed.connect(self._sync_table)
        self.canvas.delete_key_pressed.connect(self._confirm_delete_selected)

        # --- inversion dock (right, hidden by default) ---
        self.inv_dock = InversionDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inv_dock)
        self.inv_dock.hide()
        self.inv_dock.run_requested.connect(self._run_inversion)
        self.inv_dock.stop_requested.connect(self._stop_inversion)
        self.inv_dock.revert_requested.connect(self._revert_inversion)
        self._inv_worker   = None
        self._inv_snapshot = None

        # --- status bar ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._rms_label = QLabel("RMS: --")
        self.status.addPermanentWidget(self._rms_label)
        self.canvas.rms_updated.connect(
            lambda rms: self._rms_label.setText(f"RMS: {rms:.3f} mGal"))
        self.status.showMessage(
            "Select mode: click body to select | Draw mode: add vertex, RMB close")

        self._build_menu()
        self._sync_table()

    # --- toolbars ---

    def _build_toolbars(self):
        from PyQt6.QtWidgets import QToolBar
        from PyQt6.QtCore import QSize

        mode_tb = QToolBar("Editing Mode", self)
        mode_tb.setObjectName("mode_toolbar")
        mode_tb.setIconSize(QSize(24, 24))
        mode_tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, mode_tb)

        # --- polygon editing actions (from PolygonEditorActions) ---
        # These cover: New, Select, Move, Add Vert, Del Vert, Del Body + Snap
        self._poly_actions = PolygonEditorActions(ICONPATH, parent=self)
        self._poly_actions.mode_changed.connect(
            lambda em: self._on_mode_changed(Mode[em.name] if em is not None else Mode.NONE))
        self._poly_actions.snap_changed.connect(self.canvas.set_snap_enabled)
        self._poly_actions.add_to_toolbar(mode_tb, include_snap=True)

        # --- separator then Mask (application-specific, not a polygon op) ---
        mode_tb.addSeparator()

        # Mask is independent of the polygon action group — mutual exclusivity
        # is handled manually: checking mask deselects all polygon tools, and
        # checking a polygon tool unchecks mask (via _on_mode_changed).
        act_mask = QAction("Mask", self)
        act_mask.setIcon(_svg_icon(ICONPATH / "icon-mask-light-64.svg"))
        act_mask.setCheckable(True)
        act_mask.setToolTip("Toggle station mask — click stations in the gravity panel  [Ctrl+K]")
        act_mask.setShortcut(QKeySequence("Ctrl+K"))
        act_mask.triggered.connect(
            lambda checked: self._on_mode_changed(Mode.MASK if checked else Mode.NONE))
        mode_tb.addAction(act_mask)
        self._mask_action = act_mask

        # --- separator then display toggles (Gravity / Magnetics) ---
        mode_tb.addSeparator()

        act_grav = QAction("Gravity", self)
        act_grav.setIcon(_svg_icon(ICONPATH / "icon-gravity-64.svg"))
        act_grav.setCheckable(True)
        act_grav.setChecked(True)
        act_grav.setToolTip("Show / hide gravity profile")
        mode_tb.addAction(act_grav)
        self._act_show_gravity = act_grav

        act_mag = QAction("Magnetics", self)
        act_mag.setIcon(_svg_icon(ICONPATH / "icon-compass-64.svg"))
        act_mag.setCheckable(True)
        act_mag.setChecked(True)
        act_mag.setToolTip("Show / hide magnetics profile")
        mode_tb.addAction(act_mag)
        self._act_show_mag = act_mag

        act_grav.toggled.connect(lambda _: self._update_display_mode())
        act_mag.toggled.connect(lambda _: self._update_display_mode())

        # --- separator then magnetic component toggles ---
        mode_tb.addSeparator()

        self._mag_component_actions: dict[MagComponent, QAction] = {}
        _comp_tips = {
            MagComponent.TMI: "Total-field anomaly ΔT (solid line)",
            MagComponent.BX:  "Horizontal field Bx (dashed line)",
            MagComponent.BZ:  "Vertical field Bz (solid, dark orange line)",
        }
        for comp in (MagComponent.TMI, MagComponent.BX, MagComponent.BZ):
            act = QAction(comp.value, self)
            act.setCheckable(True)
            act.setToolTip(_comp_tips[comp])
            act.toggled.connect(
                lambda checked, c=comp: self.canvas.toggle_mag_component(c, checked))
            mode_tb.addAction(act)
            self._mag_component_actions[comp] = act
        self._mag_component_actions[MagComponent.TMI].setChecked(True)

    # --- menu ---

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
        act_export = QAction("&Export Forward Model CSV…", self)
        act_export.setShortcut("Ctrl+E")
        act_export.setToolTip(
            "Save the calculated gravity (gz) and magnetic fields (Bz, Bh, TMI) to a CSV file")
        act_export.triggered.connect(self._export_csv)
        file_menu.addAction(act_export)
        file_menu.addSeparator()
        act_quit = QAction("&Quit", self); act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close); file_menu.addAction(act_quit)

        edit_menu = mb.addMenu("&Edit")
        act_del = QAction("&Delete Selected Body", self); act_del.setShortcut("Delete")
        act_del.triggered.connect(self._delete_selected); edit_menu.addAction(act_del)

        view_menu = mb.addMenu("&View")
        view_menu.addAction(self.controls_dock.toggleViewAction())
        view_menu.addAction(self.table_dock.toggleViewAction())
        view_menu.addAction(self.inv_dock.toggleViewAction())

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

    # --- menu actions ---

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
        self.table_dock.show_vertices(None)

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

    # --- open data / save / load ---

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

            # --- observed data (optional) ---
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

    def _export_csv(self):
        """Export the calculated forward model fields to a CSV file."""
        import csv
        from src.physics.gravity.grav2_5d_model  import compute_gz    as _gz_25d
        from src.physics.gravity.talwani_model   import compute_gz    as _gz_2d
        from src.physics.magnetics.mag2_5d_model import compute_bt    as _bt_25d
        from src.physics.magnetics.mag2_5d_model import compute_bx_bz as _bxbz_25d
        from src.physics.magnetics.mag2d_model   import compute_bt    as _bt_2d
        from src.physics.magnetics.mag2d_model   import compute_bx_bz as _bxbz_2d

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Forward Model CSV", "",
            "CSV files (*.csv);;All files (*)")
        if not path:
            return

        canvas = self.canvas
        xp = canvas._profile_x()

        # Gravity (mGal)
        gz = np.zeros(len(xp))
        for body in canvas.bodies:
            if body.visible and body.is_complete():
                contrast = body.density - canvas.bg_density
                if canvas.is_2_5d:
                    gz += _gz_25d(xp, canvas.z_obs, body.vertices, contrast)
                else:
                    gz += _gz_2d(xp, body.vertices, contrast)

        # Magnetic components (nT)
        tmi = np.zeros(len(xp))
        bh  = np.zeros(len(xp))
        bz  = np.zeros(len(xp))
        for body in canvas.bodies:
            if not (body.visible and body.is_complete()):
                continue
            if body.susceptibility == 0.0 and body.remanence_Am == 0.0:
                continue
            if canvas.is_2_5d:
                args = (xp, canvas.z_obs, body.vertices, body.susceptibility,
                        canvas.earth_field_nT, canvas.earth_inc_deg, canvas.earth_dec_deg,
                        body.remanence_Am, body.remanence_inc_deg, body.remanence_dec_deg)
                _bt, _bxbz = _bt_25d, _bxbz_25d
            else:
                args = (xp, body.vertices, body.susceptibility,
                        canvas.earth_field_nT, canvas.earth_inc_deg,
                        body.remanence_Am, body.remanence_inc_deg)
                _bt, _bxbz = _bt_2d, _bxbz_2d
            tmi += _bt(*args)
            bh_b, bz_b = _bxbz(*args)
            bh += bh_b
            bz += bz_b

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["x_km", "gz_mGal", "Bh_nT", "Bz_nT", "TMI_nT"])
                for row in zip(xp, gz, bh, bz, tmi):
                    writer.writerow([f"{v:.6g}" for v in row])
            self.status.showMessage(
                f"Exported {len(xp)} points to {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

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

            # --- sync controls dock ---
            cd = self.controls_dock

            def _set(spin, value):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)

            _set(cd.spin_bg_density, self.canvas.bg_density)
            _set(cd.spin_field_F,    self.canvas.earth_field_nT)
            _set(cd.spin_field_IE,   self.canvas.earth_inc_deg)
            _set(cd.spin_field_DE,   self.canvas.earth_dec_deg)
            _set(cd.spin_z_obs,      self.canvas.z_obs)

            # model mode toggle (block to avoid double-triggering recompute)
            cd._model_toggle.blockSignals(True)
            cd._model_toggle.setChecked(self.canvas.is_2_5d)
            cd._is_2_5d = self.canvas.is_2_5d
            cd.spin_field_DE.setEnabled(self.canvas.is_2_5d)
            cd.spin_z_obs.setEnabled(self.canvas.is_2_5d)
            cd._model_toggle.blockSignals(False)
            self.table_dock.set_model_mode(self.canvas.is_2_5d)
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

            # --- observed data (optional) ---
            od = data.get("observed_data")
            if od is not None:
                from src.physics.gravity.data_loader import ObservedData
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

    # --- mode change ---─

    def _on_mode_changed(self, mode: Mode):
        self.canvas.set_mode(mode)
        # Sync toolbar check-states
        if mode == Mode.MASK:
            # Deselect all polygon tools, ensure mask is checked
            self._poly_actions.deselect_all()
            if not self._mask_action.isChecked():
                self._mask_action.blockSignals(True)
                self._mask_action.setChecked(True)
                self._mask_action.blockSignals(False)
        elif mode == Mode.NONE:
            # No tool active — deselect everything
            self._poly_actions.deselect_all()
            if self._mask_action.isChecked():
                self._mask_action.blockSignals(True)
                self._mask_action.setChecked(False)
                self._mask_action.blockSignals(False)
        else:
            # A polygon tool was activated — deselect mask, sync poly actions
            if self._mask_action.isChecked():
                self._mask_action.blockSignals(True)
                self._mask_action.setChecked(False)
                self._mask_action.blockSignals(False)
            from src.common.gui.polygon_editor_qt import EditMode as EM
            try:
                self._poly_actions.set_mode(EM[mode.name])
            except KeyError:
                pass
        labels = {
            Mode.NONE:          "Navigation: pan and zoom with the toolbar below",
            Mode.DRAW:          "Draw: add vertex | Enter close polygon | Esc cancel",
            Mode.SELECT:        "Move Vertex: pick body | drag a single vertex to move it",
            Mode.MOVE_BODY:     "Move Polygon: click and drag a polygon to a new location",
            Mode.ADD_VERTEX:    "Add Vertex: click on an edge to insert | click near vertex to move",
            Mode.DELETE_VERTEX: "Delete Vertex: click near a vertex to remove it (≥ 3 kept)",
            Mode.DELETE:        "Delete Polygon: click on a polygon to remove it",
            Mode.MASK:          "Mask: click on a station in the gravity panel to toggle mask",
        }
        self.status.showMessage(labels.get(mode, ""))

    # --- units / bg density ---

    def _update_display_mode(self):
        show_g = self._act_show_gravity.isChecked()
        show_m = self._act_show_mag.isChecked()
        if show_g and show_m:
            dm = DisplayMode.BOTH
        elif show_g:
            dm = DisplayMode.GRAVITY
        elif show_m:
            dm = DisplayMode.MAGNETICS
        else:
            dm = DisplayMode.NONE
        self.canvas.set_display_mode(dm)

    def _on_model_mode_changed(self, is_2_5d: bool):
        self.canvas.set_model_mode(is_2_5d)
        self.table_dock.set_model_mode(is_2_5d)

    def _on_units_changed(self, use_km: bool):
        self.canvas.set_units(use_km)
        self.table_dock.set_units(use_km)

    def _on_bg_density_changed(self, bg: float):
        self.canvas.set_bg_density(bg)
        self.table_dock.set_bg_density(bg)

    # --- table ↔ canvas ---

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

    # --- inversion ---

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


# #--------------------
#  Entry point
# #--------------------

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
