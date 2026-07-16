from __future__ import annotations
import os, sys, time, tempfile
from pathlib import Path
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from lame_core.CustomWidgets import ToggleSwitch

# ── path setup ────────────────────────────────────────────────────────────────
_COURSE = Path(__file__).resolve().parent.parent.parent.parent   # project root
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from src.physics.gravity.sphere import (
    forward_profile, forward_map, forward_sphere_r, jacobian_r, add_noise,
    forward_sphere_r_2p, forward_profile_2p, jacobian_r_2p,
)
from src.inversion.gauss_newton import gauss_newton, StopFlag
from src.common.grids.grid_worker import GridSearchWorker
from src.common.grids.global_grid_worker import GlobalGridWorker

FOUR_PI_OVER_3 = 4.0 * np.pi / 3.0


# ── Canvas widgets ─────────────────────────────────────────────────────────────
class MapModelCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(9, 8), tight_layout=True)
        self.ax_data  = fig.add_subplot(2, 1, 1)
        self.ax_model = fig.add_subplot(2, 1, 2, sharex=self.ax_data)
        super().__init__(fig)
        self.setParent(parent)
    def clear(self):
        self.ax_data.cla(); self.ax_model.cla()

class MisfitCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(10, 7), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self._layout = None
        self._cbs = []
        self.ax11 = self.ax12 = self.ax21 = self.ax22 = None
        self._configure_3p()

    def _configure_3p(self):
        self.figure.clear()
        self._cbs = []
        self.ax11 = self.figure.add_subplot(2, 2, 1)
        self.ax12 = self.figure.add_subplot(2, 2, 2)
        self.ax21 = self.figure.add_subplot(2, 2, 3)
        self.ax22 = self.figure.add_subplot(2, 2, 4, projection='3d')
        self._layout = '3p'

    def _configure_2p(self):
        self.figure.clear()
        self._cbs = []
        self.ax11 = self.figure.add_subplot(1, 2, 1)
        self.ax22 = self.figure.add_subplot(1, 2, 2, projection='3d')
        self.ax12 = None; self.ax21 = None
        self._layout = '2p'

    def ensure_3p(self):
        if self._layout != '3p':
            self._configure_3p()

    def ensure_2p(self):
        if self._layout != '2p':
            self._configure_2p()

    def clear(self):
        for cb in self._cbs:
            try: cb.remove()
            except Exception: pass
        self._cbs = []
        for ax in [self.ax11, self.ax12, self.ax21, self.ax22]:
            if ax is not None:
                ax.cla()


# ── Plotly isosurface tab ──────────────────────────────────────────────────────
class IsosurfaceWidget(QtWidgets.QWidget):
    """Plotly 3D isosurface rendered in an embedded QWebEngineView."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tmpfile = None
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            self._web = QWebEngineView(self)
            layout.addWidget(self._web)
            self._available = True
        except Exception as e:
            self._web = None
            self._available = False
            msg = QtWidgets.QLabel(
                f'Could not load QWebEngineView:\n{type(e).__name__}: {e}\n\n'
                'Install with:  pip install PyQt6-WebEngine'
            )
            msg.setWordWrap(True)
            msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(msg)
        self._show_placeholder()

    def _show_placeholder(self, msg='Run Global Search to display the misfit isosurface'):
        if not self._available:
            return
        self._web.setHtml(
            f'<html><body style="background:#1e1e2e;color:#cdd6f4;'
            f'font-family:sans-serif;display:flex;align-items:center;'
            f'justify-content:center;height:100vh;margin:0">'
            f'<h2>{msg}</h2></body></html>'
        )

    def update_isosurface(self, rms_3d, z_vals, R_vals, dr_vals,
                          global_min, p_true=None, p_est=None, pct=25.0):
        if not self._available:
            return
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except Exception as e:
            self._web.setHtml(
                f'<html><body style="color:red;font-family:sans-serif;padding:2em">'
                f'<h2>Could not import plotly</h2>'
                f'<pre>{type(e).__name__}: {e}</pre>'
                f'<p>Install with: <code>pip install plotly</code></p>'
                f'</body></html>'
            )
            return

        rms_mGal = rms_3d * 1e5
        rms_min  = float(rms_mGal.min())
        isomax   = rms_min * (1.0 + pct / 100.0)

        # rms_3d shape: (nz, nR, ndr) — build matching coordinate arrays
        z_3d, R_3d, dr_3d = np.meshgrid(z_vals, R_vals, dr_vals, indexing='ij')

        traces = [
            go.Isosurface(
                x=R_3d.flatten(), y=dr_3d.flatten(), z=z_3d.flatten(),
                value=rms_mGal.flatten(),
                isomin=rms_min, isomax=isomax, surface_count=3,
                opacity=0.4, colorscale='Plasma', showscale=True,
                colorbar=dict(title='RMS (mGal)', thickness=15),
                caps=dict(x_show=False, y_show=False, z_show=False),
                name='Isosurface',
            )
        ]
        if p_true is not None:
            traces.append(go.Scatter3d(
                x=[p_true[1]], y=[p_true[2]], z=[p_true[0]],
                name='True model', mode='markers',
                marker=dict(size=8, color='blue', symbol='circle'),
            ))
        traces.append(go.Scatter3d(
            x=[global_min['R']], y=[global_min['drho']], z=[global_min['z']],
            name=f"Global min  {global_min['rms']*1e5:.2e} mGal",
            mode='markers', marker=dict(size=8, color='red', symbol='circle'),
        ))
        if p_est is not None:
            traces.append(go.Scatter3d(
                x=[p_est[1]], y=[p_est[2]], z=[p_est[0]],
                name='Inversion', mode='markers',
                marker=dict(size=8, color='orange', symbol='circle'),
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis_title='R (m)', yaxis_title='Δρ (kg/m³)', zaxis_title='z (m)',
                zaxis=dict(autorange='reversed'),
            ),
            title=dict(
                text=(f'Misfit isosurface — {pct:.0f}% above min ({rms_min:.2e} mGal)'),
                font=dict(size=14),
            ),
            margin=dict(l=0, r=0, b=0, t=50), legend=dict(x=0, y=1),
        )
        self._load_html(pio.to_html(fig, include_plotlyjs=True, full_html=True))

    def _load_html(self, html: str):
        if self._tmpfile and os.path.exists(self._tmpfile):
            try: os.remove(self._tmpfile)
            except OSError: pass
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html)
            self._tmpfile = f.name
        from PyQt6.QtCore import QUrl
        self._web.load(QUrl.fromLocalFile(self._tmpfile))

    def closeEvent(self, event):
        if self._tmpfile and os.path.exists(self._tmpfile):
            try: os.remove(self._tmpfile)
            except OSError: pass
        super().closeEvent(event)


# ── 2-parameter (z, M) grid search worker ─────────────────────────────────────
class GridSearch2DWorker(QtCore.QObject):
    """Exhaustive 2-parameter (z, M) grid search."""
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(dict)
    aborted  = QtCore.pyqtSignal()

    def __init__(self, fwd_r, r_stations, d_obs, z_vals, M_vals, stop_flag):
        super().__init__()
        self.fwd_r      = fwd_r
        self.r_stations = r_stations
        self.d_obs      = d_obs
        self.z_vals     = z_vals
        self.M_vals     = M_vals
        self.stop_flag  = stop_flag

    def run(self):
        nz, nM = len(self.z_vals), len(self.M_vals)
        G = np.empty((nz, nM), dtype=float)
        total = max(nz * nM, 1)
        k = 0
        for iz, z in enumerate(self.z_vals):
            if self.stop_flag.stop:
                self.aborted.emit(); return
            for iM, M in enumerate(self.M_vals):
                p = np.array([z, M], float)
                g = self.fwd_r(self.r_stations, p)
                G[iz, iM] = np.sqrt(np.mean((self.d_obs - g) ** 2))
                k += 1
                if k % 200 == 0:
                    self.progress.emit(int(100 * k / total))
        self.progress.emit(100)
        iz_min, iM_min = np.unravel_index(np.argmin(G), G.shape)
        self.finished.emit({
            'G': G, 'z_vals': self.z_vals, 'M_vals': self.M_vals,
            'iz_min': int(iz_min), 'iM_min': int(iM_min),
        })


# ── Control dock ───────────────────────────────────────────────────────────────
class ControlDock(QtWidgets.QDockWidget):
    startGrid   = QtCore.pyqtSignal()
    startInv    = QtCore.pyqtSignal()
    startGlobal = QtCore.pyqtSignal()
    stopAll     = QtCore.pyqtSignal()
    regenData   = QtCore.pyqtSignal()
    modeChanged = QtCore.pyqtSignal(bool)   # True = 2-param mode

    def __init__(self, parent=None):
        super().__init__('Controls', parent)
        self._M_scale = 1e9   # updated dynamically in set_M_defaults()
        self._M_exp   = 9
        w = QtWidgets.QWidget(self)
        font = QtGui.QFont(); font.setPointSize(10)
        w.setFont(font)
        lay = QtWidgets.QVBoxLayout(w)
        AlignC = QtCore.Qt.AlignmentFlag.AlignHCenter

        # ── Parametrisation toggle ─────────────────────────────────────────────
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel('Parametrisation:'))
        mode_row.addStretch(1)
        mode_row.addWidget(QtWidgets.QLabel('3-param'))
        self.toggle_mode = ToggleSwitch(height=20)
        self.toggle_mode.setToolTip(
            'OFF = 3-parameter (z, R, Δρ)\nON = 2-parameter (z, M) '
            'where M = 4π/3·R³·Δρ is the excess mass'
        )
        mode_row.addWidget(self.toggle_mode)
        mode_row.addWidget(QtWidgets.QLabel('2-param'))
        lay.addLayout(mode_row)

        # ── True Model ────────────────────────────────────────────────────────
        synth = QtWidgets.QGroupBox('True Model')
        s = QtWidgets.QGridLayout(synth)
        self.spin_z  = QtWidgets.QDoubleSpinBox(); self.spin_z.setRange(1.0, 1e6);   self.spin_z.setDecimals(0);  self.spin_z.setValue(2000)
        self.spin_R  = QtWidgets.QDoubleSpinBox(); self.spin_R.setRange(1.0, 1e6);   self.spin_R.setDecimals(0);  self.spin_R.setValue(200)
        self.spin_dr = QtWidgets.QDoubleSpinBox(); self.spin_dr.setRange(-1e5, 1e5); self.spin_dr.setDecimals(0); self.spin_dr.setValue(1000)
        s.addWidget(QtWidgets.QLabel('z (m)'),       0, 0, AlignC)
        s.addWidget(QtWidgets.QLabel('R (m)'),       0, 1, AlignC)
        s.addWidget(QtWidgets.QLabel('Δρ (kg/m³)'), 0, 2, AlignC)
        s.addWidget(self.spin_z,  1, 0)
        s.addWidget(self.spin_R,  1, 1)
        s.addWidget(self.spin_dr, 1, 2)

        # ── Results ───────────────────────────────────────────────────────────
        res_grp = QtWidgets.QGroupBox('Results')
        r = QtWidgets.QGridLayout(res_grp)
        r.addWidget(QtWidgets.QLabel(''), 0, 0)
        r.addWidget(QtWidgets.QLabel('z (m)'), 0, 1, AlignC)
        self._res_hdr2 = QtWidgets.QLabel('R (m)');       r.addWidget(self._res_hdr2, 0, 2, AlignC)
        self._res_hdr3 = QtWidgets.QLabel('Δρ (kg/m³)'); r.addWidget(self._res_hdr3, 0, 3, AlignC)
        r.addWidget(QtWidgets.QLabel('RMS (mGal)'), 0, 4, AlignC)
        self._res_row_labels = []
        for row, name in enumerate(['Grid Δρ–R', 'Grid Δρ–z', 'Grid R–z', 'Global/2D', 'Inversion'], start=1):
            lbl = QtWidgets.QLabel(name)
            r.addWidget(lbl, row, 0)
            self._res_row_labels.append(lbl)
        self.lbl_grid_dr_R = [QtWidgets.QLabel('—') for _ in range(4)]
        self.lbl_grid_dr_z = [QtWidgets.QLabel('—') for _ in range(4)]
        self.lbl_grid_R_z  = [QtWidgets.QLabel('—') for _ in range(4)]
        self.lbl_global    = [QtWidgets.QLabel('—') for _ in range(4)]
        self.lbl_inv       = [QtWidgets.QLabel('—') for _ in range(4)]
        for col, lbl in enumerate(self.lbl_grid_dr_R, start=1): r.addWidget(lbl, 1, col, AlignC)
        for col, lbl in enumerate(self.lbl_grid_dr_z, start=1): r.addWidget(lbl, 2, col, AlignC)
        for col, lbl in enumerate(self.lbl_grid_R_z,  start=1): r.addWidget(lbl, 3, col, AlignC)
        for col, lbl in enumerate(self.lbl_global,    start=1): r.addWidget(lbl, 4, col, AlignC)
        for col, lbl in enumerate(self.lbl_inv,       start=1): r.addWidget(lbl, 5, col, AlignC)
        self._res_3p_rows = [self.lbl_grid_dr_R, self.lbl_grid_dr_z, self.lbl_grid_R_z]

        # ── QToolBox ──────────────────────────────────────────────────────────
        toolbox = QtWidgets.QToolBox()

        # ── Page 1: Sampling ──────────────────────────────────────────────────
        acq_page = QtWidgets.QWidget()
        a = QtWidgets.QGridLayout(acq_page)
        self.combo_geom = QtWidgets.QComboBox(); self.combo_geom.addItems(['Profile', 'Grid'])
        self.spin_dx    = QtWidgets.QDoubleSpinBox(); self.spin_dx.setRange(1.0, 1e6);   self.spin_dx.setDecimals(0);  self.spin_dx.setValue(50)
        self.spin_dy    = QtWidgets.QDoubleSpinBox(); self.spin_dy.setRange(1.0, 1e6);   self.spin_dy.setDecimals(0);  self.spin_dy.setValue(50)
        self.spin_xmax  = QtWidgets.QDoubleSpinBox(); self.spin_xmax.setRange(1.0, 1e6); self.spin_xmax.setDecimals(0); self.spin_xmax.setValue(2000)
        self.spin_ymax  = QtWidgets.QDoubleSpinBox(); self.spin_ymax.setRange(1.0, 1e6); self.spin_ymax.setDecimals(0); self.spin_ymax.setValue(2000)
        self.spin_sigma = QtWidgets.QDoubleSpinBox(); self.spin_sigma.setRange(0.0, 1e3); self.spin_sigma.setDecimals(4); self.spin_sigma.setSingleStep(1e-1); self.spin_sigma.setValue(1e-2)
        self.chk_weight = QtWidgets.QCheckBox('Use weighting (1/σ)'); self.chk_weight.setChecked(True)
        self.btn_regen  = QtWidgets.QPushButton('Generate')
        self.btn_regen.setToolTip('Generate synthetic data with the current true-model and noise settings')
        a.addWidget(QtWidgets.QLabel('Geometry'),          0, 0); a.addWidget(self.combo_geom, 0, 1, 1, 3)
        a.addWidget(QtWidgets.QLabel('dx (m)'),            1, 0); a.addWidget(self.spin_dx,    1, 1)
        a.addWidget(QtWidgets.QLabel('dy (m)'),            1, 2); a.addWidget(self.spin_dy,    1, 3)
        a.addWidget(QtWidgets.QLabel('x half-width (m)'), 2, 0); a.addWidget(self.spin_xmax,  2, 1)
        a.addWidget(QtWidgets.QLabel('y half-width (m)'), 2, 2); a.addWidget(self.spin_ymax,  2, 3)
        a.addWidget(QtWidgets.QLabel('Noise σ (mGal)'),   3, 0); a.addWidget(self.spin_sigma, 3, 1)
        a.addWidget(self.chk_weight,                       3, 2, 1, 2)
        a.addWidget(self.btn_regen,                        4, 0, 1, 4)
        toolbox.addItem(acq_page, 'Sampling')

        # ── Page 2: Grid Search (stacked 3-param / 2-param) ───────────────────
        grid_page = QtWidgets.QWidget()
        g_lay = QtWidgets.QVBoxLayout(grid_page)
        self.grid_stack = QtWidgets.QStackedWidget()

        # Stack page 0: 3-param grid table (z, R, Δρ)
        gs3 = QtWidgets.QWidget()
        gs3_lay = QtWidgets.QVBoxLayout(gs3)
        self.zmin  = QtWidgets.QDoubleSpinBox(); self.zmin.setRange(0.0, 1e6);   self.zmin.setDecimals(0);  self.zmin.setValue(200)
        self.zmax  = QtWidgets.QDoubleSpinBox(); self.zmax.setRange(0.0, 1e6);   self.zmax.setDecimals(0);  self.zmax.setValue(5000)
        self.zstep = QtWidgets.QDoubleSpinBox(); self.zstep.setRange(1.0, 1e6);  self.zstep.setDecimals(0); self.zstep.setValue(200)
        self.Rmin  = QtWidgets.QDoubleSpinBox(); self.Rmin.setRange(0.0, 1e6);   self.Rmin.setDecimals(0);  self.Rmin.setValue(50)
        self.Rmax  = QtWidgets.QDoubleSpinBox(); self.Rmax.setRange(0.0, 1e6);   self.Rmax.setDecimals(0);  self.Rmax.setValue(500)
        self.Rstep = QtWidgets.QDoubleSpinBox(); self.Rstep.setRange(1.0, 1e6);  self.Rstep.setDecimals(0); self.Rstep.setValue(25)
        self.drmin = QtWidgets.QDoubleSpinBox(); self.drmin.setRange(-1e5, 1e5); self.drmin.setDecimals(0); self.drmin.setValue(100)
        self.drmax = QtWidgets.QDoubleSpinBox(); self.drmax.setRange(-1e5, 1e5); self.drmax.setDecimals(0); self.drmax.setValue(3000)
        self.drstep= QtWidgets.QDoubleSpinBox(); self.drstep.setRange(1.0, 1e6); self.drstep.setDecimals(0);self.drstep.setValue(100)
        tbl3 = QtWidgets.QTableWidget(3, 3)
        tbl3.setHorizontalHeaderLabels(['z (m)', 'R (m)', 'Δρ (kg/m³)'])
        tbl3.setVerticalHeaderLabels(['min', 'max', 'step'])
        tbl3.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        tbl3.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tbl3.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl3.setCellWidget(0, 0, self.zmin);  tbl3.setCellWidget(1, 0, self.zmax);  tbl3.setCellWidget(2, 0, self.zstep)
        tbl3.setCellWidget(0, 1, self.Rmin);  tbl3.setCellWidget(1, 1, self.Rmax);  tbl3.setCellWidget(2, 1, self.Rstep)
        tbl3.setCellWidget(0, 2, self.drmin); tbl3.setCellWidget(1, 2, self.drmax); tbl3.setCellWidget(2, 2, self.drstep)
        gs3_lay.addWidget(tbl3)
        sl = QtWidgets.QHBoxLayout()
        self.combo_3d = QtWidgets.QComboBox(); self.combo_3d.addItems(['Δρ vs R', 'Δρ vs z', 'R vs z'])
        sl.addWidget(QtWidgets.QLabel('Overlay slice:')); sl.addWidget(self.combo_3d)
        gs3_lay.addLayout(sl)
        iso_row = QtWidgets.QHBoxLayout()
        self.spin_iso_thresh = QtWidgets.QDoubleSpinBox()
        self.spin_iso_thresh.setRange(1, 500); self.spin_iso_thresh.setDecimals(0)
        self.spin_iso_thresh.setValue(25); self.spin_iso_thresh.setSuffix(' %')
        self.spin_iso_thresh.setToolTip('Isosurface threshold: % above global RMS minimum')
        iso_row.addWidget(QtWidgets.QLabel('Isosurface threshold:')); iso_row.addWidget(self.spin_iso_thresh)
        gs3_lay.addLayout(iso_row)
        self.grid_stack.addWidget(gs3)   # index 0 = 3-param

        # Stack page 1: 2-param grid table (z, M)
        # z spinboxes are separate instances kept in sync with the 3-param ones
        gs2 = QtWidgets.QWidget()
        gs2_lay = QtWidgets.QVBoxLayout(gs2)
        self.zmin2  = QtWidgets.QDoubleSpinBox(); self.zmin2.setRange(0.0, 1e6);  self.zmin2.setDecimals(0);  self.zmin2.setValue(200)
        self.zmax2  = QtWidgets.QDoubleSpinBox(); self.zmax2.setRange(0.0, 1e6);  self.zmax2.setDecimals(0);  self.zmax2.setValue(5000)
        self.zstep2 = QtWidgets.QDoubleSpinBox(); self.zstep2.setRange(1.0, 1e6); self.zstep2.setDecimals(0); self.zstep2.setValue(200)
        self.Mmin  = QtWidgets.QDoubleSpinBox(); self.Mmin.setRange(0.0, 1e6);  self.Mmin.setDecimals(1);  self.Mmin.setSingleStep(1);  self.Mmin.setValue(1.0)
        self.Mmax  = QtWidgets.QDoubleSpinBox(); self.Mmax.setRange(0.0, 1e6);  self.Mmax.setDecimals(1);  self.Mmax.setSingleStep(5);  self.Mmax.setValue(200.0)
        self.Mstep = QtWidgets.QDoubleSpinBox(); self.Mstep.setRange(0.01, 1e5);self.Mstep.setDecimals(1); self.Mstep.setSingleStep(1); self.Mstep.setValue(2.0)
        # keep 2-param z range in sync with 3-param z range (bidirectional)
        self.zmin.valueChanged.connect(self.zmin2.setValue);   self.zmin2.valueChanged.connect(self.zmin.setValue)
        self.zmax.valueChanged.connect(self.zmax2.setValue);   self.zmax2.valueChanged.connect(self.zmax.setValue)
        self.zstep.valueChanged.connect(self.zstep2.setValue); self.zstep2.valueChanged.connect(self.zstep.setValue)
        tbl2 = QtWidgets.QTableWidget(3, 2)
        tbl2.setHorizontalHeaderLabels(['z (m)', 'M (×10⁹ kg)'])
        tbl2.setVerticalHeaderLabels(['min', 'max', 'step'])
        tbl2.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        tbl2.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tbl2.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl2.setCellWidget(0, 0, self.zmin2);  tbl2.setCellWidget(1, 0, self.zmax2);  tbl2.setCellWidget(2, 0, self.zstep2)
        tbl2.setCellWidget(0, 1, self.Mmin);   tbl2.setCellWidget(1, 1, self.Mmax);   tbl2.setCellWidget(2, 1, self.Mstep)
        gs2_lay.addWidget(tbl2)
        note = QtWidgets.QLabel('Note: R and Δρ cannot be separated — only excess mass M = 4π/3·R³·Δρ is recoverable.')
        note.setWordWrap(True); note.setStyleSheet('color: gray; font-size: 9pt;')
        gs2_lay.addWidget(note)
        self.grid_stack.addWidget(gs2)   # index 1 = 2-param

        g_lay.addWidget(self.grid_stack)

        # Buttons shared by both modes (Search always available; Global only meaningful in 3-param)
        btns = QtWidgets.QHBoxLayout()
        self.btn_search = QtWidgets.QPushButton('Search')
        self.btn_search.setToolTip('3-param: pairwise misfit grids  |  2-param: full z–M grid')
        self.btn_global = QtWidgets.QPushButton('Global')
        self.btn_global.setToolTip('Exhaustively search the full 3D parameter grid (3-param only)')
        btns.addWidget(self.btn_search); btns.addWidget(self.btn_global)
        g_lay.addLayout(btns)
        toolbox.addItem(grid_page, 'Grid Search')

        # ── Page 3: Inversion (stacked 3-param / 2-param) ────────────────────
        inv_page = QtWidgets.QWidget()
        inv_page_lay = QtWidgets.QVBoxLayout(inv_page)
        self.inv_stack = QtWidgets.QStackedWidget()

        # Stack page 0: 3-param start model
        ip3 = QtWidgets.QWidget()
        i3 = QtWidgets.QGridLayout(ip3)
        self.init_z  = QtWidgets.QDoubleSpinBox(); self.init_z.setRange(1.0, 1e6);   self.init_z.setDecimals(0);  self.init_z.setValue(500)
        self.init_R  = QtWidgets.QDoubleSpinBox(); self.init_R.setRange(1.0, 1e6);   self.init_R.setDecimals(0);  self.init_R.setValue(80)
        self.init_dr = QtWidgets.QDoubleSpinBox(); self.init_dr.setRange(-1e6, 1e6); self.init_dr.setDecimals(0); self.init_dr.setValue(2000)
        i3.addWidget(QtWidgets.QLabel('z₀ (m)'),       0, 0, AlignC)
        i3.addWidget(QtWidgets.QLabel('R₀ (m)'),       0, 1, AlignC)
        i3.addWidget(QtWidgets.QLabel('Δρ₀ (kg/m³)'), 0, 2, AlignC)
        i3.addWidget(self.init_z,  1, 0)
        i3.addWidget(self.init_R,  1, 1)
        i3.addWidget(self.init_dr, 1, 2)
        self.inv_stack.addWidget(ip3)   # index 0

        # Stack page 1: 2-param start model
        ip2 = QtWidgets.QWidget()
        i2 = QtWidgets.QGridLayout(ip2)
        self.init_z2 = QtWidgets.QDoubleSpinBox(); self.init_z2.setRange(1.0, 1e6);  self.init_z2.setDecimals(0); self.init_z2.setValue(500)
        self.init_M  = QtWidgets.QDoubleSpinBox(); self.init_M.setRange(0.001, 1e6); self.init_M.setDecimals(1);  self.init_M.setSingleStep(1); self.init_M.setValue(10.0)
        self.init_M.setToolTip('Starting excess mass M₀ in units of 10⁹ kg')
        i2.addWidget(QtWidgets.QLabel('z₀ (m)'),           0, 0, AlignC)
        i2.addWidget(QtWidgets.QLabel('M₀ (×10⁹ kg)'),    0, 1, AlignC)
        i2.addWidget(self.init_z2, 1, 0)
        i2.addWidget(self.init_M,  1, 1)
        self.inv_stack.addWidget(ip2)   # index 1

        inv_page_lay.addWidget(self.inv_stack)

        # Shared inversion controls
        ic = QtWidgets.QGridLayout()
        self.spin_lam   = QtWidgets.QDoubleSpinBox(); self.spin_lam.setRange(0.0, 1e3);  self.spin_lam.setDecimals(6); self.spin_lam.setValue(0.0)
        self.spin_lam.setToolTip('Damping parameter λ (0 = no damping)')
        self.spin_maxit = QtWidgets.QSpinBox();       self.spin_maxit.setRange(1, 1000);  self.spin_maxit.setValue(25)
        self.spin_tol   = QtWidgets.QSpinBox();       self.spin_tol.setRange(-12, -1);    self.spin_tol.setValue(-8)
        ic.addWidget(QtWidgets.QLabel('λ'),          0, 0); ic.addWidget(self.spin_lam,   0, 1, 1, 2)
        ic.addWidget(QtWidgets.QLabel('Max iters'),  1, 0); ic.addWidget(self.spin_maxit, 1, 1, 1, 2)
        ic.addWidget(QtWidgets.QLabel('Tol (10ⁿ)'), 2, 0); ic.addWidget(self.spin_tol,   2, 1, 1, 2)
        inv_page_lay.addLayout(ic)
        self.btn_inv  = QtWidgets.QPushButton('Invert')
        self.btn_stop = QtWidgets.QPushButton('Stop')
        inv_btns = QtWidgets.QHBoxLayout()
        inv_btns.addWidget(self.btn_inv); inv_btns.addWidget(self.btn_stop)
        inv_page_lay.addLayout(inv_btns)
        self.chk_ellipse = QtWidgets.QCheckBox('Show uncertainty ellipses (1σ)')
        self.chk_ellipse.setToolTip(
            'Draw 1σ confidence ellipses from the inversion covariance matrix\n'
            'on each pair of parameters in the misfit plots'
        )
        inv_page_lay.addWidget(self.chk_ellipse)
        toolbox.addItem(inv_page, 'Inversion (Gauss–Newton)')

        # ── Assemble ──────────────────────────────────────────────────────────
        lay.addWidget(synth)
        lay.addWidget(res_grp)
        lay.addWidget(toolbox)
        lay.addStretch(1)
        self.setWidget(w)

        # ── Signals ───────────────────────────────────────────────────────────
        self.btn_search.clicked.connect(self.startGrid)
        self.btn_global.clicked.connect(self.startGlobal)
        self.btn_inv.clicked.connect(self.startInv)
        self.btn_stop.clicked.connect(self.stopAll)
        self.btn_regen.clicked.connect(self.regenData)
        self.combo_geom.currentIndexChanged.connect(self._geom_changed)
        self.toggle_mode.stateChanged.connect(self._mode_changed)
        self._geom_changed()

    # ── Mode helpers ───────────────────────────────────────────────────────────
    def is_2p(self) -> bool:
        return self.toggle_mode.isChecked()

    def _mode_changed(self, checked: bool):
        self.grid_stack.setCurrentIndex(int(checked))
        self.inv_stack.setCurrentIndex(int(checked))
        self.btn_global.setEnabled(not checked)
        # Update Results column headers and hide 3-param-only rows
        if checked:
            self._res_hdr2.setText('M (×10⁹ kg)')
            self._res_hdr3.setText('')
            self._res_row_labels[0].setText('2D Search')
            for lbl_set in [self.lbl_grid_dr_z, self.lbl_grid_R_z]:
                for lbl in lbl_set: lbl.setVisible(False)
            for lbl in self._res_row_labels[1:3]: lbl.setVisible(False)
        else:
            self._res_hdr2.setText('R (m)')
            self._res_hdr3.setText('Δρ (kg/m³)')
            self._res_row_labels[0].setText('Grid Δρ–R')
            for lbl_set in [self.lbl_grid_dr_z, self.lbl_grid_R_z]:
                for lbl in lbl_set: lbl.setVisible(True)
            for lbl in self._res_row_labels[1:3]: lbl.setVisible(True)
        self.modeChanged.emit(checked)

    # ── M auto-scaling ─────────────────────────────────────────────────────────
    def set_M_defaults(self, M_true_si: float):
        """Set M spinbox ranges and units from the true model's excess mass (SI)."""
        if M_true_si <= 0:
            return
        exp = int(np.floor(np.log10(M_true_si))) - 1   # one decade below M_true
        self._M_exp   = exp
        self._M_scale = 10.0 ** exp
        suffix = f' ×10^{exp} kg'
        M_disp = M_true_si / self._M_scale
        for sb in [self.Mmin, self.Mmax, self.Mstep]:
            sb.blockSignals(True)
            sb.setSuffix(suffix); sb.setDecimals(2); sb.setSingleStep(max(0.01, M_disp * 0.05))
            sb.setRange(0.001, M_disp * 1000)
            sb.blockSignals(False)
        self.Mmin.setValue(max(0.001, M_disp * 0.01))
        self.Mmax.setValue(M_disp * 100)
        self.Mstep.setValue(max(0.001, M_disp * 0.02))
        self.init_M.blockSignals(True)
        self.init_M.setSuffix(suffix); self.init_M.setDecimals(2)
        self.init_M.setSingleStep(max(0.01, M_disp * 0.1))
        self.init_M.setRange(0.001, M_disp * 1000)
        self.init_M.blockSignals(False)
        self.init_M.setValue(M_disp)

    # ── Getters ────────────────────────────────────────────────────────────────
    def get_synthetic(self):
        return float(self.spin_z.value()), float(self.spin_R.value()), float(self.spin_dr.value())

    def get_noise(self):
        return float(self.spin_sigma.value()) * 1e-5   # mGal → m/s²

    def get_acq(self):
        return (self.combo_geom.currentText(),
                float(self.spin_dx.value()), float(self.spin_dy.value()),
                float(self.spin_xmax.value()), float(self.spin_ymax.value()))

    def get_grid_ranges(self):
        z_vals  = np.arange(float(self.zmin.value()),  float(self.zmax.value())  + 1e-12, float(self.zstep.value()))
        R_vals  = np.arange(float(self.Rmin.value()),  float(self.Rmax.value())  + 1e-12, float(self.Rstep.value()))
        dr_vals = np.arange(float(self.drmin.value()), float(self.drmax.value()) + 1e-12, float(self.drstep.value()))
        return z_vals, R_vals, dr_vals

    def get_grid_ranges_2p(self):
        z_vals = np.arange(float(self.zmin2.value()), float(self.zmax2.value()) + 1e-12, float(self.zstep2.value()))
        M_vals = np.arange(float(self.Mmin.value()), float(self.Mmax.value()) + 1e-12, float(self.Mstep.value())) * self._M_scale
        return z_vals, M_vals

    def get_inv_settings(self):
        p0    = np.array([float(self.init_z.value()), float(self.init_R.value()), float(self.init_dr.value())])
        lam   = float(self.spin_lam.value())
        maxit = int(self.spin_maxit.value())
        tol   = 10.0 ** self.spin_tol.value()
        return p0, lam, maxit, tol

    def get_inv_settings_2p(self):
        p0    = np.array([float(self.init_z2.value()), float(self.init_M.value()) * self._M_scale])
        lam   = float(self.spin_lam.value())
        maxit = int(self.spin_maxit.value())
        tol   = 10.0 ** self.spin_tol.value()
        return p0, lam, maxit, tol

    def get_weighting_enabled(self) -> bool:
        return bool(self.chk_weight.isChecked())

    def show_ellipses(self) -> bool:
        return bool(self.chk_ellipse.isChecked())

    def get_iso_threshold(self) -> float:
        return float(self.spin_iso_thresh.value())

    # ── Result display ─────────────────────────────────────────────────────────
    def _set_result_row(self, lbls, z, p2, p3, rms):
        lbls[0].setText(f'{z:.3g}')
        lbls[1].setText(f'{p2:.3g}')
        lbls[2].setText(f'{p3:.3g}' if p3 is not None else '—')
        lbls[3].setText(f'{rms:.2e}')

    def update_grid_results(self, grid_mins):
        R1, dr1, z1, rms1 = grid_mins['min_dr_R']
        self._set_result_row(self.lbl_grid_dr_R, z1, R1, dr1, rms1 * 1e5)
        z2, dr2, R2, rms2 = grid_mins['min_dr_z']
        self._set_result_row(self.lbl_grid_dr_z, z2, R2, dr2, rms2 * 1e5)
        z3, R3, dr3, rms3 = grid_mins['min_R_z']
        self._set_result_row(self.lbl_grid_R_z,  z3, R3, dr3, rms3 * 1e5)

    def update_2d_result(self, z, M_si, rms):
        """Update the first result row with the 2-param grid minimum."""
        self._set_result_row(self.lbl_grid_dr_R, z, M_si / self._M_scale, None, rms * 1e5)

    def update_global_result(self, best):
        self._set_result_row(self.lbl_global, best['z'], best['R'], best['drho'], best['rms'] * 1e5)

    def update_global_result_2p(self, z, M_si, rms):
        self._set_result_row(self.lbl_global, z, M_si / self._M_scale, None, rms * 1e5)

    def update_inv_result(self, p_est, rms):
        self._set_result_row(self.lbl_inv, p_est[0], p_est[1], p_est[2], rms * 1e5)

    def update_inv_result_2p(self, p_est, rms):
        self._set_result_row(self.lbl_inv, p_est[0], p_est[1] / self._M_scale, None, rms * 1e5)

    def reset_results(self):
        for lbls in (self.lbl_grid_dr_R, self.lbl_grid_dr_z, self.lbl_grid_R_z, self.lbl_global, self.lbl_inv):
            for lbl in lbls:
                lbl.setText('—')

    def _geom_changed(self):
        grid = (self.combo_geom.currentText() == 'Grid')
        self.spin_dy.setEnabled(grid)
        self.spin_ymax.setEnabled(grid)


# ── Main window ────────────────────────────────────────────────────────────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Buried Sphere — Grid Search vs Inversion')
        self.resize(1400, 900)
        self.tabs = QtWidgets.QTabWidget(self)
        self.canvas_data   = MapModelCanvas(self)
        self.canvas_misfit = MisfitCanvas(self)
        self.canvas_iso    = IsosurfaceWidget(self)
        self.tabs.addTab(self.canvas_data,   'Data && Model')
        self.tabs.addTab(self.canvas_misfit, 'Misfit')
        self.tabs.addTab(self.canvas_iso,    'Isosurface')
        self.setCentralWidget(self.tabs)
        self.dock = ControlDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        # State
        self.stop_flag    = StopFlag()
        self.p_true       = None
        self.p_est        = None      # [z, R, dr] in 3P; [z, M] in 2P
        self.Cov          = None
        self.grid_mins    = {}
        self.grid_2d_res  = None      # 2P grid result dict
        self.global_min   = None
        self.rms_3d       = None
        self.grid_3d_axes = None
        self.inv_path     = None
        self.r_stations   = None
        self.Xst = None; self.Yst = None
        self.d_obs   = None
        self.g_clean = None
        self.Wd      = None
        # Connect
        self.dock.startGrid.connect(self.on_grid)
        self.dock.startGlobal.connect(self.on_global)
        self.dock.startInv.connect(self.on_inverse)
        self.dock.stopAll.connect(self.on_stop)
        self.dock.regenData.connect(self.generate_data)
        self.dock.combo_3d.currentIndexChanged.connect(lambda _: self.draw_3d_surface())
        self.dock.spin_iso_thresh.valueChanged.connect(lambda _: self._refresh_isosurface())
        self.dock.modeChanged.connect(self._on_mode_changed)
        self.dock.chk_ellipse.stateChanged.connect(self._on_ellipse_toggled)
        self.status = self.statusBar()
        self.generate_data()

    # ── Mode switching ─────────────────────────────────────────────────────────
    def _on_mode_changed(self, is_2p: bool):
        iso_idx = self.tabs.indexOf(self.canvas_iso)
        self.tabs.setTabEnabled(iso_idx, not is_2p)
        if is_2p and self.tabs.currentIndex() == iso_idx:
            self.tabs.setCurrentIndex(1)
        if is_2p and self.p_true is not None:
            M_true = FOUR_PI_OVER_3 * self.p_true[1]**3 * self.p_true[2]
            self.dock.set_M_defaults(abs(M_true))
        # Clear results from the other parametrisation
        self.p_est = None
        self.grid_mins = {}
        self.grid_2d_res = None
        self.global_min = None
        self.rms_3d = None
        self.grid_3d_axes = None
        self.inv_path = None
        self.dock.reset_results()
        if is_2p:
            self.canvas_misfit.ensure_2p()
        else:
            self.canvas_misfit.ensure_3p()
        self.canvas_misfit.draw_idle()
        self._redraw_tab1()

    # ── Data generation ────────────────────────────────────────────────────────
    def generate_data(self):
        z, R, dr = self.dock.get_synthetic()
        self.p_true = np.array([z, R, dr], float)
        geom, dx, dy, xmax, ymax = self.dock.get_acq()
        sigma = self.dock.get_noise()
        if geom == 'Profile':
            x = np.arange(-xmax, xmax + 1e-12, dx)
            self.Xst, self.Yst = x, None
            self.g_clean = forward_profile(x, self.p_true)
            self.d_obs   = add_noise(self.g_clean, sigma)
            self.r_stations = np.abs(x)
        else:
            xs = np.arange(-xmax, xmax + 1e-12, dx)
            ys = np.arange(-ymax, ymax + 1e-12, dy)
            Xs, Ys = np.meshgrid(xs, ys)
            self.Xst, self.Yst = Xs, Ys
            self.g_clean = forward_map(Xs, Ys, self.p_true)
            self.d_obs   = add_noise(self.g_clean, sigma)
            self.r_stations = np.sqrt(Xs*Xs + Ys*Ys).ravel()
        if self.dock.get_weighting_enabled():
            w = 1.0 / max(float(sigma), 1e-20)
            self.Wd = np.full(self.d_obs.size, w, dtype=float)
        else:
            self.Wd = None
        self.p_est = None
        self.grid_mins = {}; self.grid_2d_res = None
        self.global_min = None; self.rms_3d = None; self.grid_3d_axes = None
        self.inv_path = None
        self.dock.reset_results()
        self._redraw_tab1()
        self.status.showMessage('Data generated.', 5000)

    # ── Tab-1 plotting helpers ─────────────────────────────────────────────────
    def _redraw_tab1(self):
        geom, dx, dy, xmax, ymax = self.dock.get_acq()
        if geom == 'Profile':
            x = np.arange(-float(self.dock.spin_xmax.value()),
                          float(self.dock.spin_xmax.value()) + 1e-12,
                          float(self.dock.spin_dx.value()))
            self.plot_tab1_profile(x)
        else:
            xs = np.arange(-float(self.dock.spin_xmax.value()),
                           float(self.dock.spin_xmax.value()) + 1e-12,
                           float(self.dock.spin_dx.value()))
            ys = np.arange(-float(self.dock.spin_ymax.value()),
                           float(self.dock.spin_ymax.value()) + 1e-12,
                           float(self.dock.spin_dy.value()))
            self.plot_tab1_map(xs, ys)

    def _forward_est(self, x):
        """Compute forward response for the current estimate (3P or 2P)."""
        if self.p_est is None:
            return None
        if self.dock.is_2p():
            return forward_profile_2p(x, self.p_est) * 1e5
        return forward_profile(x, self.p_est) * 1e5

    def _annotate_results(self, ax):
        lines = []
        if self.global_min is not None:
            b = self.global_min
            if self.dock.is_2p():
                sc = self.dock._M_scale; ex = self.dock._M_exp
                lines.append(f"2D min: z={b['z']:.3g}, M={b['M']/sc:.3g}×10^{ex} kg, RMS={b['rms']*1e5:.2e} mGal")
            else:
                lines.append(f"Global 3D min: z={b['z']:.3g}, R={b['R']:.3g}, Δρ={b['drho']:.3g}, RMS={b['rms']*1e5:.2e} mGal")
        if self.p_est is not None:
            if self.dock.is_2p():
                sc = self.dock._M_scale; ex = self.dock._M_exp
                lines.append(f'Inversion: z={self.p_est[0]:.3g}, M={self.p_est[1]/sc:.3g}×10^{ex} kg')
            else:
                z, R, dr = self.p_est
                lines.append(f'Inversion: z={z:.3g}, R={R:.3g}, Δρ={dr:.3g}')
        if lines:
            ax.text(0.02, 0.98, ' '.join(lines), transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='k'), fontsize=9)

    def _draw_sphere(self, ax, p, style, label):
        z, R, _ = p
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(R*np.cos(th), z + R*np.sin(th), style, lw=2, label=label)

    def _draw_sphere_2p(self, ax, z, M, style, label):
        """Draw sphere using equivalent radius for the given mass."""
        R_eq = (3.0 * abs(M) / (4.0 * np.pi * max(abs(float(self.p_true[2])), 1.0))) ** (1.0/3.0)
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(R_eq * np.cos(th), z + R_eq * np.sin(th), style, lw=2, label=label)

    def _overlay_path_on_model(self, ax):
        if self.inv_path is None:
            return
        path = np.asarray(self.inv_path)
        for k in range(len(path)):
            z, R = path[k, 0], path[k, 1]
            if not self.dock.is_2p():
                th = np.linspace(0, 2*np.pi, 60)
                ax.plot(R*np.cos(th), z + R*np.sin(th), color='orange', alpha=0.15)
        if not self.dock.is_2p():
            zf, Rf = path[-1, 0], path[-1, 1]
            th = np.linspace(0, 2*np.pi, 120)
            ax.plot(Rf*np.cos(th), zf + Rf*np.sin(th), color='orange', lw=2, label='Inv. path')

    def plot_tab1_profile(self, x):
        if self.d_obs is None:
            return
        self.canvas_data.clear()
        ax = self.canvas_data.ax_data
        ax.plot(x, self.d_obs * 1e5, 'ko', ms=3, label='Observed (noisy)')
        ax.plot(x, self.g_clean * 1e5, 'b-', lw=1.5, label='True')
        is_2p = self.dock.is_2p()
        if self.global_min is not None:
            b = self.global_min
            if is_2p:
                g_gm = forward_profile_2p(x, np.array([b['z'], b['M']]))
            else:
                g_gm = forward_profile(x, np.array([b['z'], b['R'], b['drho']]))
            ax.plot(x, g_gm * 1e5, 'm--', lw=1.5, label='Grid min')
        if not is_2p and self.grid_mins:
            which = self.dock.combo_3d.currentText()
            if which == 'Δρ vs R' and 'min_dr_R' in self.grid_mins:
                R1, dr1, z1, _ = self.grid_mins['min_dr_R']
                ax.plot(x, forward_profile(x, np.array([z1, R1, dr1])) * 1e5, 'C1--', lw=1.5, label='Slice min')
            elif which == 'Δρ vs z' and 'min_dr_z' in self.grid_mins:
                z2, dr2, R2, _ = self.grid_mins['min_dr_z']
                ax.plot(x, forward_profile(x, np.array([z2, R2, dr2])) * 1e5, 'C1--', lw=1.5, label='Slice min')
            elif which == 'R vs z' and 'min_R_z' in self.grid_mins:
                z3, R3, dr3, _ = self.grid_mins['min_R_z']
                ax.plot(x, forward_profile(x, np.array([z3, R3, dr3])) * 1e5, 'C1--', lw=1.5, label='Slice min')
        if self.p_est is not None:
            g_est = self._forward_est(x)
            ax.plot(x, g_est, 'r--', lw=1.5, label='Inversion')
        ax.set_xlabel(''); ax.set_ylabel('g (mGal)')
        ax.set_title('Profile gravity'); ax.tick_params(labelbottom=False); ax.legend(loc='best')
        axm = self.canvas_data.ax_model
        axm.axhline(0, color='k')
        self._draw_sphere(axm, self.p_true, 'b-', 'True')
        if self.global_min is not None:
            b = self.global_min
            if is_2p:
                self._draw_sphere_2p(axm, b['z'], b['M'], 'm--', 'Grid min')
            else:
                self._draw_sphere(axm, np.array([b['z'], b['R'], b['drho']]), 'm--', 'Global min')
        if self.p_est is not None:
            if is_2p:
                self._draw_sphere_2p(axm, self.p_est[0], self.p_est[1], 'r--', 'Inversion')
            else:
                self._draw_sphere(axm, self.p_est, 'r--', 'Inversion')
        self._overlay_path_on_model(axm)
        axm.set_xlabel('x (m)'); axm.set_ylabel('depth (m)'); axm.set_title('Buried sphere model')
        axm.set_aspect('equal', 'box')
        zt, Rt, _ = self.p_true
        axm.set_xlim([x[0], x[-1]]); axm.set_ylim([zt + 1.5*Rt, 0])
        try: axm.legend(loc='lower right')
        except Exception: pass
        self._annotate_results(axm)
        self.canvas_data.draw_idle()

    def plot_tab1_map(self, xs, ys):
        if self.d_obs is None:
            return
        self.canvas_data.clear()
        ax = self.canvas_data.ax_data
        im = ax.pcolormesh(xs, ys, self.d_obs * 1e5, shading='auto', cmap='viridis')
        self.canvas_data.figure.colorbar(im, ax=ax, shrink=0.9).set_label('g (mGal)')
        ax.set_xlabel(''); ax.set_ylabel('y (m)'); ax.tick_params(labelbottom=False)
        ax.set_aspect('equal', 'box'); ax.set_title('Map view: gravity (noisy)')
        axm = self.canvas_data.ax_model
        axm.axhline(0, color='k')
        self._draw_sphere(axm, self.p_true, 'b-', 'True')
        axm.set_xlabel('x (m)'); axm.set_ylabel('depth (m)'); axm.set_title('Buried sphere model')
        axm.set_aspect('equal', 'box')
        zt, Rt, _ = self.p_true
        axm.set_ylim([zt + 1.5*Rt, 0])
        try: axm.legend(loc='lower right')
        except Exception: pass
        self._annotate_results(axm)
        self.canvas_data.draw_idle()

    # ── Grid search (3-param pairwise) ────────────────────────────────────────
    def on_grid(self):
        if self.dock.is_2p():
            self._on_grid_2p()
            return
        z_vals, R_vals, dr_vals = self.dock.get_grid_ranges()
        p_fixed = self.p_est.copy() if self.p_est is not None else self.p_true.copy()
        r = self.r_stations; d_obs_vec = self.d_obs.ravel()
        self.stop_flag.stop = False
        self.thread = QtCore.QThread(self)
        self.worker = GridSearchWorker(forward_sphere_r, r, d_obs_vec, p_fixed, z_vals, R_vals, dr_vals, self.stop_flag)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda v: self.status.showMessage(f'Pairwise grid ~{v}%'))
        self.worker.finished.connect(self.on_grid_done)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_grid_done(self, res: dict):
        self.grid_res  = res
        self.grid_mins = {k: res[k] for k in ('min_dr_R', 'min_dr_z', 'min_R_z')}
        self.dock.update_grid_results(self.grid_mins)
        self.draw_misfit_tab(res)
        self._redraw_tab1()
        self.status.showMessage('Pairwise misfit grids computed.', 3000)

    # ── Grid search (2-param z, M) ────────────────────────────────────────────
    def _on_grid_2p(self):
        z_vals, M_vals = self.dock.get_grid_ranges_2p()
        self.stop_flag.stop = False
        self.thread2 = QtCore.QThread(self)
        self.worker2 = GridSearch2DWorker(
            forward_sphere_r_2p, self.r_stations, self.d_obs.ravel(),
            z_vals, M_vals, self.stop_flag
        )
        self.worker2.moveToThread(self.thread2)
        self.thread2.started.connect(self.worker2.run)
        self.worker2.progress.connect(lambda v: self.status.showMessage(f'2D grid ~{v}%'))
        self.worker2.finished.connect(self._on_grid_2p_done)
        self.worker2.finished.connect(self.thread2.quit)
        self.worker2.finished.connect(self.worker2.deleteLater)
        self.thread2.finished.connect(self.thread2.deleteLater)
        self.thread2.start()

    def _on_grid_2p_done(self, res: dict):
        self.grid_2d_res = res
        iz, iM = res['iz_min'], res['iM_min']
        z_best = res['z_vals'][iz]; M_best = res['M_vals'][iM]
        rms_best = res['G'][iz, iM]
        self.global_min = {'z': z_best, 'M': M_best, 'R': None, 'drho': None, 'rms': rms_best}
        self.dock.update_2d_result(z_best, M_best, rms_best)
        self.dock.update_global_result_2p(z_best, M_best, rms_best)
        self.draw_misfit_tab_2p(res)
        self._redraw_tab1()
        sc = self.dock._M_scale; ex = self.dock._M_exp
        self.status.showMessage(
            f'2D grid done: z={z_best:.3g} m, M={M_best/sc:.3g}×10^{ex} kg, '
            f'RMS={rms_best*1e5:.2e} mGal', 5000
        )

    # ── Global 3-param search ──────────────────────────────────────────────────
    def on_global(self):
        z_vals, R_vals, dr_vals = self.dock.get_grid_ranges()
        self.stop_flag.stop = False
        self.threadG = QtCore.QThread(self)
        self.workerG = GlobalGridWorker(
            forward_sphere_r, self.r_stations, self.d_obs.ravel(),
            z_vals, R_vals, dr_vals, self.stop_flag
        )
        self.workerG.moveToThread(self.threadG)
        self.threadG.started.connect(self.workerG.run)
        self.workerG.progress.connect(lambda v: self.status.showMessage(f'Global 3D grid ~{v}%'))
        self.workerG.finished.connect(self.on_global_done)
        self.workerG.finished.connect(self.threadG.quit)
        self.workerG.finished.connect(self.workerG.deleteLater)
        self.threadG.finished.connect(self.threadG.deleteLater)
        self.threadG.start()

    def on_global_done(self, res: dict):
        self.global_min   = res['best']
        self.rms_3d       = res['rms_3d']
        self.grid_3d_axes = {k: res[k] for k in ('z_vals', 'R_vals', 'dr_vals')}
        self.dock.update_global_result(self.global_min)
        self.status.showMessage(
            f"Global min: z={self.global_min['z']:.3g}, R={self.global_min['R']:.3g}, "
            f"Δρ={self.global_min['drho']:.3g}, RMS={self.global_min['rms']*1e5:.2e} mGal", 6000
        )
        self._refresh_isosurface()
        self._redraw_tab1()

    # ── Inversion ──────────────────────────────────────────────────────────────
    def on_inverse(self):
        is_2p = self.dock.is_2p()
        if is_2p:
            p0, lam, maxit, tol = self.dock.get_inv_settings_2p()
            fwd  = lambda _r, p: forward_sphere_r_2p(_r, p)
            jac  = lambda _r, p: jacobian_r_2p(_r, p)
        else:
            p0, lam, maxit, tol = self.dock.get_inv_settings()
            fwd  = lambda _r, p: forward_sphere_r(_r, p)
            jac  = lambda _r, p: jacobian_r(_r, p)
        r = self.r_stations; d = self.d_obs.ravel()
        t0 = time.time()
        p_est, Cov, hist = gauss_newton(fwd, jac, r, d, p0=p0, lam=lam,
                                         maxit=maxit, tol=tol, Wd=self.Wd,
                                         stop_flag=self.stop_flag)
        t1 = time.time()
        self.p_est, self.Cov = p_est, Cov
        self.inv_path = np.asarray(hist['path'])
        if is_2p:
            self.dock.update_inv_result_2p(self.p_est, hist['rms'][-1])
        else:
            self.dock.update_inv_result(self.p_est, hist['rms'][-1])
        self.status.showMessage(
            f'Inversion done in {t1-t0:.3f}s; RMS={hist["rms"][-1]*1e5:.3e} mGal', 4000
        )
        if not is_2p and hasattr(self, 'grid_res'):
            self.draw_misfit_tab(self.grid_res)
        elif is_2p and self.grid_2d_res is not None:
            self.draw_misfit_tab_2p(self.grid_2d_res)
        self._refresh_isosurface()
        self._redraw_tab1()

    def on_stop(self):
        self.stop_flag.stop = True
        self.status.showMessage('Stop requested…', 1500)

    # ── Uncertainty ellipses ───────────────────────────────────────────────────
    @staticmethod
    def _confidence_ellipse(ax, cx, cy, cov2, n_std=1.0, **kw):
        """Add a confidence ellipse to ax from a 2×2 covariance submatrix."""
        from matplotlib.patches import Ellipse
        vals, vecs = np.linalg.eigh(cov2)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w = 2.0 * n_std * np.sqrt(max(vals[0], 0.0))
        h = 2.0 * n_std * np.sqrt(max(vals[1], 0.0))
        ell = Ellipse(xy=(cx, cy), width=w, height=h, angle=angle, **kw)
        ax.add_patch(ell)
        return ell

    def _draw_ellipses_3p(self):
        """Overlay 1σ covariance ellipses on the three pairwise misfit panels."""
        if self.Cov is None or self.p_est is None:
            return
        C = self.Cov
        z_e, R_e, dr_e = self.p_est
        kw = dict(fill=False, edgecolor='cyan', linewidth=1.5, linestyle='--', zorder=6)
        # ax11: R (col 1) vs Δρ (col 2)  — x=R, y=Δρ
        sub11 = C[np.ix_([1, 2], [1, 2])]
        self._confidence_ellipse(self.canvas_misfit.ax11, R_e, dr_e, sub11, **kw)
        # ax12: z (col 0) vs Δρ (col 2)  — x=z, y=Δρ
        sub12 = C[np.ix_([0, 2], [0, 2])]
        self._confidence_ellipse(self.canvas_misfit.ax12, z_e, dr_e, sub12, **kw)
        # ax21: z (col 0) vs R (col 1)   — x=z, y=R
        sub21 = C[np.ix_([0, 1], [0, 1])]
        self._confidence_ellipse(self.canvas_misfit.ax21, z_e, R_e, sub21, **kw)

    def _draw_ellipses_2p(self, sc):
        """Overlay 1σ covariance ellipse on the z–M misfit panel (2-param)."""
        if self.Cov is None or self.p_est is None:
            return
        C = self.Cov
        z_e, M_e = self.p_est
        # x-axis = M (displayed in display units), y-axis = z
        # Cov is [[σ²z, σzM],[σzM, σ²M]], need to scale M axis by 1/sc
        S = np.diag([1.0 / sc, 1.0])   # transform: (M_si, z) → (M_disp, z)
        # cov in display space: note parameter order is [z, M] in p_est/Cov
        # axes are (M_disp, z) so we need to swap rows/cols 0↔1 and scale
        C_swap = C[np.ix_([1, 0], [1, 0])]         # now [[σ²M, σzM],[σzM, σ²z]]
        S2 = np.diag([1.0 / sc, 1.0])
        C_disp = S2 @ C_swap @ S2.T
        kw = dict(fill=False, edgecolor='cyan', linewidth=1.5, linestyle='--', zorder=6)
        self._confidence_ellipse(self.canvas_misfit.ax11, M_e / sc, z_e, C_disp, **kw)

    def _on_ellipse_toggled(self, _=None):
        if self.dock.is_2p():
            if self.grid_2d_res is not None:
                self.draw_misfit_tab_2p(self.grid_2d_res)
        else:
            if hasattr(self, 'grid_res'):
                self.draw_misfit_tab(self.grid_res)

    # ── Misfit tab — 3-param ───────────────────────────────────────────────────
    def draw_misfit_tab(self, res):
        def _loglevels(G, n=20):
            gmin = max(float(G.min()), 1e-30); gmax = float(G.max())
            return n if gmax <= gmin else np.logspace(np.log10(gmin), np.log10(gmax), n)

        self.canvas_misfit.ensure_3p()
        self.canvas_misfit.clear()
        Rv  = res['R_vals'];  drv  = res['dr_vals'];  G1 = res['G_dr_R'] * 1e5
        zv  = res['z_vals'];  drv2 = res['dr_vals2']; G2 = res['G_dr_z'] * 1e5
        zv2 = res['z_vals2']; Rv2  = res['R_vals2'];  G3 = res['G_R_z']  * 1e5

        CS1 = self.canvas_misfit.ax11.contour(Rv, drv, G1, levels=_loglevels(G1), colors='k', linewidths=0.8)
        self.canvas_misfit.ax11.clabel(CS1, inline=True, fontsize=7, fmt='%.2e')
        self.canvas_misfit.ax11.set_xlabel('R (m)'); self.canvas_misfit.ax11.set_ylabel('Δρ (kg/m³)')
        self.canvas_misfit.ax11.set_title('Misfit: Δρ vs R')
        i1, j1 = res['idx_dr_R']; self.canvas_misfit.ax11.plot([Rv[i1]], [drv[j1]], 'r*', ms=10)

        CS2 = self.canvas_misfit.ax12.contour(zv, drv2, G2, levels=_loglevels(G2), colors='k', linewidths=0.8)
        self.canvas_misfit.ax12.clabel(CS2, inline=True, fontsize=7, fmt='%.2e')
        self.canvas_misfit.ax12.set_xlabel('z (m)'); self.canvas_misfit.ax12.set_ylabel('Δρ (kg/m³)')
        self.canvas_misfit.ax12.set_title('Misfit: Δρ vs z')
        i2, j2 = res['idx_dr_z']; self.canvas_misfit.ax12.plot([zv[i2]], [drv2[j2]], 'r*', ms=10)

        CS3 = self.canvas_misfit.ax21.contour(zv2, Rv2, G3, levels=_loglevels(G3), colors='k', linewidths=0.8)
        self.canvas_misfit.ax21.clabel(CS3, inline=True, fontsize=7, fmt='%.2e')
        self.canvas_misfit.ax21.set_xlabel('z (m)'); self.canvas_misfit.ax21.set_ylabel('R (m)')
        self.canvas_misfit.ax21.set_title('Misfit: R vs z')
        i3, j3 = res['idx_R_z']; self.canvas_misfit.ax21.plot([zv2[i3]], [Rv2[j3]], 'r*', ms=10)

        if self.p_est is not None and not self.dock.is_2p():
            z_est, R_est, dr_est = self.p_est
            self.canvas_misfit.ax11.plot([R_est],  [dr_est], 'y*', ms=12, zorder=5)
            self.canvas_misfit.ax12.plot([z_est],  [dr_est], 'y*', ms=12, zorder=5)
            self.canvas_misfit.ax21.plot([z_est],  [R_est],  'y*', ms=12, zorder=5)
        if self.inv_path is not None and not self.dock.is_2p():
            path = np.asarray(self.inv_path)
            self.canvas_misfit.ax11.plot(path[:,1], path[:,2], 'r.-', lw=1.0, ms=4)
            self.canvas_misfit.ax12.plot(path[:,0], path[:,2], 'r.-', lw=1.0, ms=4)
            self.canvas_misfit.ax21.plot(path[:,0], path[:,1], 'r.-', lw=1.0, ms=4)
        if self.dock.show_ellipses():
            self._draw_ellipses_3p()

        self.draw_3d_surface()
        self.canvas_misfit.draw_idle()

    # ── Misfit tab — 2-param ───────────────────────────────────────────────────
    def draw_misfit_tab_2p(self, res):
        def _loglevels(G, n=20):
            gmin = max(float(G.min()), 1e-30); gmax = float(G.max())
            return n if gmax <= gmin else np.logspace(np.log10(gmin), np.log10(gmax), n)

        self.canvas_misfit.ensure_2p()
        self.canvas_misfit.clear()
        sc  = self.dock._M_scale
        ex  = self.dock._M_exp
        G   = res['G'] * 1e5
        zv  = res['z_vals']
        Mv  = res['M_vals'] / sc   # display units

        M_true = FOUR_PI_OVER_3 * self.p_true[1]**3 * self.p_true[2]
        x_unit = f'M (×10^{ex} kg)'

        # ax11: filled color + contour of z-M misfit
        ax = self.canvas_misfit.ax11
        im = ax.pcolormesh(Mv, zv, G, shading='auto', cmap='viridis_r')
        cb = self.canvas_misfit.figure.colorbar(im, ax=ax)
        cb.set_label('RMS (mGal)')
        self.canvas_misfit._cbs.append(cb)
        CS = ax.contour(Mv, zv, G, levels=_loglevels(G), colors='w', linewidths=0.6, alpha=0.7)
        ax.clabel(CS, inline=True, fontsize=7, fmt='%.2e', colors='w')
        iz, iM = res['iz_min'], res['iM_min']
        ax.plot([Mv[iM]], [zv[iz]], 'r*', ms=12, label='Grid min')
        if self.p_est is not None:
            ax.plot([self.p_est[1] / sc], [self.p_est[0]], 'y*', ms=12, label='Inversion')
        ax.plot([M_true / sc], [self.p_true[0]], 'b*', ms=12, label='True')
        if self.dock.show_ellipses():
            self._draw_ellipses_2p(sc)
        ax.set_xlabel(x_unit); ax.set_ylabel('z (m)')
        ax.set_title('Misfit: z vs M (excess mass)')
        ax.invert_yaxis(); ax.legend(fontsize=8)

        # ax22: 3D misfit surface
        ax3d = self.canvas_misfit.ax22
        MV, ZV = np.meshgrid(Mv, zv)
        ax3d.plot_surface(MV, ZV, G, cmap='viridis_r', rstride=1, cstride=1, linewidth=0, alpha=0.8)
        if self.p_est is not None:
            ax3d.scatter([self.p_est[1] / sc], [self.p_est[0]], [G[iz, iM]],
                         color='orange', s=60, zorder=5, label='Inversion')
        ax3d.scatter([M_true / sc], [self.p_true[0]],
                     [float(res['G'][np.argmin(np.abs(zv - self.p_true[0])),
                                    np.argmin(np.abs(res['M_vals'] - M_true))] * 1e5)],
                     color='blue', s=60, zorder=5, label='True')
        ax3d.set_xlabel(x_unit, fontsize=8); ax3d.set_ylabel('z (m)', fontsize=8)
        ax3d.set_zlabel('RMS (mGal)', fontsize=8); ax3d.set_title('Misfit surface', fontsize=9)
        ax3d.invert_yaxis()

        self.canvas_misfit.draw_idle()

    # ── 3D misfit panel (ax22, 3-param only) ──────────────────────────────────
    def draw_3d_surface(self, res=None):
        if res is None:
            res = getattr(self, 'grid_res', None)
        if res is not None:
            self._draw_pairwise_surface(res)

    def _draw_pairwise_surface(self, res):
        Rv  = res['R_vals'];  drv  = res['dr_vals'];  G1 = res['G_dr_R'] * 1e5
        zv  = res['z_vals'];  drv2 = res['dr_vals2']; G2 = res['G_dr_z'] * 1e5
        zv2 = res['z_vals2']; Rv2  = res['R_vals2'];  G3 = res['G_R_z']  * 1e5
        ax3d = self.canvas_misfit.ax22; ax3d.cla()
        which = self.dock.combo_3d.currentText()
        if which == 'Δρ vs R':
            X, Y = np.meshgrid(Rv, drv)
            ax3d.plot_surface(X, Y, G1, cmap='viridis', rstride=1, cstride=1, linewidth=0)
            ax3d.set_xlabel('R (m)'); ax3d.set_ylabel('Δρ (kg/m³)')
        elif which == 'Δρ vs z':
            X, Y = np.meshgrid(zv, drv2)
            ax3d.plot_surface(X, Y, G2, cmap='viridis', rstride=1, cstride=1, linewidth=0)
            ax3d.set_xlabel('z (m)'); ax3d.set_ylabel('Δρ (kg/m³)')
        else:
            X, Y = np.meshgrid(zv2, Rv2)
            ax3d.plot_surface(X, Y, G3, cmap='viridis', rstride=1, cstride=1, linewidth=0)
            ax3d.set_xlabel('z (m)'); ax3d.set_ylabel('R (m)')
        ax3d.set_zlabel('RMS (mGal)')
        ax3d.set_title(f'Misfit surface — {which}\n(run Global for 3D isosurface)', fontsize=8)
        self.canvas_misfit.draw_idle()

    # ── Plotly isosurface (3-param only) ──────────────────────────────────────
    def _refresh_isosurface(self):
        if self.dock.is_2p() or self.rms_3d is None or self.global_min is None:
            return
        self.canvas_iso.update_isosurface(
            self.rms_3d,
            self.grid_3d_axes['z_vals'], self.grid_3d_axes['R_vals'], self.grid_3d_axes['dr_vals'],
            self.global_min, p_true=self.p_true, p_est=self.p_est,
            pct=self.dock.get_iso_threshold(),
        )


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
