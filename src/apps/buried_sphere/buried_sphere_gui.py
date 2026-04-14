from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── path setup ────────────────────────────────────────────────────────────────
_COURSE = Path(__file__).resolve().parent.parent.parent   # new_version/
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from src.physics.gravity.sphere import forward_profile, forward_map, forward_sphere_r, jacobian_r, add_noise
from src.inversion.gauss_newton import gauss_newton, StopFlag
from src.common.grids.grid_worker import GridSearchWorker
from src.common.grids.global_grid_worker import GlobalGridWorker

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
        fig = Figure(figsize=(10,7), tight_layout=True)
        self.ax11 = fig.add_subplot(2,2,1)
        self.ax12 = fig.add_subplot(2,2,2)
        self.ax21 = fig.add_subplot(2,2,3)
        self.ax22 = fig.add_subplot(2,2,4, projection='3d')
        super().__init__(fig)
        self.setParent(parent)
        self.cb11 = None; self.cb12 = None; self.cb21 = None
    def clear(self):
        for cb in [self.cb11, self.cb12, self.cb21]:
            try: cb.remove()
            except Exception: pass
        self.cb11 = self.cb12 = self.cb21 = None
        self.ax11.cla(); self.ax12.cla(); self.ax21.cla(); self.ax22.cla()

class ControlDock(QtWidgets.QDockWidget):
    startGrid = QtCore.pyqtSignal()
    startInv  = QtCore.pyqtSignal()
    startGlobal = QtCore.pyqtSignal()
    stopAll   = QtCore.pyqtSignal()
    regenData = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__('Controls', parent)
        w = QtWidgets.QWidget(self)
        font = QtGui.QFont(); font.setPointSize(10)
        w.setFont(font)
        lay = QtWidgets.QVBoxLayout(w)
        AlignC = QtCore.Qt.AlignmentFlag.AlignHCenter

        # --- Synthetic Sphere ---
        synth = QtWidgets.QGroupBox('True Model')
        s = QtWidgets.QGridLayout(synth)
        self.spin_z = QtWidgets.QDoubleSpinBox(); self.spin_z.setRange(1.0, 1e6);    self.spin_z.setDecimals(0);  self.spin_z.setValue(2000)
        self.spin_R = QtWidgets.QDoubleSpinBox(); self.spin_R.setRange(1.0, 1e6);    self.spin_R.setDecimals(0);  self.spin_R.setValue(200)
        self.spin_dr= QtWidgets.QDoubleSpinBox(); self.spin_dr.setRange(-1e5, 1e5);  self.spin_dr.setDecimals(0); self.spin_dr.setValue(1000)
        s.addWidget(QtWidgets.QLabel('z (m)'),       0, 0, AlignC)
        s.addWidget(QtWidgets.QLabel('R (m)'),       0, 1, AlignC)
        s.addWidget(QtWidgets.QLabel('Δρ (kg/m³)'), 0, 2, AlignC)
        s.addWidget(self.spin_z,  1, 0)
        s.addWidget(self.spin_R,  1, 1)
        s.addWidget(self.spin_dr, 1, 2)

        # --- Acquisition & Noise ---
        acq = QtWidgets.QGroupBox('Sampling')
        a = QtWidgets.QGridLayout(acq)
        self.combo_geom = QtWidgets.QComboBox(); self.combo_geom.addItems(['Profile','Grid'])
        self.spin_dx  = QtWidgets.QDoubleSpinBox(); self.spin_dx.setRange(1.0, 1e6);   self.spin_dx.setDecimals(0);  self.spin_dx.setValue(50)
        self.spin_dy  = QtWidgets.QDoubleSpinBox(); self.spin_dy.setRange(1.0, 1e6);   self.spin_dy.setDecimals(0);  self.spin_dy.setValue(50)
        self.spin_xmax= QtWidgets.QDoubleSpinBox(); self.spin_xmax.setRange(1.0, 1e6); self.spin_xmax.setDecimals(0); self.spin_xmax.setValue(2000)
        self.spin_ymax= QtWidgets.QDoubleSpinBox(); self.spin_ymax.setRange(1.0, 1e6); self.spin_ymax.setDecimals(0); self.spin_ymax.setValue(2000)
        self.spin_sigma= QtWidgets.QDoubleSpinBox(); self.spin_sigma.setRange(0.0, 1e3); self.spin_sigma.setDecimals(4); self.spin_sigma.setSingleStep(1e-1); self.spin_sigma.setValue(1e-2)
        self.chk_weight = QtWidgets.QCheckBox('Use weighting (1/σ)')
        self.chk_weight.setChecked(True)  # default ON
        self.btn_regen = QtWidgets.QPushButton('Generate')
        self.btn_regen.setToolTip('Generate synthetic data with the current sphere parameters and noise level')
        a.addWidget(QtWidgets.QLabel('Geometry'), 0, 0); a.addWidget(self.combo_geom, 0, 1, 1, 3)
        a.addWidget(QtWidgets.QLabel('dx (m)'),   1, 0); a.addWidget(self.spin_dx,    1, 1)
        a.addWidget(QtWidgets.QLabel('dy (m)'),   1, 2); a.addWidget(self.spin_dy,    1, 3)
        a.addWidget(QtWidgets.QLabel('x half-width (m)'), 2, 0); a.addWidget(self.spin_xmax, 2, 1)
        a.addWidget(QtWidgets.QLabel('y half-width (m)'), 2, 2); a.addWidget(self.spin_ymax, 2, 3)
        a.addWidget(QtWidgets.QLabel('Noise σ (mGal)'),   3, 0); a.addWidget(self.spin_sigma, 3, 1, 1, 1)
        # Weighted inversion toggle
        a.addWidget(self.chk_weight, 3, 2, 1, 2)
        # Regenerate synthetic data button
        a.addWidget(self.btn_regen, 4, 0, 1, 4)

        # --- Grid Ranges (table) ---
        grid_grp = QtWidgets.QGroupBox('Grid Search')
        g_lay = QtWidgets.QVBoxLayout(grid_grp)
        self.zmin = QtWidgets.QDoubleSpinBox();  self.zmin.setRange(0.0, 1e6);    self.zmin.setDecimals(0);  self.zmin.setValue(20)
        self.zmax = QtWidgets.QDoubleSpinBox();  self.zmax.setRange(0.0, 1e6);    self.zmax.setDecimals(0);  self.zmax.setValue(300)
        self.zstep= QtWidgets.QDoubleSpinBox();  self.zstep.setRange(1.0, 1e6);   self.zstep.setDecimals(0); self.zstep.setValue(5)
        self.Rmin = QtWidgets.QDoubleSpinBox();  self.Rmin.setRange(0.0, 1e6);    self.Rmin.setDecimals(0);  self.Rmin.setValue(10)
        self.Rmax = QtWidgets.QDoubleSpinBox();  self.Rmax.setRange(0.0, 1e6);    self.Rmax.setDecimals(0);  self.Rmax.setValue(200)
        self.Rstep= QtWidgets.QDoubleSpinBox();  self.Rstep.setRange(1.0, 1e6);   self.Rstep.setDecimals(0); self.Rstep.setValue(5)
        self.drmin= QtWidgets.QDoubleSpinBox();  self.drmin.setRange(-1e5, 1e5);  self.drmin.setDecimals(0); self.drmin.setValue(1000)
        self.drmax= QtWidgets.QDoubleSpinBox();  self.drmax.setRange(-1e5, 1e5);  self.drmax.setDecimals(0); self.drmax.setValue(5000)
        self.drstep=QtWidgets.QDoubleSpinBox();  self.drstep.setRange(1.0, 1e6);  self.drstep.setDecimals(0);self.drstep.setValue(100)
        grid_table = QtWidgets.QTableWidget(3, 3)
        grid_table.setHorizontalHeaderLabels(['z (m)', 'R (m)', 'Δρ (kg/m³)'])
        grid_table.setVerticalHeaderLabels(['min', 'max', 'step'])
        grid_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        grid_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        grid_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        grid_table.setCellWidget(0, 0, self.zmin);  grid_table.setCellWidget(1, 0, self.zmax);  grid_table.setCellWidget(2, 0, self.zstep)
        grid_table.setCellWidget(0, 1, self.Rmin);  grid_table.setCellWidget(1, 1, self.Rmax);  grid_table.setCellWidget(2, 1, self.Rstep)
        grid_table.setCellWidget(0, 2, self.drmin); grid_table.setCellWidget(1, 2, self.drmax); grid_table.setCellWidget(2, 2, self.drstep)
        g_lay.addWidget(grid_table)

        sl = QtWidgets.QHBoxLayout()
        self.combo_3d = QtWidgets.QComboBox(); self.combo_3d.addItems(['Δρ vs R','Δρ vs z','R vs z'])
        sl.addWidget(QtWidgets.QLabel('Misfit Surface:'))
        sl.addWidget(self.combo_3d)
        g_lay.addLayout(sl)

        # --- Search buttons ---
        btns = QtWidgets.QHBoxLayout()
        self.btn_search = QtWidgets.QPushButton('Search')
        self.btn_search.setToolTip('Compute pairwise misfit grids (fixing the third parameter) and display contour plots')
        self.btn_global = QtWidgets.QPushButton('Global')
        self.btn_global.setToolTip('Exhaustively search the full 3D parameter grid and find the global misfit minimum')
        btns.addWidget(self.btn_search); btns.addWidget(self.btn_global)
        g_lay.addLayout(btns)

        # --- Inversion ---
        inv = QtWidgets.QGroupBox('Inversion (Gauss--Newton)')
        i = QtWidgets.QGridLayout(inv)
        self.init_z = QtWidgets.QDoubleSpinBox();  self.init_z.setRange(1.0, 1e6);   self.init_z.setDecimals(0);  self.init_z.setValue(80)
        self.init_R = QtWidgets.QDoubleSpinBox();  self.init_R.setRange(1.0, 1e6);   self.init_R.setDecimals(0);  self.init_R.setValue(30)
        self.init_dr= QtWidgets.QDoubleSpinBox();  self.init_dr.setRange(-1e6, 1e6); self.init_dr.setDecimals(0); self.init_dr.setValue(2500)
        self.spin_lam  = QtWidgets.QDoubleSpinBox(); self.spin_lam.setRange(0.0, 1e3);  self.spin_lam.setDecimals(6); self.spin_lam.setValue(0.0)
        self.spin_lam.setToolTip('Damping parameter λ for Gauss--Newton (0 = no damping, larger values = more damping)')
        self.spin_maxit= QtWidgets.QSpinBox();       self.spin_maxit.setRange(1, 1000);  self.spin_maxit.setValue(25)
        self.spin_maxit.setToolTip('Maximum number of Gauss--Newton iterations to perform')
        self.spin_tol  = QtWidgets.QSpinBox();       self.spin_tol.setRange(-12, -1);    self.spin_tol.setValue(-8)
        self.btn_inv = QtWidgets.QPushButton('Invert')
        self.btn_inv.setToolTip('Run Gauss--Newton inversion from the given starting parameters')
        self.btn_stop= QtWidgets.QPushButton('Stop')
        self.btn_stop.setToolTip('Abort the currently running grid search or inversion')
        i.addWidget(QtWidgets.QLabel('z₀ (m)'),       0, 0, AlignC)
        i.addWidget(QtWidgets.QLabel('R₀ (m)'),       0, 1, AlignC)
        i.addWidget(QtWidgets.QLabel('Δρ₀ (kg/m³)'), 0, 2, AlignC)
        i.addWidget(self.init_z,  1, 0)
        i.addWidget(self.init_R,  1, 1)
        i.addWidget(self.init_dr, 1, 2)
        i.addWidget(QtWidgets.QLabel('λ'),         2, 0); i.addWidget(self.spin_lam,   2, 1, 1, 2)
        i.addWidget(QtWidgets.QLabel('Max iters'), 3, 0); i.addWidget(self.spin_maxit, 3, 1, 1, 2)
        i.addWidget(QtWidgets.QLabel('Tol (10ⁿ)'), 4, 0); i.addWidget(self.spin_tol,   4, 1, 1, 2)
        inv_btns = QtWidgets.QHBoxLayout()
        inv_btns.addWidget(self.btn_inv)
        inv_btns.addWidget(self.btn_stop)
        i.addLayout(inv_btns, 5, 0, 1, 3)

        # --- Results ---
        res_grp = QtWidgets.QGroupBox('Results')
        r = QtWidgets.QGridLayout(res_grp)
        r.addWidget(QtWidgets.QLabel(''),            0, 0)
        r.addWidget(QtWidgets.QLabel('z (m)'),       0, 1, AlignC)
        r.addWidget(QtWidgets.QLabel('R (m)'),       0, 2, AlignC)
        r.addWidget(QtWidgets.QLabel('Δρ (kg/m³)'), 0, 3, AlignC)
        r.addWidget(QtWidgets.QLabel('RMS (mGal)'), 0, 4, AlignC)
        for row, name in enumerate(['Grid Δρ--R', 'Grid Δρ--z', 'Grid R--z', 'Global', 'Inversion'], start=1):
            r.addWidget(QtWidgets.QLabel(name), row, 0)
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

        lay.addWidget(synth)
        lay.addWidget(acq)
        lay.addWidget(grid_grp)
        lay.addWidget(inv)
        lay.addWidget(res_grp)
        lay.addStretch(1)
        self.setWidget(w)

        # Signals
        self.btn_search.clicked.connect(self.startGrid)
        self.btn_global.clicked.connect(self.startGlobal)
        self.btn_inv.clicked.connect(self.startInv)
        self.btn_stop.clicked.connect(self.stopAll)
        self.btn_regen.clicked.connect(self.regenData)
        self.combo_geom.currentIndexChanged.connect(self._geom_changed)
        self._geom_changed()

    def _geom_changed(self):
        grid = (self.combo_geom.currentText()=='Grid')
        self.spin_dy.setEnabled(grid)
        self.spin_ymax.setEnabled(grid)

    # getters
    def get_synthetic(self):
        return float(self.spin_z.value()), float(self.spin_R.value()), float(self.spin_dr.value())

    def get_noise(self):
        return float(self.spin_sigma.value()) * 1e-5  # mGal → m/s²

    def get_acq(self):
        geom = self.combo_geom.currentText()
        dx = float(self.spin_dx.value())
        dy = float(self.spin_dy.value())
        xmax = float(self.spin_xmax.value())
        ymax = float(self.spin_ymax.value())
        return geom, dx, dy, xmax, ymax

    def get_grid_ranges(self):
        import numpy as np
        z_vals = np.arange(float(self.zmin.value()), float(self.zmax.value())+1e-12, float(self.zstep.value()))
        R_vals = np.arange(float(self.Rmin.value()), float(self.Rmax.value())+1e-12, float(self.Rstep.value()))
        dr_vals= np.arange(float(self.drmin.value()), float(self.drmax.value())+1e-12, float(self.drstep.value()))
        return z_vals, R_vals, dr_vals

    def get_inv_settings(self):
        import numpy as np
        p0 = np.array([float(self.init_z.value()), float(self.init_R.value()), float(self.init_dr.value())])
        lam = float(self.spin_lam.value())
        maxit = int(self.spin_maxit.value())
        tol = 10.0**self.spin_tol.value()
        return p0, lam, maxit, tol

    def get_weighting_enabled(self) -> bool:
        return bool(self.chk_weight.isChecked())

    def _set_result_row(self, lbls, z, R, dr, rms):
        lbls[0].setText(f'{z:.3g}')
        lbls[1].setText(f'{R:.3g}')
        lbls[2].setText(f'{dr:.3g}')
        lbls[3].setText(f'{rms:.2e}')

    def update_grid_results(self, grid_mins):
        R1, dr1, z1, rms1 = grid_mins['min_dr_R']
        self._set_result_row(self.lbl_grid_dr_R, z1, R1, dr1, rms1 * 1e5)
        z2, dr2, R2, rms2 = grid_mins['min_dr_z']
        self._set_result_row(self.lbl_grid_dr_z, z2, R2, dr2, rms2 * 1e5)
        z3, R3, dr3, rms3 = grid_mins['min_R_z']
        self._set_result_row(self.lbl_grid_R_z,  z3, R3, dr3, rms3 * 1e5)

    def update_global_result(self, best):
        self._set_result_row(self.lbl_global, best['z'], best['R'], best['drho'], best['rms'] * 1e5)

    def update_inv_result(self, p_est, rms):
        self._set_result_row(self.lbl_inv, p_est[0], p_est[1], p_est[2], rms * 1e5)

    def reset_results(self):
        for lbls in (self.lbl_grid_dr_R, self.lbl_grid_dr_z, self.lbl_grid_R_z, self.lbl_global, self.lbl_inv):
            for lbl in lbls:
                lbl.setText('—')

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Buried Sphere — Grid Search vs Inversion')
        self.resize(1400, 900)
        # Tabs
        self.tabs = QtWidgets.QTabWidget(self)
        self.canvas_data = MapModelCanvas(self)
        self.canvas_misfit = MisfitCanvas(self)
        self.tabs.addTab(self.canvas_data, 'Data && Model')
        self.tabs.addTab(self.canvas_misfit, 'Misfit')
        self.setCentralWidget(self.tabs)
        # Dock
        self.dock = ControlDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        # State
        self.stop_flag = StopFlag()
        self.p_true = None
        self.p_est  = None
        self.Cov    = None
        self.grid_mins = {}
        self.global_min = None
        self.inv_path = None
        self.r_stations = None
        self.Xst = None; self.Yst = None
        self.d_obs = None
        self.g_clean = None
        # Connect
        self.dock.startGrid.connect(self.on_grid)
        self.dock.startGlobal.connect(self.on_global)
        self.dock.startInv.connect(self.on_inverse)
        self.dock.stopAll.connect(self.on_stop)
        self.dock.regenData.connect(self.generate_data)
        self.dock.combo_3d.currentIndexChanged.connect(lambda _idx: self.draw_3d_surface())
        self.status = self.statusBar()
        # Initial data
        self.generate_data()

    # ---------- Data generation and plotting (Tab 1) ----------
    def generate_data(self):
        z, R, dr = self.dock.get_synthetic()
        self.p_true = np.array([z,R,dr], float)
        geom, dx, dy, xmax, ymax = self.dock.get_acq()
        sigma = self.dock.get_noise()
        if geom=='Profile':
            x = np.arange(-xmax, xmax+1e-12, dx)
            self.Xst, self.Yst = x, None
            self.g_clean = forward_profile(x, self.p_true)
            self.d_obs = add_noise(self.g_clean, sigma)
            self.r_stations = np.abs(x)
            self.plot_tab1_profile(x)
        else:
            xs = np.arange(-xmax, xmax+1e-12, dx)
            ys = np.arange(-ymax, ymax+1e-12, dy)
            Xs, Ys = np.meshgrid(xs, ys)
            self.Xst, self.Yst = Xs, Ys
            self.g_clean = forward_map(Xs, Ys, self.p_true)
            self.d_obs = add_noise(self.g_clean, sigma)
            self.r_stations = np.sqrt(Xs*Xs + Ys*Ys).ravel()
            self.plot_tab1_map(xs, ys)
        
        # Build weights in SI if weighting is enabled; else None.
        if self.dock.get_weighting_enabled():
            sigma = self.dock.get_noise()  # mGal→SI already (yours)
            nobs = self.d_obs.size
            tiny = 1e-20
            w = 1.0 / max(float(sigma), tiny)   # uniform 1/σ (SI)
            self.Wd = np.full(nobs, w, dtype=float)
        else:
            self.Wd = None

        # reset path & mins when regenerating
        self.inv_path = None
        self.grid_mins = {}
        self.global_min = None
        self.dock.reset_results()
        self.status.showMessage('Data generated. Use Search/Global/Inverse without regenerating to keep noise.', 5000)

    def _annotate_results(self, ax):
        lines = []
        if self.global_min is not None:
            b = self.global_min
            lines.append(f"Global 3D min: z={b['z']:.3g}, R={b['R']:.3g}, Δρ={b['drho']:.3g}, RMS={b['rms']*1e5:.2e} mGal")
        if self.grid_mins:
            which = self.dock.combo_3d.currentText()
            if which=='Δρ vs R' and 'min_dr_R' in self.grid_mins:
                R1, dr1, z1, rms1 = self.grid_mins['min_dr_R']
                lines.append(f'Slice Δρ--R min: z={z1:.3g}, R={R1:.3g}, Δρ={dr1:.3g}, RMS={rms1*1e5:.2e} mGal')
            elif which=='Δρ vs z' and 'min_dr_z' in self.grid_mins:
                z2, dr2, R2, rms2 = self.grid_mins['min_dr_z']
                lines.append(f'Slice Δρ--z min: z={z2:.3g}, R={R2:.3g}, Δρ={dr2:.3g}, RMS={rms2*1e5:.2e} mGal')
            elif which=='R vs z' and 'min_R_z' in self.grid_mins:
                z3, R3, dr3, rms3 = self.grid_mins['min_R_z']
                lines.append(f'Slice R--z min: z={z3:.3g}, R={R3:.3g}, Δρ={dr3:.3g}, RMS={rms3*1e5:.2e} mGal')
        if self.p_est is not None:
            z,R,dr = self.p_est
            lines.append(f'Inversion: z={z:.3g}, R={R:.3g}, Δρ={dr:.3g}')
        if lines:
            txt = " ".join(lines)
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='k'), fontsize=9)

    def _draw_sphere(self, ax, p, style, label):
        z, R, _ = p
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(R*np.cos(th), z + R*np.sin(th), style, lw=2, label=label)

    def _overlay_path_on_model(self, ax):
        if self.inv_path is None:
            return
        path = np.asarray(self.inv_path)
        for k in range(len(path)):
            z,R,_ = path[k]
            th = np.linspace(0, 2*np.pi, 60)
            ax.plot(R*np.cos(th), z + R*np.sin(th), color='orange', alpha=0.15)
        zf,Rf,_ = path[-1]
        th = np.linspace(0, 2*np.pi, 120)
        ax.plot(Rf*np.cos(th), zf + Rf*np.sin(th), color='orange', lw=2, label='Inversion path (envelope)')

    def plot_tab1_profile(self, x):
        self.canvas_data.clear()
        ax = self.canvas_data.ax_data
        ax.plot(x, self.d_obs * 1e5, 'ko', ms=3, label='Observed (noisy)')
        ax.plot(x, self.g_clean * 1e5, 'b-', lw=1.5, label='True')
        if self.global_min is not None:
            b = self.global_min
            g_gm = forward_profile(x, np.array([b['z'], b['R'], b['drho']]))
            ax.plot(x, g_gm * 1e5, 'm--', lw=1.5, label='Global min')
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
            ax.plot(x, forward_profile(x, self.p_est) * 1e5, 'r--', lw=1.5, label='Inversion')
        ax.set_xlabel(''); ax.set_ylabel('g (mGal)')
        ax.set_title('Profile gravity with stations')
        ax.tick_params(labelbottom=False)
        ax.legend(loc='best')
        axm = self.canvas_data.ax_model
        axm.axhline(0, color='k')
        self._draw_sphere(axm, self.p_true, 'b-', 'True')
        if self.global_min is not None:
            b = self.global_min
            self._draw_sphere(axm, np.array([b['z'], b['R'], b['drho']]), 'm--', 'Global 3D min')
        if which == 'Δρ vs R' and 'min_dr_R' in self.grid_mins:
            R1, dr1, z1, _ = self.grid_mins['min_dr_R']
            self._draw_sphere(axm, np.array([z1, R1, dr1]), 'C1--', 'Slice min')
        elif which == 'Δρ vs z' and 'min_dr_z' in self.grid_mins:
            z2, dr2, R2, _ = self.grid_mins['min_dr_z']
            self._draw_sphere(axm, np.array([z2, R2, dr2]), 'C1--', 'Slice min')
        elif which == 'R vs z' and 'min_R_z' in self.grid_mins:
            z3, R3, dr3, _ = self.grid_mins['min_R_z']
            self._draw_sphere(axm, np.array([z3, R3, dr3]), 'C1--', 'Slice min')
        if self.p_est is not None:
            self._draw_sphere(axm, self.p_est, 'r--', 'Inversion')
        self._overlay_path_on_model(axm)
        axm.set_xlabel('x (m)'); axm.set_ylabel('depth (m)')
        axm.set_title('Buried sphere model')
        axm.set_aspect('equal', 'box')
        zt, Rt, _ = self.p_true
        axm.set_xlim([x[0], x[-1]])
        axm.set_ylim([zt + 1.5*Rt, 0])
        try:
            axm.legend(loc='lower right')
        except Exception:
            pass
        self._annotate_results(axm)
        self.canvas_data.draw_idle()

    def plot_tab1_map(self, xs, ys):
        self.canvas_data.clear()
        ax = self.canvas_data.ax_data
        im = ax.pcolormesh(xs, ys, self.d_obs * 1e5, shading='auto', cmap='viridis')
        cbar = self.canvas_data.figure.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label('g (mGal)')
        ax.set_xlabel(''); ax.set_ylabel('y (m)')
        ax.tick_params(labelbottom=False)
        ax.set_aspect('equal', 'box')
        ax.set_title('Map view: gravity (noisy) with station grid')
        axm = self.canvas_data.ax_model
        axm.axhline(0, color='k')
        self._draw_sphere(axm, self.p_true, 'b-', 'True')
        if self.global_min is not None:
            b = self.global_min
            self._draw_sphere(axm, np.array([b['z'], b['R'], b['drho']]), 'm--', 'Global 3D min')
        which = self.dock.combo_3d.currentText()
        if which=='Δρ vs R' and 'min_dr_R' in self.grid_mins:
            R1, dr1, z1, _ = self.grid_mins['min_dr_R']
            self._draw_sphere(axm, np.array([z1,R1,dr1]), 'C1--', 'Slice min')
        elif which=='Δρ vs z' and 'min_dr_z' in self.grid_mins:
            z2, dr2, R2, _ = self.grid_mins['min_dr_z']
            self._draw_sphere(axm, np.array([z2,R2,dr2]), 'C1--', 'Slice min')
        elif which=='R vs z' and 'min_R_z' in self.grid_mins:
            z3, R3, dr3, _ = self.grid_mins['min_R_z']
            self._draw_sphere(axm, np.array([z3,R3,dr3]), 'C1--', 'Slice min')
        if self.p_est is not None:
            self._draw_sphere(axm, self.p_est, 'r--', 'Inversion')
        self._overlay_path_on_model(axm)
        axm.set_xlabel('x (m)'); axm.set_ylabel('depth (m)')
        axm.set_title('Buried sphere model')
        axm.set_aspect('equal', 'box')
        zt, Rt, _ = self.p_true
        axm.set_ylim([zt + 1.5*Rt, 0])
        try:
            axm.legend(loc='lower right')
        except Exception:
            pass
        self._annotate_results(axm)
        self.canvas_data.draw_idle()

    # ---------- Pairwise misfit grids ----------
    def on_grid(self):
        z_vals, R_vals, dr_vals = self.dock.get_grid_ranges()
        if self.p_est is not None:
            p_fixed = self.p_est.copy()
        else:
            p_fixed = self.p_true.copy()
        r = self.r_stations; d_obs_vec = self.d_obs.ravel()
        self.stop_flag.stop = False
        self.thread = QtCore.QThread(self)
        self.worker = GridSearchWorker(forward_sphere_r, r, d_obs_vec, p_fixed, z_vals, R_vals, dr_vals, self.stop_flag)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda v: self.status.showMessage(f'Pairwise grid progress ~{v}%'))
        self.worker.finished.connect(self.on_grid_done)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_grid_done(self, res: dict):
        self.grid_res = res
        self.grid_mins = {
            'min_dr_R': res['min_dr_R'],
            'min_dr_z': res['min_dr_z'],
            'min_R_z' : res['min_R_z'],
        }
        self.dock.update_grid_results(self.grid_mins)
        self.draw_misfit_tab(res)
        # Refresh model tab overlays without regenerating data
        geom, dx, dy, xmax, ymax = self.dock.get_acq()
        if geom=='Profile':
            x = np.arange(-xmax, xmax+1e-12, dx)
            self.plot_tab1_profile(x)
        else:
            xs = np.arange(-xmax, xmax+1e-12, dx)
            ys = np.arange(-ymax, ymax+1e-12, dy)
            self.plot_tab1_map(xs, ys)
        self.status.showMessage('Pairwise misfit grids computed.', 3000)

    def draw_misfit_tab(self, res):
        def _loglevels(G, n=20):
            gmin = max(float(G.min()), 1e-30)
            gmax = float(G.max())
            if gmax <= gmin:
                return n
            return np.logspace(np.log10(gmin), np.log10(gmax), n)

        self.canvas_misfit.clear()

        Rv = res['R_vals']; drv = res['dr_vals']; G1 = res['G_dr_R'] * 1e5
        CS1 = self.canvas_misfit.ax11.contour(Rv, drv, G1, levels=_loglevels(G1), colors='k', linewidths=0.8)
        self.canvas_misfit.ax11.clabel(CS1, inline=True, fontsize=7, fmt='%.2e')
        self.canvas_misfit.cb11 = None
        self.canvas_misfit.ax11.set_xlabel('R (m)'); self.canvas_misfit.ax11.set_ylabel('Δρ (kg/m³)'); self.canvas_misfit.ax11.set_title('Misfit: Δρ vs R')
        i1,j1 = res['idx_dr_R']; self.canvas_misfit.ax11.plot([Rv[i1]],[drv[j1]], 'r*', ms=10)
        if self.inv_path is not None:
            path = np.asarray(self.inv_path)
            self.canvas_misfit.ax11.plot(path[:,1], path[:,2], 'r.-', lw=1.0, ms=4, label='Inv. path')
            self.canvas_misfit.ax11.legend(loc='best')

        zv = res['z_vals']; drv2 = res['dr_vals2']; G2 = res['G_dr_z'] * 1e5
        CS2 = self.canvas_misfit.ax12.contour(zv, drv2, G2, levels=_loglevels(G2), colors='k', linewidths=0.8)
        self.canvas_misfit.ax12.clabel(CS2, inline=True, fontsize=7, fmt='%.2e')
        self.canvas_misfit.cb12 = None
        self.canvas_misfit.ax12.set_xlabel('z (m)'); self.canvas_misfit.ax12.set_ylabel('Δρ (kg/m³)'); self.canvas_misfit.ax12.set_title('Misfit: Δρ vs z')
        i2,j2 = res['idx_dr_z']; self.canvas_misfit.ax12.plot([zv[i2]],[drv2[j2]], 'r*', ms=10)
        if self.inv_path is not None:
            path = np.asarray(self.inv_path)
            self.canvas_misfit.ax12.plot(path[:,0], path[:,2], 'r.-', lw=1.0, ms=4)

        zv2 = res['z_vals2']; Rv2 = res['R_vals2']; G3 = res['G_R_z'] * 1e5
        CS3 = self.canvas_misfit.ax21.contour(zv2, Rv2, G3, levels=_loglevels(G3), colors='k', linewidths=0.8)
        self.canvas_misfit.ax21.clabel(CS3, inline=True, fontsize=7, fmt='%.2e')
        self.canvas_misfit.cb21 = None
        self.canvas_misfit.ax21.set_xlabel('z (m)'); self.canvas_misfit.ax21.set_ylabel('R (m)'); self.canvas_misfit.ax21.set_title('Misfit: R vs z')
        i3,j3 = res['idx_R_z']; self.canvas_misfit.ax21.plot([zv2[i3]],[Rv2[j3]], 'r*', ms=10)
        if self.inv_path is not None:
            path = np.asarray(self.inv_path)
            self.canvas_misfit.ax21.plot(path[:,0], path[:,1], 'r.-', lw=1.0, ms=4)

        # Yellow star at inversion solution
        if self.p_est is not None:
            z_est, R_est, dr_est = self.p_est
            self.canvas_misfit.ax11.plot([R_est],  [dr_est], 'y*', ms=12, zorder=5)
            self.canvas_misfit.ax12.plot([z_est],  [dr_est], 'y*', ms=12, zorder=5)
            self.canvas_misfit.ax21.plot([z_est],  [R_est],  'y*', ms=12, zorder=5)

        self.draw_3d_surface(res)
        self.canvas_misfit.draw_idle()

    def draw_3d_surface(self, res=None):
        if res is None:
            res = getattr(self, 'grid_res', None)
        if res is None:
            return
        Rv = res['R_vals']; drv = res['dr_vals']; G1 = res['G_dr_R'] * 1e5
        zv = res['z_vals']; drv2 = res['dr_vals2']; G2 = res['G_dr_z'] * 1e5
        zv2 = res['z_vals2']; Rv2 = res['R_vals2']; G3 = res['G_R_z'] * 1e5
        ax3d = self.canvas_misfit.ax22
        ax3d.cla()
        which = self.dock.combo_3d.currentText()
        if which=='Δρ vs R':
            X, Y = np.meshgrid(Rv, drv)
            ax3d.plot_surface(X, Y, G1, cmap='viridis', rstride=1, cstride=1, linewidth=0)
            ax3d.set_xlabel('R (m)'); ax3d.set_ylabel('Δρ (kg/m³)')
        elif which=='Δρ vs z':
            X, Y = np.meshgrid(zv, drv2)
            ax3d.plot_surface(X, Y, G2, cmap='viridis', rstride=1, cstride=1, linewidth=0)
            ax3d.set_xlabel('z (m)'); ax3d.set_ylabel('Δρ (kg/m³)')
        else:
            X, Y = np.meshgrid(zv2, Rv2)
            ax3d.plot_surface(X, Y, G3, cmap='viridis', rstride=1, cstride=1, linewidth=0)
            ax3d.set_xlabel('z (m)'); ax3d.set_ylabel('R (m)')
        ax3d.set_zlabel('RMS (mGal)'); ax3d.set_title(f'3D Misfit Surface — {which}')
        self.canvas_misfit.draw_idle()

    # ---------- Global 3D grid search ----------
    def on_global(self):
        z_vals, R_vals, dr_vals = self.dock.get_grid_ranges()
        r = self.r_stations; d_obs_vec = self.d_obs.ravel()
        self.stop_flag.stop = False
        self.threadG = QtCore.QThread(self)
        self.workerG = GlobalGridWorker(forward_sphere_r, r, d_obs_vec, z_vals, R_vals, dr_vals, self.stop_flag)
        self.workerG.moveToThread(self.threadG)
        self.threadG.started.connect(self.workerG.run)
        self.workerG.progress.connect(lambda v: self.status.showMessage(f'Global 3D grid progress ~{v}%'))
        self.workerG.finished.connect(self.on_global_done)
        self.workerG.finished.connect(self.threadG.quit)
        self.workerG.finished.connect(self.workerG.deleteLater)
        self.threadG.finished.connect(self.threadG.deleteLater)
        self.threadG.start()

    def on_global_done(self, res: dict):
        self.global_min = res['best']
        self.dock.update_global_result(self.global_min)
        self.status.showMessage(f"Global min: z={self.global_min['z']:.3g}, R={self.global_min['R']:.3g}, Δρ={self.global_min['drho']:.3g}, RMS={self.global_min['rms']*1e5:.2e} mGal", 6000)
        geom, dx, dy, xmax, ymax = self.dock.get_acq()
        if geom=='Profile':
            x = np.arange(-xmax, xmax+1e-12, dx)
            self.plot_tab1_profile(x)
        else:
            xs = np.arange(-xmax, xmax+1e-12, dx)
            ys = np.arange(-ymax, ymax+1e-12, dy)
            self.plot_tab1_map(xs, ys)

    # ---------- Inversion ----------
    def on_inverse(self):
        p0, lam, maxit, tol = self.dock.get_inv_settings()
        r = self.r_stations; d = self.d_obs.ravel()
        t0 = time.time()
        p_est, Cov, hist = gauss_newton(
            lambda _r, p: forward_sphere_r(_r, p),
            lambda _r, p: jacobian_r(_r, p),
            r, d,
            p0=p0, lam=lam, maxit=maxit, tol=tol,
            Wd=self.Wd,
            stop_flag=self.stop_flag
        )
        # Use unweighted RMS in SI for display (convert to mGal for the label)

        t1 = time.time()
        self.p_est, self.Cov = p_est, Cov
        self.inv_path = np.asarray(hist['path'])
        self.dock.update_inv_result(self.p_est, hist['rms'][-1])
        self.status.showMessage(
            f'Inversion done in {t1-t0:.3f}s; final RMS={hist["rms"][-1]*1e5:.3e} mGal',
            4000
        )
        if hasattr(self, 'grid_res'):
            self.draw_misfit_tab(self.grid_res)
        geom, dx, dy, xmax, ymax = self.dock.get_acq()
        if geom=='Profile':
            x = np.arange(-xmax, xmax+1e-12, dx)
            self.plot_tab1_profile(x)
        else:
            xs = np.arange(-xmax, xmax+1e-12, dx)
            ys = np.arange(-ymax, ymax+1e-12, dy)
            self.plot_tab1_map(xs, ys)

    def on_stop(self):
        self.stop_flag.stop = True
        self.status.showMessage('Stop requested…', 1500)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
