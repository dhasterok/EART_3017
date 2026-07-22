# -*- coding: utf-8 -*-
"""
Mixing-Law Transport GUI (PyQt6) — V4 (definitive)
---------------------------------------------------
Combines the refined dock UI and geologic controls from V3 with the
biased random-walk (BRW) solver option from the _rw branch.

Solver modes (selectable at run time):
  • Dijkstra (shortest path)  — deterministic least-cost traversal
  • Random walk (biased)      — stochastic first-passage with eastward bias β,
                                configurable walker count, max steps, and step-time model

Geometry modes:
  • Random granular  — spatially correlated lognormal k(x,y)
  • Perpendicular    — layers ⟂ flow (built directly)
  • Parallel         — lanes ∥ flow (perpendicular rotated 90°)

Run:  python mixing_law_gui_qt_v4.py
"""

from __future__ import annotations
import sys, math
from dataclasses import dataclass
import numpy as np

# PyQt6
from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QDockWidget, QCheckBox,
    QTableWidget, QHeaderView, QToolBox, QToolBar, QLineEdit,
)

# Matplotlib embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import geometry
import flow_model as fm

# lame-core  (pip install -r requirements.txt  or  pip install -e /path/to/lame-core)
from lame_core.CustomWidgets import ToggleSwitch, CustomToolButton
from lame_core import config as lame_config


# -------------------- GUI --------------------

@dataclass
class Component:
    mean: float
    std: float
    frac: float

class MplCanvas(FigureCanvas):
    def __init__(self):
        from matplotlib.gridspec import GridSpec
        self.fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(3, 2, figure=self.fig,
                      width_ratios=[1.6, 1], hspace=0.55, wspace=0.40)
        self.ax_img  = self.fig.add_subplot(gs[:, 0])   # map — spans all 3 rows
        self.ax_time = self.fig.add_subplot(gs[0, 1])   # travel-time histogram
        self.ax_keff = self.fig.add_subplot(gs[1, 1])   # effective conductivity
        self.ax_exit = self.fig.add_subplot(gs[2, 1])   # exit distribution
        super().__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mixing-Law Transport (PyQt6) — V4")
        self.resize(1500, 900)

        # ---- State: grid ----
        self.H, self.W = 60, 90
        self.geom = 'random'
        self.ncomp = 2
        self.comps: list[Component] = [
            Component(1.2, 0.3, 0.5),
            Component(0.6, 0.2, 0.5),
            Component(0.9, 0.2, 0.0),
        ]

        # ---- State: random granular ----
        self.corr_len = 0
        self.sigma_ln_noise = 0.0

        # ---- State: layering ----
        self.N_A = 1
        self.f_A = 0.5
        self.CV_thick = 0.0
        self.dist_thick = 'lognormal'
        self.w_int = 0.0
        self.p_cont = 1.0
        self.L_gap = 10
        self.A_meander = 0.0
        self.L_meander = 30

        # ---- State: 3rd component affinity ----
        self.phi_CA = 0.8

        # ---- State: solver ----
        self.solver = 'dijkstra'       # 'dijkstra' | 'random_walk' | 'kirchhoff'
        self.kirchhoff_periodic_tb = False
        self.last_T = None             # temperature field from last Kirchhoff solve
        self.rw_beta = 1.00
        self.rw_walkers = 50
        self.rw_max_steps = 10000
        self.rw_step_time_model = '1_over_k'

        # ---- State: run / animate ----
        self.logN = 2.0
        self.fps = 30
        self.t_pred = None
        self.t_pred_label = ""
        self.times     = []   # travel time per trial
        self.keffs     = []   # effective conductivity per trial
        self.exit_rows = []   # display-window row or None (outside window)
        self.k_harm = self.k_geom = self.k_arith = None
        self.rw_start_mode = 'center'
        self.domain_pad = 0
        self.running = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._run_chunk)
        self.chunk_size = 50

        self._cbar_state = {'cax': None, 'cbar': None}
        self.view_mode = 'k'           # 'k' | 'components'
        self.seed = int(np.random.randint(0, 100000))

        self._build_ui()
        self._refresh_prediction()
        self._redraw_field()

    # ---------------- UI ----------------
    def _build_ui(self):
        # --- Toolbar (Run / Stop / Animate / Reset) ---
        def _icon(name):
            path = lame_config.ICONPATH / name
            return QIcon(str(path)) if path.exists() else QIcon()

        tb = QToolBar("Controls", self)
        tb.setIconSize(QSize(28, 28))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        self.act_run   = QAction(_icon('icon-run-64.svg'),          "Run N trials",   self)
        self.act_stop  = QAction(_icon('icon-stop-64.svg'),         "Stop",           self)
        self.act_anim  = QAction(_icon('icon-animate-path-64.svg'), "Animate 1 path", self)
        self.act_reset = QAction(_icon('icon-reset-64.svg'),        "Reset",          self)
        self.act_run.triggered.connect(self._start_run)
        self.act_stop.triggered.connect(self._stop_run)
        self.act_anim.triggered.connect(self._animate_one)
        self.act_reset.triggered.connect(self._reset_plots)
        tb.addAction(self.act_run)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        tb.addAction(self.act_anim)
        tb.addSeparator()
        tb.addAction(self.act_reset)

        # --- Central area: canvas + view toggle ---
        central = QWidget(self)
        self.setCentralWidget(central)
        hbox = QHBoxLayout(central)

        left = QWidget(); left_v = QVBoxLayout(left)
        self.canvas = MplCanvas(); self.toolbar = NavToolbar(self.canvas, self)
        view_row = QWidget()
        view_hl = QHBoxLayout(view_row); view_hl.setContentsMargins(4, 2, 4, 2)
        view_hl.addWidget(QLabel("View:"))
        view_hl.addWidget(QLabel("k-field"))
        self.toggle_view = ToggleSwitch()
        self.toggle_view.setToolTip("Off = conductivity field  |  On = component map")
        self.toggle_view.stateChanged.connect(self._toggle_view)
        view_hl.addWidget(self.toggle_view)
        view_hl.addWidget(QLabel("Components"))
        view_hl.addStretch()
        left_v.addWidget(view_row)
        left_v.addWidget(self.toolbar); left_v.addWidget(self.canvas)
        hbox.addWidget(left, 2)

        # --- Dock with QToolBox ---
        dock = QDockWidget("Controls", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock_container = QWidget()
        dock_vbox = QVBoxLayout(dock_container)
        dock_vbox.setContentsMargins(2, 2, 2, 2)
        toolbox = QToolBox()
        dock_vbox.addWidget(toolbox)
        dock.setWidget(dock_container)

        # ---- Page 1: Geometry ----
        p_geom = QWidget(); gv = QFormLayout(p_geom)
        self.cb_geom = QComboBox(); self.cb_geom.addItems(['random','parallel','perpendicular'])
        self.cb_geom.currentTextChanged.connect(self._on_geom)
        self.cb_geom.setToolTip("Choose overall fabric: Random (granular), Parallel (lanes ∥ flow), Perpendicular (layers ⟂ flow)")
        gv.addRow("Geometry:", self.cb_geom)

        self.le_seed = QLineEdit(str(self.seed))
        self.le_seed.setToolTip("Random seed — edit to reproduce a specific field, or click ⟳ for a new one")
        self.le_seed.setMaximumWidth(90)
        self.le_seed.editingFinished.connect(self._on_seed_edited)
        self.btn_reseed = CustomToolButton(
            text='', light_icon_unchecked='icon-randomize-64.svg',
            button_size=26, icon_size=20,
        )
        self.btn_reseed.setToolTip("Pick a new random seed and regenerate the field")
        self.btn_reseed.clicked.connect(self._on_reseed)
        _hw_seed = QWidget(); _hbl_seed = QHBoxLayout(_hw_seed); _hbl_seed.setContentsMargins(0, 0, 0, 0)
        _hbl_seed.addWidget(self.le_seed); _hbl_seed.addWidget(self.btn_reseed); _hbl_seed.addStretch()
        gv.addRow("Seed:", _hw_seed)

        self.sp_corr_len = QSpinBox(); self.sp_corr_len.setRange(0, 30); self.sp_corr_len.setValue(self.corr_len)
        self.sp_corr_len.valueChanged.connect(self._on_params)
        self.sp_corr_len.setToolTip("Grain-scale clustering (px) for random granular mode")
        gv.addRow("Granular corr_len (px):", self.sp_corr_len)

        self.sp_sigma_ln = QDoubleSpinBox(); self.sp_sigma_ln.setRange(0.0, 1.0); self.sp_sigma_ln.setSingleStep(0.01); self.sp_sigma_ln.setValue(self.sigma_ln_noise)
        self.sp_sigma_ln.valueChanged.connect(self._on_params)
        self.sp_sigma_ln.setToolTip("Microscale ln k noise (adds small property variability)")
        gv.addRow("ln k noise (granular/layers):", self.sp_sigma_ln)
        toolbox.addItem(p_geom, "Geometry")

        # ---- Page 2: Layering ----
        p_lay = QWidget(); lf = QFormLayout(p_lay)

        self.sp_NA = QSpinBox(); self.sp_NA.setRange(1, 50); self.sp_NA.setValue(self.N_A); self.sp_NA.valueChanged.connect(self._on_params)
        self.sp_NA.setToolTip("Number of beds per component; A and B get the same count — more beds ⇒ thinner layers")
        self.sp_fA = QDoubleSpinBox(); self.sp_fA.setRange(0.0, 1.0); self.sp_fA.setSingleStep(0.05); self.sp_fA.setDecimals(2); self.sp_fA.setValue(self.f_A)
        self.sp_fA.valueChanged.connect(self._on_params)
        self.sp_fA.setToolTip("Fraction of cross-section occupied by A-type layers; B occupies the remainder (1 − f_A)")
        _hw_nf = QWidget(); _hbl = QHBoxLayout(_hw_nf); _hbl.setContentsMargins(0, 0, 0, 0)
        _hbl.addWidget(self.sp_NA); _hbl.addWidget(QLabel("Prop. A:")); _hbl.addWidget(self.sp_fA)
        lf.addRow("No. Layers:", _hw_nf)

        self.sp_CV = QDoubleSpinBox(); self.sp_CV.setRange(0.0, 1.5); self.sp_CV.setSingleStep(0.05); self.sp_CV.setValue(self.CV_thick)
        self.sp_CV.valueChanged.connect(self._on_params)
        self.sp_CV.setToolTip("Thickness variability: CV = std/mean. 0 = equal thickness; 0.1--0.3 natural; 0.5+ very uneven")
        lf.addRow("Thickness variability (CV):", self.sp_CV)

        self.cb_dist = QComboBox(); self.cb_dist.addItems(["lognormal","uniform"]); self.cb_dist.currentTextChanged.connect(self._on_params)
        self.cb_dist.setToolTip("Distribution to draw raw bed thicknesses before rescaling: lognormal (skewed) or uniform (symmetric)")
        lf.addRow("Thickness distribution:", self.cb_dist)

        self.sp_wint = QDoubleSpinBox(); self.sp_wint.setRange(0.0, 10.0); self.sp_wint.setSingleStep(0.5); self.sp_wint.setValue(self.w_int)
        self.sp_wint.valueChanged.connect(self._on_params)
        self.sp_wint.setToolTip("Contact half-width in pixels; 0=sharp; 2--5=diffuse (error-function transition)")
        lf.addRow("Contact half-width (px):", self.sp_wint)

        self.sp_pcont = QDoubleSpinBox(); self.sp_pcont.setRange(0.0,1.0); self.sp_pcont.setSingleStep(0.05); self.sp_pcont.setValue(self.p_cont)
        self.sp_pcont.valueChanged.connect(self._on_params)
        self.sp_pcont.setToolTip("Probability a layer is continuous (1 = no leaks); lower values add realistic small bypasses")
        lf.addRow("Continuity probability:", self.sp_pcont)

        self.sp_Lgap = QSpinBox(); self.sp_Lgap.setRange(1, 500); self.sp_Lgap.setValue(self.L_gap)
        self.sp_Lgap.valueChanged.connect(self._on_params)
        self.sp_Lgap.setToolTip("Mean gap (leak) length in pixels; keep small to avoid forming full bypass lanes")
        lf.addRow("Mean gap length (px):", self.sp_Lgap)

        self.sp_Ame = QDoubleSpinBox(); self.sp_Ame.setRange(0.0, 20.0); self.sp_Ame.setSingleStep(0.5); self.sp_Ame.setValue(self.A_meander)
        self.sp_Ame.valueChanged.connect(self._on_params)
        self.sp_Ame.setToolTip("Fold amplitude (px): vertical displacement of bed traces")
        self.sp_Lme = QSpinBox(); self.sp_Lme.setRange(1, 300); self.sp_Lme.setValue(self.L_meander)
        self.sp_Lme.valueChanged.connect(self._on_params)
        self.sp_Lme.setToolTip("Fold wavelength (px): distance between fold crests")
        _hw_fold = QWidget(); _hbl2 = QHBoxLayout(_hw_fold); _hbl2.setContentsMargins(0, 0, 0, 0)
        _hbl2.addWidget(self.sp_Ame); _hbl2.addWidget(QLabel("λ (px):")); _hbl2.addWidget(self.sp_Lme)
        lf.addRow("Fold A (px):", _hw_fold)
        toolbox.addItem(p_lay, "Layering")

        # ---- Page 3: Components ----
        p_comp = QWidget(); vcomp = QVBoxLayout(p_comp)
        row1 = QHBoxLayout(); vcomp.addLayout(row1)
        row1.addWidget(QLabel("No. Components:"))
        self.cb_n = QComboBox(); self.cb_n.addItems(['2','3']); self.cb_n.setCurrentIndex(0)
        self.cb_n.currentTextChanged.connect(self._on_ncomp)
        row1.addWidget(self.cb_n)
        row1.addSpacing(12)
        row1.addWidget(QLabel("C in A:"))
        self.sp_phiCA = QDoubleSpinBox(); self.sp_phiCA.setRange(0.0, 1.0); self.sp_phiCA.setSingleStep(0.05); self.sp_phiCA.setValue(self.phi_CA)
        self.sp_phiCA.setToolTip("Fraction of C placed in A-type regions; remainder (1 − value) goes into B-type regions")
        self.sp_phiCA.valueChanged.connect(self._on_params)
        row1.addWidget(self.sp_phiCA); row1.addStretch(1)
        self.comp_table = QTableWidget(3, 3)
        self.comp_table.setHorizontalHeaderLabels(["k (conductivity)", "Std dev", "Fraction"])
        self.comp_table.setVerticalHeaderLabels(["A", "B", "C"])
        self.comp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comp_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        vcomp.addWidget(self.comp_table)
        self.le_means=[]; self.le_stds=[]; self.le_fracs=[]
        init = self.comps
        for i in range(3):
            e_m = QDoubleSpinBox(); e_m.setRange(0.6, 11.0); e_m.setSingleStep(0.1); e_m.setDecimals(3); e_m.setValue(init[i].mean)
            e_s = QDoubleSpinBox(); e_s.setRange(0.0, 3.0);  e_s.setSingleStep(0.05); e_s.setDecimals(3); e_s.setValue(init[i].std)
            e_f = QDoubleSpinBox(); e_f.setRange(0.0, 1.0);  e_f.setSingleStep(0.01); e_f.setDecimals(3); e_f.setValue(init[i].frac)
            e_m.valueChanged.connect(self._on_comp_changed)
            e_s.valueChanged.connect(self._on_comp_changed)
            e_f.valueChanged.connect(self._on_comp_changed)
            self.le_means.append(e_m); self.le_stds.append(e_s); self.le_fracs.append(e_f)
            self.comp_table.setCellWidget(i, 0, e_m)
            self.comp_table.setCellWidget(i, 1, e_s)
            self.comp_table.setCellWidget(i, 2, e_f)
        toolbox.addItem(p_comp, "Components")

        # ---- Page 4: Simulation & Solver ----
        p_sim = QWidget(); sform = QFormLayout(p_sim)

        self.sp_H = QSpinBox(); self.sp_H.setRange(20, 300); self.sp_H.setValue(self.H); self.sp_H.valueChanged.connect(self._on_grid)
        self.sp_W = QSpinBox(); self.sp_W.setRange(30, 500); self.sp_W.setValue(self.W); self.sp_W.valueChanged.connect(self._on_grid)
        _hw_grid = QWidget(); _hblg = QHBoxLayout(_hw_grid); _hblg.setContentsMargins(0, 0, 0, 0)
        _hblg.addWidget(self.sp_H); _hblg.addWidget(QLabel("W:")); _hblg.addWidget(self.sp_W)
        sform.addRow("Grid H:", _hw_grid)

        self.cb_solver = QComboBox()
        self.cb_solver.addItems([
            "Dijkstra (shortest path)",
            "Random walk (biased)",
            "Kirchhoff (direct solve)",
        ])
        self.cb_solver.currentIndexChanged.connect(self._on_solver)
        self.cb_solver.setToolTip(
            "Dijkstra: minimum-resistance path (upper-bound k_eff)\n"
            "Random walk: stochastic diffusion (k_eff ≈ k_geom)\n"
            "Kirchhoff: solve KCL exactly per realisation (exact k_eff)"
        )
        sform.addRow("Solver:", self.cb_solver)

        self.sp_beta = QDoubleSpinBox(); self.sp_beta.setRange(0.0, 5.0); self.sp_beta.setSingleStep(0.25); self.sp_beta.setValue(self.rw_beta)
        self.sp_beta.valueChanged.connect(self._on_params)
        self.sp_beta.setToolTip("Eastward bias β: 0 = isotropic diffusion; 1 = no westward steps; >1 = strongly directed east")
        sform.addRow("Bias β (eastward):", self.sp_beta)

        self.sp_walkers = QSpinBox(); self.sp_walkers.setRange(1, 5000); self.sp_walkers.setValue(self.rw_walkers)
        self.sp_walkers.valueChanged.connect(self._on_params)
        self.sp_walkers.setToolTip("Number of independent walkers per trial; more walkers ⇒ lower variance in mean FPT")
        sform.addRow("Walkers per trial:", self.sp_walkers)

        self.sp_maxsteps = QSpinBox(); self.sp_maxsteps.setRange(100, 200000); self.sp_maxsteps.setSingleStep(500); self.sp_maxsteps.setValue(self.rw_max_steps)
        self.sp_maxsteps.valueChanged.connect(self._on_params)
        self.sp_maxsteps.setToolTip("Maximum steps before a walker is declared lost (returned as inf and excluded from mean)")
        sform.addRow("Max steps per walker:", self.sp_maxsteps)

        self.cb_stepmodel = QComboBox(); self.cb_stepmodel.addItems(["1/k (target)", "constant"])
        self.cb_stepmodel.currentTextChanged.connect(self._on_params)
        self.cb_stepmodel.setToolTip("Step-time model: 1/k makes low-k cells slow to cross; constant treats all cells equally")
        sform.addRow("Step time model:", self.cb_stepmodel)

        self.sl_logN = QSlider(QtCore.Qt.Orientation.Horizontal); self.sl_logN.setRange(100, 400); self.sl_logN.setValue(int(self.logN*100)); self.sl_logN.valueChanged.connect(self._on_logN)
        self.lab_logN = QLabel(f"log10 N = {self.logN:.2f} (N≈{int(10**self.logN)})")
        sform.addRow(self.lab_logN, self.sl_logN)

        self.cb_start = QComboBox(); self.cb_start.addItems(['Center (H/2)', 'Random'])
        self.cb_start.setToolTip("Center: all walkers start at the middle of the left boundary; Random: uniform-random row each trial")
        self.cb_start.currentIndexChanged.connect(self._on_params)
        sform.addRow("Walker start:", self.cb_start)

        self.sp_pad = QSpinBox(); self.sp_pad.setRange(0, 300); self.sp_pad.setValue(0)
        self.sp_pad.setToolTip("Extra rows added above and below to reduce top/bottom edge effects on path statistics")
        self.sp_pad.valueChanged.connect(self._on_params)
        sform.addRow("Domain padding (rows):", self.sp_pad)

        self.chk_periodic = QCheckBox()
        self.chk_periodic.setChecked(False)
        self.chk_periodic.setToolTip(
            "Kirchhoff only: connect top and bottom rows with wrap-around edges.\n"
            "Makes the sample behave like an infinite statistically-homogeneous medium,\n"
            "giving better convergence to the theoretical mixing-law predictions."
        )
        self.chk_periodic.stateChanged.connect(self._on_params)
        sform.addRow("Periodic top/bottom (Kirchhoff):", self.chk_periodic)

        toolbox.addItem(p_sim, "Simulation & Solver")

        # ---- Page 5: Animation ----
        p_anim = QWidget(); af = QFormLayout(p_anim)
        self.sp_fps = QSpinBox(); self.sp_fps.setRange(5, 60); self.sp_fps.setValue(self.fps); self.sp_fps.valueChanged.connect(self._on_params)
        af.addRow("Animation FPS:", self.sp_fps)
        toolbox.addItem(p_anim, "Animation")

        self._update_comp_visibility()
        self._toggle_rw_controls()

    # ---------------- Events ----------------

    def _on_solver(self, idx):
        self.solver = ['dijkstra', 'random_walk', 'kirchhoff'][idx]
        self._toggle_rw_controls()
        self._refresh_prediction(); self._redraw_field()

    def _toggle_rw_controls(self):
        is_rw = (self.solver == 'random_walk')
        is_kh = (self.solver == 'kirchhoff')
        for w in (self.sp_beta, self.sp_walkers, self.sp_maxsteps, self.cb_stepmodel):
            w.setEnabled(is_rw)
        self.chk_periodic.setEnabled(is_kh)

    def _on_seed_edited(self):
        try:
            self.seed = int(self.le_seed.text())
        except ValueError:
            self.le_seed.setText(str(self.seed))
            return
        self._refresh_prediction(); self._redraw_field()

    def _on_reseed(self):
        self.seed = int(np.random.randint(0, 100000))
        self.le_seed.setText(str(self.seed))
        self._refresh_prediction(); self._redraw_field()

    def _on_geom(self, text):
        self.geom = text
        self._refresh_prediction(); self._redraw_field()

    def _on_params(self, *args):
        self.corr_len = self.sp_corr_len.value()
        self.sigma_ln_noise = self.sp_sigma_ln.value()
        self.N_A = self.sp_NA.value()
        self.f_A = self.sp_fA.value()
        self.CV_thick = self.sp_CV.value(); self.dist_thick = self.cb_dist.currentText()
        self.w_int = self.sp_wint.value(); self.p_cont = self.sp_pcont.value(); self.L_gap = self.sp_Lgap.value()
        self.A_meander = self.sp_Ame.value(); self.L_meander = self.sp_Lme.value()
        self.phi_CA = self.sp_phiCA.value()
        self.rw_beta = self.sp_beta.value()
        self.rw_walkers = self.sp_walkers.value()
        self.rw_max_steps = self.sp_maxsteps.value()
        self.rw_step_time_model = '1_over_k' if self.cb_stepmodel.currentText().startswith('1/') else 'constant'
        self.rw_start_mode = 'center' if self.cb_start.currentIndex() == 0 else 'random'
        self.domain_pad = self.sp_pad.value()
        self.kirchhoff_periodic_tb = self.chk_periodic.isChecked()
        self.fps = self.sp_fps.value()
        self._refresh_prediction(); self._redraw_field()

    def _on_ncomp(self, value):
        self.ncomp = int(value)
        self._update_comp_visibility(); self._refresh_prediction(); self._redraw_field()

    def _on_comp_changed(self, *args):
        for i in range(3):
            self.comps[i].mean = self.le_means[i].value()
            self.comps[i].std  = self.le_stds[i].value()
            self.comps[i].frac = self.le_fracs[i].value()
        self._normalize_fracs(); self._refresh_prediction(); self._redraw_field()

    def _normalize_fracs(self):
        f = [self.comps[i].frac for i in range(self.ncomp)]
        s = sum(f)
        if s <= 1e-12:
            for i in range(self.ncomp): self.comps[i].frac = 1.0/self.ncomp
        else:
            for i in range(self.ncomp): self.comps[i].frac = f[i]/s
        for i in range(3):
            if i < self.ncomp:
                self.le_fracs[i].blockSignals(True); self.le_fracs[i].setValue(self.comps[i].frac); self.le_fracs[i].blockSignals(False)
            else:
                self.le_fracs[i].blockSignals(True); self.le_fracs[i].setValue(0.0); self.le_fracs[i].blockSignals(False)

    def _on_logN(self, val):
        self.logN = val/100.0
        self.lab_logN.setText(f"log10 N = {self.logN:.2f} (N≈{int(10**self.logN)})")

    def _on_grid(self, *args):
        self.H = self.sp_H.value(); self.W = self.sp_W.value()
        # Prior run results (e.g. self.last_T) are sized to the old grid and
        # would no longer broadcast against a freshly rebuilt field.
        self.times = []; self.keffs = []; self.exit_rows = []; self.last_T = None
        self._refresh_prediction(); self._redraw_field()

    def _update_comp_visibility(self):
        enable3 = (self.ncomp == 3)
        for w in (self.le_means[2], self.le_stds[2], self.le_fracs[2]): w.setEnabled(enable3)

    # ---------------- Build + predict ----------------

    def _build_field(self, use_seed=True):
        fr   = [self.comps[i].frac for i in range(self.ncomp)]
        means= [self.comps[i].mean for i in range(self.ncomp)]
        stds = [self.comps[i].std  for i in range(self.ncomp)]
        mus, sigs = [], []
        for m, s in zip(means, stds):
            mu, sg = geometry.lognormal_mu_sigma_from_mean_std(m, s)
            mus.append(mu); sigs.append(sg)
        mus = np.array(mus); sigs = np.array(sigs)

        # For display/preview: use the stored seed so the field is reproducible.
        # For simulation trials: let numpy draw from its current state so each
        # trial gets an independent field realization.
        if use_seed:
            np.random.seed(self.seed)
            rng = np.random.default_rng(self.seed)
        else:
            rng = np.random.default_rng()

        if self.geom == 'random':
            k_raw, comp_img = geometry.build_random_field(self.H, self.W, fr, mus, sigs, self.corr_len, self.sigma_ln_noise)
        else:
            if self.ncomp == 2:
                k_perp, comp_img = geometry.build_two_component_layers(
                    self.H, self.W,
                    f_A=self.f_A, N_A=self.N_A, N_B=self.N_A,
                    CV_thick=self.CV_thick, dist_thick=self.dist_thick,
                    w_int=self.w_int, p_cont=self.p_cont, L_gap=self.L_gap,
                    A_meander=self.A_meander, L_meander=self.L_meander,
                    muA=mus[0], sigA=sigs[0], muB=mus[1], sigB=sigs[1],
                    sigma_ln_noise=self.sigma_ln_noise, rng=rng,
                )
            else:
                f_B = max(0.0, 1.0 - self.f_A - fr[2])
                k_perp, comp_img = geometry.build_layers_ABC_affinity(
                    self.H, self.W,
                    fA=self.f_A, fB=f_B, fC=fr[2], N_A=self.N_A, N_B=self.N_A,
                    CV_thick=self.CV_thick, dist_thick=self.dist_thick,
                    w_int=self.w_int, p_cont=self.p_cont, L_gap=self.L_gap,
                    A_meander=self.A_meander, L_meander=self.L_meander,
                    phi_CA=self.phi_CA,
                    muA=mus[0], sigA=sigs[0], muB=mus[1], sigB=sigs[1], muC=mus[2], sigC=sigs[2],
                    sigma_ln_noise=self.sigma_ln_noise, rng=rng,
                )
            if self.geom == 'parallel':
                k_perp = np.rot90(k_perp)
                comp_img = np.rot90(comp_img)
            k_raw = k_perp

        k = k_raw
        geom_pred = 'parallel' if self.geom=='parallel' else ('perpendicular' if self.geom=='perpendicular' else 'random')
        keff = fm.predicted_keff(geom_pred, fr, mus, sigs)
        t_pred = self.W / max(1e-9, keff)

        # Padded simulation domain (wrap-pad vertically to reduce edge effects)
        pad = self.domain_pad
        k_sim = np.pad(k, ((pad, pad), (0, 0)), mode='wrap') if pad > 0 else k

        return k, comp_img, k_sim, mus, sigs, fr, t_pred

    def _refresh_prediction(self):
        k, _, _k_sim, mus, sigs, fr, t_pred = self._build_field()
        self.t_pred = t_pred
        self.k_harm, self.k_geom, self.k_arith = fm.mixing_law_predictions(fr, mus, sigs)
        self.t_pred_label = (
            f"Predicted t ≈ {t_pred:.2f}  ({self.geom})"
        )

    def _update_colorbar(self, im):
        """Horizontal colorbar below ax_img; created once and reused."""
        if self._cbar_state['cax'] is None or self._cbar_state['cax'] not in self.canvas.fig.axes:
            divider = make_axes_locatable(self.canvas.ax_img)
            cax = divider.append_axes("bottom", size="4%", pad=0.5)
            self._cbar_state['cax'] = cax
        else:
            self._cbar_state['cax'].cla()
        cbar = self.canvas.fig.colorbar(im, cax=self._cbar_state['cax'],
                                        orientation='horizontal')
        self._cbar_state['cbar'] = cbar
        return cbar

    def _toggle_view(self, checked):
        self.view_mode = 'components' if checked else 'k'
        self._redraw_field()

    # --- helpers ---

    def _ml_vlines(self, ax, kind='time'):
        """Draw harmonic/geometric/arithmetic mixing-law lines on a stat axis."""
        if None in (self.k_harm, self.k_geom, self.k_arith):
            return
        W = self.W
        specs = [
            (self.k_harm,  'Harmonic',   'C0'),
            (self.k_geom,  'Geometric',  'C1'),
            (self.k_arith, 'Arithmetic', 'C2'),
        ]
        for k_ml, lbl, col in specs:
            x = W / k_ml if kind == 'time' else k_ml
            ax.axvline(x, color=col, ls='--', lw=1.2, label=lbl, zorder=2)
        ax.legend(fontsize=7, loc='upper right')

    def _update_exit_strip(self):
        """Overlay a thin normalized exit-density strip on the right edge of ax_img."""
        from matplotlib.patches import Rectangle
        if hasattr(self, '_exit_patches'):
            for p in self._exit_patches:
                try: p.remove()
                except Exception: pass
        self._exit_patches = []
        valid = [r for r in self.exit_rows if r is not None]
        if not valid:
            return
        counts = np.bincount(valid, minlength=self.H).astype(float)
        mx = counts.max()
        if mx <= 0:
            return
        counts_norm = counts / mx
        strip_w = max(1.0, self.W * 0.04)
        x0 = self.W - 0.5 - strip_w
        import matplotlib.pyplot as _plt
        cmap = _plt.cm.YlOrRd
        ax = self.canvas.ax_img
        for row in range(self.H):
            rect = Rectangle((x0, row - 0.5), strip_w, 1.0,
                             color=cmap(counts_norm[row]), alpha=0.9,
                             zorder=4, transform=ax.transData)
            ax.add_patch(rect)
            self._exit_patches.append(rect)

    def _clear_exit_strip(self):
        if hasattr(self, '_exit_patches'):
            for p in self._exit_patches:
                try: p.remove()
                except Exception: pass
            self._exit_patches = []

    def _start_row(self, H_sim):
        """Return the walker start row for the simulation domain."""
        if self.rw_start_mode == 'center':
            return H_sim // 2
        return None  # random, chosen inside one_trial

    def _filter_exit(self, exit_row):
        """Map exit_row in simulation domain to display-window row, or None."""
        pad = self.domain_pad
        if pad > 0:
            if self.domain_pad <= exit_row < self.domain_pad + self.H:
                return exit_row - pad
            return None
        return exit_row

    # --- map ---

    def _redraw_field(self):
        import matplotlib as mpl
        k, comp_img, _k_sim, _mus, _sigs, _fr, _ = self._build_field()
        self._clear_exit_strip()

        ax = self.canvas.ax_img
        ax.cla()
        if self.view_mode == 'components':
            n = self.ncomp
            cmap = mpl.colormaps['Set1'].resampled(n)
            im = ax.imshow(comp_img, cmap=cmap, origin='lower',
                           vmin=-0.5, vmax=n - 0.5, interpolation='nearest')
            ax.set_title(f"{self.geom.upper()} — components")
            cbar = self._update_colorbar(im)
            cbar.set_ticks(list(range(n)))
            cbar.set_ticklabels(['A', 'B', 'C'][:n])
        else:
            im = ax.imshow(k, cmap='magma', origin='lower')
            ax.set_title(f"{self.geom.upper()} — k-field")
            self._update_colorbar(im)
        ax.set_xlabel("column  →  flow direction")
        ax.set_ylabel("row")

        # Stat panels
        n_t = len(self.keffs)
        is_kh = (self.solver == 'kirchhoff')
        if self.solver == 'dijkstra':
            keff_title  = f"k_eff = W/t  optimal path — upper bound  (n={n_t})"
            keff_xlabel = "k_eff = W / Σ(1/k)  along path"
        elif self.solver == 'random_walk':
            keff_title  = f"k_eff = geom. mean k along path  ≈ field k_geom  (n={n_t})"
            keff_xlabel = "k_eff = exp( mean(log k) )  along path"
        else:
            keff_title  = f"k_eff  Kirchhoff direct solve  (n={n_t})"
            keff_xlabel = "k_eff = Q W / H  (total flux × W / H)"

        ax_t = self.canvas.ax_time
        ax_t.cla()
        if is_kh:
            ax_t.set_title(f"k_eff convergence  (n={n_t})", fontsize=9)
            ax_t.set_xlabel("realization", fontsize=8)
            ax_t.set_ylabel("k_eff  (running mean)", fontsize=8)
            self._ml_vlines(ax_t, 'keff')
        else:
            ax_t.set_title(f"Travel time  (n={n_t})", fontsize=9)
            ax_t.set_xlabel("t = Σ 1/k  along path", fontsize=8)
            ax_t.set_ylabel("count", fontsize=8)
            self._ml_vlines(ax_t, 'time')

        self.canvas.ax_keff.cla()
        self.canvas.ax_keff.set_title(keff_title, fontsize=9)
        self.canvas.ax_keff.set_xlabel(keff_xlabel, fontsize=8)
        self.canvas.ax_keff.set_ylabel("count", fontsize=8)
        self._ml_vlines(self.canvas.ax_keff, 'keff')

        n_ex = sum(1 for r in self.exit_rows if r is not None)
        ax_e = self.canvas.ax_exit
        ax_e.cla()
        if is_kh and self.last_T is not None:
            ax_e.set_title("Right-boundary exit flux  (last realisation)", fontsize=9)
            ax_e.set_xlabel("flux", fontsize=8)
        else:
            ax_e.set_title(f"Right-boundary exit distribution  (n={n_ex})", fontsize=9)
            ax_e.set_xlabel("count", fontsize=8)
        ax_e.set_ylabel("row", fontsize=8)
        ax_e.set_ylim(-0.5, self.H - 0.5)

        # Fill with existing data if any
        if n_t > 0:
            finite_k = [x for x in self.keffs if np.isfinite(x)]
            if is_kh:
                if len(finite_k) > 1:
                    arr = np.array(finite_k)
                    run_mean = np.cumsum(arr) / np.arange(1, len(arr) + 1)
                    run_std  = np.array([arr[:i+1].std(ddof=0) for i in range(len(arr))])
                    xs = np.arange(1, len(arr) + 1)
                    ax_t.plot(xs, run_mean, color='#4477AA', lw=1.5)
                    ax_t.fill_between(xs, run_mean - run_std, run_mean + run_std,
                                      color='#4477AA', alpha=0.2)
                    self._ml_vlines(ax_t, 'keff')
                if self.last_T is not None:
                    k_disp = k
                    g_r = 2*k_disp[:, -2]*k_disp[:, -1] / (k_disp[:, -2] + k_disp[:, -1] + 1e-30)
                    flux = g_r * (self.last_T[:, -2] - self.last_T[:, -1])
                    ax_e.barh(np.arange(self.H), flux, height=0.85, color='#4477AA')
            else:
                finite_t = [x for x in self.times if np.isfinite(x)]
                if finite_t:
                    ax_t.hist(finite_t, bins=24, color='#4477AA', edgecolor='white')
                    self._ml_vlines(ax_t, 'time')
            if finite_k:
                self.canvas.ax_keff.hist(finite_k, bins=24, color='#4477AA', edgecolor='white')
                self._ml_vlines(self.canvas.ax_keff, 'keff')
            valid_exits = [r for r in self.exit_rows if r is not None]
            if valid_exits and not is_kh:
                counts = np.bincount(valid_exits, minlength=self.H)
                self.canvas.ax_exit.barh(np.arange(self.H), counts, height=0.85, color='#4477AA')
                self._update_exit_strip()

        if self.t_pred_label and not is_kh:
            self.canvas.ax_time.annotate(
                self.t_pred_label, xy=(0.02, 0.97), xycoords='axes fraction',
                va='top', ha='left', fontsize=7, color='gray')

        self.canvas.fig.canvas.draw_idle()

    # ---------------- Simulation ----------------

    def _start_run(self):
        if self.running: return
        self.times = []; self.keffs = []; self.exit_rows = []; self.last_T = None
        self.running = True
        self.act_run.setEnabled(False); self.act_anim.setEnabled(False); self.act_reset.setEnabled(False)
        self.timer.start(1)

    def _stop_run(self):
        if not self.running: return
        self.timer.stop(); self.running = False
        self.act_run.setEnabled(True); self.act_anim.setEnabled(True); self.act_reset.setEnabled(True)

    def _run_chunk(self):
        try:
            N_target = int(10**self.logN)
        except Exception:
            N_target = 100
        remaining = N_target - len(self.keffs)
        if remaining <= 0:
            self._stop_run(); return
        m = min(self.chunk_size, remaining)
        rng = np.random.default_rng()
        for _ in range(m):
            # Rebuild the field for every trial so each realization is independent.
            _, _, k_sim, _, _, _, _ = self._build_field(use_seed=False)
            H_sim = k_sim.shape[0]
            if self.solver == 'kirchhoff':
                k_eff, T = fm.solve_kirchhoff(k_sim, periodic_tb=self.kirchhoff_periodic_tb)
                self.keffs.append(float(k_eff))
                self.last_T = T[:self.H, :]   # strip padding rows for display
            else:
                t, k_eff, exit_row, _ = fm.one_trial(
                    k_sim, self.solver, self.W,
                    start_row=self._start_row(H_sim),
                    beta=self.rw_beta, max_steps=self.rw_max_steps,
                    step_time_model=self.rw_step_time_model, rng=rng,
                )
                self.times.append(float(t))
                self.keffs.append(float(k_eff))
                self.exit_rows.append(self._filter_exit(exit_row))

        n = len(self.keffs)
        finite_k = [x for x in self.keffs if np.isfinite(x)]

        ax_t = self.canvas.ax_time
        ax_t.cla()
        if self.solver == 'kirchhoff':
            ax_t.set_title(f"k_eff convergence  (n={n})", fontsize=9)
            ax_t.set_xlabel("realization", fontsize=8)
            ax_t.set_ylabel("k_eff  (running mean)", fontsize=8)
            if len(finite_k) > 1:
                arr = np.array(finite_k)
                run_mean = np.cumsum(arr) / np.arange(1, len(arr) + 1)
                run_std  = np.array([arr[:i+1].std(ddof=0) for i in range(len(arr))])
                xs = np.arange(1, len(arr) + 1)
                ax_t.plot(xs, run_mean, color='#4477AA', lw=1.5)
                ax_t.fill_between(xs, run_mean - run_std, run_mean + run_std,
                                  color='#4477AA', alpha=0.2)
            self._ml_vlines(ax_t, 'keff')  # same x-scale as k_eff
        else:
            finite_t = [x for x in self.times if np.isfinite(x)]
            ax_t.set_title(f"Travel time  (n={n})", fontsize=9)
            ax_t.set_xlabel("t = Σ 1/k  along path", fontsize=8)
            ax_t.set_ylabel("count", fontsize=8)
            if finite_t:
                ax_t.hist(finite_t, bins=24, color='#4477AA', edgecolor='white')
            self._ml_vlines(ax_t, 'time')
            if self.t_pred_label:
                ax_t.annotate(self.t_pred_label, xy=(0.02, 0.97), xycoords='axes fraction',
                              va='top', ha='left', fontsize=7, color='gray')

        ax_k = self.canvas.ax_keff
        ax_k.cla()
        if self.solver == 'dijkstra':
            keff_title  = f"k_eff = W/t  optimal path — upper bound  (n={n})"
            keff_xlabel = "k_eff = W / Σ(1/k)  along path"
        elif self.solver == 'random_walk':
            keff_title  = f"k_eff = geom. mean k along path  ≈ field k_geom  (n={n})"
            keff_xlabel = "k_eff = exp( mean(log k) )  along path"
        else:
            keff_title  = f"k_eff  Kirchhoff direct solve  (n={n})"
            keff_xlabel = "k_eff = Q W / H  (total flux × W / H)"
        ax_k.set_title(keff_title, fontsize=8)
        ax_k.set_xlabel(keff_xlabel, fontsize=8)
        ax_k.set_ylabel("count", fontsize=8)
        if finite_k:
            ax_k.hist(finite_k, bins=24, color='#4477AA', edgecolor='white')
        self._ml_vlines(ax_k, 'keff')

        ax_e = self.canvas.ax_exit
        ax_e.cla()
        if self.solver == 'kirchhoff' and self.last_T is not None:
            # Right-boundary flux: g_{r,W-2 → r,W-1} × (T[r,W-2] – T[r,W-1])
            T = self.last_T
            k_disp = self._build_field()[0]
            g_r = 2*k_disp[:, -2]*k_disp[:, -1] / (k_disp[:, -2] + k_disp[:, -1] + 1e-30)
            flux = g_r * (T[:, -2] - T[:, -1])
            ax_e.set_title("Right-boundary exit flux  (last realisation)", fontsize=9)
            ax_e.set_xlabel("flux", fontsize=8)
            ax_e.barh(np.arange(self.H), flux, height=0.85, color='#4477AA')
        else:
            valid_exits = [r for r in self.exit_rows if r is not None]
            ax_e.set_title(f"Exit distribution  (n={len(valid_exits)})", fontsize=9)
            ax_e.set_xlabel("count", fontsize=8)
            ax_e.set_ylabel("row", fontsize=8)
            ax_e.set_ylim(-0.5, self.H - 0.5)
            if valid_exits:
                counts = np.bincount(valid_exits, minlength=self.H)
                ax_e.barh(np.arange(self.H), counts, height=0.85, color='#4477AA')
                self._update_exit_strip()
        if self.solver == 'kirchhoff':
            ax_e.set_ylabel("row", fontsize=8)
            ax_e.set_ylim(-0.5, self.H - 0.5)

        self.canvas.fig.canvas.draw_idle()
        if n >= N_target:
            self._stop_run()

    def _animate_one(self):
        k, _, k_sim, _, _, _, _ = self._build_field()
        H_sim = k_sim.shape[0]
        start_row = self._start_row(H_sim)
        rng = np.random.default_rng()

        if self.solver == 'kirchhoff':
            k_eff, T = fm.solve_kirchhoff(k_sim, periodic_tb=self.kirchhoff_periodic_tb)
            T_disp = T[:self.H, :]
            ax = self.canvas.ax_img
            ax.cla()
            import matplotlib as _mpl
            im_k = ax.imshow(k, cmap='magma', origin='lower', alpha=0.5)
            ax.imshow(T_disp, cmap='RdBu_r', origin='lower', alpha=0.7,
                      vmin=0, vmax=1)
            ax.set_title(f"{self.geom.upper()} — Kirchhoff T-field  k_eff={k_eff:.3f}")
            ax.set_xlabel("column  →  flow direction")
            ax.set_ylabel("row")
            self._update_colorbar(im_k)
            self.canvas.fig.canvas.draw_idle()
            return

        if self.solver == 'dijkstra':
            cost = 1.0 / (k_sim + 1e-9)
            t, exit_row, path, explored = fm.dijkstra_random_start_free_exit(
                cost, rng, start_row, return_explored=True)
            solver_label = 'Dijkstra'
        else:
            t, exit_row, path, _ = fm.first_passage_time_random_walk(
                k_sim, beta=self.rw_beta, max_steps=self.rw_max_steps,
                step_time_model=self.rw_step_time_model, rng=rng,
                record_path=True, start_row=start_row)
            explored = None
            solver_label = 'Random walk'

        # Map path coordinates back to display window
        pad = self.domain_pad
        def to_display(r, c):
            return r - pad, c  # c is always in [0, W)

        ax = self.canvas.ax_img
        ax.cla()
        im = ax.imshow(k, cmap='magma', origin='lower')
        ax.set_title(f"{self.geom.upper()} — {solver_label}  (t={t:.2f})")
        ax.set_xlabel("column  →  flow direction")
        ax.set_ylabel("row")
        self._update_colorbar(im)

        if self.solver == 'fm.dijkstra' and explored:
            # Phase 1: scatter of explored cells (clipped to display window)
            sc = ax.scatter([], [], c='yellow', alpha=0.4, s=6, zorder=3)
            (ln,) = ax.plot([], [], color='cyan', lw=2, zorder=4)
            (pt,) = ax.plot([], [], 'wo', ms=5, zorder=5)
            self.canvas.fig.canvas.draw_idle()

            disp_explored = [(r - pad, c) for r, c in explored
                             if -0.5 <= r - pad < self.H + 0.5]
            disp_path = [(r - pad, c) for r, c in path
                         if -0.5 <= r - pad < self.H + 0.5]
            exp_xs = [e[1] for e in disp_explored]
            exp_ys = [e[0] for e in disp_explored]
            path_xs = [p[1] for p in disp_path]
            path_ys = [p[0] for p in disp_path]

            self._anim_phase = 1
            self._anim_index = 0
            batch = max(1, len(disp_explored) // 200)
            self._anim_timer = QtCore.QTimer(self)
            interval_ms = max(10, int(1000 / max(1, self.fps)))

            def step_dijkstra():
                if self._anim_phase == 1:
                    j = min(self._anim_index + batch, len(exp_xs))
                    sc.set_offsets(np.c_[exp_xs[:j], exp_ys[:j]])
                    if j >= len(exp_xs):
                        self._anim_phase = 2
                        self._anim_index = 0
                        sc.set_alpha(0.1)
                    else:
                        self._anim_index = j
                else:
                    j = self._anim_index
                    if j >= len(path_xs):
                        self._anim_timer.stop(); return
                    ln.set_data(path_xs[:j+1], path_ys[:j+1])
                    pt.set_data([path_xs[j]], [path_ys[j]])
                    self._anim_index += 1
                self.canvas.fig.canvas.draw_idle()

            self._anim_timer.timeout.connect(step_dijkstra)
        else:
            # Random walk: simple step-by-step path animation
            disp_path = [(r - pad, c) for r, c in path]
            xs = [p[1] for p in disp_path]
            ys = [p[0] for p in disp_path]
            (ln,) = ax.plot([], [], color='cyan', lw=1.8, zorder=4)
            (pt,) = ax.plot([], [], 'wo', ms=6, zorder=5)
            self.canvas.fig.canvas.draw_idle()
            self._anim_index = 0
            self._anim_timer = QtCore.QTimer(self)
            interval_ms = max(10, int(1000 / max(1, self.fps)))

            def step_rw():
                if self._anim_index >= len(xs):
                    self._anim_timer.stop(); return
                j = self._anim_index
                ln.set_data(xs[:j+1], ys[:j+1])
                pt.set_data([xs[j]], [ys[j]])
                self.canvas.fig.canvas.draw_idle()
                self._anim_index += 1

            self._anim_timer.timeout.connect(step_rw)

        self._anim_timer.start(interval_ms)

    def _reset_plots(self):
        self.times = []; self.keffs = []; self.exit_rows = []; self.last_T = None
        self._redraw_field()

# -------------------- Entry --------------------

def main():
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
