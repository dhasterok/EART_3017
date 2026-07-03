# -*- coding: utf-8 -*-
"""
Mixing-Law Transport GUI (PyQt6)
--------------------------------
A classroom-ready GUI to demonstrate how geometry + mineral components control
effective transport (conductivity k) and traversal times. Students can:
  • Pick geometry: RANDOM (granular), PARALLEL (lanes), SERIES (barriers)
  • Set 2--3 components (means, stds, fractions) assuming lognormal k
  • Adjust grain correlation, fuzziness, corridor/barrier fractions
  • Choose number of realizations with a LOG-SCALED control; run and watch the
    histogram grow in real time
  • Animate a single least-cost path across the field at selectable FPS
  • See mixing-law predictions (arithmetic / geometric / harmonic) as a
    predicted traversal-time line before the run

Requires: PyQt6, matplotlib, numpy
Install (if needed):
  pip install PyQt6 matplotlib numpy

Run:
  python mixing_law_gui_qt.py
"""
from __future__ import annotations
import sys, math, time
from dataclasses import dataclass
import numpy as np

# PyQt6
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget, QLabel, QSlider, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    )

# Matplotlib embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
import matplotlib.pyplot as plt

# -------------------- Math helpers --------------------

def lognormal_mu_sigma_from_mean_std(m: float, s: float):
    m = max(1e-9, float(m))
    s = max(1e-12, float(s))
    sigma2 = float(np.log(1.0 + (s*s)/(m*m)))
    sigma = math.sqrt(sigma2)
    mu = math.log(m) - 0.5*sigma2
    return mu, sigma


def moving_average_2d(arr: np.ndarray, win: int, iters: int = 1) -> np.ndarray:
    if win <= 1:
        return arr
    a = arr.copy()
    H, W = a.shape
    for _ in range(iters):
        py = px = win//2
        p = np.pad(a, ((py,py),(px,px)), mode='reflect')
        ii = p.cumsum(0).cumsum(1)
        y2 = np.arange(win, H+win)
        x2 = np.arange(win, W+win)
        y1 = y2 - win
        x1 = x2 - win
        s = ii[y2[:,None], x2] - ii[y1[:,None], x2] - ii[y2[:,None], x1] + ii[y1[:,None], x1]
        a = s / float(win*win)
    return a


def normalize_mean_k(k: np.ndarray):
    mk = float(k.mean())
    if mk <= 0:
        return k, 1.0
    return k / mk, 1.0/mk

# -------------------- Field builders --------------------

def sample_component_k(comp_idx: int, shape, mu: float, sig: float):
    return np.exp(np.random.normal(mu, sig, size=shape))


def build_random_field(H: int, W: int, fracs, mus, sigs, corr_len: int, fuzz: float):
    K = len(fracs)
    comps = np.random.choice(K, size=(H,W), p=np.array(fracs))
    if corr_len and corr_len > 1:
        score = []
        for i in range(K):
            m = (comps==i).astype(float)
            sc = moving_average_2d(m, corr_len, 1)
            score.append(sc)
        score = np.stack(score, axis=-1)
        comps = np.argmax(score, axis=-1)
    if fuzz > 0:
        mask = np.random.rand(H,W) < min(0.49, fuzz)
        comps[mask] = np.random.randint(0, K, size=mask.sum())
    k = np.zeros((H,W), float)
    for i in range(K):
        mu, sig = mus[i], sigs[i]
        mask = (comps==i)
        if mask.any():
            k[mask] = sample_component_k(i, mask.sum(), mu, sig).reshape(-1)
    # Fill any remaining zeros (edge case) with average of components
    if (k==0).any():
        k[k==0] = np.mean([np.exp(mus[i]+0.5*sigs[i]**2) for i in range(K)])
    return k


def build_parallel_field(H: int, W: int, fracs, mus, sigs, fuzz: float, corridor_frac: float):
    K = len(fracs)
    lane_ends = np.floor(np.cumsum(np.array(fracs))*W).astype(int)
    comps = np.zeros((H,W), int)
    start = 0
    for i, stop in enumerate(lane_ends):
        comps[:, start:stop] = i
        start = stop
    # jitter boundaries per row
    rng = np.random.default_rng()
    for i in range(1, len(lane_ends)):
        b = lane_ends[i-1]
        shifts = rng.integers(-2,3,size=H)
        for r in range(H):
            c = int(np.clip(b + shifts[r], 1, W-2))
            comps[r, c-1:c+1] = rng.integers(0, K)
    if fuzz>0:
        mask = np.random.rand(H,W) < min(0.49, fuzz)
        comps[mask] = np.random.randint(0, K, size=mask.sum())
    k = np.zeros((H,W), float)
    for i in range(K):
        mu, sig = mus[i], sigs[i]
        mask = (comps==i)
        if mask.any():
            k[mask] = sample_component_k(i, mask.sum(), mu, sig).reshape(-1)
    # add high-k corridors along columns
    if corridor_frac > 0:
        num = max(1, int(W*corridor_frac))
        cols = np.linspace(0, W-1, num, dtype=int)
        for c in cols:
            k[:, c] *= 1.5
    return k


def build_series_field(H: int, W: int, fracs, mus, sigs, fuzz: float, barrier_frac: float, thickness: int, strict: bool):
    K = len(fracs)
    layer_ends = np.floor(np.cumsum(np.array(fracs))*H).astype(int)
    comps = np.zeros((H,W), int)
    start = 0
    for i, stop in enumerate(layer_ends):
        comps[start:stop, :] = i
        start = stop
    rng = np.random.default_rng()
    for i in range(1, len(layer_ends)):
        b = layer_ends[i-1]
        shifts = rng.integers(-2,3,size=W)
        for c in range(W):
            r = int(np.clip(b + shifts[c], 1, H-2))
            comps[r-1:r+1, c] = rng.integers(0, K)
    if not strict:
        gap_cols = rng.choice(W, size=max(1, W//20), replace=False)
        comps[::max(1,H//20), gap_cols] = rng.integers(0, K, size=(len(range(0,H,max(1,H//20))), len(gap_cols)))
    if fuzz>0:
        mask = np.random.rand(H,W) < min(0.49, fuzz)
        comps[mask] = np.random.randint(0, K, size=mask.sum())
    k = np.zeros((H,W), float)
    for i in range(K):
        mu, sig = mus[i], sigs[i]
        mask = (comps==i)
        if mask.any():
            k[mask] = sample_component_k(i, mask.sum(), mu, sig).reshape(-1)
    if barrier_frac > 0:
        num = max(1, int(H*barrier_frac))
        rows = np.linspace(0, H-1, num, dtype=int)
        for r in rows:
            k[r:r+thickness, :] *= 0.5
    return k

# -------------------- Path solver --------------------

def dijkstra(cost: np.ndarray, start: tuple[int,int], goal: tuple[int,int]):
    H, W = cost.shape
    INF = 1e18
    dist = np.full((H,W), INF, float)
    prev = np.full((H,W,2), -1, int)
    sx, sy = start; gx, gy = goal
    dist[sx,sy] = cost[sx,sy]
    from heapq import heappush, heappop
    pq = [(dist[sx,sy], (sx,sy))]
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    while pq:
        d,(x,y) = heappop(pq)
        if (x,y)==(gx,gy):
            break
        if d != dist[x,y]:
            continue
        for dx,dy in moves:
            nx,ny = x+dx, y+dy
            if 0<=nx<H and 0<=ny<W:
                nd = d + cost[nx,ny]
                if nd < dist[nx,ny]:
                    dist[nx,ny] = nd
                    prev[nx,ny] = [x,y]
                    heappush(pq, (nd,(nx,ny)))
    path=[]; cx,cy=gx,gy
    if dist[gx,gy] >= INF/2:
        return float('inf'), path
    while not (cx==sx and cy==sy):
        path.append((cx,cy))
        px,py = prev[cx,cy]
        cx,cy = px,py
    path.append((sx,sy)); path.reverse()
    return dist[gx,gy], path

# --- New: Dijkstra with uniform-random start on left edge and free exit on right edge ---
def dijkstra_random_start_free_exit(cost: np.ndarray, rng: np.random.Generator | None = None):
    """
    Choose start row uniformly from [0, H-1] on the left edge (col = 0).
    Run Dijkstra and STOP as soon as we pop any node on the right edge (col = W-1).
    Returns: (time, path)
    """
    H, W = cost.shape
    if rng is None:
        rng = np.random.default_rng()
    start_row = int(rng.integers(0, H))
    sx, sy = start_row, 0

    INF = 1e18
    dist = np.full((H, W), INF, float)
    prev = np.full((H, W, 2), -1, int)
    dist[sx, sy] = cost[sx, sy]

    from heapq import heappush, heappop
    pq = [(dist[sx, sy], (sx, sy))]
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    goal = None
    while pq:
        d, (x, y) = heappop(pq)
        if d != dist[x, y]:
            continue
        # free exit: first time we pop any node on right edge, it is globally minimal
        if y == W - 1:
            goal = (x, y)
            break
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                nd = d + cost[nx, ny]
                if nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    prev[nx, ny] = [x, y]
                    heappush(pq, (nd, (nx, ny)))

    if goal is None or dist[goal[0], goal[1]] >= INF / 2:
        return float('inf'), []

    # backtrack the path
    path = []
    cx, cy = goal
    while not (cx == sx and cy == sy):
        path.append((cx, cy))
        px, py = prev[cx, cy]
        cx, cy = px, py
    path.append((sx, sy))
    path.reverse()
    return float(dist[goal[0], goal[1]]), path

# -------------------- Predictions --------------------

def predicted_keff(geom: str, fracs, mus, sigs) -> float:
    fracs = np.array(fracs)
    means = np.exp(mus + 0.5*(sigs**2))         # E[k]
    inv_means = np.exp(-mus + 0.5*(sigs**2))    # E[1/k] for lognormal
    mean_lnk = mus                               # E[ln k] = mu
    if geom=='parallel':
        keff = float(np.sum(fracs * means))
    elif geom=='series':
        keff = 1.0 / float(np.sum(fracs * inv_means))
    else:
        keff = float(np.exp(np.sum(fracs * mean_lnk)))
    return keff

# -------------------- GUI --------------------
@dataclass
class Component:
    mean: float
    std: float
    frac: float

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig, (self.ax_img, self.ax_hist) = plt.subplots(2, 1, figsize=(7.8, 8.4))
        super().__init__(self.fig)
        self.fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mixing-Law Transport (PyQt6)")
        self.resize(1250, 780)
        # --- State ---
        self.H, self.W = 60, 90
        self.geom = 'random'  # 'random'|'parallel'|'series'
        self.ncomp = 2
        self.comps: list[Component] = [Component(1.2,0.3,0.5), Component(0.6,0.2,0.5), Component(0.9,0.2,0.0)]
        self.corr_len = 0
        self.fuzz = 0.10
        self.corridor_frac = 0.25
        self.barrier_frac  = 0.18
        self.strict_barriers = False
        self.logN = 2.0
        self.fps = 30
        # last predicted time
        self.t_pred = None
        self.times = []
        self.running = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._run_chunk)
        self.chunk_size = 50
        # --- UI ---
        self._build_ui()
        self._refresh_prediction()
        self._redraw_field()

    # ---------------- UI Layout ----------------
    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        hbox = QHBoxLayout(central)

        # Left: Matplotlib canvas + toolbar
        left = QWidget()
        left_v = QVBoxLayout(left)
        self.canvas = MplCanvas()
        self.toolbar = NavToolbar(self.canvas, self)
        left_v.addWidget(self.toolbar)
        left_v.addWidget(self.canvas)
        hbox.addWidget(left, 2)

        # Right: Controls in tabs
        right = QDockWidget("Controls", self)

        container = QWidget()
        dock_layout = QVBoxLayout()
        container.setLayout(dock_layout)

        right.setWidget(container)
        hbox.addWidget(right, 1)


        # --- Tab 1: Geometry ---
        tab_geom = QGroupBox(); dock_layout.addWidget(tab_geom)
        gform = QFormLayout(tab_geom)
        self.cb_geom = QComboBox(); self.cb_geom.addItems(['random','parallel','series'])
        self.cb_geom.currentTextChanged.connect(self._on_geom)
        gform.addRow("Geometry:", self.cb_geom)

        self.sp_corr_len = QSpinBox(); self.sp_corr_len.setRange(0, 30); self.sp_corr_len.setValue(self.corr_len)
        self.sp_corr_len.valueChanged.connect(self._on_params)
        gform.addRow("Grain corr_len (px):", self.sp_corr_len)

        self.sp_fuzz = QDoubleSpinBox(); self.sp_fuzz.setRange(0.0, 0.49); self.sp_fuzz.setSingleStep(0.01); self.sp_fuzz.setDecimals(2); self.sp_fuzz.setValue(self.fuzz)
        self.sp_fuzz.valueChanged.connect(self._on_params)
        gform.addRow("Boundary fuzziness:", self.sp_fuzz)

        self.sp_corridor = QDoubleSpinBox(); self.sp_corridor.setRange(0.0, 0.8); self.sp_corridor.setSingleStep(0.01); self.sp_corridor.setValue(self.corridor_frac)
        self.sp_corridor.valueChanged.connect(self._on_params)
        gform.addRow("Parallel corridors (frac):", self.sp_corridor)

        self.sp_barrier = QDoubleSpinBox(); self.sp_barrier.setRange(0.0, 0.8); self.sp_barrier.setSingleStep(0.01); self.sp_barrier.setValue(self.barrier_frac)
        self.sp_barrier.valueChanged.connect(self._on_params)
        gform.addRow("Series barriers (frac):", self.sp_barrier)

        self.chk_strict = QCheckBox("Strict continuous barriers (series)")
        self.chk_strict.stateChanged.connect(self._on_params)
        gform.addRow(self.chk_strict)

        # --- Tab 2: Components ---
        tab_comp = QGroupBox("Components"); dock_layout.addWidget(tab_comp)
        vcomp = QVBoxLayout(tab_comp)

        row1 = QHBoxLayout(); vcomp.addLayout(row1)
        row1.addWidget(QLabel("# Components:"))
        self.cb_n = QComboBox(); self.cb_n.addItems(['2','3']); self.cb_n.setCurrentIndex(0)
        self.cb_n.currentTextChanged.connect(self._on_ncomp)
        row1.addWidget(self.cb_n); row1.addStretch(1)

        grid = QGridLayout(); vcomp.addLayout(grid)
        headers = ["","k mean","k std","fraction f"]
        for j,h in enumerate(headers):
            lab = QLabel(f"{h}")
            lab.setStyleSheet("font-weight:bold")
            grid.addWidget(lab, 0, j)
        self.le_means = []
        self.le_stds  = []
        self.le_fracs = []
        names = ["1","2","3"]
        for i in range(3):
            grid.addWidget(QLabel(names[i]), i+1, 0)
            e_m = QDoubleSpinBox(); e_m.setRange(1e-6, 1e6); e_m.setDecimals(4); e_m.setValue(self.comps[i].mean)
            e_s = QDoubleSpinBox(); e_s.setRange(0.0, 1e6); e_s.setDecimals(4); e_s.setValue(self.comps[i].std)
            e_f = QDoubleSpinBox(); e_f.setRange(0.0, 1.0); e_f.setSingleStep(0.01); e_f.setDecimals(3); e_f.setValue(self.comps[i].frac)
            e_m.valueChanged.connect(self._on_comp_changed)
            e_s.valueChanged.connect(self._on_comp_changed)
            e_f.valueChanged.connect(self._on_comp_changed)
            self.le_means.append(e_m); self.le_stds.append(e_s); self.le_fracs.append(e_f)
            grid.addWidget(e_m, i+1, 1); grid.addWidget(e_s, i+1, 2); grid.addWidget(e_f, i+1, 3)
        self._update_comp_visibility()

        # --- Tab 3: Simulation ---
        tab_sim = QGroupBox(); dock_layout.addWidget(tab_sim)
        sform = QFormLayout(tab_sim)

        self.sp_H = QSpinBox(); self.sp_H.setRange(20, 300); self.sp_H.setValue(self.H); self.sp_H.valueChanged.connect(self._on_grid)
        self.sp_W = QSpinBox(); self.sp_W.setRange(30, 500); self.sp_W.setValue(self.W); self.sp_W.valueChanged.connect(self._on_grid)
        sform.addRow("Grid H:", self.sp_H); sform.addRow("Grid W:", self.sp_W)

        # log10 N slider via int -> float mapping
        self.sl_logN = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_logN.setRange(100, 400)  # 1.00 .. 4.00
        self.sl_logN.setValue(int(self.logN*100))
        self.sl_logN.valueChanged.connect(self._on_logN)
        self.lab_logN = QLabel(f"log10 N = {self.logN:.2f} (N≈{int(10**self.logN)})")
        sform.addRow(self.lab_logN, self.sl_logN)

        self.sp_fps = QSpinBox(); self.sp_fps.setRange(5, 60); self.sp_fps.setValue(self.fps); self.sp_fps.valueChanged.connect(self._on_params)
        sform.addRow("Animation FPS:", self.sp_fps)

        # Buttons
        self.btn_run = QPushButton("Run")
        self.btn_stop= QPushButton("Stop")
        self.btn_anim= QPushButton("Animate")
        self.btn_reset=QPushButton("Reset")
        self.btn_run.clicked.connect(self._start_run)
        self.btn_stop.clicked.connect(self._stop_run)
        self.btn_anim.clicked.connect(self._animate_one)
        self.btn_reset.clicked.connect(self._reset_plots)
        hb = QHBoxLayout(); hb.addWidget(self.btn_run); hb.addWidget(self.btn_stop); hb.addWidget(self.btn_anim); hb.addWidget(self.btn_reset)
        sform.addRow(hb)

        # prediction readout
        self.lab_pred = QLabel("Prediction: keff=?, t_pred=?")
        sform.addRow(self.lab_pred)

    # ---------------- Event handlers ----------------
    def _on_geom(self, text):
        self.geom = text
        self._refresh_prediction()
        self._redraw_field()

    def _on_params(self, *args):
        self.corr_len = self.sp_corr_len.value()
        self.fuzz = self.sp_fuzz.value()
        self.corridor_frac = self.sp_corridor.value()
        self.barrier_frac = self.sp_barrier.value()
        self.strict_barriers = self.chk_strict.isChecked()
        self.fps = self.sp_fps.value()
        self._refresh_prediction()
        self._redraw_field()

    def _on_ncomp(self, value):
        self.ncomp = int(value)
        self._update_comp_visibility()
        self._refresh_prediction()
        self._redraw_field()

    def _on_comp_changed(self, *args):
        # Update internal comp list
        for i in range(3):
            self.comps[i].mean = self.le_means[i].value()
            self.comps[i].std  = self.le_stds[i].value()
            self.comps[i].frac = self.le_fracs[i].value()
        self._normalize_fracs()
        self._refresh_prediction()
        self._redraw_field()

    def _normalize_fracs(self):
        f = [self.comps[i].frac for i in range(self.ncomp)]
        s = sum(f)
        if s <= 1e-12:
            # default even split
            for i in range(self.ncomp):
                self.comps[i].frac = 1.0/self.ncomp
        else:
            for i in range(self.ncomp):
                self.comps[i].frac = f[i]/s
        # write back to widgets
        for i in range(3):
            if i < self.ncomp:
                self.le_fracs[i].blockSignals(True)
                self.le_fracs[i].setValue(self.comps[i].frac)
                self.le_fracs[i].blockSignals(False)
            else:
                self.le_fracs[i].blockSignals(True); self.le_fracs[i].setValue(0.0); self.le_fracs[i].blockSignals(False)

    def _on_logN(self, val):
        self.logN = val/100.0
        self.lab_logN.setText(f"log10 N = {self.logN:.2f} (N≈{int(10**self.logN)})")

    def _on_grid(self, *args):
        self.H = self.sp_H.value(); self.W = self.sp_W.value()
        self._refresh_prediction(); self._redraw_field()

    def _update_comp_visibility(self):
        # Show/hide row 3
        enable3 = (self.ncomp == 3)
        for w in (self.le_means[2], self.le_stds[2], self.le_fracs[2]):
            w.setEnabled(enable3)
        self._normalize_fracs()

    # ---------------- Build + predict ----------------
    def _build_field(self):
        fr = [self.comps[i].frac for i in range(self.ncomp)]
        means = [self.comps[i].mean for i in range(self.ncomp)]
        stds  = [self.comps[i].std  for i in range(self.ncomp)]
        mus, sigs = [], []
        for m,s in zip(means, stds):
            mu, sg = lognormal_mu_sigma_from_mean_std(m, s)
            mus.append(mu); sigs.append(sg)
        mus = np.array(mus); sigs = np.array(sigs)
        if self.geom == 'random':
            k_raw = build_random_field(self.H, self.W, fr, mus, sigs, self.corr_len, self.fuzz)
        elif self.geom == 'parallel':
            k_raw = build_parallel_field(self.H, self.W, fr, mus, sigs, self.fuzz, self.corridor_frac)
        else:
            k_raw = build_series_field(self.H, self.W, fr, mus, sigs, self.fuzz, self.barrier_frac, 1, self.strict_barriers)
        k, alpha = normalize_mean_k(k_raw)
        keff_raw = predicted_keff(self.geom, fr, mus, sigs)
        keff = alpha * keff_raw
        t_pred = self.W / max(1e-9, keff)
        return k, t_pred

    def _refresh_prediction(self):
        k, t_pred = self._build_field()
        self.t_pred = t_pred
        self.lab_pred.setText(f"Prediction: t_pred ≈ {t_pred:.2f} (geometry={self.geom})")

    def _update_colorbar(self, im):
        """
        Create the colorbar once and reuse it.
        Uses an axes appended to ax_img so we never add extra axes.
        """
        # First time: create cax + colorbar
        if not hasattr(self, '_cax') or self._cax is None or not self._cax in self.canvas.fig.axes:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(self.canvas.ax_img)
            # reserve 4% width on the right with small pad
            self._cax = divider.append_axes("right", size="4%", pad=0.04)
            self._cbar = self.canvas.fig.colorbar(im, cax=self._cax)
        else:
            # Subsequent draws: just update the existing colorbar
            try:
                self._cbar.update_normal(im)
            except Exception:
                # If something went wrong (e.g., artist removed), recreate once
                self._cax.cla()
                self._cbar = self.canvas.fig.colorbar(im, cax=self._cax)

    def _redraw_field(self):
        k, _ = self._build_field()
        self.canvas.ax_img.cla()
        im = self.canvas.ax_img.imshow(k, cmap='magma', origin='lower')
        self.canvas.ax_img.set_title(f"{self.geom.upper()} k-field (mean≈1)")
        self._update_colorbar(im)  # <- reuse, don't add new axes

        # histogram
        self.canvas.ax_hist.cla()
        self.canvas.ax_hist.set_title("Traversal-time histogram")
        self.canvas.ax_hist.set_xlabel("time (sum 1/k)")
        self.canvas.ax_hist.set_ylabel("count")
        if self.t_pred is not None:
            self.canvas.ax_hist.axvline(self.t_pred, color='orange', lw=2, ls='--', label='predicted')
            self.canvas.ax_hist.legend(loc='upper right')
        self.canvas.draw_idle()

    # ---------------- Simulation controls ----------------
    def _start_run(self):
        if self.running:
            return
        self.times = []
        self.running = True
        self.btn_run.setEnabled(False)
        self.btn_anim.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.timer.start(1)

    def _stop_run(self):
        if not self.running:
            return
        self.timer.stop()
        self.running = False
        self.btn_run.setEnabled(True)
        self.btn_anim.setEnabled(True)
        self.btn_reset.setEnabled(True)

    def _run_chunk(self):
        # run a chunk of realizations then update histogram
        try:
            N_target = int(10**self.logN)
        except Exception:
            N_target = 100
        remaining = N_target - len(self.times)
        if remaining <= 0:
            self._stop_run()
            return
        m = min(self.chunk_size, remaining)
        for _ in range(m):
            k, _ = self._build_field()
            cost = 1.0/(k+1e-9)
            # OLD (fixed center start/exit)
            #t, path = dijkstra(cost, (self.H//2,0), (self.H//2,self.W-1))

            # NEW: uniform-random start on left, free exit on right
            t, path = dijkstra_random_start_free_exit(cost)
            self.times.append(float(t))
        # redraw histogram incrementally
        ax = self.canvas.ax_hist
        ax.cla()
        ax.hist(self.times, bins=24, color='#4477AA', edgecolor='white')
        if self.t_pred is not None:
            ax.axvline(self.t_pred, color='orange', lw=2, ls='--', label='predicted')
            ax.legend(loc='upper right')
        ax.set_title(f"Traversal-time histogram (n={len(self.times)})")
        ax.set_xlabel("time (sum 1/k)")
        ax.set_ylabel("count")
        self.canvas.draw_idle()
        # if finished, stop
        if len(self.times) >= N_target:
            self._stop_run()

    def _animate_one(self):
        # build one field and animate at self.fps using a local QTimer
        k, _ = self._build_field()
        cost = 1.0/(k+1e-9)
        # OLD (fixed center start/exit)
        # t, path = dijkstra(cost, (self.H//2,0), (self.H//2,self.W-1))

        # NEW
        t, path = dijkstra_random_start_free_exit(cost)

        xs = [p[1] for p in path]; ys=[p[0] for p in path]
        self.canvas.ax_img.cla()
        im = self.canvas.ax_img.imshow(k, cmap='magma', origin='lower')
        self.canvas.ax_img.set_title(f"{self.geom.upper()} path animation (t={t:.2f})")
        self._update_colorbar(im)  # <- reuse, don't add new axes

        (ln,) = self.canvas.ax_img.plot([], [], color='cyan', lw=1.8)
        (pt,) = self.canvas.ax_img.plot([], [], 'wo', ms=6)
        self.canvas.draw_idle()
        # timer-driven frame advance
        self._anim_index = 0
        self._anim_timer = QtCore.QTimer(self)
        interval_ms = max(10, int(1000/self.fps))
        def step():
            j = min(self._anim_index, len(xs)-1)
            ln.set_data(xs[:j+1], ys[:j+1])
            pt.set_data([xs[j]], [ys[j]])
            self.canvas.draw_idle()
            self._anim_index += 1
            if self._anim_index >= len(xs):
                self._anim_timer.stop()
        self._anim_timer.timeout.connect(step)
        self._anim_timer.start(interval_ms)

    def _reset_plots(self):
        self.times = []
        self._redraw_field()

# -------------------- Entry --------------------
def main():
    app = QApplication(sys.argv)
    # High-DPI on modern displays
    #QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    #QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
