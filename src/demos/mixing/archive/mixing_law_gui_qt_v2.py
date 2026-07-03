# -*- coding: utf-8 -*-
"""
Mixing-Law Transport GUI (PyQt6) — Layering V2
-----------------------------------------------
This version integrates intuitive layering controls:
  • Two-component layers: N_A, N_B (thick↔thin), contact sharpness (erf half-width w_int),
    continuity p_cont and mean gap length L_gap, gentle meander (A_meander, L_meander),
    thickness variability (CV_thick, distribution).
  • Three-component (A--B--C) with affinities (phi_CA, phi_CB), attach fraction (eta_attach),
    and threshold (tau_aff); optional standalone C layers.
  • Parallel geometry reuses the series builder via 90° rotation.
  • Dijkstra with uniform-random start (left) + free exit (right); stable colorbar; animation fix.

Dependencies: PyQt6, matplotlib, numpy (no SciPy required; uses math.erf via vectorization).
Run:  python mixing_law_gui_qt_v2.py
"""

from __future__ import annotations
import sys, math
from dataclasses import dataclass
import numpy as np
from math import erf as _erf

# PyQt6
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QDockWidget,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtGui import QFont

# Matplotlib embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------- Small helpers --------------------

def lognormal_mu_sigma_from_mean_std(m: float, s: float):
    m = max(1e-9, float(m))
    s = max(1e-12, float(s))
    sigma2 = float(np.log(1.0 + (s*s)/(m*m)))
    sigma = math.sqrt(sigma2)
    mu = math.log(m) - 0.5*sigma2
    return mu, sigma

# vectorized erf without SciPy
_erf_vec = np.vectorize(_erf, otypes=[float])

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

# -------------------- Random granular field (unchanged mode) --------------------

def build_random_field(H: int, W: int, fracs, mus, sigs, corr_len: int, sigma_ln_noise: float):
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
    k = np.zeros((H,W), float)
    for i in range(K):
        mu, sig = mus[i], sigs[i]
        mask = (comps==i)
        if mask.any():
            lnk = np.random.normal(mu, sig, size=mask.sum())
            if sigma_ln_noise>0:
                lnk += np.random.normal(0, sigma_ln_noise, size=mask.sum())
            k[mask] = np.exp(lnk)
    mk = k.mean()
    if mk>0: k /= mk
    return k

# -------------------- Layer builders (intuitive controls) --------------------

def _draw_thicknesses(rng, N, target_sum, CV, dist):
    if N <= 0:
        return np.array([])
    if CV <= 1e-12:
        t = np.ones(N)
    else:
        if dist == "lognormal":
            s = np.sqrt(np.log(1.0 + CV**2))
            m = -0.5*s**2
            t = np.exp(rng.normal(m, s, size=N))
        else:
            a = np.sqrt(3.0) * CV
            t = np.clip(1.0 + rng.uniform(-a, +a, size=N), 1e-3, None)
    ssum = t.sum()
    if ssum>0:
        t *= (target_sum/ssum)
    return t


def build_two_component_layers(
    H, W,
    f_A=0.5,
    N_A=1, N_B=2,
    CV_thick=0.0, dist_thick="lognormal",
    w_int=0.0,
    p_cont=1.0, L_gap=10,
    A_meander=0.0, L_meander=999999,
    muA=0.0, sigA=0.3,
    muB=-0.4, sigB=0.3,
    sigma_ln_noise=0.0,
    rng=None
):
    """Series-oriented layering (layers ⟂ flow). Parallel achieved by rotation."""
    if rng is None:
        rng = np.random.default_rng()
    f_B = 1.0 - f_A
    N_A = max(1, int(N_A)); N_B = max(2, int(N_B))
    tA = _draw_thicknesses(rng, N_A, f_A*H, CV_thick, dist_thick)
    tB = _draw_thicknesses(rng, N_B, f_B*H, CV_thick, dist_thick)

    # Interleave starting with larger fraction
    seq, iA, iB, turnA = [], 0, 0, (f_A >= f_B)
    while iA < N_A or iB < N_B:
        if turnA and iA < N_A: seq.append(("A", tA[iA])); iA += 1
        elif (not turnA) and iB < N_B: seq.append(("B", tB[iB])); iB += 1
        turnA = not turnA
    for j in range(iA, N_A): seq.append(("A", tA[j]))
    for j in range(iB, N_B): seq.append(("B", tB[j]))

    # Straight cumulative edges
    y_edges = [0.0]
    for _, th in seq: y_edges.append(y_edges[-1] + th)
    y_edges[-1] = H

    # Meandering interfaces
    x = np.arange(W)
    phases = np.random.uniform(0, 2*np.pi, size=len(seq)-1) if L_meander>0 else np.zeros(len(seq)-1)
    y_if = []
    for i in range(1, len(seq)):
        y0 = 0.5*(y_edges[i-1] + y_edges[i])
        if A_meander>0 and L_meander>0:
            yi = y0 + A_meander*np.sin(2*np.pi*x/max(1, L_meander) + phases[i-1])
        else:
            yi = np.full(W, y0)
        y_if.append(yi)

    # Hard labels + leaks
    label = np.zeros((H,W), np.uint8)
    for i,(lab,th) in enumerate(seq):
        r0,r1 = int(round(y_edges[i])), int(round(y_edges[i+1])); r1 = max(r0+1, min(H, r1))
        label[r0:r1, :] = 1 if lab=='A' else 0
    if p_cont < 1.0:
        for i,(lab,th) in enumerate(seq):
            if np.random.random() > p_cont:
                ngaps = np.random.randint(1,4)
                r0,r1 = int(round(y_edges[i])), int(round(y_edges[i+1]))
                for _ in range(ngaps):
                    glen = max(1, int(np.random.normal(L_gap, 0.3*L_gap)))
                    cx = np.random.randint(0, max(1, W-glen))
                    label[r0:r1, cx:cx+glen] = 1 - (1 if lab=='A' else 0)

    # Soft alpha via erf distance to nearest interface
    if w_int <= 1e-12 or not y_if:
        alphaA = label.astype(float)
    else:
        yy = np.arange(H)[:, None] * np.ones((1,W))
        dist = np.full((H,W), np.inf)
        for yi in y_if:
            dist = np.minimum(dist, np.abs(yy - yi[None,:]))
        sign = np.where(label>0, 1.0, -1.0)
        d_signed = sign * dist
        alphaA = 0.5*(1.0 + _erf_vec(d_signed/(np.sqrt(2.0)*max(1e-9, w_int))))
    alphaB = 1.0 - alphaA

    # ln k fields and geometric blend
    lnkA = np.random.normal(muA, sigA, size=(H, W))
    lnkB = np.random.normal(muB, sigB, size=(H, W))
    if sigma_ln_noise>0:
        noise = np.random.normal(0, sigma_ln_noise, size=(H, W))
        lnkA += noise; lnkB += noise
    lnk = alphaA*lnkA + alphaB*lnkB
    k = np.exp(lnk)
    mk = k.mean()
    if mk>0: k /= mk
    return k


def build_layers_ABC_affinity(
    H, W,
    fA=0.45, fB=0.45, fC=0.10,
    N_A=1, N_B=2,
    CV_thick=0.0, dist_thick="lognormal",
    w_int=0.0, p_cont=1.0, L_gap=10,
    A_meander=0.0, L_meander=999999,
    phi_CA=0.8, phi_CB=0.2, eta_attach=0.7, tau_aff=0.2, N_C_min=0,
    muA=0.0,  sigA=0.3,
    muB=-0.4, sigB=0.3,
    muC=-0.2, sigC=0.35,
    sigma_ln_noise=0.0,
):
    # 1) Build A/B scaffold to get alphaA (ignore absolute k)
    alphaA = build_two_component_layers(
        H, W, f_A=fA, N_A=N_A, N_B=N_B, CV_thick=CV_thick, dist_thick=dist_thick,
        w_int=w_int, p_cont=p_cont, L_gap=L_gap, A_meander=A_meander, L_meander=L_meander,
        muA=0.0, sigA=1e-6, muB=-10.0, sigB=1e-6, sigma_ln_noise=0.0
    )
    alphaA = np.clip(alphaA, 0.0, 1.0)
    alphaB = 1.0 - alphaA

    # 2) Split C into attached and standalone
    eps = 1e-12
    fC_attach = eta_attach * fC
    if (phi_CA < tau_aff) and (phi_CB < tau_aff):
        fC_attach = 0.0
    denom = max(eps, phi_CA + phi_CB)
    fC_toA = fC_attach * (phi_CA/denom)
    fC_toB = fC_attach - fC_toA
    fC_standalone = max(0.0, fC - (fC_toA + fC_toB))

    # 3) Attach C proportionally in A/B
    WA = phi_CA * alphaA; WB = phi_CB * alphaB
    SA, SB = WA.sum(), WB.sum()
    if SA>eps: WA *= (fC_toA * H * W) / SA
    else: WA[:] = 0
    if SB>eps: WB *= (fC_toB * H * W) / SB
    else: WB[:] = 0
    alphaC = WA + WB
    overflow = np.maximum(0.0, (alphaA + alphaB + alphaC) - 1.0)
    if np.any(overflow>0):
        denomAB = alphaA + alphaB + eps
        alphaA -= overflow * (alphaA/denomAB)
        alphaB -= overflow * (alphaB/denomAB)

    # 4) Standalone C layers (soft bands)
    if fC_standalone > 1e-9:
        N_C = max(int(N_C_min), 1)
        t_total = fC_standalone * H
        y_edges = np.linspace(0, H, N_C+1)
        x = np.arange(W)
        phases = np.random.uniform(0, 2*np.pi, size=N_C) if L_meander>0 else np.zeros(N_C)
        for i in range(N_C):
            y0 = 0.5*(y_edges[i] + y_edges[i+1])
            yc = y0 + (A_meander*np.sin(2*np.pi*x/max(1,L_meander) + phases[i]) if A_meander>0 and L_meander>0 else 0.0)
            yy = np.arange(H)[:, None] * np.ones((1, W))
            half = 0.5 * (t_total / N_C)
            if w_int<=1e-12:
                maskC = (np.abs(yy - yc[None,:]) <= half).astype(float)
            else:
                maskC = 0.5 * (_erf_vec((half - np.abs(yy - yc[None,:]))/(np.sqrt(2.0)*max(1e-9, w_int))) + 1.0)
            alphaC += maskC
            total = alphaA + alphaB + alphaC + eps
            alphaA *= 1.0/total; alphaB *= 1.0/total; alphaC *= 1.0/total

    # 5) ln k and geometric blend
    lnkA = np.random.normal(muA, sigA, size=(H, W))
    lnkB = np.random.normal(muB, sigB, size=(H, W))
    lnkC = np.random.normal(muC, sigC, size=(H, W))
    if sigma_ln_noise>0:
        noise = np.random.normal(0, sigma_ln_noise, size=(H, W))
        lnkA += noise; lnkB += noise; lnkC += noise
    lnk = alphaA*lnkA + alphaB*lnkB + alphaC*lnkC
    k = np.exp(lnk)
    mk = k.mean()
    if mk>0: k /= mk
    return k

# -------------------- Dijkstra (with random start + free exit) --------------------

def dijkstra_random_start_free_exit(cost, rng=None):
    H, W = cost.shape
    if rng is None: rng = np.random.default_rng()
    start_row = int(rng.integers(0, H)); sx, sy = start_row, 0
    INF = 1e18
    dist = np.full((H, W), INF, float)
    prev = np.full((H, W, 2), -1, int)
    dist[sx, sy] = cost[sx, sy]
    from heapq import heappush, heappop
    pq = [(dist[sx, sy], (sx, sy))]
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    goal = None
    while pq:
        d,(x,y) = heappop(pq)
        if d != dist[x,y]:
            continue
        if y == W-1:  # free exit
            goal = (x,y); break
        for dx,dy in moves:
            nx,ny = x+dx, y+dy
            if 0<=nx<H and 0<=ny<W:
                nd = d + cost[nx,ny]
                if nd < dist[nx,ny]:
                    dist[nx,ny] = nd
                    prev[nx,ny] = [x,y]
                    heappush(pq, (nd,(nx,ny)))
    if goal is None:
        return float('inf'), []
    path=[]; cx,cy = goal
    while not (cx==sx and cy==sy):
        path.append((cx,cy)); cx,cy = prev[cx,cy]
    path.append((sx,sy)); path.reverse()
    return float(dist[goal]), path

# -------------------- Prediction (simple proxy) --------------------

def predicted_keff(geom: str, fracs, mus, sigs) -> float:
    fracs = np.array(fracs)
    means = np.exp(mus + 0.5*(sigs**2))      # E[k]
    inv_means = np.exp(-mus + 0.5*(sigs**2)) # E[1/k]
    mean_lnk = mus                            # E[ln k] = mu
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
        self.fig, (self.ax_img, self.ax_hist) = plt.subplots(2, 1, figsize=(8.2, 8.8))
        super().__init__(self.fig)
        self.fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mixing-Law Transport (PyQt6) — Layering V2")
        self.resize(1280, 820)
        # ---- State ----
        self.H, self.W = 60, 90
        self.geom = 'random'          # 'random' | 'parallel' | 'series'
        self.ncomp = 2
        self.comps: list[Component] = [Component(1.2,0.3,0.5), Component(0.6,0.2,0.5), Component(0.9,0.2,0.0)]
        # Random granular params
        self.corr_len = 0
        self.sigma_ln_noise = 0.0
        # Layering params
        self.N_A = 1; self.N_B = 2
        self.CV_thick = 0.0
        self.dist_thick = 'lognormal'
        self.w_int = 0.0
        self.p_cont = 1.0
        self.L_gap = 10
        self.A_meander = 0.0
        self.L_meander = 999999
        # 3rd component affinity
        self.phi_CA = 0.8; self.phi_CB = 0.2
        self.eta_attach = 0.7; self.tau_aff = 0.2; self.N_C_min = 0
        # Run / animate
        self.logN = 2.0
        self.fps = 30
        self.t_pred = None
        self.times = []
        self.running = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._run_chunk)
        self.chunk_size = 50
        # colorbar state
        self._cbar_state = {'cax': None, 'cbar': None}
        # UI
        self._build_ui()
        self._refresh_prediction()
        self._redraw_field()

        # --- Field cache (prevents re-randomizing unless params change) ---
        self._field_cache = {
            'params_key': None,   # tuple snapshot of parameters
            'k': None,            # cached normalized k(H,W)
        }
        
        # fixed RNG seed per field realization for reproducibility
        self._field_seed = np.random.SeedSequence().entropy  # or any int; changes only on regenerate

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        hbox = QHBoxLayout(central)

        # Left: Matplotlib
        left = QWidget()
        left_v = QVBoxLayout(left)
        self.canvas = MplCanvas()
        self.toolbar = NavToolbar(self.canvas, self)
        left_v.addWidget(self.toolbar)
        left_v.addWidget(self.canvas)
        hbox.addWidget(left, 2)

        # Right: Controls
        dock = QDockWidget("Controls", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

        dock_contianer = QWidget()
        rv = QVBoxLayout(dock_contianer)
        dock.setWidget(dock_contianer)

        # Geometry group
        g_geom = QGroupBox("Geometry & Random Granular")
        gv = QFormLayout(g_geom)
        self.cb_geom = QComboBox(); self.cb_geom.addItems(['random','parallel','series'])
        self.cb_geom.currentTextChanged.connect(self._on_geom)
        gv.addRow("Geometry:", self.cb_geom)
        self.sp_corr_len = QSpinBox(); self.sp_corr_len.setRange(0, 30); self.sp_corr_len.setValue(self.corr_len)
        self.sp_corr_len.valueChanged.connect(self._on_params)
        gv.addRow("Granular corr_len (px):", self.sp_corr_len)
        self.sp_sigma_ln = QDoubleSpinBox(); self.sp_sigma_ln.setRange(0.0, 1.0); self.sp_sigma_ln.setSingleStep(0.01); self.sp_sigma_ln.setValue(self.sigma_ln_noise)
        self.sp_sigma_ln.valueChanged.connect(self._on_params)
        gv.addRow("ln k noise (granular/layers):", self.sp_sigma_ln)
        rv.addWidget(g_geom)

        # Layering controls
        layer_group = QGroupBox("Layering (Series ⟂ flow; Parallel by rotation)")
        layer_layout = QVBoxLayout(layer_group)

        # Number of layers controls
        nlayer_layout = QHBoxLayout()
        self.sp_NA = QSpinBox(); self.sp_NA.setRange(1, 10); self.sp_NA.setValue(self.N_A); self.sp_NA.valueChanged.connect(self._on_params)
        self.sp_NB = QSpinBox(); self.sp_NB.setRange(2, 11); self.sp_NB.setValue(self.N_B); self.sp_NB.valueChanged.connect(self._on_params)
        nlayer_layout.addWidget(QLabel("N Layers"))
        nlayer_layout.addWidget(QLabel("A:"))
        nlayer_layout.addWidget(self.sp_NA)
        nlayer_layout.addWidget(QLabel("B:"))
        nlayer_layout.addWidget(self.sp_NB)
        
        # Thickness variability (coefficient of variation, CV) controls
        thickness_layout = QHBoxLayout()
        self.sp_CV = QDoubleSpinBox()
        self.sp_CV.setRange(0.0, 1.5)
        self.sp_CV.setSingleStep(0.05)
        self.sp_CV.setValue(self.CV_thick)
        self.sp_CV.valueChanged.connect(self._on_params)
        self.sp_CV.setToolTip("Thickness variability: std/mean 0 = all beds same thickness; 0.1--0.3 = natural variation; 0.5+ = very uneven.")

        # Thickness distribution
        self.cb_dist = QComboBox()
        self.cb_dist.addItems(["lognormal","uniform"])
        self.cb_dist.currentTextChanged.connect(self._on_params)
        thickness_layout.addWidget(QLabel("Thickness"))
        thickness_layout.addWidget(QLabel("variance:"))
        thickness_layout.addWidget(self.sp_CV)
        thickness_layout.addWidget(QLabel("distribution:"))
        thickness_layout.addWidget(self.cb_dist)

        # Contact sharpness via erf half-width
        layer_form_layout = QFormLayout()
        self.sp_wint = QDoubleSpinBox()
        self.sp_wint.setRange(0.0, 10.0)
        self.sp_wint.setSingleStep(0.5)
        self.sp_wint.setValue(self.w_int)
        self.sp_wint.valueChanged.connect(self._on_params)
        self.sp_wint.setToolTip("Contact sharpness: half-width of erf transition (in pixels); 0 = sharp; 1--3 = natural diffusion; 5+ = very diffuse.")
        layer_form_layout.addRow("half-width:", self.sp_wint)

        # Layer continuity and gaps
        gap_layout = QHBoxLayout()
        self.sp_pcont = QDoubleSpinBox()
        self.sp_pcont.setRange(0.0,1.0)
        self.sp_pcont.setSingleStep(0.05)
        self.sp_pcont.setValue(self.p_cont)
        self.sp_pcont.valueChanged.connect(self._on_params)
        self.sp_pcont.setToolTip("Probability that a given layer is continuous across the profile (1.0 = all continuous; 0.5 = half have gaps; 0.0 = all have gaps)")

        # Gaps (only if not continuous)
        self.sp_Lgap = QSpinBox()
        self.sp_Lgap.setRange(1, 500)
        self.sp_Lgap.setValue(self.L_gap)
        self.sp_Lgap.valueChanged.connect(self._on_params)
        self.sp_Lgap.setToolTip("Mean gap length in pixels (only if layers are not continuous)")

        gap_layout.addWidget(QLabel("Continuity"))
        gap_layout.addWidget(QLabel("Probability:"))
        gap_layout.addWidget(self.sp_pcont)
        gap_layout.addWidget(QLabel("Mean gap:"))
        gap_layout.addWidget(self.sp_Lgap)

        # Folded interfaces
        fold_layout = QHBoxLayout()

        # Fold amplitude
        self.sp_Ame = QDoubleSpinBox()
        self.sp_Ame.setRange(0.0, 20.0)
        self.sp_Ame.setSingleStep(0.5)
        self.sp_Ame.setValue(self.A_meander)
        self.sp_Ame.valueChanged.connect(self._on_params)
        self.sp_Ame.setToolTip("Amplitude of folded interfaces (in pixels)")

        # Fold wavelength controls
        self.sp_Lme = QSpinBox()
        self.sp_Lme.setRange(1, 999999)
        self.sp_Lme.setValue(self.L_meander)
        self.sp_Lme.valueChanged.connect(self._on_params)
        self.sp_Lme.setToolTip("Wavelength of folded interfaces (in pixels)")

        fold_layout.addWidget(QLabel("Folds"))
        fold_layout.addWidget(QLabel("A:"))
        fold_layout.addWidget(self.sp_Ame)
        fold_layout.addWidget(QLabel("λ:"))
        fold_layout.addWidget(self.sp_Lme)

        layer_layout.addLayout(nlayer_layout)
        layer_layout.addLayout(thickness_layout)
        layer_layout.addLayout(layer_form_layout)
        layer_layout.addLayout(gap_layout)
        layer_layout.addLayout(fold_layout)

        rv.addWidget(layer_group)

        # Components controls
        mineral_group = QGroupBox("Minerals (means, stds, fractions)")
        vcomp = QVBoxLayout(mineral_group)
        row1 = QHBoxLayout(); vcomp.addLayout(row1)
        row1.addWidget(QLabel("# Minerals:"))
        self.cb_n = QComboBox()
        self.cb_n.addItems(['2','3'])
        self.cb_n.setCurrentIndex(0)
        self.cb_n.currentTextChanged.connect(self._on_ncomp)
        row1.addWidget(self.cb_n); row1.addStretch(1)
        grid = QGridLayout()
        vcomp.addLayout(grid)

        headers = ["", "k mean", "k std", "fraction f"]
        for j,h in enumerate(headers):
            lab = QLabel(h); lab.setStyleSheet("font-weight:bold"); grid.addWidget(lab, 0, j)
        self.le_means=[]; self.le_stds=[]; self.le_fracs=[]
        names=["A","B","C"]
        init = self.comps
        for i in range(3):
            grid.addWidget(QLabel(names[i]), i+1, 0)
            e_m = QDoubleSpinBox(); e_m.setRange(1e-6, 1e6); e_m.setDecimals(4); e_m.setValue(init[i].mean)
            e_s = QDoubleSpinBox(); e_s.setRange(0.0, 1e6); e_s.setDecimals(4); e_s.setValue(init[i].std)
            e_f = QDoubleSpinBox(); e_f.setRange(0.0, 1.0); e_f.setSingleStep(0.01); e_f.setDecimals(3); e_f.setValue(init[i].frac)
            e_m.valueChanged.connect(self._on_comp_changed)
            e_s.valueChanged.connect(self._on_comp_changed)
            e_f.valueChanged.connect(self._on_comp_changed)
            self.le_means.append(e_m); self.le_stds.append(e_s); self.le_fracs.append(e_f)
            grid.addWidget(e_m, i+1, 1); grid.addWidget(e_s, i+1, 2); grid.addWidget(e_f, i+1, 3)
        rv.addWidget(mineral_group)

        # Affinity controls (only meaningful when 3 components)
        affinity_group = QGroupBox("Third component C — affinities & attach")
        affinity_layout = QVBoxLayout(affinity_group)

        component_layout = QHBoxLayout()
        self.sp_phiCA = QDoubleSpinBox()
        self.sp_phiCA.setRange(0.0,1.0)
        self.sp_phiCA.setSingleStep(0.05)
        self.sp_phiCA.setValue(self.phi_CA)
        self.sp_phiCA.valueChanged.connect(self._on_params)
        self.sp_phiCA.setToolTip("Affinity of C to A (0 = no affinity, 1 = perfect affinity). Only matters if phi_CB < tau_aff.")

        self.sp_phiCB = QDoubleSpinBox()
        self.sp_phiCB.setRange(0.0,1.0)
        self.sp_phiCB.setSingleStep(0.05)
        self.sp_phiCB.setValue(self.phi_CB)
        self.sp_phiCB.valueChanged.connect(self._on_params)
        self.sp_phiCB.setToolTip("Affinity of C to B (0 = no affinity, 1 = perfect affinity). Only matters if phi_CA < tau_aff.")

        component_layout.addWidget(QLabel("Affinities:"))
        component_layout.addWidget(QLabel("C→A:"))
        component_layout.addWidget(self.sp_phiCA)
        component_layout.addWidget(QLabel("C→B:"))
        component_layout.addWidget(self.sp_phiCB)

        af = QFormLayout(affinity_group)
        self.sp_eta = QDoubleSpinBox()
        self.sp_eta.setRange(0.0,1.0)
        self.sp_eta.setSingleStep(0.05)
        self.sp_eta.setValue(self.eta_attach)
        self.sp_eta.valueChanged.connect(self._on_params)
        self.sp_eta.setToolTip("Fraction of C layers that are attached to A or B (0 = none, 1 = all).")

        self.sp_tau = QDoubleSpinBox()
        self.sp_tau.setRange(0.0,1.0)
        self.sp_tau.setSingleStep(0.05)
        self.sp_tau.setValue(self.tau_aff)
        self.sp_tau.valueChanged.connect(self._on_params)
        self.sp_tau.setToolTip("Threshold for considering an affinity significant.")

        self.sp_NCmin = QSpinBox()
        self.sp_NCmin.setRange(0,50)
        self.sp_NCmin.setValue(self.N_C_min)
        self.sp_NCmin.valueChanged.connect(self._on_params)

        af.addRow("Attach fraction (eta_attach):", self.sp_eta)
        af.addRow("Affinity threshold (tau_aff):", self.sp_tau)
        af.addRow("Min C-only layers (N_C_min):", self.sp_NCmin)

        affinity_layout.addLayout(component_layout)
        affinity_layout.addLayout(af)

        rv.addWidget(affinity_group)

        # Simulation
        simulation_control_group = QGroupBox("Simulation & Prediction")
        simulation_control_layout = QVBoxLayout(simulation_control_group)

        grid_layout = QHBoxLayout()

        self.sp_H = QSpinBox()
        self.sp_H.setRange(20, 300)
        self.sp_H.setValue(self.H)
        self.sp_H.valueChanged.connect(self._on_grid)
        self.sp_H.setToolTip("Grid height in pixels")

        self.sp_W = QSpinBox()
        self.sp_W.setRange(30, 500)
        self.sp_W.setValue(self.W)
        self.sp_W.valueChanged.connect(self._on_grid)
        self.sp_W.setToolTip("Grid width in pixels")

        grid_layout.addWidget(QLabel("Grid size:"))
        grid_layout.addWidget(QLabel("H:"))
        grid_layout.addWidget(self.sp_H)
        grid_layout.addWidget(QLabel("W:"))
        grid_layout.addWidget(self.sp_W)

        sform = QFormLayout()
        self.sl_logN = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_logN.setRange(100, 400)
        self.sl_logN.setValue(int(self.logN*100))
        self.sl_logN.valueChanged.connect(self._on_logN)
        self.lab_logN = QLabel(f"log10 N = {self.logN:.2f} (N≈{int(10**self.logN)})")
        self.lab_logN.setToolTip("log10 of number of random realizations to run for prediction (higher = slower but more accurate)")
        sform.addRow(self.lab_logN, self.sl_logN)

        self.sp_fps = QSpinBox()
        self.sp_fps.setRange(5, 60)
        self.sp_fps.setValue(self.fps)
        self.sp_fps.valueChanged.connect(self._on_params)
        self.sp_fps.setToolTip("Animation frames per second (higher = faster animation)")
        sform.addRow("Animation FPS:", self.sp_fps)

        self.btn_run = QPushButton("Run")
        self.btn_run.setToolTip("Run the simulation")
        self.btn_stop= QPushButton("Stop")
        self.btn_stop.setToolTip("Stop the simulation")
        self.btn_anim= QPushButton("Animate")
        self.btn_anim.setToolTip("Animate a single path")
        self.btn_reset=QPushButton("Reset")
        self.btn_reset.setToolTip("Reset all plots and data")
        self.btn_run.clicked.connect(self._start_run)
        self.btn_stop.clicked.connect(self._stop_run)
        self.btn_anim.clicked.connect(self._animate_one)
        self.btn_reset.clicked.connect(self._reset_plots)

        hb = QHBoxLayout()
        hb.addWidget(self.btn_run)
        hb.addWidget(self.btn_stop)
        hb.addWidget(self.btn_anim)
        hb.addWidget(self.btn_reset)

        self.lab_pred = QLabel("Prediction: keff=?, t_pred=?")
        simulation_control_layout.addLayout(grid_layout)
        simulation_control_layout.addLayout(sform)
        simulation_control_layout.addWidget(self.lab_pred)
        simulation_control_layout.addLayout(hb)

        rv.addWidget(simulation_control_group)

        self._update_comp_visibility()

    # ---------------- Parameter Cache ----------------
    def _params_key(self) -> tuple:
        """Build a tuple that uniquely represents all inputs affecting k(x,y)."""
        # geometry
        key = [self.geom, int(self.H), int(self.W)]

        # components (means, stds, fractions) — include only the active ncomp
        for i in range(self.ncomp):
            key += [float(self.comps[i].mean), float(self.comps[i].std), float(self.comps[i].frac)]
        key += [int(self.ncomp)]

        # granular mode knobs
        key += [int(self.corr_len), float(self.sigma_ln_noise)]

        # layering (series builder; parallel is rotation so same knobs apply)
        key += [
            int(self.N_A), int(self.N_B),
            float(self.CV_thick), self.dist_thick,
            float(self.w_int), float(self.p_cont), int(self.L_gap),
            float(self.A_meander), int(self.L_meander)
        ]

        # third-component affinities (even if ncomp==2 they won't be used, but harmless)
        key += [
            float(self.phi_CA), float(self.phi_CB),
            float(self.eta_attach), float(self.tau_aff), int(self.N_C_min)
        ]

        # include the current field seed so identical params & seed ⇒ identical pixels
        key += [int(self._field_seed)]
        return tuple(key)

    def _invalidate_cache(self):
        """Call whenever a control that changes the field is modified."""
        self._field_cache['params_key'] = None
        self._field_cache['k'] = None

    # ---------------- Events ----------------
    def _on_geom(self, text):
        self.geom = text
        self._refresh_prediction(); self._redraw_field()

    def _on_params(self, *args):
        # granular + shared
        self.corr_len = self.sp_corr_len.value()
        self.sigma_ln_noise = self.sp_sigma_ln.value()
        # layering
        self.N_A = self.sp_NA.value(); self.N_B = self.sp_NB.value()
        self.CV_thick = self.sp_CV.value(); self.dist_thick = self.cb_dist.currentText()
        self.w_int = self.sp_wint.value(); self.p_cont = self.sp_pcont.value(); self.L_gap = self.sp_Lgap.value()
        self.A_meander = self.sp_Ame.value(); self.L_meander = self.sp_Lme.value()
        # affinity
        self.phi_CA = self.sp_phiCA.value(); self.phi_CB = self.sp_phiCB.value()
        self.eta_attach = self.sp_eta.value(); self.tau_aff = self.sp_tau.value(); self.N_C_min = self.sp_NCmin.value()
        # run settings
        self.fps = self.sp_fps.value()

        self._invalidate_cache()
        self._refresh_prediction()
        self._redraw_field()

    def _on_ncomp(self, value):
        self.ncomp = int(value)

        self._update_comp_visibility()
        self._invalidate_cache()
        self._refresh_prediction()
        self._redraw_field()

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
        self.H = self.sp_H.value()
        self.W = self.sp_W.value()

        self._invalidate_cache()
        self._refresh_prediction()
        self._redraw_field()

    def _update_comp_visibility(self):
        enable3 = (self.ncomp == 3)
        for w in (self.le_means[2], self.le_stds[2], self.le_fracs[2]): w.setEnabled(enable3)

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
            k_raw = build_random_field(self.H, self.W, fr, mus, sigs, self.corr_len, self.sigma_ln_noise)
        else:
            if self.ncomp == 2:
                f_A = fr[0]
                k_series = build_two_component_layers(
                    self.H, self.W,
                    f_A=f_A, N_A=self.N_A, N_B=self.N_B,
                    CV_thick=self.CV_thick, dist_thick=self.dist_thick,
                    w_int=self.w_int, p_cont=self.p_cont, L_gap=self.L_gap,
                    A_meander=self.A_meander, L_meander=self.L_meander,
                    muA=mus[0], sigA=sigs[0], muB=mus[1], sigB=sigs[1],
                    sigma_ln_noise=self.sigma_ln_noise
                )
                k_raw = np.rot90(k_series) if self.geom=='parallel' else k_series
            else:  # 3 components with affinity rules
                k_series = build_layers_ABC_affinity(
                    self.H, self.W,
                    fA=fr[0], fB=fr[1], fC=fr[2], N_A=self.N_A, N_B=self.N_B,
                    CV_thick=self.CV_thick, dist_thick=self.dist_thick,
                    w_int=self.w_int, p_cont=self.p_cont, L_gap=self.L_gap,
                    A_meander=self.A_meander, L_meander=self.L_meander,
                    phi_CA=self.phi_CA, phi_CB=self.phi_CB, eta_attach=self.eta_attach,
                    tau_aff=self.tau_aff, N_C_min=self.N_C_min,
                    muA=mus[0], sigA=sigs[0], muB=mus[1], sigB=sigs[1], muC=mus[2], sigC=sigs[2],
                    sigma_ln_noise=self.sigma_ln_noise
                )
                k_raw = np.rot90(k_series) if self.geom=='parallel' else k_series

        # normalize + simple prediction
        mk = k_raw.mean(); k = k_raw/mk if mk>0 else k_raw
        # mixing-law proxy
        geom_for_pred = 'parallel' if self.geom=='parallel' else ('series' if self.geom=='series' else 'random')
        keff_raw = predicted_keff(geom_for_pred, fr, mus, sigs)
        keff = (1.0/mk)*keff_raw if mk>0 else keff_raw
        t_pred = self.W / max(1e-9, keff)
        return k, t_pred

    def _refresh_prediction(self):
        k, t_pred = self._build_field()
        self.t_pred = t_pred
        self.lab_pred.setText(f"Prediction: t_pred ≈ {t_pred:.2f} (geometry={self.geom})")

    def _update_colorbar(self, im):
        if self._cbar_state['cax'] is None or self._cbar_state['cax'] not in self.canvas.fig.axes:
            divider = make_axes_locatable(self.canvas.ax_img)
            cax = divider.append_axes("right", size="4%", pad=0.04)
            cbar = self.canvas.fig.colorbar(im, cax=cax)
            self._cbar_state['cax'], self._cbar_state['cbar'] = cax, cbar
        else:
            try:
                self._cbar_state['cbar'].update_normal(im)
            except Exception:
                self._cbar_state['cax'].cla()
                self._cbar_state['cbar'] = self.canvas.fig.colorbar(im, cax=self._cbar_state['cax'])

    def _redraw_field(self):
        k, _ = self._build_field()
        self.canvas.ax_img.cla()
        im = self.canvas.ax_img.imshow(k, cmap='magma', origin='lower')
        self.canvas.ax_img.set_title(f"{self.geom.upper()} k-field (mean≈1)")
        self._update_colorbar(im)
        # histogram frame
        self.canvas.ax_hist.cla()
        self.canvas.ax_hist.set_title("Traversal-time histogram")
        self.canvas.ax_hist.set_xlabel("time (sum 1/k)")
        self.canvas.ax_hist.set_ylabel("count")
        if self.t_pred is not None:
            self.canvas.ax_hist.axvline(self.t_pred, color='orange', lw=2, ls='--', label='predicted')
            self.canvas.ax_hist.legend(loc='upper right')
        self.canvas.draw_idle()

    # ---------------- Simulation ----------------
    def _start_run(self):
        if self.running: return
        self.times = []
        self.running = True
        self.btn_run.setEnabled(False); self.btn_anim.setEnabled(False); self.btn_reset.setEnabled(False)
        self.timer.start(1)

    def _stop_run(self):
        if not self.running: return
        self.timer.stop(); self.running = False
        self.btn_run.setEnabled(True); self.btn_anim.setEnabled(True); self.btn_reset.setEnabled(True)

    def _run_chunk(self):
        try:
            N_target = int(10**self.logN)
        except Exception:
            N_target = 100
        remaining = N_target - len(self.times)
        if remaining <= 0:
            self._stop_run(); return
        m = min(self.chunk_size, remaining)
        for _ in range(m):
            k, _ = self._build_field()
            cost = 1.0/(k+1e-9)
            t, path = dijkstra_random_start_free_exit(cost)
            self.times.append(float(t))
        ax = self.canvas.ax_hist
        ax.cla(); ax.hist(self.times, bins=24, color='#4477AA', edgecolor='white')
        if self.t_pred is not None:
            ax.axvline(self.t_pred, color='orange', lw=2, ls='--', label='predicted')
            ax.legend(loc='upper right')
        ax.set_title(f"Traversal-time histogram (n={len(self.times)})")
        ax.set_xlabel("time (sum 1/k)"); ax.set_ylabel("count")
        self.canvas.draw_idle()
        if len(self.times) >= N_target:
            self._stop_run()

    def _animate_one(self):
        k, _ = self._build_field()
        cost = 1.0/(k+1e-9)
        t, path = dijkstra_random_start_free_exit(cost)
        xs = [p[1] for p in path]; ys = [p[0] for p in path]
        self.canvas.ax_img.cla()
        im = self.canvas.ax_img.imshow(k, cmap='magma', origin='lower')
        self.canvas.ax_img.set_title(f"{self.geom.upper()} path animation (t={t:.2f})")
        self._update_colorbar(im)
        (ln,) = self.canvas.ax_img.plot([], [], color='cyan', lw=1.8)
        (pt,) = self.canvas.ax_img.plot([], [], 'wo', ms=6)
        self.canvas.draw_idle()
        self._anim_index = 0
        self._anim_timer = QtCore.QTimer(self)
        interval_ms = max(10, int(1000/max(1,int(self.fps))))
        def step():
            if self._anim_index >= len(xs):
                self._anim_timer.stop(); return
            j = self._anim_index
            ln.set_data(xs[:j+1], ys[:j+1])
            pt.set_data([xs[j]], [ys[j]])  # 1-element sequences
            self.canvas.draw_idle()
            self._anim_index += 1
        self._anim_timer.timeout.connect(step)
        self._anim_timer.start(interval_ms)

    def _reset_plots(self):
        self.times = []
        self._redraw_field()

# -------------------- Entry --------------------
def main():
    app = QApplication(sys.argv)

    font = QFont()
    font.setPointSize(10)   # Set default size
    app.setFont(font)

    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
