#!/usr/bin/env python3
"""
Isostasy Demo — PyQt6 interactive demonstration of Airy, Pratt and Flexural isostasy.

Gravity anomalies are computed with a Bouguer-slab (infinite-sheet) approximation,
which is accurate for features whose horizontal scale greatly exceeds their depth.
The cross-section uses exact column geometry.

Supports three independent crustal column profiles (left flank, mountain/centre,
right flank), multi-layer crust with distinct densities, erosion/sedimentation
simulation, and time-evolution plots.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import fft, ifft, fftfreq

from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib

# ── path setup ────────────────────────────────────────────────────────────────
_COURSE = Path(__file__).resolve().parent.parent.parent   # new_version/
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

# ── Physical constants ────────────────────────────────────────────────────────
G         = 6.674e-11   # m³ kg⁻¹ s⁻²
g_surf    = 9.81        # m s⁻²
E_lith    = 70.0e9      # Pa  (Young's modulus, typical lithosphere)
nu_lith   = 0.25        # Poisson's ratio
mGal      = 1e5         # 1 m/s² = 1e5 mGal

# ── Colours ───────────────────────────────────────────────────────────────────
C_MANTLE = '#7B3F00'
C_WATER  = '#4A90D9'
C_SEALVL = '#2171B5'

# Cycling palette for new user-added layers
_LAYER_COLORS = ['#E8D5A3', '#C8A882', '#A07850', '#805030',
                 '#D0B090', '#B89070', '#987050', '#786040']


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer data model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Layer:
    name:      str
    density:   float    # kg/m³
    thickness: float    # km (reference column)
    color:     str      # hex color

    def clone(self) -> 'Layer':
        return Layer(self.name, self.density, self.thickness, self.color)


# Column labels and keys used throughout
COLUMNS       = ['left', 'center', 'right']
COLUMN_LABELS = {'left': '← Left', 'center': '▲ Center', 'right': '→ Right'}

# Default crustal profiles
#   Left   — average continental crust
#   Center — mountain column (no sediments)
#   Right  — average oceanic crust
DEFAULT_PROFILES: dict[str, list[Layer]] = {
    'left': [
        Layer('Sediments',   2200.0,  2.0,  '#E8D5A3'),
        Layer('Upper Crust', 2700.0, 20.0,  '#C8A882'),
        Layer('Lower Crust', 2900.0, 13.0,  '#A07850'),
    ],
    'center': [
        Layer('Upper Crust', 2700.0, 22.0,  '#C8A882'),
        Layer('Lower Crust', 2900.0, 13.0,  '#A07850'),
    ],
    'right': [
        Layer('Sediments', 2000.0,  0.01, '#E8D5A3'),   # 10 m
        Layer('Basalt',    2900.0,  2.0,  '#708CA0'),
        Layer('Gabbro',    3100.0,  5.0,  '#4A6080'),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Physics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def total_thickness(layers: list[Layer]) -> float:
    """Sum of layer thicknesses (km)."""
    return sum(l.thickness for l in layers)


def effective_density(layers: list[Layer]) -> float:
    """Weighted-average crustal density: ρ_eff = Σ(ρᵢ·dᵢ) / Σdᵢ"""
    H = total_thickness(layers)
    if H == 0:
        return 2800.0
    return sum(l.density * l.thickness for l in layers) / H


def make_column_arrays(x: np.ndarray, profiles: dict[str, list[Layer]],
                       w: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return rho_eff(x) and H_c(x) arrays based on which column each point
    falls in.  Column boundaries are placed at x = ±w.

    Left  column : x < −w
    Centre column: −w ≤ x ≤ +w
    Right column : x >  +w
    """
    left_rho = effective_density(profiles['left'])
    left_Hc  = total_thickness(profiles['left'])
    ctr_rho  = effective_density(profiles['center'])
    ctr_Hc   = total_thickness(profiles['center'])
    rgt_rho  = effective_density(profiles['right'])
    rgt_Hc   = total_thickness(profiles['right'])

    rho_arr = np.where(x < -w, left_rho, np.where(x > w, rgt_rho, ctr_rho))
    Hc_arr  = np.where(x < -w, left_Hc,  np.where(x > w, rgt_Hc,  ctr_Hc))
    return rho_arr, Hc_arr


def ocean_water_depth(left_layers: list[Layer], right_layers: list[Layer],
                      rho_m: float, rho_water: float = 1030.0) -> float:
    """
    Isostatic water depth (km) for the right (oceanic) column, computed by
    balancing column pressures against the left (continental) reference at
    sea level (h_left = 0).

    Derivation (pressure equality at compensation depth D):
        ρ_w · d_w + ρ_eff_R · H_R + ρ_m · (D − d_w − H_R)
            = ρ_eff_L · H_L + ρ_m · (D − H_L)

    Solving for d_w:
        d_w = [H_L·(ρ_L − ρ_m) − H_R·(ρ_R − ρ_m)] / (ρ_w − ρ_m)

    Returns max(0, d_w) — negative result means the right surface sits above
    sea level (uncommon with oceanic parameters, shown as dry land).
    """
    H_L   = total_thickness(left_layers)
    rho_L = effective_density(left_layers)
    H_R   = total_thickness(right_layers)
    rho_R = effective_density(right_layers)
    d_w   = (H_L * (rho_L - rho_m) - H_R * (rho_R - rho_m)) / (rho_water - rho_m)
    return max(0.0, d_w)


# ═══════════════════════════════════════════════════════════════════════════════
#  Physics
# ═══════════════════════════════════════════════════════════════════════════════

def make_topography(x: np.ndarray, h_peak: float, w: float, shape: str) -> np.ndarray:
    """Return surface elevation h(x) in km.  h_peak and w in km."""
    if shape == 'Gaussian':
        sigma = w / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return h_peak * np.exp(-x**2 / (2.0 * sigma**2))
    elif shape == 'Box':
        return np.where(np.abs(x) <= w, float(h_peak), 0.0)
    else:  # Triangle
        return np.maximum(h_peak * (1.0 - np.abs(x) / w), 0.0)


def erode_topography(h0: np.ndarray, t: float, rate: float,
                     erosion_type: str, tau: float) -> np.ndarray:
    """
    Apply erosion or sedimentation to h0 at time t (Ma).

    rate > 0 → erosion (height decreases)
    rate < 0 → sedimentation (height increases)

    Constant:    h(t) = h0 − rate·t
    Exponential: h(t) = h0 · exp(−t / τ)
    """
    if erosion_type == 'None' or t <= 0:
        return h0.copy()
    if erosion_type == 'Constant':
        return h0 - rate * t
    else:  # Exponential
        if tau <= 0:
            return h0.copy()
        return h0 * np.exp(-t / tau)


def compute_time_series(
        h0: np.ndarray, x: np.ndarray,
        profiles: dict[str, list[Layer]], rho_m: float,
        rate: float, erosion_type: str, tau: float,
        model: str, model_params: dict,
        duration: float, n_steps: int = 150
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute h_peak(t) and moho_root_peak(t) over [0, duration] Ma.
    Uses the centre column's properties as the reference.

    Returns (t_arr, h_peak_arr, root_peak_arr).
    """
    ctr_layers = profiles['center']
    ctr_rho    = effective_density(ctr_layers)
    ctr_Hc     = total_thickness(ctr_layers)
    t_arr  = np.linspace(0, duration, n_steps)
    h_pk   = np.zeros(n_steps)
    rt_pk  = np.zeros(n_steps)

    for j, t in enumerate(t_arr):
        h_t = erode_topography(h0, t, rate, erosion_type, tau)
        if model == 'Flexure':
            moho_t = flexure_moho(x, h_t, ctr_rho, rho_m,
                                  model_params['rho_i'], model_params['Te'], ctr_Hc)
        else:
            moho_t = airy_moho(h_t, ctr_rho, rho_m, ctr_Hc)

        i_pk      = int(np.argmax(np.abs(h_t))) if h_t.max() > h_t.min() else 0
        h_pk[j]   = h_t[i_pk]
        rt_pk[j]  = moho_t[i_pk] - ctr_Hc

    return t_arr, h_pk, rt_pk


def airy_moho(h: np.ndarray, rho_c, rho_m: float, H_c) -> np.ndarray:
    """
    Moho depth (km, positive downward) for Airy isostasy.
    rho_c and H_c may be scalars or arrays (for variable-column mode).
    """
    root = rho_c / (rho_m - rho_c) * h
    return H_c + root


def pratt_density(h: np.ndarray, rho_ref: float, D: float) -> np.ndarray:
    """
    Column density for Pratt isostasy.
    Pressure at compensation depth D is uniform ⟹ ρ(x)·(D + h(x)) = ρ_ref·D.
    """
    return rho_ref * D / (D + h)


def flexure_moho(x: np.ndarray, h: np.ndarray,
                 rho_c: float, rho_m: float, rho_i: float,
                 Te: float, H_c: float) -> np.ndarray:
    """
    Flexural isostasy via the thin-elastic-plate equation (Fourier domain).

    For multi-column mode, rho_c and H_c should be the centre-column values;
    the per-column H_c offset is applied by the caller.
    """
    if Te <= 0.0:
        return airy_moho(h, rho_c, rho_m, H_c)

    dx_m = (x[1] - x[0]) * 1e3
    N    = len(x)
    Te_m = Te * 1e3
    D_f  = E_lith * Te_m**3 / (12.0 * (1.0 - nu_lith**2))

    q   = rho_c * g_surf * h * 1e3
    k   = 2.0 * np.pi * fftfreq(N, dx_m)
    Q_k = fft(q)

    dr = (rho_m - rho_i) * g_surf
    with np.errstate(divide='ignore', invalid='ignore'):
        W_k = np.where(k == 0, Q_k / dr, Q_k / (D_f * k**4 + dr))

    w_m = np.real(ifft(W_k))
    return H_c + w_m / 1e3


def gravity_anomalies(h: np.ndarray, moho: np.ndarray,
                      H_c, rho_c, rho_m: float,
                      rho_bouguer=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Bouguer-slab approximation for Free-air and Bouguer anomalies (mGal).
    H_c, rho_c, and rho_bouguer may be scalars or arrays.
    """
    if rho_bouguer is None:
        rho_bouguer = rho_c

    h_m      = h    * 1e3
    delta_m  = (moho - H_c) * 1e3

    g_topo    = 2.0 * np.pi * G * rho_c           * h_m    * mGal
    g_root    = 2.0 * np.pi * G * (rho_m - rho_c) * delta_m * mGal

    g_faa     = g_topo - g_root
    g_bouguer = g_faa  - 2.0 * np.pi * G * rho_bouguer * h_m * mGal

    return g_faa, g_bouguer


# ═══════════════════════════════════════════════════════════════════════════════
#  Canvas
# ═══════════════════════════════════════════════════════════════════════════════

class IsostasyCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 11))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 2, 1], hspace=0.45)
        self.ax_grav = self.fig.add_subplot(gs[0])
        self.ax_sec  = self.fig.add_subplot(gs[1])
        self.ax_time = self.fig.add_subplot(gs[2])
        super().__init__(self.fig)
        self.setParent(parent)
        self._cb = None   # colorbar handle (Pratt density plot)


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer table dock  (bottom)
# ═══════════════════════════════════════════════════════════════════════════════

class LayerTableDock(QtWidgets.QDockWidget):
    """
    Displays and edits a three-column crustal profile (left, centre, right).
    A combo-box at the top selects which column's layer stack is shown in the
    table.  All three stacks are kept in memory; layersChanged is emitted
    whenever any column changes.
    """
    layersChanged = QtCore.pyqtSignal()

    _COL_IDX   = 0
    _COL_NAME  = 1
    _COL_DENS  = 2
    _COL_THICK = 3
    _COL_COLOR = 4
    _NCOLS     = 5

    def __init__(self, parent=None):
        super().__init__('Crustal Layers', parent)
        self.setMinimumHeight(160)

        # Deep-copy the default profiles so defaults are never mutated
        self._profiles: dict[str, list[Layer]] = {
            col: [l.clone() for l in layers]
            for col, layers in DEFAULT_PROFILES.items()
        }
        self._active      = 'left'   # currently displayed column
        self._updating    = False
        self._ocean_depth = 0.0     # updated by MainWindow after each solve

        container = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(container)
        vlay.setContentsMargins(4, 4, 4, 4)
        vlay.setSpacing(3)

        # ── Column selector ───────────────────────────────────────────────
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel('Column:'))
        self.combo_col = QtWidgets.QComboBox()
        for col in COLUMNS:
            self.combo_col.addItem(COLUMN_LABELS[col], col)
        self.combo_col.currentIndexChanged.connect(self._on_col_changed)
        top_row.addWidget(self.combo_col, 1)

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = QtWidgets.QToolBar()
        toolbar.setIconSize(QtCore.QSize(16, 16))
        self._act_add = toolbar.addAction('＋  Add layer')
        self._act_del = toolbar.addAction('－  Remove')
        toolbar.addSeparator()
        self._act_up  = toolbar.addAction('▲  Move up')
        self._act_dn  = toolbar.addAction('▼  Move down')
        self._act_add.triggered.connect(self._add_layer)
        self._act_del.triggered.connect(self._remove_layer)
        self._act_up.triggered.connect(self._move_up)
        self._act_dn.triggered.connect(self._move_down)

        # ── Table ─────────────────────────────────────────────────────────
        self.table = QtWidgets.QTableWidget(0, self._NCOLS)
        self.table.setHorizontalHeaderLabels(
            ['#', 'Name', 'Density (kg/m³)', 'Thickness (km)', 'Color'])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(self._COL_NAME, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(self._COL_IDX,  QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(self._COL_IDX,   28)
        self.table.setColumnWidth(self._COL_DENS, 120)
        self.table.setColumnWidth(self._COL_THICK, 110)
        self.table.setColumnWidth(self._COL_COLOR,  52)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)

        # ── Status bar ────────────────────────────────────────────────────
        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setStyleSheet('color: #444; font-size: 9pt; padding: 2px;')

        vlay.addLayout(top_row)
        vlay.addWidget(toolbar)
        vlay.addWidget(self.table)
        vlay.addWidget(self.lbl_status)
        self.setWidget(container)

        self._rebuild_table()

    # ── Public API ────────────────────────────────────────────────────────────
    def profiles(self) -> dict[str, list[Layer]]:
        """Return deep-copies of all three column layer stacks."""
        return {col: [l.clone() for l in layers]
                for col, layers in self._profiles.items()}

    # ── Column switching ──────────────────────────────────────────────────────
    def _on_col_changed(self, idx: int):
        self._active = self.combo_col.itemData(idx)
        self._rebuild_table()

    def _active_layers(self) -> list[Layer]:
        return self._profiles[self._active]

    # ── Table build ───────────────────────────────────────────────────────────
    def _rebuild_table(self):
        self._updating = True
        layers = self._active_layers()
        self.table.setRowCount(len(layers))
        for row, lyr in enumerate(layers):
            self._populate_row(row, lyr)
        self._update_status()
        self._updating = False

    def _populate_row(self, row: int, lyr: Layer):
        # Col 0: index
        idx = QtWidgets.QTableWidgetItem(str(row + 1))
        idx.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        idx.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, self._COL_IDX, idx)

        # Col 1: name — editable QLineEdit
        le = QtWidgets.QLineEdit(lyr.name)
        le.setFrame(False)
        le.editingFinished.connect(
            lambda r=row, w=le: self._on_name(r, w.text()))
        self.table.setCellWidget(row, self._COL_NAME, le)

        # Col 2: density spinner
        sp_d = QtWidgets.QDoubleSpinBox()
        sp_d.setRange(1000, 3500)
        sp_d.setDecimals(0)
        sp_d.setSuffix(' kg/m³')
        sp_d.setValue(lyr.density)
        sp_d.valueChanged.connect(lambda v, r=row: self._on_density(r, v))
        self.table.setCellWidget(row, self._COL_DENS, sp_d)

        # Col 3: thickness spinner
        sp_t = QtWidgets.QDoubleSpinBox()
        sp_t.setRange(0.001, 200)
        sp_t.setDecimals(3)
        sp_t.setSuffix(' km')
        sp_t.setValue(lyr.thickness)
        sp_t.valueChanged.connect(lambda v, r=row: self._on_thickness(r, v))
        self.table.setCellWidget(row, self._COL_THICK, sp_t)

        # Col 4: color button
        btn = QtWidgets.QPushButton()
        btn.setFixedWidth(44)
        btn.setStyleSheet(f'background-color: {lyr.color}; border: 1px solid #888;')
        btn.clicked.connect(lambda _, r=row: self._pick_color(r))
        self.table.setCellWidget(row, self._COL_COLOR, btn)

        self.table.setRowHeight(row, 28)

    def _update_status(self):
        layers = self._active_layers()
        H_c   = total_thickness(layers)
        r_eff = effective_density(layers)
        col   = COLUMN_LABELS[self._active]
        extra = (f'  |  Ocean depth = {self._ocean_depth:.2f} km'
                 if self._active == 'right' else '')
        self.lbl_status.setText(
            f'{col}:  H_c = {H_c:.2f} km  |  ρ_eff = {r_eff:.0f} kg/m³{extra}')

    def set_ocean_depth(self, d_water: float):
        """Called by MainWindow after each update to display the computed depth."""
        self._ocean_depth = d_water
        if self._active == 'right':
            self._update_status()

    # ── Cell callbacks ────────────────────────────────────────────────────────
    def _on_name(self, row: int, text: str):
        if self._updating or row >= len(self._active_layers()):
            return
        self._active_layers()[row].name = text
        # Don't rebuild the table (would unfocus the line edit); status only
        self._update_status()
        self.layersChanged.emit()

    def _on_density(self, row: int, value: float):
        if self._updating or row >= len(self._active_layers()):
            return
        self._active_layers()[row].density = value
        self._update_status()
        self.layersChanged.emit()

    def _on_thickness(self, row: int, value: float):
        if self._updating or row >= len(self._active_layers()):
            return
        self._active_layers()[row].thickness = value
        self._update_status()
        self.layersChanged.emit()

    def _pick_color(self, row: int):
        if row >= len(self._active_layers()):
            return
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self._active_layers()[row].color), self, 'Layer color')
        if color.isValid():
            self._active_layers()[row].color = color.name()
            btn = self.table.cellWidget(row, self._COL_COLOR)
            if btn:
                btn.setStyleSheet(
                    f'background-color: {color.name()}; border: 1px solid #888;')
            self.layersChanged.emit()

    # ── Add / remove / reorder ────────────────────────────────────────────────
    def _add_layer(self):
        layers = self._active_layers()
        n      = len(layers)
        color  = _LAYER_COLORS[n % len(_LAYER_COLORS)]
        layers.append(Layer(f'Layer {n + 1}', 2750.0, 10.0, color))
        self._rebuild_table()
        self.layersChanged.emit()

    def _remove_layer(self):
        row    = self.table.currentRow()
        layers = self._active_layers()
        if row < 0 or len(layers) <= 1:
            return
        layers.pop(row)
        self._rebuild_table()
        self.layersChanged.emit()

    def _move_up(self):
        row    = self.table.currentRow()
        layers = self._active_layers()
        if row <= 0:
            return
        layers[row - 1], layers[row] = layers[row], layers[row - 1]
        self._rebuild_table()
        self.table.selectRow(row - 1)
        self.layersChanged.emit()

    def _move_down(self):
        row    = self.table.currentRow()
        layers = self._active_layers()
        if row < 0 or row >= len(layers) - 1:
            return
        layers[row], layers[row + 1] = layers[row + 1], layers[row]
        self._rebuild_table()
        self.table.selectRow(row + 1)
        self.layersChanged.emit()


# ═══════════════════════════════════════════════════════════════════════════════
#  Control dock  (left)
# ═══════════════════════════════════════════════════════════════════════════════

class ControlDock(QtWidgets.QDockWidget):
    paramsChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__('Controls', parent)
        self.setMinimumWidth(270)
        self.setMaximumWidth(340)

        container = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setPointSize(10)
        container.setFont(font)
        vlay = QtWidgets.QVBoxLayout(container)
        vlay.setSpacing(6)
        vlay.setContentsMargins(6, 6, 6, 6)

        # ── Model selector ────────────────────────────────────────────────
        grp_model = QtWidgets.QGroupBox('Isostasy Model')
        ml = QtWidgets.QVBoxLayout(grp_model)
        self.combo_model = QtWidgets.QComboBox()
        self.combo_model.addItems(['Airy', 'Pratt', 'Flexure'])
        self.combo_model.setToolTip(
            'Airy  — uniform-density crust; Moho depth varies\n'
            'Pratt — flat Moho; column density varies\n'
            'Flexure — elastic plate bending under topographic load')
        ml.addWidget(self.combo_model)

        # ── Topography ────────────────────────────────────────────────────
        grp_topo = QtWidgets.QGroupBox('Topography')
        tl = QtWidgets.QGridLayout(grp_topo)
        self.combo_shape = QtWidgets.QComboBox()
        self.combo_shape.addItems(['Gaussian', 'Box', 'Triangle'])
        self.spin_h = self._ds(-9.0, 9.0, 1, 4.0, ' km',
                               'Peak height (> 0 = mountain, < 0 = basin)')
        self.spin_w = self._ds(10, 2000, 0, 200, ' km',
                               'Half-width (also sets column boundaries at ±w)')
        tl.addWidget(QtWidgets.QLabel('Shape'),      0, 0)
        tl.addWidget(self.combo_shape,               0, 1)
        tl.addWidget(QtWidgets.QLabel('Height'),     1, 0)
        tl.addWidget(self.spin_h,                    1, 1)
        tl.addWidget(QtWidgets.QLabel('Half-width'), 2, 0)
        tl.addWidget(self.spin_w,                    2, 1)

        # ── Physical properties ────────────────────────────────────────────
        grp_dens = QtWidgets.QGroupBox('Physical Properties')
        dl = QtWidgets.QGridLayout(grp_dens)
        self.spin_rho_m = self._ds(3000, 3500, 0, 3300, ' kg/m³', 'Upper mantle density')
        self.spin_rho_w = self._ds(900, 1100, 0, 1030, ' kg/m³',
                                   'Water density (used for ocean depth computation)')
        dl.addWidget(QtWidgets.QLabel('ρ mantle'), 0, 0)
        dl.addWidget(self.spin_rho_m,              0, 1)
        dl.addWidget(QtWidgets.QLabel('ρ water'),  1, 0)
        dl.addWidget(self.spin_rho_w,              1, 1)

        # ── Pratt parameters ──────────────────────────────────────────────
        self.grp_pratt = QtWidgets.QGroupBox('Pratt Parameters')
        pl = QtWidgets.QGridLayout(self.grp_pratt)
        self.spin_D       = self._ds(20, 300, 0, 100, ' km',
                                     'Depth of compensation (flat Moho)')
        self.spin_rho_ref = self._ds(2400, 3100, 0, 2800, ' kg/m³',
                                     'Reference column density at sea level')
        pl.addWidget(QtWidgets.QLabel('Depth D'), 0, 0)
        pl.addWidget(self.spin_D,                 0, 1)
        pl.addWidget(QtWidgets.QLabel('ρ ref'),   1, 0)
        pl.addWidget(self.spin_rho_ref,           1, 1)

        # ── Flexure parameters ────────────────────────────────────────────
        self.grp_flex = QtWidgets.QGroupBox('Flexure Parameters')
        fl = QtWidgets.QGridLayout(self.grp_flex)
        self.spin_Te    = self._ds(0, 150, 0, 30, ' km',
                                   'Effective elastic thickness  (0 km → Airy limit)')
        self.spin_rho_i = self._ds(0, 3300, 0, 2300, ' kg/m³',
                                   'Density of material filling the flexural moat')
        fl.addWidget(QtWidgets.QLabel('Tₑ'),       0, 0)
        fl.addWidget(self.spin_Te,                  0, 1)
        fl.addWidget(QtWidgets.QLabel('ρ infill'), 1, 0)
        fl.addWidget(self.spin_rho_i,               1, 1)

        # ── Erosion / sedimentation ───────────────────────────────────────
        grp_eros = QtWidgets.QGroupBox('Erosion / Sedimentation')
        el = QtWidgets.QGridLayout(grp_eros)

        self.combo_eros = QtWidgets.QComboBox()
        self.combo_eros.addItems(['None', 'Constant', 'Exponential'])
        self.combo_eros.setToolTip(
            'None        — no erosion\n'
            'Constant    — height decreases at fixed rate (km/Ma)\n'
            'Exponential — height decays as h₀·exp(−t/τ)')

        self.spin_rate     = self._ds(-10, 10, 3, 0.04, ' km/Ma',
                                      'Erosion rate (positive = removal, '
                                      'negative = deposition)')
        self.spin_tau      = self._ds(0.1, 1000, 1, 30.0, ' Ma',
                                      'Exponential time constant τ')
        self.spin_duration = self._ds(0.1, 2000, 0, 100, ' Ma',
                                      'Total duration of simulation')

        self.lbl_time = QtWidgets.QLabel('t = 0.0 Ma')
        self.lbl_time.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.slider_time = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_time.setRange(0, 1000)
        self.slider_time.setValue(0)

        self.btn_play = QtWidgets.QPushButton('▶  Play')
        self.btn_stop = QtWidgets.QPushButton('■  Stop')
        self.btn_stop.setEnabled(False)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_stop)

        self._lbl_tau = QtWidgets.QLabel('τ (time const)')

        el.addWidget(QtWidgets.QLabel('Type'),      0, 0)
        el.addWidget(self.combo_eros,               0, 1)
        el.addWidget(QtWidgets.QLabel('Rate'),      1, 0)
        el.addWidget(self.spin_rate,                1, 1)
        el.addWidget(self._lbl_tau,                 2, 0)
        el.addWidget(self.spin_tau,                 2, 1)
        el.addWidget(QtWidgets.QLabel('Duration'),  3, 0)
        el.addWidget(self.spin_duration,            3, 1)
        el.addWidget(self.slider_time,              4, 0, 1, 2)
        el.addWidget(self.lbl_time,                 5, 0, 1, 2)
        el.addLayout(btn_row,                       6, 0, 1, 2)

        # ── Display options ───────────────────────────────────────────────
        grp_disp = QtWidgets.QGroupBox('Display')
        disp_l = QtWidgets.QVBoxLayout(grp_disp)
        self.chk_faa      = QtWidgets.QCheckBox('Free-air anomaly')
        self.chk_bouguer  = QtWidgets.QCheckBox('Bouguer anomaly')
        self.chk_labels   = QtWidgets.QCheckBox('Layer labels')
        self.chk_moho_ref = QtWidgets.QCheckBox('Reference Moho')
        self.chk_root_ann = QtWidgets.QCheckBox('Annotate root / deflection')
        self.chk_col_bdy  = QtWidgets.QCheckBox('Column boundaries')
        for chk, on in [(self.chk_faa, True), (self.chk_bouguer, True),
                        (self.chk_labels, True), (self.chk_moho_ref, False),
                        (self.chk_root_ann, True), (self.chk_col_bdy, True)]:
            chk.setChecked(on)
            disp_l.addWidget(chk)

        # ── Status label ──────────────────────────────────────────────────
        self.lbl_status = QtWidgets.QLabel('')
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet('color: #444; font-size: 9pt;')

        # ── Assemble ──────────────────────────────────────────────────────
        for grp in [grp_model, grp_topo, grp_dens,
                    self.grp_pratt, self.grp_flex,
                    grp_eros, grp_disp]:
            vlay.addWidget(grp)
        vlay.addWidget(self.lbl_status)
        vlay.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(container)
        self.setWidget(scroll)

        # ── Signals ───────────────────────────────────────────────────────
        self.combo_model.currentIndexChanged.connect(self._on_model)
        self.combo_model.currentIndexChanged.connect(self.paramsChanged)
        self.combo_shape.currentIndexChanged.connect(self.paramsChanged)
        self.combo_eros.currentIndexChanged.connect(self._on_eros_type)
        self.combo_eros.currentIndexChanged.connect(self.paramsChanged)

        for w in [self.spin_h, self.spin_w, self.spin_rho_m, self.spin_rho_w,
                  self.spin_D, self.spin_rho_ref,
                  self.spin_Te, self.spin_rho_i,
                  self.spin_rate, self.spin_tau, self.spin_duration]:
            w.valueChanged.connect(self.paramsChanged)

        self.slider_time.valueChanged.connect(self._on_slider)
        self.spin_duration.valueChanged.connect(self._on_duration_changed)

        for chk in [self.chk_faa, self.chk_bouguer, self.chk_labels,
                    self.chk_moho_ref, self.chk_root_ann, self.chk_col_bdy]:
            chk.toggled.connect(self.paramsChanged)

        self._on_model()
        self._on_eros_type()

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _ds(lo, hi, dec, val, suffix='', tip=''):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        if tip:
            s.setToolTip(tip)
        return s

    def _on_model(self):
        m = self.combo_model.currentText()
        self.grp_pratt.setVisible(m == 'Pratt')
        self.grp_flex.setVisible(m == 'Flexure')

    def _on_eros_type(self):
        t = self.combo_eros.currentText()
        active = (t != 'None')
        self.spin_rate.setEnabled(active)
        self.spin_duration.setEnabled(active)
        self.slider_time.setEnabled(active)
        self.btn_play.setEnabled(active)
        self._lbl_tau.setVisible(t == 'Exponential')
        self.spin_tau.setVisible(t == 'Exponential')

    def _on_slider(self, val: int):
        t = val / 1000.0 * self.spin_duration.value()
        self.lbl_time.setText(f't = {t:.1f} Ma')
        self.paramsChanged.emit()

    def _on_duration_changed(self):
        self._on_slider(self.slider_time.value())

    def current_time(self) -> float:
        return self.slider_time.value() / 1000.0 * self.spin_duration.value()

    def set_slider_position(self, frac: float):
        """Move slider to frac ∈ [0, 1] without firing paramsChanged."""
        self.slider_time.blockSignals(True)
        self.slider_time.setValue(int(frac * 1000))
        self.slider_time.blockSignals(False)
        self.lbl_time.setText(f't = {frac * self.spin_duration.value():.1f} Ma')

    def params(self) -> dict:
        return dict(
            model         = self.combo_model.currentText(),
            shape         = self.combo_shape.currentText(),
            h             = self.spin_h.value(),
            w             = self.spin_w.value(),
            rho_m         = self.spin_rho_m.value(),
            rho_water     = self.spin_rho_w.value(),
            D             = self.spin_D.value(),
            rho_ref       = self.spin_rho_ref.value(),
            Te            = self.spin_Te.value(),
            rho_i         = self.spin_rho_i.value(),
            erosion_type  = self.combo_eros.currentText(),
            rate          = self.spin_rate.value(),
            tau           = self.spin_tau.value(),
            duration      = self.spin_duration.value(),
            current_t     = self.current_time(),
            show_faa      = self.chk_faa.isChecked(),
            show_bouguer  = self.chk_bouguer.isChecked(),
            show_labels   = self.chk_labels.isChecked(),
            show_moho_ref = self.chk_moho_ref.isChecked(),
            show_root_ann = self.chk_root_ann.isChecked(),
            show_col_bdy  = self.chk_col_bdy.isChecked(),
        )

    def set_status(self, text: str):
        self.lbl_status.setText(text)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Isostasy Demo')
        self.resize(1350, 950)

        self.canvas     = IsostasyCanvas(self)
        self.setCentralWidget(self.canvas)

        self.dock       = ControlDock(self)
        self.layer_dock = LayerTableDock(self)

        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,   self.dock)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.layer_dock)

        # ── Animation timer ───────────────────────────────────────────────
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._step_animation)
        self.dock.btn_play.clicked.connect(self._start_animation)
        self.dock.btn_stop.clicked.connect(self._stop_animation)

        self.dock.paramsChanged.connect(self.update_plot)
        self.layer_dock.layersChanged.connect(self.update_plot)

        self.update_plot()

    # ── Animation ─────────────────────────────────────────────────────────────
    def _start_animation(self):
        if self.dock.slider_time.value() >= 999:
            self.dock.set_slider_position(0.0)
        self._timer.start()
        self.dock.btn_play.setEnabled(False)
        self.dock.btn_stop.setEnabled(True)

    def _stop_animation(self):
        self._timer.stop()
        self.dock.btn_play.setEnabled(True)
        self.dock.btn_stop.setEnabled(False)

    def _step_animation(self):
        val = self.dock.slider_time.value()
        if val >= 1000:
            self._stop_animation()
            return
        self.dock.set_slider_position(min(val + 10, 1000) / 1000.0)
        self.update_plot()

    # ── Main update ───────────────────────────────────────────────────────────
    def update_plot(self):
        p        = self.dock.params()
        profiles = self.layer_dock.profiles()
        model    = p['model']
        rho_m    = p['rho_m']
        rho_water = p['rho_water']
        w        = p['w']

        # x-grid
        half = max(600.0, w * 3.0)
        x    = np.linspace(-half, half, 600)

        # Column-variable effective density and reference H_c arrays
        rho_arr, Hc_arr = make_column_arrays(x, profiles, w)
        ctr_rho = effective_density(profiles['center'])
        ctr_Hc  = total_thickness(profiles['center'])

        # Isostatic ocean water depth (right column vs. left reference)
        d_water  = ocean_water_depth(profiles['left'], profiles['right'],
                                     rho_m, rho_water)
        H_c_right = total_thickness(profiles['right'])
        rgt_mask  = x > w
        lft_mask  = x < -w

        # Build topography:
        #   left  → flat continental (h = 0)
        #   centre → user-controlled mountain shape
        #   right  → isostatically computed ocean floor (h = −d_water)
        h0 = make_topography(x, p['h'], w, p['shape'])
        h0[lft_mask] = 0.0
        h0[rgt_mask] = -d_water

        # Apply erosion only to mountain (centre column); ocean floor unchanged
        h = erode_topography(h0, p['current_t'], p['rate'],
                             p['erosion_type'], p['tau'])
        h[lft_mask] = 0.0
        h[rgt_mask] = -d_water

        rho_col = None

        if model == 'Airy':
            moho = airy_moho(h, rho_arr, rho_m, Hc_arr)
            # Right column: use isostatic Moho directly (water load ≠ rock load)
            moho[rgt_mask] = d_water + H_c_right
            g_faa, g_bouguer = gravity_anomalies(h, moho, Hc_arr, rho_arr, rho_m)
            self._set_status_airy(h, moho, profiles, rho_m, d_water)

        elif model == 'Pratt':
            rho_col   = pratt_density(h, p['rho_ref'], p['D'])
            moho      = np.full_like(h, p['D'])
            g_faa     = np.zeros_like(h)
            g_bouguer = -2.0 * np.pi * G * p['rho_ref'] * h * 1e3 * mGal
            self._set_status_pratt(h, rho_col, p['D'])

        else:  # Flexure — compute deflection with centre-column parameters,
               # then offset for each column's reference H_c
            moho_flex = flexure_moho(x, h, ctr_rho, rho_m,
                                     p['rho_i'], p['Te'], ctr_Hc)
            w_flex    = moho_flex - ctr_Hc
            moho      = Hc_arr + w_flex
            moho[rgt_mask] = d_water + H_c_right   # isostatic override for right
            g_faa, g_bouguer = gravity_anomalies(h, moho, Hc_arr, rho_arr, rho_m)
            self._set_status_flex(h, moho, profiles, rho_m, p['Te'], d_water)

        # Notify layer dock of computed water depth (shown in right-column status)
        self.layer_dock.set_ocean_depth(d_water)

        # Time series
        t_arr = h_pk = rt_pk = None
        if p['erosion_type'] != 'None' and p['duration'] > 0:
            t_arr, h_pk, rt_pk = compute_time_series(
                h0, x, profiles, rho_m,
                p['rate'], p['erosion_type'], p['tau'],
                model, p, p['duration'])

        self._draw_gravity(x, g_faa, g_bouguer, model, p)
        self._draw_section(x, h, moho, profiles, w, rho_col, model, p)
        self._draw_time_series(t_arr, h_pk, rt_pk, p)
        self.canvas.fig.canvas.draw_idle()

    # ── Gravity panel ─────────────────────────────────────────────────────────
    def _draw_gravity(self, x, g_faa, g_bouguer, model, p):
        ax = self.canvas.ax_grav
        ax.cla()

        if p['show_faa']:
            ax.plot(x, g_faa,     'b-', lw=2, label='Free-air anomaly')
        if p['show_bouguer']:
            ax.plot(x, g_bouguer, 'r-', lw=2, label='Bouguer anomaly')

        if p['show_col_bdy']:
            for xb in (-p['w'], p['w']):
                ax.axvline(xb, color='#888', lw=1, ls='--', alpha=0.6)

        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylabel('Gravity anomaly (mGal)')
        ax.set_title(f'{model} Isostasy — Gravity Anomalies  '
                     r'(Bouguer-slab approximation)',
                     fontsize=10, fontweight='bold')
        if p['show_faa'] or p['show_bouguer']:
            ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    # ── Cross-section panel ───────────────────────────────────────────────────
    def _draw_section(self, x, h, moho, profiles, w, rho_col, model, p):
        ax  = self.canvas.ax_sec
        fig = self.canvas.fig
        ax.cla()

        if self.canvas._cb is not None:
            try:
                self.canvas._cb.remove()
            except Exception:
                pass
            self.canvas._cb = None

        z_top = max(h.max() + 3.0, 5.0)
        z_bot = moho.max() + 20.0

        # ── Mantle background ──────────────────────────────────────────────
        ax.fill_between(x, -z_bot, -moho,
                        color=C_MANTLE, alpha=0.85, zorder=1,
                        label='Mantle lithosphere')

        # ── Crust ─────────────────────────────────────────────────────────
        if model == 'Pratt' and rho_col is not None:
            norm = mcolors.Normalize(vmin=rho_col.min() - 1, vmax=rho_col.max() + 1)
            cmap = matplotlib.colormaps['RdYlBu_r']
            N_z  = 200
            z_g  = np.linspace(-p['D'], float(h.max()) + 0.5, N_z)
            XX, ZZ = np.meshgrid(x, z_g)
            density = np.full_like(XX, np.nan)
            for i in range(len(x)):
                inside = (z_g >= -p['D']) & (z_g <= h[i])
                density[inside, i] = rho_col[i]
            pcm = ax.pcolormesh(XX, ZZ, density,
                                cmap=cmap, norm=norm,
                                shading='auto', zorder=2)
            self.canvas._cb = fig.colorbar(
                pcm, ax=ax, label='Crustal density (kg/m³)',
                pad=0.02, shrink=0.65, fraction=0.03)
        else:
            # Draw each column segment with its own layer stack
            col_masks = {
                'left':   x < -w,
                'center': (x >= -w) & (x <= w),
                'right':  x > w,
            }
            drawn_names: set[str] = set()
            for col_key in COLUMNS:
                mask   = col_masks[col_key]
                if not np.any(mask):
                    continue
                layers = profiles[col_key]
                H_c_col = total_thickness(layers)
                if H_c_col <= 0:
                    continue

                x_seg    = x[mask]
                h_seg    = h[mask]
                moho_seg = moho[mask]
                col_h    = h_seg + moho_seg   # surface → Moho height

                cum = 0.0
                for lyr in layers:
                    frac  = lyr.thickness / H_c_col
                    top_z = h_seg - cum * col_h
                    bot_z = h_seg - (cum + frac) * col_h
                    label = lyr.name if lyr.name not in drawn_names else '_'
                    ax.fill_between(x_seg, bot_z, top_z,
                                    color=lyr.color, alpha=0.90, zorder=2,
                                    label=label)
                    drawn_names.add(lyr.name)
                    cum += frac

        # ── Ocean water where h < 0 ────────────────────────────────────────
        if h.min() < 0:
            ax.fill_between(x, h, 0.0, where=(h < 0),
                            color=C_WATER, alpha=0.7, zorder=3,
                            label='Ocean water')

        # ── Sea level ──────────────────────────────────────────────────────
        ax.axhline(0, color=C_SEALVL, lw=1.5, alpha=0.6, zorder=4)

        # ── Column boundary lines ──────────────────────────────────────────
        if p['show_col_bdy']:
            for xb in (-w, w):
                ax.axvline(xb, color='#555', lw=1.2, ls='--',
                           alpha=0.75, zorder=5)

        # ── Reference Moho (centre column) ────────────────────────────────
        if p['show_moho_ref']:
            ctr_Hc = total_thickness(profiles['center'])
            ax.axhline(-ctr_Hc, color='silver', lw=1.5, ls='--',
                       alpha=0.85, zorder=4,
                       label=f'Ref. Moho centre (−{ctr_Hc:.0f} km)')

        # ── Surface and Moho outlines ──────────────────────────────────────
        ax.plot(x, h,     'k-', lw=1.5, zorder=6)
        ax.plot(x, -moho, 'k-', lw=1.5, zorder=6)

        # ── Layer labels ───────────────────────────────────────────────────
        if p['show_labels'] and model != 'Pratt':
            label_specs = [
                ('left',   x[int(len(x) * 0.15)]),
                ('center', x[int(len(x) * 0.50)]),
                ('right',  x[int(len(x) * 0.85)]),
            ]
            for col_key, x_lbl in label_specs:
                mask   = col_masks[col_key] if model != 'Pratt' else (x >= -w) & (x <= w)
                if not np.any(mask):
                    continue
                layers  = profiles[col_key]
                H_c_col = total_thickness(layers)
                if H_c_col <= 0:
                    continue
                i_lbl   = np.argmin(np.abs(x - x_lbl))
                col_h_l = h[i_lbl] + moho[i_lbl]
                cum = 0.0
                for lyr in layers:
                    frac  = lyr.thickness / H_c_col
                    mid_z = h[i_lbl] - (cum + frac / 2.0) * col_h_l
                    ax.text(x[i_lbl], mid_z, lyr.name,
                            ha='center', va='center', fontsize=8,
                            color='#1A0A00', fontweight='bold', zorder=7,
                            bbox=dict(boxstyle='round,pad=0.12',
                                      fc=lyr.color, ec='none', alpha=0.75))
                    cum += frac
            # Mantle label in each column
            for col_key, x_lbl in label_specs:
                i_lbl = np.argmin(np.abs(x - x_lbl))
                ax.text(x[i_lbl], -(moho[i_lbl] + 8),
                        'Mantle', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold', zorder=7)

        # ── Root / deflection annotation (centre column only) ──────────────
        if p['show_root_ann'] and model in ('Airy', 'Flexure'):
            i_pk    = int(np.argmax(np.abs(h)))
            x_pk    = x[i_pk]
            ctr_Hc  = total_thickness(profiles['center'])
            root    = moho[i_pk] - ctr_Hc
            if abs(root) > 0.5:
                y_top = -ctr_Hc
                y_bot = -moho[i_pk]
                ax.annotate('',
                            xy=(x_pk, y_bot), xytext=(x_pk, y_top),
                            arrowprops=dict(arrowstyle='<->',
                                            color='white', lw=1.8),
                            zorder=8)
                tag = 'root' if model == 'Airy' else 'deflection'
                ax.text(x_pk + (x[-1] - x[0]) * 0.025,
                        0.5 * (y_top + y_bot),
                        f'{tag}\n{abs(root):.1f} km',
                        color='white', fontsize=9, va='center', zorder=8)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-z_bot, z_top)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (km)')
        ax.set_title('Geological Cross-section', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)

    # ── Time-series panel ─────────────────────────────────────────────────────
    def _draw_time_series(self, t_arr, h_pk, rt_pk, p):
        ax = self.canvas.ax_time
        ax.cla()

        if t_arr is None or p['erosion_type'] == 'None':
            ax.text(0.5, 0.5,
                    'Select an erosion / sedimentation type\nto see time evolution',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='#888888')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Elevation / Root vs Time', fontsize=10, fontweight='bold')
            return

        ax.plot(t_arr, h_pk,  'b-',  lw=2, label='Surface elevation (centre)')
        ax.plot(t_arr, rt_pk, 'r--', lw=2, label='Moho root depth (centre)')
        ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)

        t_now = p['current_t']
        ax.axvline(t_now, color='gray', lw=1.2, ls=':', alpha=0.9)
        ylims = ax.get_ylim()
        ax.text(t_now, ylims[1], f'  t={t_now:.0f} Ma',
                fontsize=8, color='gray', va='top')

        ax.set_xlim(0, p['duration'])
        ax.set_xlabel('Time (Ma)')
        ax.set_ylabel('km')
        ax.set_title('Elevation / Root vs Time', fontsize=10, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    # ── Status text helpers ───────────────────────────────────────────────────
    def _set_status_airy(self, h, moho, profiles, rho_m, d_water):
        i_pk    = int(np.argmax(np.abs(h)))
        ctr_lyr = profiles['center']
        rho_eff = effective_density(ctr_lyr)
        ctr_Hc  = total_thickness(ctr_lyr)
        root    = moho[i_pk] - ctr_Hc
        ratio   = rho_eff / (rho_m - rho_eff)
        self.dock.set_status(
            f'Peak h = {h[i_pk]:.1f} km  (centre col.)\n'
            f'Root depth = {root:.1f} km\n'
            f'Root ratio ρ_eff/(ρ_m−ρ_eff) = {ratio:.2f}\n'
            f'Ocean depth = {d_water:.2f} km'
        )

    def _set_status_pratt(self, h, rho_col, D):
        i_pk    = int(np.argmax(np.abs(h)))
        rho_min = rho_col.min()
        rho_max = rho_col.max()
        self.dock.set_status(
            f'Peak h = {h[i_pk]:.1f} km\n'
            f'Column ρ range: {rho_min:.0f}--{rho_max:.0f} kg/m³\n'
            f'Compensation depth D = {D:.0f} km'
        )

    def _set_status_flex(self, h, moho, profiles, rho_m, Te, d_water):
        i_pk    = int(np.argmax(np.abs(h)))
        ctr_lyr = profiles['center']
        rho_eff = effective_density(ctr_lyr)
        ctr_Hc  = total_thickness(ctr_lyr)
        defl    = moho[i_pk] - ctr_Hc
        airy_r  = rho_eff / (rho_m - rho_eff) * h[i_pk]
        frac    = defl / airy_r if abs(airy_r) > 0.01 else float('nan')
        self.dock.set_status(
            f'Peak h = {h[i_pk]:.1f} km  (centre col.)\n'
            f'Tₑ = {Te:.0f} km\n'
            f'Deflection = {defl:.1f} km\n'
            f'Airy root = {airy_r:.1f} km\n'
            f'Compensation fraction = {frac:.0%}\n'
            f'Ocean depth = {d_water:.2f} km'
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
