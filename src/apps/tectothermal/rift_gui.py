"""
rift_gui.py

Interactive explorer for the McKenzie (1978) pure shear rifting model.

Tabs
----
1. Full Solution   — T(z) profiles at multiple times + surface heat flow q(t)
2. Fourier Construction — step-by-step build-up of the initial condition
3. Temporal Decay  — mode-decay amplitudes + temperature evolution over time

Controls (left dock)
---------------------
  Spinboxes : L (km), Ts, Tm (°C), k (W m⁻¹ K⁻¹), N (Fourier terms), t_max (Ma)
  Sliders   : κ (km² Ma⁻¹, 24–48), γ (thinning factor, 0–1)

Table (right dock): n and aₙ for the current γ.
"""

from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2]  # src/apps/tectothermal -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QDockWidget, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from src.common.gui.CustomWidgets import CustomSlider
from src.physics.geothermics.pure_shear_rift import (
    decay_curves,
    equilibrium_temperature,
    fourier_coefs,
    fourier_terms,
    heat_flow,
    initial_temperature,
    temperature,
)


# ---------------------------------------------------------------------------
# k ↔ κ empirical linkage
#   κ [m²/s] = -0.99882e-6 + 0.8478e-6 * k [W/(m·K)]
#   App units: κ in km² Ma⁻¹  →  multiply m²/s by _KAPPA_CONV
# ---------------------------------------------------------------------------
_KAPPA_CONV = 1e6 * 365.25 * 24 * 3600 / 1e6   # m²/s → km² Ma⁻¹  (≈ 3.156e7)


def _k_to_kappa(k):
    """Thermal conductivity [W/(m·K)] → diffusivity [km² Ma⁻¹]."""
    return (-0.99882e-6 + 0.8478e-6 * k) * _KAPPA_CONV


def _kappa_to_k(kappa):
    """Thermal diffusivity [km² Ma⁻¹] → conductivity [W/(m·K)]."""
    return (kappa / _KAPPA_CONV + 0.99882e-6) / 0.8478e-6


# ---------------------------------------------------------------------------
# Reusable multi-axes canvas
# ---------------------------------------------------------------------------
class TabCanvas(QWidget):
    """Matplotlib figure with *nrows × ncols* axes embedded in a QWidget."""

    def __init__(self, nrows=1, ncols=1, parent=None):
        super().__init__(parent)
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        axes = self.fig.subplots(nrows, ncols)
        try:
            self.axes = list(axes.flat)
        except AttributeError:
            self.axes = [axes]

    def clear(self):
        for ax in self.axes:
            ax.cla()

    def draw(self):
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class RiftWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Pure Shear Rifting Explorer')
        self.setMinimumSize(1280, 800)

        self._cb_full     = None   # colorbar for Tab 1 T(z) panel
        self._cb_temporal = None   # colorbar for Tab 3 T(z) panel
        self._cb_hf       = None   # colorbar for Tab 4 heat flow survey
        self._syncing     = False  # prevents k ↔ κ signal loops

        self._setup_central()
        self._setup_controls_dock()
        self._setup_table_dock()
        self._update()

    # ------------------------------------------------------------------
    # Central widget — three tab canvases
    # ------------------------------------------------------------------
    def _setup_central(self):
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._w_full  = TabCanvas(nrows=1, ncols=2)   # T(z) | q(t)
        self._w_four  = TabCanvas(nrows=2, ncols=2)   # 4-panel Fourier build
        self._w_temp  = TabCanvas(nrows=1, ncols=2)   # decay curves | T(z,t)
        self._w_hf    = TabCanvas(nrows=1, ncols=1)   # q(t) for range of γ

        self._tabs.addTab(self._w_full,  'Full Solution')
        self._tabs.addTab(self._w_four,  'Fourier Construction')
        self._tabs.addTab(self._w_temp,  'Temporal Decay')
        self._tabs.addTab(self._w_hf,    'Heat Flow Survey')
        self._tabs.currentChanged.connect(self._refresh_tab)

    # ------------------------------------------------------------------
    # Left dock — controls
    # ------------------------------------------------------------------
    def _setup_controls_dock(self):
        dock = QDockWidget('Parameters', self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setSpacing(8)

        # ── Lithosphere ────────────────────────────────────────────
        grp_litho = QGroupBox('Lithosphere')
        vl_litho = QVBoxLayout(grp_litho)
        fl = QFormLayout()
        self._sb_L  = self._dsb(125.0,  50.0, 300.0,  5.0, 0, ' km')
        self._sb_L.setToolTip('Lithospheric thickness')
        self._sb_Ts = self._dsb(  0.0, -10.0,  50.0,  1.0, 0, ' °C')
        self._sb_Ts.setToolTip('Surface temperature')
        self._sb_Tm = self._dsb(1300.0, 900.0, 1600.0, 10.0, 0, ' °C')
        self._sb_Tm.setToolTip('Mantle temperature')
        fl.addRow('Ts:', self._sb_Ts)
        fl.addRow('Tm:', self._sb_Tm)
        fl.addRow('L:', self._sb_L)
        vl_litho.addLayout(fl)
        vl_litho.addWidget(QLabel('k  (W m⁻¹ K⁻¹)'))
        self._sl_k = CustomSlider(
            min_value=1.5, max_value=5.0, step=0.05,
            initial_value=2.5, precision=2)
        self._sl_k.setToolTip('Thermal conductivity')
        vl_litho.addWidget(self._sl_k)
        vbox.addWidget(grp_litho)

        # ── Rift parameters (sliders) ─────────────────────────────
        grp_rift = QGroupBox('Rift parameters')
        vl = QVBoxLayout(grp_rift)

        vl.addWidget(QLabel('κ  (km² Ma⁻¹)'))
        self._sl_kappa = CustomSlider(
            min_value=5.0, max_value=110.0, step=0.1,
            initial_value=_k_to_kappa(2.5), precision=3)
        vl.addWidget(self._sl_kappa)

        vl.addWidget(QLabel('\u03b3  (thinning factor)'))
        self._sl_gamma = CustomSlider(
            min_value=0.01, max_value=0.99, step=0.01,
            initial_value=0.50, precision=2)
        vl.addWidget(self._sl_gamma)
        vbox.addWidget(grp_rift)

        # ── Fourier series ────────────────────────────────────────────
        grp_f = QGroupBox('Fourier series')
        fl2 = QFormLayout(grp_f)
        self._sb_N = QSpinBox()
        self._sb_N.setRange(1, 200)
        self._sb_N.setValue(20)
        fl2.addRow('N terms:', self._sb_N)
        vbox.addWidget(grp_f)

        # ── Time range ────────────────────────────────────────────────
        grp_t = QGroupBox('Time range')
        fl3 = QFormLayout(grp_t)
        self._sb_tmax = self._dsb(250.0, 1.0, 2000.0, 10.0, 0, ' Ma')
        self._sb_nt = self._dsb(5.0, 1.0, 100.0, 1.0, 0, ' Ma')
        fl3.addRow('\u0394t:', self._sb_nt)
        fl3.addRow('t max:', self._sb_tmax)
        vbox.addWidget(grp_t)

        # ── Heat flow survey ──────────────────────────────────────────
        grp_hf = QGroupBox('Heat flow survey')
        fl4 = QFormLayout(grp_hf)
        self._sb_gmin = self._dsb(0.10, 0.01, 0.98, 0.05, 2)
        self._sb_gmax = self._dsb(0.90, 0.02, 0.99, 0.05, 2)
        self._sb_ng   = QSpinBox()
        self._sb_ng.setRange(2, 20)
        self._sb_ng.setValue(9)
        fl4.addRow('γ min:', self._sb_gmin)
        fl4.addRow('γ max:', self._sb_gmax)
        fl4.addRow('N curves:', self._sb_ng)
        vbox.addWidget(grp_hf)

        vbox.addStretch()

        for w in [self._sb_L, self._sb_Ts, self._sb_Tm, self._sb_tmax, self._sb_nt]:
            w.valueChanged.connect(self._update)
        for w in [self._sb_N, self._sb_ng]:
            w.valueChanged.connect(self._update)
        for w in [self._sb_gmin, self._sb_gmax]:
            w.valueChanged.connect(self._update)
        self._sl_k.valueChanged.connect(lambda _: self._on_k_changed())
        self._sl_kappa.valueChanged.connect(lambda _: self._on_kappa_changed())
        self._sl_gamma.valueChanged.connect(lambda _: self._update())

        scroll.setWidget(content)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    # ------------------------------------------------------------------
    # Right dock — Fourier coefficient table
    # ------------------------------------------------------------------
    def _setup_table_dock(self):
        dock = QDockWidget('Fourier Coefficients', self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(['n', 'aₙ  (°C)'])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)

        dock.setWidget(self._table)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dsb(val, lo, hi, step, dec, suffix=''):
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(dec)
        sb.setValue(val)
        if suffix:
            sb.setSuffix(suffix)
        return sb

    def _params(self):
        return dict(
            L=self._sb_L.value(),
            Ts=self._sb_Ts.value(),
            Tm=self._sb_Tm.value(),
            k=self._sl_k.value(),
            kappa=self._sl_kappa.value(),
            gamma=self._sl_gamma.value(),
            N=self._sb_N.value(),
            t_max=self._sb_tmax.value(),
            n_t=max(2, round(self._sb_tmax.value() / self._sb_nt.value())),
            g_min=self._sb_gmin.value(),
            g_max=self._sb_gmax.value(),
            n_gamma=self._sb_ng.value(),
        )

    # ------------------------------------------------------------------
    # k ↔ κ sync (bidirectional, guarded against re-entrancy)
    # ------------------------------------------------------------------
    def _on_k_changed(self):
        if self._syncing:
            return
        self._syncing = True
        self._sl_kappa.setValue(_k_to_kappa(self._sl_k.value()))
        self._syncing = False
        self._update()

    def _on_kappa_changed(self):
        if self._syncing:
            return
        self._syncing = True
        self._sl_k.setValue(_kappa_to_k(self._sl_kappa.value()))
        self._syncing = False
        self._update()

    # ------------------------------------------------------------------
    # Update entry point
    # ------------------------------------------------------------------
    def _update(self, _=None):
        p = self._params()
        self._update_table(p)
        self._refresh_tab(self._tabs.currentIndex(), p)

    def _refresh_tab(self, idx, p=None):
        if p is None:
            p = self._params()
        [self._plot_full, self._plot_fourier, self._plot_temporal,
         self._plot_hf_survey][idx](p)

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    def _update_table(self, p):
        N = p['N']
        n = np.arange(1, N + 1)
        an = fourier_coefs(n, p['gamma'], p['Ts'], p['Tm'])
        self._table.setRowCount(N)
        for i in range(N):
            self._table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._table.setItem(i, 1, QTableWidgetItem(f'{an[i]:.4f}'))

    # ------------------------------------------------------------------
    # Tab 1 — Full solution: T(z,t) + q(t)
    # ------------------------------------------------------------------
    def _plot_full(self, p):
        self._w_full.clear()
        ax_T, ax_q = self._w_full.axes

        L, Ts, Tm = p['L'], p['Ts'], p['Tm']
        kappa, gamma = p['kappa'], p['gamma']
        k, N = p['k'], p['N']
        t_max, n_t = p['t_max'], p['n_t']

        z = np.linspace(0, L, 300)
        t = np.linspace(0, t_max, n_t)

        T  = temperature(z, t, gamma, kappa, Ts, Tm, L, N)   # (nz, nt)
        q  = heat_flow(t, gamma, kappa, Ts, Tm, L, k, N)
        Te = equilibrium_temperature(z, Ts, Tm, L)
        Ti = initial_temperature(z, gamma, Ts, Tm, L)

        # Color profiles by time
        cmap_t = plt.get_cmap('plasma')
        norm_t = Normalize(vmin=0, vmax=t_max)

        ax_T.plot(Te, z, 'b--', linewidth=1.8, label='Equilibrium (t→∞)', zorder=3)
        ax_T.plot(Ti, z, 'r-',  linewidth=1.8, label='Initial (t=0)', zorder=3)
        for i in range(n_t):
            ax_T.plot(T[:, i], z, color=cmap_t(norm_t(t[i])),
                      linewidth=0.9, alpha=0.7)

        sm = ScalarMappable(cmap=cmap_t, norm=norm_t)
        sm.set_array([])
        if self._cb_full is None:
            self._cb_full = self._w_full.fig.colorbar(sm, ax=ax_T, label='Time (Ma)')
        else:
            self._cb_full.update_normal(sm)

        ax_T.set_xlim(Ts, Tm)
        ax_T.set_ylim(L, 0)
        ax_T.set_xlabel('Temperature (°C)')
        ax_T.set_ylabel('Depth (km)')
        ax_T.set_title('Temperature Profiles')
        ax_T.legend(fontsize=8)
        ax_T.grid(True, alpha=0.3)

        # Heat flow vs time (skip t=0 where q can be very large)
        t_pos  = t[t > 0]
        q_pos  = q[t > 0]
        q_eq   = k * (Tm - Ts) / L
        ax_q.plot(t_pos, q_pos, 'k-', linewidth=1.8)
        ax_q.axhline(q_eq, color='b', linestyle='--', linewidth=1.2,
                     label=f'Equilibrium  {q_eq:.1f} mW m⁻²')
        ax_q.axhline(q[0], color='r', linestyle='--', linewidth=1.2,
                     label=f'Post-rift  {q[0]:.1f} mW m⁻²')
        ax_q.set_xlim(0, t_max)
        ax_q.set_xlabel('Time (Ma)')
        ax_q.set_ylabel('Heat flow (mW m⁻²)')
        ax_q.set_title('Surface Heat Flow')
        ax_q.legend(fontsize=8)
        ax_q.grid(True, alpha=0.3)

        self._w_full.draw()

    # ------------------------------------------------------------------
    # Tab 2 — Fourier construction (follows terms.m figure structure)
    # ------------------------------------------------------------------
    def _plot_fourier(self, p):
        self._w_four.clear()
        ax_an, ax_ind, ax_cum, ax_cmp = self._w_four.axes

        L, Ts, Tm = p['L'], p['Ts'], p['Tm']
        gamma, N = p['gamma'], p['N']

        z  = np.linspace(0, L, 300)
        n  = np.arange(1, N + 1)
        an, terms_z, cumsum_z = fourier_terms(z, gamma, Ts, Tm, L, N)
        Te = equilibrium_temperature(z, Ts, Tm, L)
        Ti = initial_temperature(z, gamma, Ts, Tm, L)

        cmap = plt.get_cmap('tab20')
        colors = [cmap(i / max(N - 1, 1)) for i in range(N)]

        # ── a_n bar chart ─────────────────────────────────────────────
        ax_an.bar(n, an, color=colors, alpha=0.85, edgecolor='none')
        ax_an.axhline(0, color='k', linewidth=0.8)
        ax_an.set_xlabel('Index  n')
        ax_an.set_ylabel('aₙ  (°C)')
        ax_an.set_title('Fourier Coefficients  aₙ')
        ax_an.grid(True, alpha=0.3, axis='y')

        # ── Individual terms ──────────────────────────────────────────
        for i in range(N):
            ax_ind.plot(terms_z[i], z, color=colors[i], linewidth=1.2,
                        alpha=0.8, label=f'n={i+1}' if N <= 10 else None)
        ax_ind.axvline(0, color='k', linewidth=0.6)
        ax_ind.invert_yaxis()
        ax_ind.set_xlabel('aₙ sin(nπz/L)  (°C)')
        ax_ind.set_ylabel('Depth (km)')
        ax_ind.set_title('Individual Terms')
        ax_ind.grid(True, alpha=0.3)
        if N <= 10:
            ax_ind.legend(fontsize=6, loc='lower right')

        # ── Cumulative sum ────────────────────────────────────────────
        for i in range(N):
            ax_cum.plot(cumsum_z[i], z, color=colors[i], linewidth=1.2,
                        alpha=0.8, label=f'N={i+1}' if N <= 10 else None)
        ax_cum.plot(Ti - Te, z, 'k--', linewidth=1.5, label='Tᵢ − Tₑ', zorder=3)
        ax_cum.axvline(0, color='k', linewidth=0.6)
        ax_cum.invert_yaxis()
        ax_cum.set_xlabel('Σ aₙ sin(nπz/L)  (°C)')
        ax_cum.set_ylabel('Depth (km)')
        ax_cum.set_title('Cumulative Sum')
        ax_cum.grid(True, alpha=0.3)
        if N <= 10:
            ax_cum.legend(fontsize=6, loc='lower right')
        else:
            ax_cum.legend(handles=[ax_cum.lines[-1]], fontsize=6, loc='lower right')

        # ── Comparison: Tₑ, Tᵢ, Tₑ + series ─────────────────────────
        ax_cmp.plot(Te, z, 'b--', linewidth=1.8, label='Equilibrium  Tₑ')
        ax_cmp.plot(Ti, z, 'r-',  linewidth=1.8, label='Initial  Tᵢ')
        ax_cmp.plot(Te + cumsum_z[-1], z, 'k-', linewidth=1.8,
                    label=f'Tₑ + Σ  (N={N})')
        ax_cmp.invert_yaxis()
        ax_cmp.set_xlabel('Temperature (°C)')
        ax_cmp.set_ylabel('Depth (km)')
        ax_cmp.set_title('Comparison at  t = 0')
        ax_cmp.legend(fontsize=8)
        ax_cmp.grid(True, alpha=0.3)

        self._w_four.draw()

    # ------------------------------------------------------------------
    # Tab 3 — Temporal decay: mode amplitudes + T(z,t) evolution
    # ------------------------------------------------------------------
    def _plot_temporal(self, p):
        self._w_temp.clear()
        ax_dec, ax_T = self._w_temp.axes

        L, Ts, Tm = p['L'], p['Ts'], p['Tm']
        kappa, gamma = p['kappa'], p['gamma']
        N, t_max, n_t = p['N'], p['t_max'], p['n_t']

        t   = np.linspace(0, t_max, n_t)
        z   = np.linspace(0, L, 300)
        n   = np.arange(1, N + 1)
        dec = decay_curves(t, kappa, L, N)           # (N, nt)
        T   = temperature(z, t, gamma, kappa, Ts, Tm, L, N)

        cmap_n = plt.get_cmap('tab20')
        colors_n = [cmap_n(i / max(N - 1, 1)) for i in range(N)]

        cmap_t = plt.get_cmap('plasma')
        norm_t = Normalize(vmin=0, vmax=t_max)

        # ── Decay curves ──────────────────────────────────────────────
        for i in range(N):
            ax_dec.plot(t, dec[i], color=colors_n[i], linewidth=1.5,
                        alpha=0.85, label=f'n={i+1}' if N <= 10 else None)
        ax_dec.set_xlabel('Time (Ma)')
        ax_dec.set_ylabel('exp(−n²π²κt / L²)')
        ax_dec.set_title('Mode Decay Amplitudes')
        ax_dec.set_ylim(-0.02, 1.05)
        ax_dec.grid(True, alpha=0.3)
        if N <= 10:
            ax_dec.legend(fontsize=7, loc='upper right')
        else:
            # Annotate first and last mode only
            tau1  = L ** 2 / (np.pi ** 2 * kappa)
            tauN  = L ** 2 / (N ** 2 * np.pi ** 2 * kappa)
            ax_dec.text(0.98, 0.92, f'n=1  τ={tau1:.0f} Ma',
                        transform=ax_dec.transAxes, ha='right', fontsize=7,
                        color=colors_n[0])
            ax_dec.text(0.98, 0.80, f'n={N}  τ={tauN:.1f} Ma',
                        transform=ax_dec.transAxes, ha='right', fontsize=7,
                        color=colors_n[-1])

        # ── Temperature evolution ─────────────────────────────────────
        for i in range(n_t):
            ax_T.plot(T[:, i], z, color=cmap_t(norm_t(t[i])),
                      linewidth=0.9, alpha=0.7)

        sm = ScalarMappable(cmap=cmap_t, norm=norm_t)
        sm.set_array([])
        if self._cb_temporal is None:
            self._cb_temporal = self._w_temp.fig.colorbar(
                sm, ax=ax_T, label='Time (Ma)')
        else:
            self._cb_temporal.update_normal(sm)

        ax_T.set_xlim(Ts, Tm)
        ax_T.set_ylim(L, 0)
        ax_T.set_xlabel('Temperature (°C)')
        ax_T.set_ylabel('Depth (km)')
        ax_T.set_title('Temperature Evolution')
        ax_T.grid(True, alpha=0.3)

        self._w_temp.draw()


    # ------------------------------------------------------------------
    # Tab 4 — Heat flow survey: q(t) for a range of thinning factors
    # ------------------------------------------------------------------
    def _plot_hf_survey(self, p):
        self._w_hf.clear()
        ax = self._w_hf.axes[0]

        L, Ts, Tm = p['L'], p['Ts'], p['Tm']
        kappa, k  = p['kappa'], p['k']
        N, t_max, n_t = p['N'], p['t_max'], p['n_t']
        g_min, g_max, n_gamma = p['g_min'], p['g_max'], p['n_gamma']

        g_min = min(g_min, g_max)
        g_max = max(p['g_min'], p['g_max'])

        t     = np.linspace(0, t_max, n_t)
        t_pos = t[t > 0]
        gammas = np.linspace(g_min, g_max, n_gamma)

        cmap_g = plt.get_cmap('viridis')
        norm_g = Normalize(vmin=g_min, vmax=g_max)
        q_eq   = k * (Tm - Ts) / L

        for gamma in gammas:
            q = heat_flow(t_pos, gamma, kappa, Ts, Tm, L, k, N)
            ax.plot(t_pos, q, color=cmap_g(norm_g(gamma)), linewidth=1.5)

        # Highlight current slider gamma if it falls within the survey range
        gamma_cur = p['gamma']
        if g_min <= gamma_cur <= g_max:
            q_cur = heat_flow(t_pos, gamma_cur, kappa, Ts, Tm, L, k, N)
            ax.plot(t_pos, q_cur, color='k', linewidth=2.5,
                    label=f'γ = {gamma_cur:.2f}  (current)')

        ax.axhline(q_eq, color='b', linestyle='--', linewidth=1.2,
                   label=f'Equilibrium  {q_eq:.1f} mW m⁻²')

        sm = ScalarMappable(cmap=cmap_g, norm=norm_g)
        sm.set_array([])
        if self._cb_hf is None:
            self._cb_hf = self._w_hf.fig.colorbar(
                sm, ax=ax, label='γ  (thinning factor)')
        else:
            self._cb_hf.update_normal(sm)

        ax.set_xlim(0, t_max)
        ax.set_xlabel('Time (Ma)')
        ax.set_ylabel('Heat flow (mW m⁻²)')
        ax.set_title('Surface Heat Flow — Range of Thinning Factors')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        self._w_hf.draw()


# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(11)
    app.setFont(font)
    win = RiftWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
