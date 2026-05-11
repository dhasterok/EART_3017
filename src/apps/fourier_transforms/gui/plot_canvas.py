"""Central three-panel matplotlib canvas for the FFT teaching tool."""
import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class PlotCanvas(QWidget):
    """Three-panel canvas: original signal | spectrum | filtered signal.

    A fourth phase-spectrum panel is inserted between the spectrum and filtered
    panels when ``set_phase_visible(True)`` is called.  Custom filter painting
    is enabled whenever *ftype* is 'Custom': left-drag passes (H=1), right-drag
    blocks (H=0).
    """

    custom_filter_changed = pyqtSignal(object)   # emits H ndarray

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig    = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Custom filter state
        self._painting   = False
        self._paint_val  = 1.0
        self.custom_H: np.ndarray | None = None
        self._freqs:   np.ndarray | None = None
        self._custom_active = False   # only paint when filter type == Custom

        self._show_phase = False
        self._build_layout(show_phase=False)

        self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self, show_phase):
        self.fig.clf()
        if show_phase:
            gs = self.fig.add_gridspec(4, 1, hspace=0.55,
                                       height_ratios=[2, 2, 1.2, 2])
            self.ax_sig  = self.fig.add_subplot(gs[0])
            self.ax_amp  = self.fig.add_subplot(gs[1])
            self.ax_pha  = self.fig.add_subplot(gs[2])
            self.ax_filt = self.fig.add_subplot(gs[3])
        else:
            gs = self.fig.add_gridspec(3, 1, hspace=0.50)
            self.ax_sig  = self.fig.add_subplot(gs[0])
            self.ax_amp  = self.fig.add_subplot(gs[1])
            self.ax_pha  = None
            self.ax_filt = self.fig.add_subplot(gs[2])

        # Secondary axis for filter overlay (shared x with amplitude axis)
        self.ax_H = self.ax_amp.twinx()
        self._show_phase = show_phase
        self._style_axes()

    def _style_axes(self):
        def _fmt(ax, title, xl, yl):
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        _fmt(self.ax_sig,  'Original Signal',  'Time (s)',       'Amplitude')
        _fmt(self.ax_amp,  'Frequency Domain', 'Frequency (Hz)', 'Amplitude')
        _fmt(self.ax_filt, 'Filtered Signal',  'Time (s)',       'Amplitude')
        if self.ax_pha is not None:
            _fmt(self.ax_pha, 'Phase Spectrum', 'Frequency (Hz)', 'Phase (rad)')

        self.ax_H.set_ylabel('Filter  H(f)', fontsize=7, color='C1')
        self.ax_H.set_ylim(-0.05, 1.35)
        self.ax_H.set_yticks([0, 0.5, 1])
        self.ax_H.tick_params(axis='y', labelsize=7, labelcolor='C1')

    def set_phase_visible(self, visible):
        if visible != self._show_phase:
            self._build_layout(show_phase=visible)

    # ── Custom filter painting ────────────────────────────────────────────────

    def init_custom_filter(self, freqs):
        self._freqs = freqs
        if self.custom_H is None or len(self.custom_H) != len(freqs):
            self.custom_H = np.ones(len(freqs))

    def reset_custom_filter(self):
        if self.custom_H is not None:
            self.custom_H[:] = 1.0
            self.custom_filter_changed.emit(self.custom_H.copy())

    def set_custom_active(self, active):
        self._custom_active = active

    def _on_press(self, event):
        if not self._custom_active:
            return
        if event.inaxes not in (self.ax_amp, self.ax_H):
            return
        self._painting  = True
        self._paint_val = 0.0 if event.button == 3 else 1.0
        self._paint(event.xdata)

    def _on_motion(self, event):
        if not self._painting:
            return
        if event.inaxes in (self.ax_amp, self.ax_H):
            self._paint(event.xdata)

    def _on_release(self, _event):
        self._painting = False

    def _paint(self, x):
        if x is None or self.custom_H is None or self._freqs is None:
            return
        idx = int(np.argmin(np.abs(self._freqs - x)))
        lo  = max(0, idx - 1)
        hi  = min(len(self.custom_H) - 1, idx + 1)
        self.custom_H[lo:hi + 1] = self._paint_val
        self.custom_filter_changed.emit(self.custom_H.copy())

    # ── Drawing ───────────────────────────────────────────────────────────────

    def update_signal(self, t, signal, components=None, show_components=False):
        ax = self.ax_sig
        ax.cla()
        ax.set_title('Original Signal', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        if show_components and components:
            for label, arr in components:
                ax.plot(t, arr, lw=0.7, alpha=0.7, label=label)
            ax.legend(fontsize=6, loc='upper right', ncol=2)

        ax.plot(t, signal, color='k', lw=0.9)

    def update_spectrum(self, freqs, F, H, show_filter=True, log_scale=False):
        # One-sided amplitude spectrum (normalised so sine amplitude → 1)
        N   = 2 * (len(freqs) - 1)
        amp = 2.0 * np.abs(F) / N
        amp[0]  /= 2   # DC  appears once
        amp[-1] /= 2   # Nyquist appears once

        # Amplitude axis
        ax = self.ax_amp
        ax.cla()
        ax.set_title('Frequency Domain', fontsize=9)
        ax.set_xlabel('Frequency (Hz)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, zorder=0)
        if log_scale:
            ax.set_xscale('log')

        ax.fill_between(freqs, 0, amp, alpha=0.35, color='C0', zorder=2)
        ax.plot(freqs, amp, color='C0', lw=0.8, zorder=3)

        # Phase axis
        if self.ax_pha is not None:
            self.ax_pha.cla()
            self.ax_pha.set_title('Phase Spectrum', fontsize=9)
            self.ax_pha.set_xlabel('Frequency (Hz)', fontsize=8)
            self.ax_pha.set_ylabel('Phase (rad)', fontsize=8)
            self.ax_pha.tick_params(labelsize=7)
            self.ax_pha.grid(True, alpha=0.3)
            if log_scale:
                self.ax_pha.set_xscale('log')
            self.ax_pha.plot(freqs, np.angle(F), color='C2', lw=0.6, alpha=0.8)
            self.ax_pha.set_ylim(-np.pi - 0.2, np.pi + 0.2)
            self.ax_pha.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            self.ax_pha.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'], fontsize=7)

        # Filter overlay
        self.ax_H.cla()
        self.ax_H.set_ylabel('Filter  H(f)', fontsize=7, color='C1')
        self.ax_H.set_ylim(-0.05, 1.35)
        self.ax_H.set_yticks([0, 0.5, 1])
        self.ax_H.tick_params(axis='y', labelsize=7, labelcolor='C1')
        if log_scale:
            self.ax_H.set_xscale('log')

        if show_filter:
            self.ax_H.fill_between(freqs, 0, H, color='C1', alpha=0.20, step='mid', zorder=1)
            self.ax_H.plot(freqs, H, color='C1', lw=1.4, zorder=4)

    def update_filtered(self, t, sig_orig, sig_filt, show_original=True):
        ax = self.ax_filt
        ax.cla()
        ax.set_title('Filtered Signal', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        if show_original:
            ax.plot(t, sig_orig, color='0.70', lw=0.8, label='Original', zorder=2)
        ax.plot(t, sig_filt, color='C3', lw=1.0, label='Filtered', zorder=3)
        if show_original:
            ax.legend(fontsize=7, loc='upper right')

    def redraw(self):
        self.canvas.draw_idle()
