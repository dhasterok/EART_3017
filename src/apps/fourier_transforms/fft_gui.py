"""Fourier Transform Filtering GUI — interactive teaching tool.

Run directly:
    python src/apps/fourier_transforms/fft_gui.py

Layout
------
Left dock   : signal preset, noise level, display toggles
Central     : three matplotlib panels (signal | spectrum + filter | filtered)
Right dock  : filter type & cutoffs, taper, phase manipulation

Custom filter
-------------
Select "Custom" as filter type, then left-drag on the spectrum panel to paint
pass regions (H=1) and right-drag to paint stop regions (H=0).
"""
import sys
from pathlib import Path
import numpy as np

_HERE   = Path(__file__).resolve().parent
_COURSE = _HERE.parents[2]
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow

from src.apps.fourier_transforms.fft_signals import make_signal, get_freqs
from src.apps.fourier_transforms.fft_filters import make_filter, apply_filter
from src.apps.fourier_transforms.gui.plot_canvas import PlotCanvas
from src.apps.fourier_transforms.gui.signal_dock import SignalDock
from src.apps.fourier_transforms.gui.filter_dock import FilterDock


class FFTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Transform Explorer")
        self.resize(1300, 800)

        self.canvas = PlotCanvas(self)
        self.setCentralWidget(self.canvas)

        self.sig_dock = SignalDock(self)
        self.flt_dock = FilterDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,  self.sig_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.flt_dock)

        self._freqs = get_freqs()
        self.canvas.init_custom_filter(self._freqs)

        self._wire_signals()
        self._update()

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _wire_signals(self):
        sd = self.sig_dock
        fd = self.flt_dock

        # Signal controls
        sd.preset_cb.currentTextChanged.connect(self._update)
        sd.noise_spin.valueChanged.connect(self._update)
        sd.show_comps_chk.stateChanged.connect(self._update)

        # Display toggles
        sd.overlay_orig_chk.stateChanged.connect(self._update)
        sd.log_freq_chk.stateChanged.connect(self._update)
        sd.show_filter_chk.stateChanged.connect(self._update)
        sd.show_phase_chk.stateChanged.connect(self._on_phase_toggle)

        # Filter controls
        fd.type_cb.currentTextChanged.connect(self._on_filter_type_changed)
        fd.fc1_spin.valueChanged.connect(self._update)
        fd.fc2_spin.valueChanged.connect(self._update)
        fd.taper_cb.currentTextChanged.connect(self._update)
        fd.taper_w_spin.valueChanged.connect(self._update)
        fd.reset_custom_btn.clicked.connect(self._reset_custom)

        # Phase controls
        for btn in fd._phase_btns:
            btn.toggled.connect(self._update)
        fd.phase_shift_spin.valueChanged.connect(self._update)

        # Custom filter painting
        self.canvas.custom_filter_changed.connect(lambda _: self._update())

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_phase_toggle(self):
        self.canvas.set_phase_visible(self.sig_dock.show_phase_chk.isChecked())
        self._update()

    def _on_filter_type_changed(self, ftype):
        self.canvas.set_custom_active(ftype == 'Custom')
        self._update()

    def _reset_custom(self):
        self.canvas.reset_custom_filter()

    # ── Master update ─────────────────────────────────────────────────────────

    def _update(self):
        sd = self.sig_dock
        fd = self.flt_dock

        # 1. Signal
        t, signal, components = make_signal(
            sd.preset_cb.currentText(),
            sd.noise_spin.value(),
        )

        # 2. Forward FFT
        F = np.fft.rfft(signal)

        # 3. Build filter
        ftype = fd.type_cb.currentText()
        if ftype == 'Custom':
            H = (self.canvas.custom_H.copy()
                 if self.canvas.custom_H is not None
                 else np.ones(len(self._freqs)))
        else:
            H = make_filter(
                self._freqs, ftype,
                fd.fc1_spin.value(),
                fd.fc2_spin.value(),
                fd.taper_cb.currentText(),
                fd.taper_w_spin.value(),
            )

        # 4. Apply filter + phase manipulation
        F_filt   = apply_filter(F, H, fd.phase_mode,
                                np.deg2rad(fd.phase_shift_spin.value()))
        sig_filt = np.fft.irfft(F_filt, n=len(signal))

        # 5. Redraw
        self.canvas.update_signal(t, signal, components,
                                  sd.show_comps_chk.isChecked())
        self.canvas.update_spectrum(self._freqs, F, H,
                                    sd.show_filter_chk.isChecked(),
                                    sd.log_freq_chk.isChecked())
        self.canvas.update_filtered(t, signal, sig_filt,
                                    sd.overlay_orig_chk.isChecked())
        self.canvas.redraw()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    win = FFTApp()
    win.show()
    sys.exit(app.exec())
