"""Signal & display controls dock for the FFT teaching tool."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QCheckBox, QDoubleSpinBox,
)
from src.apps.fourier_transforms.fft_signals import PRESET_NAMES


class SignalDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Signal & Display", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setWidget(widget)

        # ── Signal ────────────────────────────────────────────────────────────
        sig_group  = QGroupBox("Signal")
        sig_layout = QFormLayout(sig_group)

        self.preset_cb = QComboBox()
        self.preset_cb.addItems(PRESET_NAMES)
        sig_layout.addRow("Preset:", self.preset_cb)

        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 2.0)
        self.noise_spin.setSingleStep(0.05)
        self.noise_spin.setDecimals(2)
        self.noise_spin.setValue(0.2)
        sig_layout.addRow("Noise level:", self.noise_spin)

        self.show_comps_chk = QCheckBox("Show individual components")
        sig_layout.addRow(self.show_comps_chk)

        layout.addWidget(sig_group)

        # ── Display ───────────────────────────────────────────────────────────
        disp_group  = QGroupBox("Display")
        disp_layout = QVBoxLayout(disp_group)

        self.overlay_orig_chk = QCheckBox("Overlay original on filtered")
        self.overlay_orig_chk.setChecked(True)
        self.log_freq_chk     = QCheckBox("Log frequency scale")
        self.show_filter_chk  = QCheckBox("Show filter curve")
        self.show_filter_chk.setChecked(True)
        self.show_phase_chk   = QCheckBox("Show phase spectrum")

        for w in (self.overlay_orig_chk, self.log_freq_chk,
                  self.show_filter_chk, self.show_phase_chk):
            disp_layout.addWidget(w)

        layout.addWidget(disp_group)
        layout.addStretch()
