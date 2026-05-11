"""Filter and phase-manipulation controls dock for the FFT teaching tool."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QPushButton, QButtonGroup, QRadioButton, QLabel,
)
from src.apps.fourier_transforms.fft_filters import FILTER_TYPES, TAPER_TYPES, PHASE_MODES
from src.apps.fourier_transforms.fft_signals import FS


_FMAX = FS / 2.0   # Nyquist


class FilterDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Filter & Phase", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setWidget(widget)

        # ── Filter ────────────────────────────────────────────────────────────
        filt_group  = QGroupBox("Filter")
        filt_layout = QFormLayout(filt_group)

        self.type_cb = QComboBox()
        self.type_cb.addItems(FILTER_TYPES)
        filt_layout.addRow("Type:", self.type_cb)

        def _dsb(lo, hi, val, step=1.0, dec=1, suffix=' Hz'):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(val)
            sb.setSingleStep(step)
            sb.setDecimals(dec)
            sb.setSuffix(suffix)
            return sb

        self.fc1_spin  = _dsb(0.5, _FMAX - 0.5, 30.0)
        self.fc2_spin  = _dsb(0.5, _FMAX - 0.5, 80.0)
        self.fc1_label = QLabel("Cutoff:")
        self.fc2_label = QLabel("Upper cutoff:")
        filt_layout.addRow(self.fc1_label, self.fc1_spin)
        filt_layout.addRow(self.fc2_label, self.fc2_spin)

        self.reset_custom_btn = QPushButton("Reset custom filter")
        self.reset_custom_btn.setEnabled(False)
        filt_layout.addRow(self.reset_custom_btn)

        self.taper_cb = QComboBox()
        self.taper_cb.addItems(TAPER_TYPES)
        filt_layout.addRow("Taper:", self.taper_cb)

        self.taper_w_spin = _dsb(0.5, 100.0, 10.0)
        self.taper_w_spin.setEnabled(False)
        filt_layout.addRow("Taper width:", self.taper_w_spin)

        layout.addWidget(filt_group)

        # ── Phase manipulation ────────────────────────────────────────────────
        pha_group  = QGroupBox("Phase Manipulation")
        pha_layout = QVBoxLayout(pha_group)

        self._phase_bg   = QButtonGroup(self)
        self._phase_btns = []
        for mode in PHASE_MODES:
            rb = QRadioButton(mode)
            self._phase_bg.addButton(rb)
            self._phase_btns.append(rb)
            pha_layout.addWidget(rb)
        self._phase_btns[0].setChecked(True)

        self.phase_shift_spin = _dsb(0.0, 360.0, 0.0, step=5.0, dec=1, suffix=' °')
        pha_layout.addWidget(QLabel("Phase shift (all components):"))
        pha_layout.addWidget(self.phase_shift_spin)

        layout.addWidget(pha_group)
        layout.addStretch()

        # Internal wiring
        self.type_cb.currentTextChanged.connect(self._on_type_changed)
        self.taper_cb.currentTextChanged.connect(self._on_taper_changed)
        self._on_type_changed(FILTER_TYPES[0])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _on_type_changed(self, ftype):
        band = ftype in ('Band-pass', 'Band-stop')
        self.fc1_label.setText('Lower cutoff:' if band else 'Cutoff:')
        self.fc2_spin.setEnabled(band)
        self.fc2_label.setEnabled(band)
        custom = ftype == 'Custom'
        self.reset_custom_btn.setEnabled(custom)
        has_taper = ftype not in ('None', 'Custom')
        self.taper_cb.setEnabled(has_taper)
        self.taper_w_spin.setEnabled(
            has_taper and self.taper_cb.currentText() != 'None'
        )

    def _on_taper_changed(self, taper):
        self.taper_w_spin.setEnabled(
            taper != 'None' and self.type_cb.currentText() not in ('None', 'Custom')
        )

    @property
    def phase_mode(self):
        for btn in self._phase_btns:
            if btn.isChecked():
                return btn.text()
        return 'Normal'
