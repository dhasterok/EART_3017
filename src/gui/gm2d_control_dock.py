from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QGroupBox, QFormLayout,
)
# QPushButton kept for the Apply Profile button
from PyQt6.QtCore import Qt, pyqtSignal

from src.utils.CustomWidgets import ToggleSwitch

class ControlsDock(QDockWidget):
    profile_changed      = pyqtSignal(float, float, int)
    depth_changed        = pyqtSignal(float)
    units_changed        = pyqtSignal(bool)
    bg_density_changed   = pyqtSignal(float)
    earth_field_changed  = pyqtSignal(float, float, float)   # F_nT, IE_deg, DE_deg
    z_obs_changed        = pyqtSignal(float)                 # observation height (km)
    model_mode_changed   = pyqtSignal(bool)                  # True = 2.5D, False = 2D
    snap_changed         = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                             Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        w = QWidget()
        self.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._use_km  = True
        self._is_2_5d = True   # default: 2.5D model

        # --- forward model mode ---
        model_group  = QGroupBox("Forward Model")
        model_layout = QHBoxLayout(model_group)
        model_layout.addWidget(QLabel("2D"))
        self._model_toggle = ToggleSwitch(height=20, bg_left_color="#669940", bg_right_color="#567fd0")
        self._model_toggle.setChecked(True)   # start in 2.5D
        self._model_toggle.stateChanged.connect(self._on_model_toggled)
        model_layout.addWidget(self._model_toggle)
        model_layout.addWidget(QLabel("2.5D"))
        model_layout.addStretch()
        layout.addWidget(model_group)

        # --- distance units ---
        units_group  = QGroupBox("Distance Units")
        units_layout = QHBoxLayout(units_group)
        units_layout.addWidget(QLabel("km"))
        self._unit_toggle = ToggleSwitch(height=20, bg_left_color="#669940", bg_right_color="#567fd0")
        self._unit_toggle.stateChanged.connect(self._on_unit_toggled)
        units_layout.addWidget(self._unit_toggle)
        units_layout.addWidget(QLabel("m"))
        units_layout.addStretch()
        layout.addWidget(units_group)

        # --- background density ---
        bg_group = QGroupBox("Background Density")
        bg_form  = QFormLayout(bg_group)
        self.spin_bg_density = QDoubleSpinBox()
        self.spin_bg_density.setRange(0.0, 1e5)
        self.spin_bg_density.setValue(2670.0)
        self.spin_bg_density.setSuffix(" kg/m³")
        self.spin_bg_density.setSingleStep(10.0)
        self.spin_bg_density.setDecimals(1)
        self.spin_bg_density.setToolTip(
            "Reference density.  Forward model uses: contrast = body density − background")
        self.spin_bg_density.valueChanged.connect(
            lambda v: self.bg_density_changed.emit(v))
        bg_form.addRow("ρ background:", self.spin_bg_density)
        layout.addWidget(bg_group)

        # --- Earth field ---
        ef_group = QGroupBox("Earth Field (magnetics)")
        ef_form  = QFormLayout(ef_group)

        self.spin_field_F = QDoubleSpinBox()
        self.spin_field_F.setRange(0.0, 100_000.0)
        self.spin_field_F.setValue(50_000.0)
        self.spin_field_F.setSuffix(" nT")
        self.spin_field_F.setSingleStep(1000.0)
        self.spin_field_F.setDecimals(0)
        self.spin_field_F.setToolTip("Total intensity of Earth's field (nT)")
        ef_form.addRow("F (intensity):", self.spin_field_F)

        self.spin_field_IE = QDoubleSpinBox()
        self.spin_field_IE.setRange(-90.0, 90.0)
        self.spin_field_IE.setValue(60.0)
        self.spin_field_IE.setSuffix(" °")
        self.spin_field_IE.setSingleStep(5.0)
        self.spin_field_IE.setDecimals(1)
        self.spin_field_IE.setToolTip(
            "Inclination of Earth's field (degrees, + downward from horizontal)")
        ef_form.addRow("Inclination IE:", self.spin_field_IE)

        self.spin_field_DE = QDoubleSpinBox()
        self.spin_field_DE.setRange(-180.0, 180.0)
        self.spin_field_DE.setValue(0.0)
        self.spin_field_DE.setSuffix(" °")
        self.spin_field_DE.setSingleStep(5.0)
        self.spin_field_DE.setDecimals(1)
        self.spin_field_DE.setToolTip(
            "Declination of Earth's field (degrees, + east of north)")
        ef_form.addRow("Declination DE:", self.spin_field_DE)

        self.spin_z_obs = QDoubleSpinBox()
        self.spin_z_obs.setRange(-100.0, 100.0)
        self.spin_z_obs.setValue(0.0)
        self.spin_z_obs.setSuffix(" km")
        self.spin_z_obs.setSingleStep(0.1)
        self.spin_z_obs.setDecimals(2)
        self.spin_z_obs.setToolTip(
            "Observation height (km; 0 = surface, negative = airborne above surface)")
        ef_form.addRow("Obs. height:", self.spin_z_obs)

        self.spin_field_F.valueChanged.connect(self._emit_earth_field)
        self.spin_field_IE.valueChanged.connect(self._emit_earth_field)
        self.spin_field_DE.valueChanged.connect(self._emit_earth_field)
        self.spin_z_obs.valueChanged.connect(
            lambda v: self.z_obs_changed.emit(v))
        layout.addWidget(ef_group)
        # ---------------─

        # --- profile ---─
        profile_group = QGroupBox("Gravity Profile")
        pform = QFormLayout(profile_group)

        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setRange(-1e6, 0)
        self.spin_xmin.setValue(-50.0)
        self.spin_xmin.setSuffix(" km")
        self.spin_xmin.setSingleStep(10.0)
        self.spin_xmin.setDecimals(1)

        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setRange(0, 1e6)
        self.spin_xmax.setValue(50.0)
        self.spin_xmax.setSuffix(" km")
        self.spin_xmax.setSingleStep(10.0)
        self.spin_xmax.setDecimals(1)

        self.spin_npts = QSpinBox()
        self.spin_npts.setRange(10, 2000)
        self.spin_npts.setValue(201)
        self.spin_npts.setSingleStep(10)

        pform.addRow("x min:", self.spin_xmin)
        pform.addRow("x max:", self.spin_xmax)
        pform.addRow("Points:", self.spin_npts)

        apply_btn = QPushButton("Apply Profile")
        apply_btn.clicked.connect(self._emit_profile)
        pform.addRow(apply_btn)
        layout.addWidget(profile_group)

        # --- depth range ---
        depth_group = QGroupBox("Model Depth")
        dform = QFormLayout(depth_group)
        self.spin_zmax = QDoubleSpinBox()
        self.spin_zmax.setRange(0.1, 1e6)
        self.spin_zmax.setValue(20.0)
        self.spin_zmax.setSuffix(" km")
        self.spin_zmax.setSingleStep(5.0)
        self.spin_zmax.setDecimals(1)
        self.spin_zmax.valueChanged.connect(lambda v: self.depth_changed.emit(v))
        dform.addRow("Max depth:", self.spin_zmax)
        layout.addWidget(depth_group)

        layout.addStretch()

    # --- unit toggle ---─

    def _on_unit_toggled(self, is_m: bool):
        use_km = not is_m
        if use_km == self._use_km:
            return
        factor = 1e-3 if use_km else 1e3
        self._use_km = use_km
        suffix = " km" if use_km else " m"
        step   = 10.0 if use_km else 10_000.0
        for spin in (self.spin_xmin, self.spin_xmax, self.spin_zmax):
            spin.blockSignals(True)
            spin.setValue(spin.value() * factor)
            spin.setSuffix(suffix)
            spin.setSingleStep(step)
            spin.blockSignals(False)
        self.units_changed.emit(use_km)

    def _emit_profile(self):
        self.profile_changed.emit(
            self.spin_xmin.value(), self.spin_xmax.value(), self.spin_npts.value())

    def _on_model_toggled(self, is_2_5d: bool):
        self._is_2_5d = is_2_5d
        self.spin_field_DE.setEnabled(is_2_5d)
        self.spin_z_obs.setEnabled(is_2_5d)
        self.model_mode_changed.emit(is_2_5d)

    def _emit_earth_field(self):
        self.earth_field_changed.emit(
            self.spin_field_F.value(),
            self.spin_field_IE.value(),
            self.spin_field_DE.value())