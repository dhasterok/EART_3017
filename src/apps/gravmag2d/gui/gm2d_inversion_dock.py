from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox, QCheckBox, QFormLayout, QHBoxLayout, QDoubleSpinBox, QSpinBox

class InversionDock(QDockWidget):
    run_requested    = pyqtSignal(object)
    stop_requested   = pyqtSignal()
    revert_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Inversion", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                             Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        w  = QWidget()
        self.setWidget(w)
        vb = QVBoxLayout(w)
        vb.setAlignment(Qt.AlignmentFlag.AlignTop)

        inv_group = QGroupBox("Invert")
        inv_vb    = QVBoxLayout(inv_group)
        self.chk_vertices  = QCheckBox("Vertex positions");  self.chk_vertices.setChecked(True)
        self.chk_densities = QCheckBox("Density contrasts"); self.chk_densities.setChecked(True)
        self.chk_weights   = QCheckBox("Use data weights (1/σ²)")
        self.chk_weights.setToolTip("Weight residuals by 1/σ² when uncertainty column is loaded")
        inv_vb.addWidget(self.chk_vertices)
        inv_vb.addWidget(self.chk_densities)
        inv_vb.addWidget(self.chk_weights)
        vb.addWidget(inv_group)

        param_group = QGroupBox("Solver Parameters")
        pform       = QFormLayout(param_group)
        self.spin_damping = QDoubleSpinBox()
        self.spin_damping.setRange(1e-8, 1e4); self.spin_damping.setValue(1e-2)
        self.spin_damping.setDecimals(6); self.spin_damping.setSingleStep(0.001)
        self.spin_damping.setToolTip("Tikhonov/LM damping λ")
        pform.addRow("Damping λ:", self.spin_damping)
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(1, 500); self.spin_max_iter.setValue(20)
        pform.addRow("Max iterations:", self.spin_max_iter)
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(1e-10, 1.0); self.spin_tol.setValue(1e-4)
        self.spin_tol.setDecimals(8); self.spin_tol.setSingleStep(1e-5)
        self.spin_tol.setToolTip("Convergence tolerance on ||Δp||")
        pform.addRow("Tolerance:", self.spin_tol)
        vb.addWidget(param_group)

        btn_row = QHBoxLayout()
        self.btn_run  = QPushButton("Run");  self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        btn_row.addWidget(self.btn_run); btn_row.addWidget(self.btn_stop)
        vb.addLayout(btn_row)

        self.btn_revert = QPushButton("Revert to Starting Model")
        self.btn_revert.setEnabled(False)
        self.btn_revert.setToolTip(
            "Restore the model to the state before inversion was run")
        self.btn_revert.clicked.connect(self.revert_requested.emit)
        vb.addWidget(self.btn_revert)

        self.lbl_progress = QLabel("Ready")
        self.lbl_progress.setWordWrap(True)
        vb.addWidget(self.lbl_progress)
        vb.addStretch()

    def _on_run(self):
        from src.inversion.gravity_inversion import InversionConfig
        cfg = InversionConfig(
            invert_vertices  = self.chk_vertices.isChecked(),
            invert_densities = self.chk_densities.isChecked(),
            use_weights      = self.chk_weights.isChecked(),
            damping          = self.spin_damping.value(),
            max_iter         = self.spin_max_iter.value(),
            tol              = self.spin_tol.value(),
        )
        self.run_requested.emit(cfg)

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def update_progress(self, it: int, rms: float):
        self.lbl_progress.setText(
            f"Iter: {it} / {self.spin_max_iter.value()}  |  RMS: {rms:.4f} mGal")

    def show_converged(self, rms: float):
        self.lbl_progress.setText(f"Converged  |  RMS: {rms:.4f} mGal")
        self.set_running(False)
        self.btn_revert.setEnabled(True)

    def show_stopped(self):
        self.lbl_progress.setText("Stopped by user")
        self.set_running(False)
        self.btn_revert.setEnabled(True)