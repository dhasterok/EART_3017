from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox, QTableWidget,
    QDockWidget
)
from PyQt6.QtCore import Qt

from src.common.gui.CustomWidgets import RangeWidget

# ============================================================
# Control Dock
# ============================================================
class ControlDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Controls", parent)

        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setWidget(widget)

        # ---------------- Geotherm Parameters ----------------
        geo_group = QGroupBox("Geotherm Parameters")
        geo_layout = QVBoxLayout(geo_group)

        self.layer_table = QTableWidget(3, 4)
        self.layer_table.setHorizontalHeaderLabels(
            ["layer", "ztop [km]", "k [W/m/K]", "A [µW/m³]"]
        )
        geo_layout.addWidget(self.layer_table)

        self.compute_btn = QPushButton("Compute Geotherm")
        geo_layout.addWidget(self.compute_btn)

        layer_btns = QHBoxLayout()
        self.add_layer_btn = QPushButton("Add Layer")
        self.remove_layer_btn = QPushButton("Remove Layer")
        self.remove_layer_btn.setEnabled(False)
        layer_btns.addWidget(self.add_layer_btn)
        layer_btns.addWidget(self.remove_layer_btn)
        geo_layout.addLayout(layer_btns)

        self.pt_checkbox = QCheckBox("P–T dependence")
        geo_layout.addWidget(self.pt_checkbox)

        layout.addWidget(geo_group)

        # ---------------- Elevation Parameters ----------------
        elev_group = QGroupBox("Elevation Curve")
        elev_layout = QGridLayout(elev_group)

        self.hf_range = RangeWidget(min_val=20, max_val=150, step=5.0, decimals=1, suffix="mW m⁻²")
        self.hf_range.setValues(35, 100)
        self.ref_hf = QDoubleSpinBox()
        self.ref_hf.setRange(20, 150)
        self.ref_hf.setDecimals(1)
        self.ref_hf.setValue(40)

        self.expansivity = QDoubleSpinBox()
        self.expansivity.setRange(1, 6)
        self.expansivity.setValue(3)
        self.partition = QDoubleSpinBox()
        self.partition.setRange(0, 1)
        self.partition.setSingleStep(0.05)
        self.partition.setValue(0.75)

        elev_layout.addWidget(QLabel("Heat Flow Range"), 0, 0)
        elev_layout.addWidget(self.hf_range, 0, 1)
        elev_layout.addWidget(QLabel("Expansivity (×10⁻⁵/K)"), 2, 0)
        elev_layout.addWidget(self.expansivity, 2, 1)
        elev_layout.addWidget(QLabel("Partition Coefficient"), 3, 0)
        elev_layout.addWidget(self.partition, 3, 1)
        elev_layout.addWidget(QLabel("Reference Heat Flow"), 4, 0)
        elev_layout.addWidget(self.ref_hf, 4, 1)

        self.isostatic_btn = QPushButton("Isostatic Curve")
        elev_layout.addWidget(self.isostatic_btn, 5, 0, 1, 2)

        layout.addWidget(elev_group)

        # ---------------- Selection ----------------
        select_group = QGroupBox("Selected Geotherm")
        select_layout = QHBoxLayout(select_group)

        self.spinner = QSpinBox()
        self.spinner.setRange(0, 1)
        self.spinner.setEnabled(False)

        self.remove_geo_btn = QPushButton("Remove")
        self.remove_geo_btn.setEnabled(False)

        select_layout.addWidget(QLabel("Index"))
        select_layout.addWidget(self.spinner)
        select_layout.addWidget(self.remove_geo_btn)

        layout.addWidget(select_group)

        # ---------------- Reset ----------------
        self.reset_btn = QPushButton("Reset Axes")
        layout.addWidget(self.reset_btn)

        layout.addStretch()

