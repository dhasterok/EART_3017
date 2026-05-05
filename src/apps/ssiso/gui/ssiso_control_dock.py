from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox, QTableWidget,
    QAbstractItemView, QHeaderView, QToolButton, QMessageBox, QDockWidget,
    QStyledItemDelegate,
)


class _CenterDelegate(QStyledItemDelegate):
    """Renders and edits all table cells with centred alignment."""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignmentFlag.AlignCenter

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        if editor is not None:
            editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return editor

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

        self.layer_table = QTableWidget(3, 3)
        self.layer_table.setHorizontalHeaderLabels(
            ["ztop\n[km]", "k\n[W/m/K]", "A\n[µW/m³]"]
        )
        self.layer_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.layer_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.layer_table.setItemDelegate(_CenterDelegate(self.layer_table))
        self.layer_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.layer_table.installEventFilter(self)
        geo_layout.addWidget(self.layer_table)

        geo_params = QFormLayout()
        self.geo_T0 = QDoubleSpinBox()
        self.geo_T0.setRange(-10, 50)
        self.geo_T0.setDecimals(1)
        self.geo_T0.setValue(20.0)
        self.geo_T0.setSuffix(" °C")
        geo_params.addRow("Surface temperature:", self.geo_T0)

        self.geo_qs = QDoubleSpinBox()
        self.geo_qs.setRange(20, 150)
        self.geo_qs.setDecimals(1)
        self.geo_qs.setValue(60.0)
        self.geo_qs.setSuffix(" mW m⁻²")
        geo_params.addRow("Surface heat flow:", self.geo_qs)
        geo_layout.addLayout(geo_params)

        self.compute_btn = QPushButton("Compute Geotherm")
        geo_layout.addWidget(self.compute_btn)

        layer_btns = QHBoxLayout()
        self.add_layer_btn = QPushButton("Add Layer")

        self.move_up_btn = QToolButton()
        self.move_up_btn.setText("▲")
        self.move_up_btn.setToolTip("Move selected row up")

        self.move_down_btn = QToolButton()
        self.move_down_btn.setText("▼")
        self.move_down_btn.setToolTip("Move selected row down")

        layer_btns.addWidget(self.add_layer_btn)
        layer_btns.addStretch()
        layer_btns.addWidget(self.move_up_btn)
        layer_btns.addWidget(self.move_down_btn)
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

        # Wire internal layer buttons
        self.add_layer_btn.clicked.connect(self._add_layer)
        self.move_up_btn.clicked.connect(self._move_row_up)
        self.move_down_btn.clicked.connect(self._move_row_down)

    # ------------------------------------------------------------------
    # Layer management (internal)
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self.layer_table and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Delete:
                self._delete_selected_rows()
                return True
        return super().eventFilter(obj, event)

    def _add_layer(self):
        t = self.layer_table
        rows = sorted({i.row() for i in t.selectedIndexes()})
        insert_at = rows[-1] + 1 if rows else t.rowCount()
        t.insertRow(insert_at)
        t.selectRow(insert_at)

    def _delete_selected_rows(self):
        t = self.layer_table
        rows = sorted({i.row() for i in t.selectedIndexes()}, reverse=True)
        if not rows:
            return
        n = len(rows)
        msg = QMessageBox(self)
        msg.setWindowTitle("Confirm Deletion")
        msg.setText(f"Delete {n} selected layer{'s' if n > 1 else ''}?")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        if msg.exec() == QMessageBox.StandardButton.Yes:
            for r in rows:
                t.removeRow(r)

    def _move_row_up(self):
        t = self.layer_table
        rows = sorted({i.row() for i in t.selectedIndexes()})
        if len(rows) != 1 or rows[0] == 0:
            return
        r = rows[0]
        for col in range(t.columnCount()):
            above = t.takeItem(r - 1, col)
            curr  = t.takeItem(r, col)
            t.setItem(r - 1, col, curr)
            t.setItem(r,     col, above)
        t.selectRow(r - 1)

    def _move_row_down(self):
        t = self.layer_table
        rows = sorted({i.row() for i in t.selectedIndexes()})
        if len(rows) != 1 or rows[0] == t.rowCount() - 1:
            return
        r = rows[0]
        for col in range(t.columnCount()):
            below = t.takeItem(r + 1, col)
            curr  = t.takeItem(r, col)
            t.setItem(r + 1, col, curr)
            t.setItem(r,     col, below)
        t.selectRow(r + 1)
