from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QMessageBox, QColorDialog,
    QAbstractItemView
)
from PyQt6.QtGui import QColor

from src.gui.gm2d_types import PolygonBody

COL_VIS      = 0
COL_COLOR    = 1
COL_NAME     = 2
COL_DENSITY  = 3
COL_CONTRAST = 4
COL_SUSCEPT  = 5
COL_REM_J    = 6   # remanence intensity (A/m)
COL_REM_INC  = 7   # remanence inclination (degrees)
COL_REM_DEC  = 8   # remanence declination (degrees)
COL_NVERTS   = 9
N_COLS       = 10

VCOL_IDX = 0
VCOL_X   = 1
VCOL_Z   = 2


class PolygonDock(QDockWidget):
    """
    Bottom dock – polygon table (left) and vertex table (right).
    Columns: Vis | Color | Name | Density | Contrast | χ (SI) | Verts
    """

    body_selected        = pyqtSignal(object)
    body_changed         = pyqtSignal(object)
    body_vertex_changed  = pyqtSignal(object)
    body_vertex_deleted  = pyqtSignal(object)   # emitted after a vertex is removed
    body_delete_requested = pyqtSignal(list)    # emitted when user confirms polygon deletion

    def __init__(self, parent=None):
        super().__init__("Polygon Bodies", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea |
                             Qt.DockWidgetArea.TopDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- left: polygon table ---
        left_w  = QWidget()
        left_vb = QVBoxLayout(left_w)
        left_vb.setContentsMargins(0, 0, 0, 0)
        left_vb.addWidget(QLabel("Polygon Bodies"))

        self.table = QTableWidget(0, N_COLS)
        self.table.setHorizontalHeaderLabels(
            ["Vis", "Color", "Name",
             "Density\n(kg/m³)", "Contrast\n(kg/m³)",
             "χ (SI)",
             "J_rem\n(A/m)", "Inc_rem\n(°)", "Dec_rem\n(°)",
             "Verts"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_NAME,     QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(COL_DENSITY,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_CONTRAST, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_SUSCEPT,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_REM_J,    QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_REM_INC,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_REM_DEC,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_NVERTS,   QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_VIS,      QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_COLOR,    QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(120)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.installEventFilter(self)
        left_vb.addWidget(self.table)

        self.btn_del_poly = QPushButton("Delete Selected Polygon(s)")
        self.btn_del_poly.setToolTip("Remove the selected polygon(s) from the model")
        self.btn_del_poly.setEnabled(False)
        self.btn_del_poly.clicked.connect(self._confirm_delete_polygons)
        left_vb.addWidget(self.btn_del_poly)

        splitter.addWidget(left_w)

        # --- right: vertex table ---
        right_w  = QWidget()
        right_vb = QVBoxLayout(right_w)
        right_vb.setContentsMargins(0, 0, 0, 0)
        self._vert_label = QLabel("Vertices")
        right_vb.addWidget(self._vert_label)

        self.vert_table = QTableWidget(0, 3)
        self.vert_table.setHorizontalHeaderLabels(["#", "X (km)", "Z (km)"])
        vh = self.vert_table.horizontalHeader()
        vh.setSectionResizeMode(VCOL_IDX, QHeaderView.ResizeMode.ResizeToContents)
        vh.setSectionResizeMode(VCOL_X,   QHeaderView.ResizeMode.Stretch)
        vh.setSectionResizeMode(VCOL_Z,   QHeaderView.ResizeMode.Stretch)
        self.vert_table.setMinimumHeight(120)
        self.vert_table.setAlternatingRowColors(True)
        self.vert_table.itemChanged.connect(self._on_vert_item_changed)
        self.vert_table.itemSelectionChanged.connect(self._on_vert_selection_changed)
        self.vert_table.installEventFilter(self)
        right_vb.addWidget(self.vert_table)

        self.btn_del_vert = QPushButton("Delete Selected Vertex")
        self.btn_del_vert.setToolTip(
            "Remove the selected vertex (body must keep at least 3 vertices)")
        self.btn_del_vert.setEnabled(False)
        self.btn_del_vert.clicked.connect(self._delete_selected_vertex)
        right_vb.addWidget(self.btn_del_vert)

        splitter.addWidget(right_w)

        splitter.setSizes([600, 300])
        outer.addWidget(splitter)
        self.setWidget(container)

        self._bodies:   List[PolygonBody]    = []
        self._sel_body: Optional[PolygonBody] = None
        self._updating  = False
        self._bg_density: float = 2670.0
        self._use_km:     bool  = True
        self._is_2_5d:    bool  = True

    # --- synchronise with canvas ---

    def sync(self, bodies: List[PolygonBody]):
        self._updating = True
        self._bodies   = bodies
        self.table.setRowCount(len(bodies))
        for row, body in enumerate(bodies):
            self._populate_row(row, body)
        self._updating = False

    def _populate_row(self, row: int, body: PolygonBody):
        # Visible
        vis_item = QTableWidgetItem()
        vis_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        vis_item.setCheckState(
            Qt.CheckState.Checked if body.visible else Qt.CheckState.Unchecked)
        self.table.setItem(row, COL_VIS, vis_item)

        # Color button
        color_btn = QPushButton()
        color_btn.setFixedSize(28, 22)
        self._set_btn_color(color_btn, body.color)
        color_btn.clicked.connect(lambda _, b=body, btn=color_btn:
                                  self._pick_color(b, btn))
        self.table.setCellWidget(row, COL_COLOR, color_btn)

        # Name
        self.table.setItem(row, COL_NAME, QTableWidgetItem(body.name))

        # Density
        dens_item = QTableWidgetItem(f"{body.density:.1f}")
        dens_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_DENSITY, dens_item)

        # Contrast (editable; editing back-calculates density)
        cont_item = QTableWidgetItem(f"{body.density - self._bg_density:.1f}")
        cont_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_CONTRAST, cont_item)

        # Susceptibility
        susc_item = QTableWidgetItem(f"{body.susceptibility:.5f}")
        susc_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_SUSCEPT, susc_item)

        # Remanence intensity (A/m)
        rem_j_item = QTableWidgetItem(f"{body.remanence_Am:.4f}")
        rem_j_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_REM_J, rem_j_item)

        # Remanence inclination (degrees)
        rem_inc_item = QTableWidgetItem(f"{body.remanence_inc_deg:.1f}")
        rem_inc_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, COL_REM_INC, rem_inc_item)

        # Remanence declination (degrees) — editable only in 2.5D mode
        rem_dec_item = QTableWidgetItem(f"{body.remanence_dec_deg:.1f}")
        rem_dec_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if not self._is_2_5d:
            rem_dec_item.setFlags(rem_dec_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, COL_REM_DEC, rem_dec_item)

        # #Vertices (read-only)
        nv_item = QTableWidgetItem(str(len(body.vertices)))
        nv_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.table.setItem(row, COL_NVERTS, nv_item)

    def update_row(self, body: PolygonBody):
        self._updating = True
        try:
            row = self._bodies.index(body)
        except ValueError:
            self._updating = False
            return
        nv = self.table.item(row, COL_NVERTS)
        if nv:
            nv.setText(str(len(body.vertices)))
        ct = self.table.item(row, COL_CONTRAST)
        if ct:
            ct.setText(f"{body.density - self._bg_density:.1f}")
        self._updating = False
        if body is self._sel_body:
            self._populate_vert_table(body)

    def set_bg_density(self, bg: float):
        self._bg_density = bg
        self._updating   = True
        for row, body in enumerate(self._bodies):
            ct = self.table.item(row, COL_CONTRAST)
            if ct:
                ct.setText(f"{body.density - bg:.1f}")
        self._updating = False

    def set_model_mode(self, is_2_5d: bool):
        """Enable or disable editing of the Dec_rem column based on model mode."""
        self._is_2_5d = is_2_5d
        for row in range(self.table.rowCount()):
            item = self.table.item(row, COL_REM_DEC)
            if item is None:
                continue
            if is_2_5d:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def set_units(self, use_km: bool):
        self._use_km = use_km
        unit = "km" if use_km else "m"
        self.vert_table.setHorizontalHeaderLabels(
            ["#", f"X ({unit})", f"Z ({unit})"])
        if self._sel_body is not None:
            self._populate_vert_table(self._sel_body)

    # --- vertex table ---

    def show_vertices(self, body: Optional[PolygonBody]):
        self._sel_body = body
        self._populate_vert_table(body)

    def _populate_vert_table(self, body: Optional[PolygonBody]):
        self._updating = True
        self.vert_table.setRowCount(0)
        if body is None or not body.vertices:
            self._vert_label.setText("Vertices")
            self._updating = False
            return
        self._vert_label.setText(f"Vertices -- {body.name}")
        scale = 1.0 if self._use_km else 1000.0
        self.vert_table.setRowCount(len(body.vertices))
        for i, (vx, vz) in enumerate(body.vertices):
            idx_item = QTableWidgetItem(str(i))
            idx_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.vert_table.setItem(i, VCOL_IDX, idx_item)

            x_item = QTableWidgetItem(f"{vx * scale:.4f}")
            x_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.vert_table.setItem(i, VCOL_X, x_item)

            z_item = QTableWidgetItem(f"{vz * scale:.4f}")
            z_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.vert_table.setItem(i, VCOL_Z, z_item)
        self._updating = False

    def _on_vert_selection_changed(self):
        """Enable the delete button only when a row is selected and body has > 3 verts."""
        rows = {idx.row() for idx in self.vert_table.selectedIndexes()}
        can_delete = (
            bool(rows)
            and self._sel_body is not None
            and len(self._sel_body.vertices) > 3
        )
        self.btn_del_vert.setEnabled(can_delete)

    def _delete_selected_vertex(self):
        if self._sel_body is None:
            return
        rows = sorted({idx.row() for idx in self.vert_table.selectedIndexes()})
        if not rows:
            return
        if len(self._sel_body.vertices) - len(rows) < 3:
            return   # would leave fewer than 3 vertices — refuse silently
        # When multiple vertices share the same coordinates, only remove the
        # one at the selected index (not all duplicates).  Removing in reverse
        # order keeps earlier indices valid.
        for row in reversed(rows):
            if 0 <= row < len(self._sel_body.vertices):
                del self._sel_body.vertices[row]
        self._populate_vert_table(self._sel_body)
        self.body_vertex_deleted.emit(self._sel_body)

    def eventFilter(self, source, event):
        """Intercept Delete / Backspace on both tables."""
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.KeyPress and event.key() in (
                Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if source is self.table:
                if self.btn_del_poly.isEnabled():
                    self._confirm_delete_polygons()
                return True
            if source is self.vert_table:
                if self.btn_del_vert.isEnabled():
                    self._delete_selected_vertex()
                return True
        return super().eventFilter(source, event)

    def _on_vert_item_changed(self, item: QTableWidgetItem):
        if self._updating or self._sel_body is None:
            return
        row, col = item.row(), item.column()
        if col not in (VCOL_X, VCOL_Z):
            return
        if row < 0 or row >= len(self._sel_body.vertices):
            return
        scale = 1.0 if self._use_km else 1e-3
        try:
            val_km = float(item.text()) * scale
        except ValueError:
            self._populate_vert_table(self._sel_body)
            return
        coord = 0 if col == VCOL_X else 1
        self._sel_body.vertices[row][coord] = val_km
        self.body_vertex_changed.emit(self._sel_body)

    def _set_btn_color(self, btn: QPushButton, color: str):
        c = QColor(color)
        btn.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #555;")

    def _pick_color(self, body: PolygonBody, btn: QPushButton):
        c = QColorDialog.getColor(QColor(body.color), self, "Choose color")
        if c.isValid():
            body.color = c.name()
            self._set_btn_color(btn, body.color)
            self.body_changed.emit(body)

    # --- table item changed ------------------------------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._updating:
            return
        row = item.row()
        if row < 0 or row >= len(self._bodies):
            return
        body = self._bodies[row]
        col  = item.column()

        if col == COL_VIS:
            body.visible = (item.checkState() == Qt.CheckState.Checked)
            self.body_changed.emit(body)

        elif col == COL_NAME:
            body.name = item.text().strip() or body.name

        elif col == COL_DENSITY:
            try:
                body.density = float(item.text())
                self._updating = True
                ct = self.table.item(row, COL_CONTRAST)
                if ct:
                    ct.setText(f"{body.density - self._bg_density:.1f}")
                self._updating = False
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.density:.1f}")
                self._updating = False

        elif col == COL_CONTRAST:
            try:
                contrast = float(item.text())
                body.density = contrast + self._bg_density
                self._updating = True
                dn = self.table.item(row, COL_DENSITY)
                if dn:
                    dn.setText(f"{body.density:.1f}")
                self._updating = False
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.density - self._bg_density:.1f}")
                self._updating = False

        elif col == COL_SUSCEPT:
            try:
                body.susceptibility = float(item.text())
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.susceptibility:.5f}")
                self._updating = False

        elif col == COL_REM_J:
            try:
                body.remanence_Am = float(item.text())
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.remanence_Am:.4f}")
                self._updating = False

        elif col == COL_REM_INC:
            try:
                body.remanence_inc_deg = float(item.text())
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.remanence_inc_deg:.1f}")
                self._updating = False

        elif col == COL_REM_DEC:
            try:
                body.remanence_dec_deg = float(item.text())
                self.body_changed.emit(body)
            except ValueError:
                self._updating = True
                item.setText(f"{body.remanence_dec_deg:.1f}")
                self._updating = False

    # --- row selection ------------------------------------------------------------------------------─

    def _on_selection_changed(self):
        if self._updating:
            return
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        self.btn_del_poly.setEnabled(bool(rows))
        if rows:
            row = min(rows)
            if 0 <= row < len(self._bodies):
                body = self._bodies[row]
                self._sel_body = body
                self._populate_vert_table(body)
                self.body_selected.emit(body)
                return
        self._sel_body = None
        self._populate_vert_table(None)
        self.body_selected.emit(None)

    def _confirm_delete_polygons(self):
        """Show a confirmation dialog and emit body_delete_requested if confirmed."""
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        bodies = [self._bodies[r] for r in rows if 0 <= r < len(self._bodies)]
        if not bodies:
            return
        names = ", ".join(f'"{b.name}"' for b in bodies)
        reply = QMessageBox.question(
            self, "Delete Polygon(s)",
            f"Delete selected polygon(s)?\n{names}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.body_delete_requested.emit(bodies)

    def select_body(self, body: Optional[PolygonBody]):
        self._updating = True
        self.table.clearSelection()
        if body is not None and body in self._bodies:
            row = self._bodies.index(body)
            self.table.selectRow(row)
        self._updating = False