import sys
from pathlib import Path
import numpy as np

_HERE   = Path(__file__).resolve().parent          # src/gui/
_COURSE = _HERE.parent.parent                       # project root
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QHBoxLayout,
    QTableWidgetItem
)

from src.geothermics.ss_thermal import compute_temperature, compute_elevation
from src.gui.ssiso_control_dock import ControlDock
from src.utils.mpl_widget import MplWidget


# ============================================================
# Main Window
# ============================================================
class TISApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Thermal Isostasy Calculator")
        self.resize(1400, 700)

        # ---------------- State ----------------
        self.depth = np.arange(0, 251, 1)
        self.gtherm = []
        self.refcurve = None
        self.isoflag = False
        self.ghandle = []
        self.ehandle = []

        # ---------------- Central plots ----------------
        central = QWidget()
        clayout = QHBoxLayout(central)

        self.geo_plot = MplWidget(
            "Geotherms", "Temperature (°C)", "Depth (km)",
            xlim=(0, 1400), ylim=(0, 250), invert_y=True
        )
        self.elev_plot = MplWidget(
            "Thermal Isostasy", "Heat Flow (mW m$^{-2}$)", "Elevation (km)",
            xlim=(20, 120), ylim=(-1, 3.5)
        )

        clayout.addWidget(self.geo_plot)
        clayout.addWidget(self.elev_plot)

        self.setCentralWidget(central)

        # ---------------- Dock ----------------
        self.controls = ControlDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.controls)

        self._startup()
        self._wire_signals()

    # ======================================================
    # Startup initialization
    # ======================================================
    def _startup(self):
        defaults = [
            (False, 0, 3.0, 1.5),
            (False, 16, 2.5, 0.45),
            (False, 39, 2.5, 0.02),
        ]

        for r, row in enumerate(defaults):
            for c, val in enumerate(row):
                self.controls.layer_table.setItem(
                    r, c, QTableWidgetItem(str(val))
                )

    # ======================================================
    # Utilities
    # ======================================================
    def read_layer_table(self):
        t = self.controls.layer_table
        ztop, k, A, layer_flag = [], [], [], []

        for r in range(t.rowCount()):
            layer_flag.append(t.item(r, 0).text().lower() == "true")
            ztop.append(float(t.item(r, 1).text()))
            k.append(float(t.item(r, 2).text()))
            A.append(float(t.item(r, 3).text()))

        return np.array(ztop), np.array(k), np.array(A), np.array(layer_flag)

    # ======================================================
    # Callbacks
    # ======================================================
    def compute_geotherm(self):
        c = self.controls
        ztop, k, A, _ = self.read_layer_table()

        T0 = 20.0
        qs = c.ref_hf.value()

        T = compute_temperature(self.depth, T0, qs, ztop, k, A)

        entry = {"temperature": T, "qs": qs}
        self.gtherm.append(entry)

        line, = self.geo_plot.ax.plot(T, self.depth, "b-")
        self.ghandle.append(line)
        self.geo_plot.canvas.draw_idle()

        c.spinner.setMaximum(len(self.gtherm))
        c.spinner.setEnabled(True)
        c.remove_geo_btn.setEnabled(True)

        if not self.isoflag:
            return

        alpha = c.expansivity.value() * 1e-5
        dz = self.depth[1] - self.depth[0]
        elev = compute_elevation(dz, T, self.refcurve["temperature"], alpha)

        entry["elevation"] = elev
        e_line, = self.elev_plot.ax.plot(qs, elev, "ob")
        self.ehandle.append(e_line)
        self.elev_plot.canvas.draw_idle()

    def compute_isostatic(self):
        c = self.controls
        ztop, k, A, _ = self.read_layer_table()

        T0 = 20.0
        qs = c.ref_hf.value()
        P = c.partition.value()

        A[0] = (1 - P) * qs / ztop[1]
        Tref = compute_temperature(self.depth, T0, qs, ztop, k, A)

        self.refcurve = {"temperature": Tref, "qs": qs}
        self.isoflag = True

        self.geo_plot.ax.plot(Tref, self.depth, "k--")
        self.geo_plot.canvas.draw_idle()

        self.elev_plot.ax.axhline(0, color="k", linestyle=":")
        self.elev_plot.ax.axvline(qs, color="k", linestyle=":")

        q_min, q_max = c.hf_range.values()
        q = np.arange(q_min, q_max + 1)
        dz = self.depth[1] - self.depth[0]
        alpha = c.expansivity.value() * 1e-5

        de = []
        for qi in q:
            A[0] = (1 - P) * qi / ztop[1]
            T = compute_temperature(self.depth, T0, qi, ztop, k, A)
            de.append(compute_elevation(dz, T, Tref, alpha))

        self.elev_plot.ax.plot(q, de, "r-")
        self.elev_plot.canvas.draw_idle()

    def spinner_changed(self, val):
        if val == 0:
            return
        idx = val - 1
        for i, ln in enumerate(self.ghandle):
            ln.set_color("r" if i == idx else "b")
        for i, ln in enumerate(self.ehandle):
            ln.set_color("r" if i == idx else "b")
        self.geo_plot.canvas.draw_idle()
        self.elev_plot.canvas.draw_idle()

    def remove_geotherm(self):
        c = self.controls
        idx = c.spinner.value() - 1
        if idx < 0:
            return

        self.ghandle[idx].remove()
        del self.ghandle[idx]
        del self.gtherm[idx]

        if self.isoflag:
            self.ehandle[idx].remove()
            del self.ehandle[idx]

        c.spinner.setMaximum(len(self.gtherm))
        if not self.gtherm:
            c.spinner.setEnabled(False)
            c.remove_geo_btn.setEnabled(False)
            c.spinner.setValue(0)

        self.geo_plot.canvas.draw_idle()
        self.elev_plot.canvas.draw_idle()

    def add_layer(self):
        t = self.controls.layer_table
        t.insertRow(t.rowCount())

    def remove_layer(self):
        t = self.controls.layer_table
        rows = sorted({i.row() for i in t.selectedIndexes()}, reverse=True)
        for r in rows:
            t.removeRow(r)

    def reset_axes(self):
        self.geo_plot.ax.cla()
        self.elev_plot.ax.cla()

        self.geo_plot.ax.set_title("Geotherms")
        self.geo_plot.ax.set_xlabel("Temperature (°C)")
        self.geo_plot.ax.set_ylabel("Depth (km)")
        self.geo_plot.ax.invert_yaxis()

        self.elev_plot.ax.set_title("Thermal Isostasy")
        self.elev_plot.ax.set_xlabel("Heat Flow (mW m$^{-2}$)")
        self.elev_plot.ax.set_ylabel("Elevation (km)")

        self.gtherm.clear()
        self.ghandle.clear()
        self.ehandle.clear()
        self.isoflag = False

        c = self.controls
        c.spinner.setEnabled(False)
        c.remove_geo_btn.setEnabled(False)
        c.spinner.setValue(0)

        self.geo_plot.canvas.draw_idle()
        self.elev_plot.canvas.draw_idle()

    # ======================================================
    # Signal wiring
    # ======================================================
    def _wire_signals(self):
        c = self.controls
        c.compute_btn.clicked.connect(self.compute_geotherm)
        c.isostatic_btn.clicked.connect(self.compute_isostatic)
        c.reset_btn.clicked.connect(self.reset_axes)
        c.add_layer_btn.clicked.connect(self.add_layer)
        c.remove_layer_btn.clicked.connect(self.remove_layer)
        c.spinner.valueChanged.connect(self.spinner_changed)
        c.remove_geo_btn.clicked.connect(self.remove_geotherm)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    win = TISApp()
    win.show()
    sys.exit(app.exec())