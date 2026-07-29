"""
Interactive demo of numerical stability in 1-D transient heat conduction.

Where ``finitediff_demo.py`` relaxes a column to steady state with an
implicit unit node spacing, this demo solves the actual time-dependent
diffusion equation

.. math::

    \\frac{\\partial T}{\\partial z} = \\kappa \\frac{\\partial^2 T}{\\partial z^2}

with an explicit depth step ``dz`` and time step ``dt`` the student sets
directly, so the stability parameter

.. math::

    r = \\frac{\\kappa\\, dt}{dz^2}

is under their control. Three finite-difference formulations are
available -- explicit (FTCS), fully implicit (BTCS), and Crank-Nicolson
-- so the same instability that breaks the explicit scheme once
``r > 0.5`` can be directly compared against the unconditionally stable
implicit schemes at the same ``r``.

Reuses ``render_mathtext`` and ``MplCanvas`` from ``finitediff_demo.py``
rather than redefining them.
"""

import csv
import io
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
)

import matplotlib as mpl

from src.demos.finitediff.finitediff_demo import render_mathtext, MplCanvas
from src.common.gui import gui_io

SCHEMES = ["Explicit (FTCS)", "Implicit (BTCS)", "Crank-Nicolson"]
STABILITY_LIMIT = 0.5   # explicit FTCS: stable only for r <= 0.5
MAX_ITER_TOTAL = 200_000   # hard cap so a tiny dt / huge total-time can't hang the GUI

# Default control values, factored out so __init__ and "Reset to defaults"
# share one source of truth.
DEFAULTS = dict(
    n_nodes=21,
    dz=1.0,
    dt=0.4,
    kappa=1.0,
    total_time=16.0,
    n_display=40,
    scheme=SCHEMES[0],
    speed_ms=150,
    preset="spike",
)


# ----------------------------------------------------------------------
# Numerical core -- independent of Qt, so it can be tested/imported on
# its own.
# ----------------------------------------------------------------------

def initial_condition(n_nodes, kind="spike", amplitude=100.0):
    """
    Build a starting temperature profile designed to excite every
    wavelength the grid can represent, so an unstable scheme has
    something to amplify.

    Parameters
    ----------
    n_nodes : int
    kind : str
        ``"spike"`` -- a single hot node at the column's midpoint,
        zero elsewhere (the classic worst-case for FTCS instability).
        ``"step"`` -- a half-and-half discontinuity.
    amplitude : float

    Returns
    -------
    numpy.ndarray, shape (n_nodes,)
    """
    T = np.zeros(n_nodes)
    if kind == "step":
        T[: n_nodes // 2] = amplitude
    else:
        T[n_nodes // 2] = amplitude
    T[0] = 0.0
    T[-1] = 0.0
    return T


def step_explicit(T, r):
    """One FTCS (forward-time, centred-space) explicit step. Dirichlet
    boundary nodes are left untouched."""
    Tnew = T.copy()
    Tnew[1:-1] = T[1:-1] + r * (T[2:] - 2.0 * T[1:-1] + T[:-2])
    return Tnew


def _theta_step(T, r, theta):
    """
    Generalized theta-method step, solved as a tridiagonal system:

        theta=1   -> fully implicit (BTCS), unconditionally stable
        theta=0.5 -> Crank-Nicolson, unconditionally stable, 2nd order

    Boundary rows are held at their (fixed, Dirichlet) initial value by
    making that row of the system the identity.
    """
    n = len(T)
    ab = np.zeros((3, n))   # banded storage for solve_banded((1, 1), ...)
    b = np.empty(n)

    # interior rows: -r*theta*T[i-1] + (1+2*r*theta)*T[i] - r*theta*T[i+1]
    #              = r*(1-theta)*T_old[i-1] + (1-2*r*(1-theta))*T_old[i] + r*(1-theta)*T_old[i+1]
    ab[1, 1:-1] = 1.0 + 2.0 * r * theta
    ab[0, 2:] = -r * theta        # superdiagonal, ab[0, j] = a[j-1, j]
    ab[2, :-2] = -r * theta       # subdiagonal,   ab[2, j] = a[j+1, j]
    b[1:-1] = (
        r * (1.0 - theta) * T[:-2]
        + (1.0 - 2.0 * r * (1.0 - theta)) * T[1:-1]
        + r * (1.0 - theta) * T[2:]
    )

    # boundary rows: identity, holding the Dirichlet value fixed
    ab[1, 0] = 1.0
    ab[0, 1] = 0.0
    b[0] = T[0]

    ab[1, -1] = 1.0
    ab[2, -2] = 0.0
    b[-1] = T[-1]

    return solve_banded((1, 1), ab, b)


def step_implicit(T, r):
    """One fully implicit (BTCS) step."""
    return _theta_step(T, r, theta=1.0)


def step_crank_nicolson(T, r):
    """One Crank-Nicolson step."""
    return _theta_step(T, r, theta=0.5)


STEP_FUNCS = {
    "Explicit (FTCS)": step_explicit,
    "Implicit (BTCS)": step_implicit,
    "Crank-Nicolson": step_crank_nicolson,
}


def run_scheme_for_time(T0, r, scheme, n_iter_total, n_display):
    """
    Advance `T0` for `n_iter_total` physical time steps, but keep only
    `n_display` (+1, for the initial condition) evenly spaced snapshots.

    Changing `dt` changes how many physical iterations are needed to
    reach a given total time, but a student watching the animation
    doesn't need -- and, for a fine `dt`, can't practically be shown --
    one frame per physical step. Separating "how far to integrate" from
    "how many snapshots to keep" lets `n_iter_total` be whatever the
    requested total time demands while the table/animation stay a
    manageable, fixed size.

    Returns
    -------
    results : list of numpy.ndarray
        Sampled profiles, starting with T0 itself.
    step_numbers : list of int
        The physical iteration number each entry in `results`
        corresponds to (``step_numbers[0] == 0``).
    """
    step = STEP_FUNCS[scheme]
    n_display = max(1, min(n_display, n_iter_total))
    snapshot_indices = sorted(set(
        np.linspace(0, n_iter_total, n_display + 1).astype(int).tolist()
    ))

    results = []
    step_numbers = []
    T = T0.copy()

    pos = 0
    if snapshot_indices[pos] == 0:
        results.append(T.copy())
        step_numbers.append(0)
        pos += 1

    for i in range(1, n_iter_total + 1):
        T = step(T, r)
        if pos < len(snapshot_indices) and i == snapshot_indices[pos]:
            results.append(T.copy())
            step_numbers.append(i)
            pos += 1

    return results, step_numbers


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------

class StabilityGUI(QMainWindow):
    """
    Interactive GUI for exploring explicit/implicit/Crank-Nicolson
    stability in 1-D transient heat conduction.

    Attributes
    ----------
    canvas : MplCanvas
        Matplotlib canvas used to plot the depth profiles.
    animation_results : list of numpy.ndarray or None
        Profiles from the most recent run, used to drive the
        frame-by-frame animation.
    animation_index : int
        Index of the next animation frame to draw.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Numerical Stability: Explicit vs. Implicit vs. Crank-Nicolson")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # -------------------------------
        # Left: controls
        # -------------------------------
        left_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Nodes"))
        self.node_spin = QSpinBox()
        self.node_spin.setRange(5, 200)
        self.node_spin.setValue(DEFAULTS["n_nodes"])
        row1.addWidget(self.node_spin)

        row1.addWidget(QLabel("dz (m)"))
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.01, 1000.0)
        self.dz_spin.setDecimals(3)
        self.dz_spin.setSingleStep(0.1)
        self.dz_spin.setValue(DEFAULTS["dz"])
        row1.addWidget(self.dz_spin)

        row1.addWidget(QLabel("dt (s)"))
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(1e-4, 1e12)
        self.dt_spin.setDecimals(4)
        self.dt_spin.setSingleStep(0.1)
        self.dt_spin.setValue(DEFAULTS["dt"])
        row1.addWidget(self.dt_spin)

        left_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("κ (m²/s)"))
        self.kappa_spin = QDoubleSpinBox()
        self.kappa_spin.setRange(1e-6, 1e6)
        self.kappa_spin.setDecimals(6)
        self.kappa_spin.setSingleStep(0.1)
        self.kappa_spin.setValue(DEFAULTS["kappa"])
        row2.addWidget(self.kappa_spin)

        row2.addWidget(QLabel("Total time (s)"))
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(1e-4, 1e12)
        self.time_spin.setDecimals(4)
        self.time_spin.setSingleStep(1.0)
        self.time_spin.setValue(DEFAULTS["total_time"])
        self.time_spin.setToolTip(
            "Total physical time to integrate. Combined with dt this sets "
            "how many iterations actually run -- see 'Iterations needed' "
            "below."
        )
        row2.addWidget(self.time_spin)

        row2.addWidget(QLabel("Snapshots"))
        self.display_spin = QSpinBox()
        self.display_spin.setRange(1, 500)
        self.display_spin.setValue(DEFAULTS["n_display"])
        self.display_spin.setToolTip(
            "Number of evenly spaced profiles to keep and show (table "
            "columns / animation frames) -- independent of how many "
            "physical iterations are actually needed to reach the "
            "requested total time."
        )
        row2.addWidget(self.display_spin)

        left_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Scheme"))
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems(SCHEMES)
        row3.addWidget(self.scheme_combo)

        row3.addWidget(QLabel("Speed (ms)"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(10, 5000)
        self.speed_spin.setValue(DEFAULTS["speed_ms"])
        row3.addWidget(self.speed_spin)

        left_layout.addLayout(row3)

        # stability readout: r = kappa*dt/dz^2, with a visible warning
        # for the explicit scheme once r exceeds the 0.5 limit.
        r_caption = QLabel("Stability parameter:")
        left_layout.addWidget(r_caption)

        self.r_equation_label = QLabel()
        self.r_equation_label.setPixmap(
            render_mathtext(r"$r = \kappa\, dt / dz^2$")
        )
        left_layout.addWidget(self.r_equation_label)

        self.r_value_label = QLabel()
        left_layout.addWidget(self.r_value_label)

        self.iter_needed_label = QLabel()
        left_layout.addWidget(self.iter_needed_label)

        # -------------------------------
        # Initial-condition table: rows are depth nodes, column 0 is the
        # user-editable initial temperature, later columns are the
        # computed (read-only) snapshots from the most recent run.
        # -------------------------------
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Spike", "spike")
        self.preset_combo.addItem("Step", "step")
        preset_layout.addWidget(self.preset_combo)

        self.apply_preset_button = QPushButton("Apply to initial column")
        preset_layout.addWidget(self.apply_preset_button)
        preset_layout.addStretch()
        left_layout.addLayout(preset_layout)

        self.table = QTableWidget()
        left_layout.addWidget(self.table)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.clear_button = QPushButton("Clear")
        self.reset_button = QPushButton("Reset to defaults")
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.reset_button)
        left_layout.addLayout(button_layout)

        export_layout = QHBoxLayout()
        self.copy_csv_button = QPushButton("Copy table as CSV")
        self.export_csv_button = QPushButton("Export table (.csv)")
        self.export_pdf_button = QPushButton("Export figure (.pdf)")
        export_layout.addWidget(self.copy_csv_button)
        export_layout.addWidget(self.export_csv_button)
        export_layout.addWidget(self.export_pdf_button)
        left_layout.addLayout(export_layout)

        # -------------------------------
        # Right: plot
        # -------------------------------
        self.canvas = MplCanvas()

        main_layout.addLayout(left_layout, 2)
        main_layout.addWidget(self.canvas, 3)

        # -------------------------------
        # Signals
        # -------------------------------
        for w in (self.dz_spin, self.dt_spin, self.kappa_spin):
            w.valueChanged.connect(self.update_r_label)
        self.scheme_combo.currentIndexChanged.connect(self.update_r_label)
        for w in (self.dt_spin, self.time_spin):
            w.valueChanged.connect(self.update_iter_needed_label)

        self.node_spin.valueChanged.connect(self.build_table)
        self.dz_spin.valueChanged.connect(self.update_row_labels)
        self.apply_preset_button.clicked.connect(self.apply_preset)

        self.run_button.clicked.connect(self.run_model)
        self.clear_button.clicked.connect(self.clear_results)
        self.reset_button.clicked.connect(self.reset_to_defaults)

        self.copy_csv_button.clicked.connect(self.copy_table_csv)
        self.export_csv_button.clicked.connect(self.export_table_csv)
        self.export_pdf_button.clicked.connect(self.export_figure_pdf)

        self.timer = QTimer()
        self.timer.timeout.connect(self.animation_step)
        self.animation_results = None
        self.animation_step_numbers = None
        self.animation_index = 0

        self.build_table()
        self.update_r_label()
        self.update_iter_needed_label()

    # --------------------------------------------------

    def current_r(self):
        dz = self.dz_spin.value()
        dt = self.dt_spin.value()
        kappa = self.kappa_spin.value()
        return kappa * dt / dz**2

    def update_r_label(self):
        r = self.current_r()
        scheme = self.scheme_combo.currentText()

        text = f"r = {r:.3f}"
        if scheme == "Explicit (FTCS)":
            if r > STABILITY_LIMIT:
                text += f"  ⚠ unstable (limit r ≤ {STABILITY_LIMIT})"
                self.r_value_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                text += f"  ✓ stable (limit r ≤ {STABILITY_LIMIT})"
                self.r_value_label.setStyleSheet("color: green;")
        else:
            text += "  ✓ unconditionally stable"
            self.r_value_label.setStyleSheet("color: green;")

        self.r_value_label.setText(text)

    def iterations_needed(self):
        """Physical iterations required to reach the requested total
        time at the current dt, clamped to `MAX_ITER_TOTAL` so a tiny
        dt / large total-time combination can't hang the GUI."""
        dt = self.dt_spin.value()
        total_time = self.time_spin.value()
        n_iter = max(1, int(np.ceil(total_time / dt)))
        return min(n_iter, MAX_ITER_TOTAL), n_iter

    def update_iter_needed_label(self):
        n_iter, n_iter_unclamped = self.iterations_needed()
        if n_iter_unclamped > MAX_ITER_TOTAL:
            self.iter_needed_label.setText(
                f"Iterations needed: {n_iter_unclamped:,} "
                f"-- clamped to {MAX_ITER_TOTAL:,} (reached time "
                f"{n_iter * self.dt_spin.value():.3g} s instead)"
            )
            self.iter_needed_label.setStyleSheet("color: darkorange;")
        else:
            self.iter_needed_label.setText(f"Iterations needed: {n_iter:,}")
            self.iter_needed_label.setStyleSheet("")

    # --------------------------------------------------

    def value(self, row, col, default=0.0):
        """Read a single table cell as a float, falling back to
        `default` if it's empty or not a valid number."""
        item = self.table.item(row, col)
        if item is None:
            return default
        try:
            return float(item.text())
        except ValueError:
            return default

    def build_table(self):
        """(Re)size the table to the current node count. Existing
        column-0 (initial condition) entries are preserved; only
        missing cells are filled with a default of 0."""
        n_nodes = self.node_spin.value()

        self.table.blockSignals(True)
        self.table.setRowCount(n_nodes)
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["T (initial)"])

        for row in range(n_nodes):
            if not self.table.item(row, 0):
                self.table.setItem(row, 0, QTableWidgetItem("0"))

        self.update_row_labels()
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.blockSignals(False)

    def update_row_labels(self):
        """Label each row with its physical depth (row index * dz)."""
        dz = self.dz_spin.value()
        n_nodes = self.table.rowCount()
        self.table.setVerticalHeaderLabels(
            [f"{i * dz:.2f}" for i in range(n_nodes)]
        )

    def apply_preset(self):
        """Fill the editable initial-condition column from the selected
        preset, overwriting any hand-entered values."""
        n_nodes = self.table.rowCount()
        kind = self.preset_combo.currentData()
        T0 = initial_condition(n_nodes, kind=kind)

        self.table.blockSignals(True)
        for row in range(n_nodes):
            self.table.setItem(row, 0, QTableWidgetItem(f"{T0[row]:g}"))
        self.table.blockSignals(False)

    def reset_to_defaults(self):
        """Restore every control and the initial-condition table to the
        values set at startup, discarding hand edits and computed
        results."""
        self.node_spin.setValue(DEFAULTS["n_nodes"])
        self.dz_spin.setValue(DEFAULTS["dz"])
        self.dt_spin.setValue(DEFAULTS["dt"])
        self.kappa_spin.setValue(DEFAULTS["kappa"])
        self.time_spin.setValue(DEFAULTS["total_time"])
        self.display_spin.setValue(DEFAULTS["n_display"])
        self.speed_spin.setValue(DEFAULTS["speed_ms"])
        self.scheme_combo.setCurrentText(DEFAULTS["scheme"])

        preset_index = self.preset_combo.findData(DEFAULTS["preset"])
        if preset_index >= 0:
            self.preset_combo.setCurrentIndex(preset_index)

        # rebuild from scratch so every row goes back to "0", not just
        # rows that happen to be missing an item
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.blockSignals(False)
        self.build_table()

        self.clear_results()
        self.update_r_label()
        self.update_iter_needed_label()

    # --------------------------------------------------

    def _row_depths(self):
        """Depth (row index * dz) for every current table row, used as
        the leading column when copying/exporting the table."""
        dz = self.dz_spin.value()
        return [row * dz for row in range(self.table.rowCount())]

    def copy_table_csv(self):
        gui_io.copy_table_csv(
            self.table, row_labels=self._row_depths(), row_label_header="Depth (m)"
        )

    def export_table_csv(self):
        gui_io.export_table_csv(
            self.table, parent=self, default_name="stability_table.csv",
            row_labels=self._row_depths(), row_label_header="Depth (m)",
        )

    def export_figure_pdf(self):
        gui_io.export_figure_pdf(
            self.canvas.fig, parent=self, default_name="stability_figure.pdf"
        )

    # --------------------------------------------------

    def clear_results(self):
        self.timer.stop()
        self.canvas.ax.clear()
        self.canvas.draw()
        self.animation_results = None
        self.animation_step_numbers = None

        # drop the computed (non-editable) columns, keep column 0
        self.table.blockSignals(True)
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["T (initial)"])
        self.table.blockSignals(False)

    def run_model(self):
        self.timer.stop()

        n_nodes = self.table.rowCount()
        dt = self.dt_spin.value()
        n_iter_total, _ = self.iterations_needed()
        n_display = self.display_spin.value()
        r = self.current_r()
        scheme = self.scheme_combo.currentText()

        self.update_iter_needed_label()

        T0 = np.array([self.value(row, 0, 0.0) for row in range(n_nodes)])
        results, step_numbers = run_scheme_for_time(
            T0, r, scheme, n_iter_total, n_display
        )

        # write the computed (read-only) snapshots into the table,
        # leaving column 0 (the initial condition) untouched
        self.table.blockSignals(True)
        self.table.setColumnCount(len(results))
        headers = ["T (initial)"]
        for s in step_numbers[1:]:
            headers.append(f"n={s}\nt={s * dt:.3g}s")
        self.table.setHorizontalHeaderLabels(headers)

        for col, profile in enumerate(results[1:], start=1):
            for row in range(n_nodes):
                item = QTableWidgetItem(f"{profile[row]:.3f}")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)

        self.animation_results = results
        self.animation_step_numbers = step_numbers
        self.animation_index = 0
        self.timer.start(self.speed_spin.value())

    def plot_results(self, results, step_numbers):
        ax = self.canvas.ax
        ax.clear()

        dt = self.dt_spin.value()
        depth = np.arange(len(results[0])) * self.dz_spin.value()

        ax.plot(results[0], depth, "--", linewidth=2, color="black",
                marker="o", markersize=4, label="Initial")

        cmap = mpl.colormaps["viridis"]
        ncurves = len(results) - 1
        for i in range(1, len(results)):
            c = cmap(i / max(ncurves, 1))
            ax.plot(results[i], depth, color=c, linewidth=1.5, alpha=0.9,
                     marker="o", markersize=4)

        last_n = step_numbers[-1]
        ax.plot(results[-1], depth, color=cmap(1.0), linewidth=2.5,
                marker="o", markersize=4,
                label=f"n={last_n} (t={last_n * dt:.3g}s)")

        r = self.current_r()
        scheme = self.scheme_combo.currentText()
        ax.set_title(f"{scheme}  (r = {r:.3f})")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Depth")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def animation_step(self):
        results = self.animation_results
        step_numbers = self.animation_step_numbers
        if results is None:
            return

        n_steps = len(results) - 1
        if self.animation_index >= n_steps:
            self.timer.stop()
            return

        end = self.animation_index + 2
        self.plot_results(results[:end], step_numbers[:end])
        self.animation_index += 1


def main():
    app = QApplication(sys.argv)
    window = StabilityGUI()
    window.resize(1200, 650)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
