"""
Interactive demo of an explicit 1-D finite-difference heat conduction solver.

Students set up a column of nodes -- each with an internal heat
source/sink ``A``, a thermal conductivity ``k``, and an initial value
``f(T)`` -- and step a node-by-node finite-difference update forward one
iteration at a time. Results are written into a table (one column per
iteration) and plotted with depth increasing downward, as is
conventional in geophysics.
"""

import sys
import numpy as np

from PyQt6.QtCore import Qt, QTimer

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from PyQt6.QtGui import QImage, QPixmap

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgba as mpl_to_rgba


def render_mathtext(mathtext, fontsize=14, dpi=150, color="black"):
    """
    Rasterize a Matplotlib mathtext string into a ``QPixmap``.

    This lets a formula be shown in an ordinary ``QLabel`` (via
    ``QLabel.setPixmap``), fully typeset, without embedding another
    interactive Matplotlib canvas or requiring a LaTeX installation.

    Parameters
    ----------
    mathtext : str
        A Matplotlib mathtext string wrapped in ``$...$``, e.g.
        ``r"$x^2$"``.
    fontsize : float, optional
        Font size in points used to render the glyphs.
    dpi : float, optional
        Resolution used when rasterizing the text.
    color : str, optional
        Matplotlib color name used for the glyphs.

    Returns
    -------
    PyQt6.QtGui.QPixmap
        A pixmap containing the rendered equation with a transparent
        background.
    """
    parser = MathTextParser("agg")
    parsed = parser.parse(mathtext, dpi=dpi, prop=FontProperties(size=fontsize))

    # `image` is a single-channel ink mask (0 = background, 0xff = fully
    # inked); build an RGBA image by using it as the alpha channel of a
    # solid-color fill.
    mask = np.asarray(parsed.image, dtype=np.uint8)
    height, width = mask.shape

    r, g, b, _ = mpl_to_rgba(color)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = int(r * 255)
    rgba[..., 1] = int(g * 255)
    rgba[..., 2] = int(b * 255)
    rgba[..., 3] = mask

    image = QImage(
        rgba.tobytes(),
        width,
        height,
        QImage.Format.Format_RGBA8888,
    )

    # QImage wraps the bytes object above without copying it, so make a
    # deep copy before the temporary buffer goes out of scope.
    return QPixmap.fromImage(image.copy())


class MplCanvas(FigureCanvasQTAgg):
    """
    A minimal Matplotlib canvas embedded in a Qt widget.

    Wraps a single :class:`~matplotlib.figure.Figure` with one axes so
    it can be dropped into a PyQt6 layout like any other widget.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure that owns the canvas.
    ax : matplotlib.axes.Axes
        The single axes used for all plotting in the demo.
    """

    def __init__(self):
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class HeatEquationGUI(QMainWindow):
    """
    Interactive GUI for exploring a 1-D explicit finite-difference solver.

    The window lets a student set up a column of nodes and step a
    finite-difference update forward node by node and iteration by
    iteration, either all at once or animated frame by frame.

    Attributes
    ----------
    table : PyQt6.QtWidgets.QTableWidget
        Per-node inputs (``A``, ``k``, ``f(T)``) in the first three
        columns, followed by one results column per iteration.
    canvas : MplCanvas
        Matplotlib canvas used to plot the node profiles.
    animation_results : list of numpy.ndarray or None
        Node values at each iteration from the most recent run, used to
        drive the step-by-step animation.
    animation_index : int
        Index of the next animation frame to draw.
    animation_A : numpy.ndarray or None
        Source/sink term from the most recent run, kept around so the
        animation can keep annotating sources and sinks each frame.
    """

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Student Node Solver")

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # -------------------------------
        # Left side
        # -------------------------------

        left_layout = QVBoxLayout()

        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Nodes"))

        self.node_spin = QSpinBox()
        self.node_spin.setRange(3, 100)
        self.node_spin.setValue(7)
        controls_layout.addWidget(self.node_spin)

        controls_layout.addWidget(QLabel("Iterations"))

        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 200)
        self.iter_spin.setValue(10)
        controls_layout.addWidget(self.iter_spin)

        controls_layout.addStretch()

        left_layout.addLayout(controls_layout)

        self.uniform_k_checkbox = QCheckBox("Uniform k")
        self.uniform_k_checkbox.setChecked(True)

        controls_layout.addWidget(self.uniform_k_checkbox)

        self.table = QTableWidget()
        left_layout.addWidget(self.table)

        # -------------------------------
        # Finite-difference equation display
        # -------------------------------

        equation_caption = QLabel("Update equation for interior node i:")
        equation_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(equation_caption)

        self.equation_label = QLabel()
        self.equation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.equation_label.setPixmap(
            render_mathtext(
                r"$T_i^{\,n+1} = \frac{k_{i-1} T_{i-1}^{\,n} + "
                r"k_{i+1} T_{i+1}^{\,n}}{k_{i-1} + k_{i+1}} "
                r"+ \frac{A_i}{k_i}$"
            )
        )
        left_layout.addWidget(self.equation_label)

        self.animate_checkbox = QCheckBox("Animate")
        self.animate_checkbox.setChecked(True)
        controls_layout.addWidget(self.animate_checkbox)

        controls_layout.addWidget(QLabel("ms"))

        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(50, 5000)
        self.speed_spin.setValue(500)
        controls_layout.addWidget(self.speed_spin)

        button_layout = QHBoxLayout()

        self.run_button = QPushButton("Run")
        self.clear_button = QPushButton("Clear")

        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)

        left_layout.addLayout(button_layout)

        # -------------------------------
        # Right side
        # -------------------------------

        self.canvas = MplCanvas()

        main_layout.addLayout(left_layout, 3)
        main_layout.addWidget(self.canvas, 2)

        self.node_spin.valueChanged.connect(self.build_table)
        self.iter_spin.valueChanged.connect(self.build_table)
        self.uniform_k_checkbox.toggled.connect(self.update_conductivity_mode)

        self.run_button.clicked.connect(self.run_model)
        self.clear_button.clicked.connect(self.clear_results)

        self.build_table()

        self.table.itemChanged.connect(self.sync_uniform_k)


        # -------------------------------
        # Animation timer
        # -------------------------------
        self.timer = QTimer()

        self.timer.timeout.connect(self.animation_step)

        self.animation_results = None
        self.animation_index = 0
        self.animation_A = None

    # --------------------------------------------------

    def build_table(self):
        """
        (Re)size the table to match the current node/iteration counts.

        Called whenever the node or iteration spin boxes change. Existing
        per-node inputs (columns 0-2) are preserved; only missing cells
        are filled with defaults. The first and last rows' ``f(T)``
        values are always reset to the default boundary conditions
        (0 and 100) since a changed node count shifts which row is last.
        """

        n_nodes = self.node_spin.value()
        n_steps = self.iter_spin.value()

        cols = n_steps + 3

        # Rebuilding the table fires itemChanged for every cell we
        # touch; block signals so sync_uniform_k doesn't run mid-rebuild.
        self.table.blockSignals(True)

        self.table.setRowCount(n_nodes)
        self.table.setColumnCount(cols)

        headers = ["A", "k", "f(T)"]

        for i in range(n_steps):
            headers.append(f"Step {i+1}")

        self.table.setHorizontalHeaderLabels(headers)

        for row in range(n_nodes):

            if not self.table.item(row, 0):
                self.table.setItem(row, 0, QTableWidgetItem("0"))

            if not self.table.item(row, 1):
                self.table.setItem(row, 1, QTableWidgetItem("1"))

            if not self.table.item(row, 2):
                self.table.setItem(row, 2, QTableWidgetItem("100"))

        # default boundary conditions

        self.table.item(0, 2).setText("0")
        self.table.item(n_nodes - 1, 2).setText("100")

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.update_conductivity_mode()

        self.table.blockSignals(False)

    # --------------------------------------------------

    def value(self, row, col, default=0.0):
        """
        Read a single table cell as a float, falling back to a default.

        Parameters
        ----------
        row : int
            Table row index.
        col : int
            Table column index.
        default : float, optional
            Value returned if the cell is empty or not a valid float.

        Returns
        -------
        float
            The parsed cell value, or `default`.
        """

        item = self.table.item(row, col)

        if item is None:
            return default

        try:
            return float(item.text())
        except Exception:
            return default

    def update_conductivity_mode(self):
        """
        Apply the "Uniform k" setting to the conductivity column.

        When uniform conductivity is enabled, every row's ``k`` value is
        overwritten with row 0's value and locked read-only. When
        disabled, every row becomes independently editable. Called on
        table rebuild and whenever the checkbox is toggled.
        """
        n_nodes = self.table.rowCount()
        uniform = self.uniform_k_checkbox.isChecked()

        for row in range(1, n_nodes):
            item = self.table.item(row, 1)
            if item is None:
                continue
            if uniform:
                item.setText(self.table.item(0, 1).text())
                item.setFlags(
                    item.flags()
                    & ~Qt.ItemFlag.ItemIsEditable
                )
            else:
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsEditable
                )

    def sync_uniform_k(self, item):
        """
        Propagate an edit of row 0's conductivity to every other row.

        Connected to ``table.itemChanged``. Only acts when uniform
        conductivity is enabled and the edited cell is row 0's ``k``
        column; any other edit is ignored.

        Parameters
        ----------
        item : PyQt6.QtWidgets.QTableWidgetItem
            The table item that changed.
        """

        if not self.uniform_k_checkbox.isChecked():
            return
        if item.column() != 1:
            return
        if item.row() != 0:
            return

        text = item.text()

        # Writing to other rows below would itself fire itemChanged;
        # block signals so this doesn't recurse.
        self.table.blockSignals(True)

        for row in range(1, self.table.rowCount()):
            other = self.table.item(row, 1)
            if other is not None:
                other.setText(text)

        self.table.blockSignals(False)

    # --------------------------------------------------

    def clear_results(self):
        """Erase every iteration column's values and clear the plot."""

        n_rows = self.table.rowCount()

        for row in range(n_rows):
            for col in range(3, self.table.columnCount()):
                self.table.setItem(row, col, QTableWidgetItem(""))

        self.canvas.ax.clear()
        self.canvas.draw()

    # --------------------------------------------------

    def run_model(self):
        """
        Run the finite-difference solver and display the results.

        Reads the per-node ``A``, ``k``, and initial ``f(T)`` values
        from the table, iterates the update below for the configured
        number of steps, and either starts the frame-by-frame animation
        or writes every column and plots the full result immediately,
        depending on the "Animate" checkbox.

        Notes
        -----
        Interior nodes are updated with

        .. math::

            T_i^{n+1} = \\frac{k_{i-1} T_{i-1}^{n} + k_{i+1} T_{i+1}^{n}}
            {k_{i-1} + k_{i+1}} + \\frac{A_i}{k_i}

        (the same formula shown beneath the table), which relaxes the
        column toward the steady-state solution of the 1-D conduction
        equation :math:`d(k\\, dT/dz)/dz = -A`. Boundary nodes (index 0
        and ``n_nodes - 1``) are held fixed at their ``f(T)`` values.
        """
        self.timer.stop()
        self.clear_results()

        n_nodes = self.node_spin.value()
        n_steps = self.iter_spin.value()

        A = np.zeros(n_nodes)
        k = np.ones(n_nodes)
        T0 = np.zeros(n_nodes)

        for i in range(n_nodes):
            A[i] = self.value(i, 0, 0)
            T0[i] = self.value(i, 2, 0)

        if self.uniform_k_checkbox.isChecked():

            k_value = self.value(0, 1, 1)
            k[:] = k_value

        else:

            for i in range(n_nodes):
                k[i] = self.value(i, 1, 1)

        results = [T0.copy()]

        T = T0.copy()

        for step in range(n_steps):

            Tnew = T.copy()

            # Boundary nodes (0 and n_nodes - 1) are never touched here,
            # so they stay fixed at their initial f(T) values.
            for i in range(1, n_nodes - 1):

                left_k = k[i - 1]
                right_k = k[i + 1]

                Tnew[i] = (
                    (left_k * T[i - 1] + right_k * T[i + 1])
                    / (left_k + right_k)
                    + A[i] / max(k[i], 1e-12)
                )

            T = Tnew
            results.append(T.copy())

        self.animation_results = results
        self.animation_A = A

        if self.animate_checkbox.isChecked():

            self.animation_index = 0
            self.timer.start(self.speed_spin.value())

        else:

            for step in range(n_steps):

                vals = results[step + 1]

                for row in range(n_nodes):

                    self.table.setItem(
                        row,
                        step + 3,
                        QTableWidgetItem(f"{vals[row]:.2f}")
                    )

            self.plot_results(results, A)

    # --------------------------------------------------

    def plot_results(self, results, A):
        """
        Plot every iteration's node profile, depth increasing downward.

        Parameters
        ----------
        results : list of numpy.ndarray
            Node values at each iteration, starting with the initial
            condition (index 0).
        A : numpy.ndarray
            Per-node source/sink term, used to annotate nonzero entries
            as "Source" or "Sink" on the final profile.
        """

        ax = self.canvas.ax
        ax.clear()

        depth = np.arange(len(results[0]))

        ax.plot(
            results[0],
            depth,
            "--",
            linewidth=2,
            color="black",
            label="Initial"
        )

        cmap = mpl.colormaps["viridis"]

        ncurves = len(results) - 1

        for i in range(1, len(results)):

            c = cmap(i / max(ncurves, 1))

            ax.plot(
                results[i],
                depth,
                color=c,
                linewidth=2,
                alpha=0.9,
                label=f"Step {i}"
            )

        latest = results[-1]

        ax.scatter(
            latest,
            depth,
            s=60,
            edgecolor="black",
            zorder=10
        )

        ax.scatter(
            [latest[0], latest[-1]],
            [0, len(depth)-1],
            s=120,
            marker="s",
            color="red",
            label="Boundary"
        )

        for i, val in enumerate(A):

            if abs(val) > 1e-12:

                label = "Source" if val > 0 else "Sink"

                ax.annotate(
                    f"{label} ({val:g})",
                    (latest[i], i),
                    xytext=(10, 10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->")
                )

        ax.set_xlabel("Temperature")
        ax.set_ylabel("Node")
        ax.set_title("Iteration Progression")

        # Depth convention: node 0 (surface) at the top, increasing downward.
        ax.invert_yaxis()

        ax.grid(True, alpha=0.3)

        ax.legend(
            fontsize=8,
            loc="lower left",
            ncol=2
        )

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def animation_step(self):
        """
        Advance the animation by one iteration.

        Connected to the ``QTimer`` timeout signal. Writes the current
        step's values into the table, re-plots every iteration up to
        and including this one, and stops the timer once the last
        stored iteration has been shown.
        """

        results = self.animation_results

        if results is None:
            return

        n_steps = len(results) - 1

        if self.animation_index >= n_steps:

            self.timer.stop()
            return

        step = self.animation_index

        vals = results[step + 1]

        for row in range(len(vals)):

            self.table.setItem(
                row,
                step + 3,
                QTableWidgetItem(f"{vals[row]:.2f}")
            )

        partial = results[:step + 2]

        self.plot_results(
            partial,
            self.animation_A
        )

        self.animation_index += 1


def main():
    """Launch the ``HeatEquationGUI`` application."""

    app = QApplication(sys.argv)

    window = HeatEquationGUI()
    window.resize(1400, 700)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
