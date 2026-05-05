from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure

class MplWidget(QWidget):
    def __init__(self, title, xlabel, ylabel, xlim=None, ylim=None, invert_y=False):
        super().__init__()

        self._title    = title
        self._xlabel   = xlabel
        self._ylabel   = ylabel
        self._xlim     = xlim
        self._ylim     = ylim
        self._invert_y = invert_y

        layout = QVBoxLayout(self)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self._apply_axes_defaults()

    def _apply_axes_defaults(self):
        self.ax.set_title(self._title)
        self.ax.set_xlabel(self._xlabel)
        self.ax.set_ylabel(self._ylabel)
        if self._xlim:
            self.ax.set_xlim(*self._xlim)
        if self._ylim:
            self.ax.set_ylim(*self._ylim)
        if self._invert_y:
            self.ax.invert_yaxis()
        self.ax.grid(True)

    def reset(self):
        self.ax.cla()
        self._apply_axes_defaults()
        self.canvas.draw_idle()