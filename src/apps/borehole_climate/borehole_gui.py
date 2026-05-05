"""
borehole_gui.py

PyQt6 GUI for borehole palaeoclimate analysis.

Layout
------
Central area   : QTabWidget with four Matplotlib canvases
Right dock     : parameter controls and action buttons
Bottom dock    : table of loaded boreholes with computed results

Usage
-----
    python borehole_gui.py
"""

import sys
import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDockWidget,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QDoubleSpinBox,
    QComboBox, QFileDialog, QMessageBox,
    QAbstractItemView, QStatusBar,
)

# Common project widgets
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.common.gui.mpl_widget import MplWidget
from src.common.gui.CustomWidgets import ToggleSwitch, RangeWidget, CustomPage

# Physics module
from src.physics.geothermics.borehole_climate import (
    load_boreholes, process_borehole, forward_step,
    invert_newton, invert_gridsearch, air_temp_to_reduced,
)

# Default data directory (relative to repo root)
_DEFAULT_DIR = str(
    Path(__file__).resolve().parents[3]
    / 'data' / 'geothermics' / 'borehole_temperatures'
)

_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

TABLE_COLS = [
    'Name', 'Max Depth (m)', 'Fit Depth (m)',
    'Grad (°C/km)', 'SD Grad', 'T_surf (°C)', 'SD T_surf',
    '|Curv|×10⁶', 'dT (°C)', 'τ (yr)', 'RMS (°C)',
]


# ---------------------------------------------------------------------------
# Multi-axis canvas (for curvature and inversion tabs)
# ---------------------------------------------------------------------------

class MultiAxCanvas(QWidget):
    """Matplotlib figure with *n* axes arranged in *nrows* × *ncols*."""

    def __init__(self, nrows=1, ncols=1, parent=None):
        super().__init__(parent)
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        axes = self.fig.subplots(nrows, ncols)
        # flatten to list regardless of shape
        try:
            self.axes = list(axes.flat)
        except AttributeError:
            self.axes = [axes]

    def clear(self):
        for ax in self.axes:
            ax.cla()

    def draw(self):
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Borehole Climate Analysis')
        self.setMinimumSize(QSize(1280, 800))

        self._boreholes = []
        self._air_years = None
        self._air_T = None
        self._inv_colorbar = None

        self._setup_central()
        self._setup_controls_dock()
        self._setup_table_dock()
        self._setup_menu()
        self.setStatusBar(QStatusBar())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_central(self):
        self._tabs = QTabWidget()

        # Single-axis tabs: use MplWidget
        self._w_tz = MplWidget('Temperature–Depth Profiles',
                               'Temperature (°C)', 'Depth (m)',
                               invert_y=True)
        self._w_red = MplWidget('Reduced Temperature Profiles',
                                'Reduced Temperature (°C)', 'Depth (m)',
                                invert_y=True)

        # Multi-axis tabs: use MultiAxCanvas
        self._w_curv = MultiAxCanvas(nrows=2, ncols=2)
        self._w_inv  = MultiAxCanvas(nrows=1, ncols=2)

        self._tabs.addTab(self._w_tz,   'T–z Profiles')
        self._tabs.addTab(self._w_curv, 'Curvature Analysis')
        self._tabs.addTab(self._w_red,  'Reduced Temperature')
        self._tabs.addTab(self._w_inv,  'Inversion')

        self._tabs.currentChanged.connect(self._refresh_current_tab)
        self.setCentralWidget(self._tabs)

    def _setup_controls_dock(self):
        dock = QDockWidget('Controls', self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        page = CustomPage(layout_cls=QVBoxLayout)
        page.setSpacing(8)

        # --- Data group ---
        grp_data = QGroupBox('Data')
        fl = QFormLayout(grp_data)

        self._dir_edit = QLineEdit(_DEFAULT_DIR)
        btn_browse = QPushButton('Browse…')
        btn_browse.clicked.connect(self._browse_directory)
        dir_row = QWidget()
        hl = QHBoxLayout(dir_row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self._dir_edit, 1)
        hl.addWidget(btn_browse)
        fl.addRow('Directory:', dir_row)

        self._sb_start = QDoubleSpinBox()
        self._sb_start.setRange(0, 500)
        self._sb_start.setValue(20)
        self._sb_start.setSuffix(' m')
        fl.addRow('Start depth:', self._sb_start)

        self._sb_min = QDoubleSpinBox()
        self._sb_min.setRange(0, 5000)
        self._sb_min.setValue(140)
        self._sb_min.setSuffix(' m')
        fl.addRow('Min depth:', self._sb_min)

        # Fit criterion as a ToggleSwitch
        crit_row = QWidget()
        crit_hl = QHBoxLayout(crit_row)
        crit_hl.setContentsMargins(0, 0, 0, 0)
        crit_hl.addWidget(QLabel('Min Curv'))
        self._toggle_criterion = ToggleSwitch()
        crit_hl.addWidget(self._toggle_criterion)
        crit_hl.addWidget(QLabel('Min SD Grad'))
        crit_hl.addStretch()
        fl.addRow('Fit criterion:', crit_row)

        btn_load = QPushButton('Load && Process')
        btn_load.setStyleSheet('font-weight: bold;')
        btn_load.clicked.connect(self._load_and_process)
        fl.addRow(btn_load)

        # Color mode toggle
        mode_row = QWidget()
        mode_hl = QHBoxLayout(mode_row)
        mode_hl.setContentsMargins(0, 0, 0, 0)
        mode_hl.addWidget(QLabel('Individual'))
        self._toggle_overlay = ToggleSwitch()
        mode_hl.addWidget(self._toggle_overlay)
        mode_hl.addWidget(QLabel('Uniform'))
        mode_hl.addStretch()
        fl.addRow('Color mode:', mode_row)
        self._toggle_overlay.stateChanged.connect(self._refresh_current_tab)

        # Air temperature
        self._air_edit = QLineEdit()
        self._air_edit.setPlaceholderText('(optional)')
        btn_air = QPushButton('Browse…')
        btn_air.clicked.connect(self._browse_air_temp)
        air_row = QWidget()
        air_hl = QHBoxLayout(air_row)
        air_hl.setContentsMargins(0, 0, 0, 0)
        air_hl.addWidget(self._air_edit, 1)
        air_hl.addWidget(btn_air)
        fl.addRow('Air temp file:', air_row)

        btn_air_fwd = QPushButton('Compute Air-Temp Forward')
        btn_air_fwd.clicked.connect(self._run_air_forward)
        fl.addRow(btn_air_fwd)

        page.addWidget(grp_data)

        # --- Inversion group ---
        grp_inv = QGroupBox('Inversion')
        fl2 = QFormLayout(grp_inv)

        self._cb_method = QComboBox()
        self._cb_method.addItems(['Newton (step change)', 'Grid search (step change)'])
        self._cb_method.currentIndexChanged.connect(self._toggle_inv_params)
        fl2.addRow('Method:', self._cb_method)

        self._sb_alpha = QDoubleSpinBox()
        self._sb_alpha.setRange(0.1, 1000)
        self._sb_alpha.setValue(31.5)
        self._sb_alpha.setSuffix(' m²/yr')
        self._sb_alpha.setDecimals(1)
        fl2.addRow('Diffusivity α:', self._sb_alpha)

        # Newton initial guesses
        self._newton_widget = QWidget()
        nfl = QFormLayout(self._newton_widget)
        nfl.setContentsMargins(0, 0, 0, 0)
        self._sb_dT0 = QDoubleSpinBox()
        self._sb_dT0.setRange(-20, 20); self._sb_dT0.setValue(1.0)
        self._sb_dT0.setSuffix(' °C'); self._sb_dT0.setDecimals(2)
        nfl.addRow('Initial dT:', self._sb_dT0)
        self._sb_tau0 = QDoubleSpinBox()
        self._sb_tau0.setRange(1, 2000); self._sb_tau0.setValue(150)
        self._sb_tau0.setSuffix(' yr')
        nfl.addRow('Initial τ:', self._sb_tau0)
        fl2.addRow(self._newton_widget)

        # Grid search — use RangeWidget for min/max, spinbox for step
        self._grid_widget = QWidget()
        gfl = QFormLayout(self._grid_widget)
        gfl.setContentsMargins(0, 0, 0, 0)

        self._range_dT = RangeWidget(
            min_val=-10.0, max_val=10.0, step=0.5, decimals=1, suffix='°C')
        self._range_dT.setValues(-5.0, 5.0)
        gfl.addRow('dT range:', self._range_dT)

        self._sb_dT_step = QDoubleSpinBox()
        self._sb_dT_step.setRange(0.01, 2.0); self._sb_dT_step.setValue(0.1)
        self._sb_dT_step.setSuffix(' °C'); self._sb_dT_step.setDecimals(2)
        gfl.addRow('dT step:', self._sb_dT_step)

        self._range_tau = RangeWidget(
            min_val=1.0, max_val=2000.0, step=10.0, decimals=0, suffix='yr')
        self._range_tau.setValues(10.0, 500.0)
        gfl.addRow('τ range:', self._range_tau)

        self._sb_tau_step = QDoubleSpinBox()
        self._sb_tau_step.setRange(1.0, 50.0); self._sb_tau_step.setValue(5.0)
        self._sb_tau_step.setSuffix(' yr')
        gfl.addRow('τ step:', self._sb_tau_step)

        self._grid_widget.setVisible(False)
        fl2.addRow(self._grid_widget)

        inv_btn_row = QWidget()
        inv_btn_hl = QHBoxLayout(inv_btn_row)
        inv_btn_hl.setContentsMargins(0, 0, 0, 0)
        btn_invert = QPushButton('Invert Selected')
        btn_invert.setStyleSheet('font-weight: bold;')
        btn_invert.clicked.connect(self._run_inversion)
        inv_btn_hl.addWidget(btn_invert)
        btn_invert_all = QPushButton('Invert All')
        btn_invert_all.setStyleSheet('font-weight: bold;')
        btn_invert_all.clicked.connect(self._run_inversion_all)
        inv_btn_hl.addWidget(btn_invert_all)
        fl2.addRow(inv_btn_row)

        page.addWidget(grp_inv)

        # --- Export group ---
        grp_exp = QGroupBox('Export')
        efl = QFormLayout(grp_exp)
        btn_save = QPushButton('Save Results CSV…')
        btn_save.clicked.connect(self._save_csv)
        efl.addRow(btn_save)
        page.addWidget(grp_exp)

        page.content_layout.addStretch()

        dock.setWidget(page)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _setup_table_dock(self):
        dock = QDockWidget('Boreholes', self)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._table = QTableWidget(0, len(TABLE_COLS))
        self._table.setHorizontalHeaderLabels(TABLE_COLS)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        dock.setWidget(self._table)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [200], Qt.Orientation.Vertical)

    def _setup_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu('File')
        for label, shortcut, slot in [
            ('Load Directory…', 'Ctrl+O', self._browse_directory),
            ('Save Results CSV…', 'Ctrl+S', self._save_csv),
            ('Quit', 'Ctrl+Q', self.close),
        ]:
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            if label == 'Quit':
                fm.addSeparator()
            fm.addAction(act)

    # ------------------------------------------------------------------
    # Slots: controls
    # ------------------------------------------------------------------

    def _toggle_inv_params(self, index):
        self._newton_widget.setVisible(index == 0)
        self._grid_widget.setVisible(index == 1)

    def _criterion(self):
        return 'sd_grad' if self._toggle_criterion.isChecked() else 'curvature'

    def _browse_directory(self):
        d = QFileDialog.getExistingDirectory(
            self, 'Select Borehole Data Directory', self._dir_edit.text())
        if d:
            self._dir_edit.setText(d)

    def _browse_air_temp(self):
        f, _ = QFileDialog.getOpenFileName(
            self, 'Select Air Temperature File', '',
            'Data files (*.csv *.dat *.txt);;All files (*)')
        if f:
            self._air_edit.setText(f)

    def _load_and_process(self):
        directory = self._dir_edit.text()
        if not Path(directory).is_dir():
            QMessageBox.warning(self, 'Directory not found',
                                f'Cannot find:\n{directory}')
            return

        self.statusBar().showMessage('Loading boreholes…')
        QApplication.processEvents()

        start = self._sb_start.value()
        min_d = self._sb_min.value()
        criterion = self._criterion()

        raw = load_boreholes(directory)
        if not raw:
            QMessageBox.warning(self, 'No data',
                                'No readable .dat files found.')
            self.statusBar().showMessage('No data loaded.')
            return

        self._boreholes = []
        for log in raw:
            result = process_borehole(log, start, min_d, criterion)
            if result is not None:
                self._boreholes.append(result)

        n_kept = len(self._boreholes)
        self._populate_table()
        self._table.selectAll()
        self.statusBar().showMessage(
            f'Loaded {n_kept} of {len(raw)} boreholes '
            f'({len(raw) - n_kept} rejected below min depth).')

    def _run_inversion(self):
        selected = self._selected_boreholes()
        if not selected:
            QMessageBox.information(self, 'Nothing selected',
                                    'Select at least one borehole.')
            return

        method = self._cb_method.currentIndex()
        alpha = self._sb_alpha.value()

        for log in selected:
            if log['Tred'] is None:
                continue
            z = log['z']
            Tred = log['Tred']

            if method == 0:   # Newton / Gauss-Newton
                dT, tau, rms, Cov, history = invert_newton(
                    z, Tred, alpha,
                    dT0=self._sb_dT0.value(),
                    tau0=self._sb_tau0.value(),
                )
                log['dT'] = dT
                log['tau'] = tau
                log['inv_rms'] = rms
                log['inv_cov'] = Cov
                log['_grid_data'] = None

            else:             # Grid search
                dT_min, dT_max = self._range_dT.values()
                tau_min, tau_max = self._range_tau.values()
                dT, tau, rms, grid, dT_vals, tau_vals = invert_gridsearch(
                    z, Tred, alpha,
                    dT_min=dT_min, dT_max=dT_max,
                    dT_step=self._sb_dT_step.value(),
                    tau_min=tau_min, tau_max=tau_max,
                    tau_step=self._sb_tau_step.value(),
                )
                log['dT'] = dT
                log['tau'] = tau
                log['inv_rms'] = rms
                log['inv_cov'] = None
                log['_grid_data'] = (grid, dT_vals, tau_vals)

        self._populate_table()
        self._restore_selection(selected)
        self._tabs.setCurrentIndex(3)
        self._plot_inversion(selected)
        self.statusBar().showMessage(
            f'Inversion complete for {len(selected)} borehole(s).')

    def _run_inversion_all(self):
        if not self._boreholes:
            QMessageBox.information(self, 'No data', 'No boreholes loaded.')
            return
        self._table.selectAll()
        self._run_inversion()

    def _run_air_forward(self):
        filepath = self._air_edit.text()
        if not filepath or not Path(filepath).is_file():
            QMessageBox.warning(self, 'No file',
                                'Select an air temperature file first.')
            return
        try:
            data = np.loadtxt(filepath, delimiter=None)
            if data.ndim == 1 or data.shape[1] < 2:
                raise ValueError('Need at least 2 columns: year, temperature.')
            years = data[:, 0]
            T_air = data[:, 1]
        except Exception as e:
            QMessageBox.critical(self, 'Load error', str(e))
            return

        self._air_years = years
        self._air_T = T_air
        alpha = self._sb_alpha.value()

        for log in self._selected_boreholes():
            log['Tred_air'] = air_temp_to_reduced(
                log['z'], years, T_air, alpha)

        self._tabs.setCurrentIndex(3)
        self._plot_inversion(self._selected_boreholes())
        self.statusBar().showMessage('Air-temperature forward model computed.')

    def _save_csv(self):
        if not self._boreholes:
            QMessageBox.information(self, 'No data', 'No boreholes loaded.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Results', 'borehole_results.csv',
            'CSV files (*.csv);;All files (*)')
        if not path:
            return
        with open(path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow([
                'Name', 'Max_Depth_m', 'Fit_Depth_m',
                'Grad_oCpkm', 'SD_Grad', 'Tsurf_oC', 'SD_Tsurf',
                'AbsCurv_e6', 'dT_oC', 'tau_yr', 'Inv_RMS_oC',
            ])
            for log in self._boreholes:
                writer.writerow(self._table_row_values(log))
        self.statusBar().showMessage(f'Saved to {path}')

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _populate_table(self):
        self._table.setRowCount(0)
        for log in self._boreholes:
            row = self._table.rowCount()
            self._table.insertRow(row)
            for col, val in enumerate(self._table_row_values(log)):
                item = QTableWidgetItem(val if isinstance(val, str) else '')
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, col, item)

    def _table_row_values(self, log):
        def fmt(v, d=3):
            return f'{v:.{d}f}' if v is not None else ''

        z = log['z']
        max_z = float(np.max(z)) if len(z) else None
        crit = self._criterion()

        if log['ztop'] is not None:
            idx = log['ind_curv'] if crit == 'curvature' else log['ind_grad']
            return [
                log['name'],
                fmt(max_z, 1),
                fmt(log['ztop'][idx], 1),
                fmt(log['grad'][idx], 3),
                fmt(log['sd_grad'][idx], 5),
                fmt(log['tsurf'][idx], 3),
                fmt(log['sd_tsurf'][idx], 4),
                fmt(abs(log['curv'][idx]) * 1e6 if not np.isnan(log['curv'][idx]) else None, 2),
                fmt(log.get('dT'), 3),
                fmt(log.get('tau'), 1),
                fmt(log.get('inv_rms'), 4),
            ]

        return [log['name'], fmt(max_z, 1)] + [''] * (len(TABLE_COLS) - 2)

    def _selected_boreholes(self):
        rows = {idx.row() for idx in self._table.selectedIndexes()}
        return [self._boreholes[r] for r in sorted(rows)
                if r < len(self._boreholes)]

    def _restore_selection(self, boreholes):
        names = {b['name'] for b in boreholes}
        self._table.clearSelection()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and item.text() in names:
                self._table.selectRow(row)

    # ------------------------------------------------------------------
    # Plot refresh
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        self._refresh_current_tab()

    def _refresh_current_tab(self, _=None):
        idx = self._tabs.currentIndex()
        sel = self._selected_boreholes()
        [self._plot_tz, self._plot_curvature,
         self._plot_reduced, self._plot_inversion][idx](sel)

    def _color(self, i):
        return _COLORS[i % len(_COLORS)]

    # ------------------------------------------------------------------
    # Individual plot methods
    # ------------------------------------------------------------------

    def _plot_tz(self, logs):
        ax = self._w_tz.ax
        ax.cla()
        ax.set_title('Temperature–Depth Profiles')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        for i, log in enumerate(logs):
            ax.plot(log['T'], log['z'],
                    color=self._color(i),
                    linewidth=1.2,
                    label=log['name'])
        if logs:
            ax.legend(fontsize=7)
        self._w_tz.canvas.draw_idle()

    def _plot_curvature(self, logs):
        self._w_curv.clear()
        axs = self._w_curv.axes   # 4 axes in 2×2 layout

        log = next((b for b in logs if b['ztop'] is not None), None)
        if log is None:
            self._w_curv.draw()
            return

        crit = self._criterion()
        idx = log['ind_curv'] if crit == 'curvature' else log['ind_grad']
        fit_depth = log['ztop'][idx]
        vline_kw = dict(color='r', linestyle='--', linewidth=1)

        panels = [
            ('Gradient (°C/km)', 'grad'),
            ('SD Gradient', 'sd_grad'),
            ('T_surf (°C)', 'tsurf'),
            ('Curvature', 'curv'),
        ]
        for ax, (ylabel, key) in zip(axs, panels):
            ax.plot(log['ztop'], log[key], 'b-', linewidth=1)
            ax.axvline(fit_depth, **vline_kw,
                       label=f'Fit start {fit_depth:.0f} m')
            ax.set_xlabel('Fit-start depth (m)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} — {log["name"]}')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        self._w_curv.draw()

    def _plot_reduced(self, logs):
        ax = self._w_red.ax
        ax.cla()
        ax.set_title('Reduced Temperature Profiles')
        ax.set_xlabel('Reduced Temperature (°C)')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        show_all = self._toggle_overlay.isChecked()
        data = self._boreholes if show_all else logs
        for i, log in enumerate(data):
            if log['Tred'] is None:
                continue
            ax.plot(log['Tred'], log['z'],
                    color=_COLORS[0] if show_all else self._color(i),
                    linewidth=1.2,
                    alpha=0.5 if show_all else 1.0,
                    label='_nolegend_' if show_all else log['name'])
        ax.axvline(0, color='k', linestyle=':', linewidth=1)
        if any(b['Tred'] is not None for b in data):
            if show_all:
                n = sum(1 for b in data if b['Tred'] is not None)
                ax.plot([], [], color=_COLORS[0], linewidth=1.2,
                        label=f'{n} borehole(s)')
            ax.legend(fontsize=7)
        self._w_red.canvas.draw_idle()

    def _plot_inversion(self, logs):
        self._w_inv.clear()
        ax_fit, ax_grid = self._w_inv.axes

        alpha = self._sb_alpha.value()

        for i, log in enumerate(logs):
            if log['Tred'] is None:
                continue
            color = self._color(i)
            ax_fit.plot(log['Tred'], log['z'], color=color,
                        linewidth=1.2, alpha=0.7, label=log['name'])

            if log.get('dT') is not None:
                T_fwd = forward_step(log['dT'], log['tau'], alpha, log['z'])
                bias = float(np.mean(log['Tred'] - T_fwd))
                ax_fit.plot(T_fwd + bias, log['z'], color=color,
                            linewidth=2, linestyle='--',
                            label=(f'{log["name"]} '
                                   f'dT={log["dT"]:+.2f}°C  '
                                   f'τ={log["tau"]:.0f} yr'))

            if log.get('Tred_air') is not None:
                ax_fit.plot(log['Tred_air'], log['z'], color=color,
                            linewidth=1.5, linestyle=':',
                            label=f'{log["name"]} (air-temp fwd)')

        ax_fit.axvline(0, color='k', linestyle=':', linewidth=1)
        ax_fit.invert_yaxis()
        ax_fit.set_xlabel('Reduced Temperature (°C)')
        ax_fit.set_ylabel('Depth (m)')
        ax_fit.set_title('Inversion Result')
        if logs:
            ax_fit.legend(fontsize=6)
        ax_fit.grid(True, alpha=0.3)

        # RMS grid from the last selected borehole with grid data
        grid_log = next(
            (b for b in reversed(logs) if b.get('_grid_data') is not None),
            None)

        if grid_log:
            grid, dT_vals, tau_vals = grid_log['_grid_data']
            ax_grid.set_visible(True)
            pcm = ax_grid.pcolormesh(tau_vals, dT_vals, grid,
                                     cmap='viridis_r', shading='auto')
            if self._inv_colorbar is not None:
                self._inv_colorbar.remove()
            self._inv_colorbar = self._w_inv.fig.colorbar(
                pcm, ax=ax_grid, label='RMS (°C)')
            ax_grid.set_xlabel('τ (yr)')
            ax_grid.set_ylabel('dT (°C)')
            ax_grid.set_title(f'RMS grid — {grid_log["name"]}')
            if grid_log.get('dT') is not None:
                ax_grid.plot(grid_log['tau'], grid_log['dT'],
                             'r*', markersize=12, label='Best fit')
                ax_grid.legend(fontsize=7)
        else:
            if self._inv_colorbar is not None:
                self._inv_colorbar.remove()
                self._inv_colorbar = None
            ax_grid.set_visible(False)

        self._w_inv.draw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
