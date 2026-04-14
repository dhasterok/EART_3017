"""
data_loader.py
--------------
CSV gravity data loading utilities for the 2-D gravity modelling GUI.

Provides:
  ObservedData          -- dataclass holding loaded/projected gravity data
  ColumnSelectorDialog  -- map CSV columns to model variables
  ProfileSelectorDialog -- interactive map-based profile selection (with
                          zoom/pan toolbar; haversine distances for lon/lat)
  load_csv_data()       -- top-level entry point
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QLabel, QDialogButtonBox,
    QDoubleSpinBox, QMessageBox, QWidget,
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
#  Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObservedData:
    """
    Observed gravity data for comparison with the forward model.

    Attributes
    ----------
    x          : observation positions in km (along-profile for 2-D selection)
    gz         : Bouguer anomaly or FAA in mGal
    gz_unc     : per-station uncertainty (mGal), or None
    masked     : boolean array -- True means the station is excluded from misfit
    profile_start, profile_end : original (x_raw, y_raw) endpoints if 2-D profile
    swath_half_width : half-width used for station selection (raw input units)
    x_col, y_col, gz_col, gz_unc_col : column names as read from CSV
    x_scale_km : factor applied to convert raw x (or along-profile dist.) → km
                 (1.0 when haversine conversion was applied automatically)
    source_file : path to the CSV that was loaded
    """
    x:           np.ndarray
    gz:          np.ndarray
    gz_unc:      Optional[np.ndarray]  = None
    masked:      Optional[np.ndarray]  = None   # bool
    profile_start: Optional[tuple]     = None
    profile_end:   Optional[tuple]     = None
    swath_half_width: float            = 5.0
    x_col:       str = ""
    y_col:       str = ""
    gz_col:      str = ""
    gz_unc_col:  str = ""
    x_scale_km:  float = 1.0
    source_file: str = ""

    def __post_init__(self):
        if self.masked is None:
            self.masked = np.zeros(len(self.x), dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
#  Haversine helper
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(lon0: float, lat0: float,
                  lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Great-circle distance in km from (lon0, lat0) to arrays of (lons, lats).
    All angles in degrees.
    """
    R = 6371.0
    lat0_r = math.radians(lat0)
    dlat   = np.radians(lats - lat0)
    dlon   = np.radians(lons - lon0)
    lat1_r = np.radians(lats)
    a = (np.sin(dlat / 2.0) ** 2
         + math.cos(lat0_r) * np.cos(lat1_r) * np.sin(dlon / 2.0) ** 2)
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ─────────────────────────────────────────────────────────────────────────────
#  Column selector dialog
# ─────────────────────────────────────────────────────────────────────────────

class ColumnSelectorDialog(QDialog):
    """
    Let the user map CSV columns to X, Y (optional), Gravity, Uncertainty.
    Also sets a scale factor to convert X values to km (for 1-D data).
    When lat/lon columns are chosen and a 2-D profile is used, the
    ProfileSelectorDialog applies haversine distances automatically and
    x_scale is ignored for the profile projection.
    """

    _NONE = "<none>"

    def __init__(self, columns: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Data Columns")
        self.setMinimumWidth(400)

        self._cols = list(columns)
        all_cols_opt = [self._NONE] + self._cols

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Map CSV columns to model variables.  "
                   "X and Gravity are required."))

        form = QFormLayout()

        self.cmb_x   = QComboBox(); self.cmb_x.addItems(self._cols)
        self.cmb_y   = QComboBox(); self.cmb_y.addItems(all_cols_opt)
        self.cmb_gz  = QComboBox(); self.cmb_gz.addItems(self._cols)
        self.cmb_unc = QComboBox(); self.cmb_unc.addItems(all_cols_opt)

        # x scale (only relevant for 1-D data; lat/lon profile auto-converts)
        self.spin_xscale = QDoubleSpinBox()
        self.spin_xscale.setRange(1e-12, 1e6)
        self.spin_xscale.setDecimals(6)
        self.spin_xscale.setValue(1.0)
        self.spin_xscale.setToolTip(
            "Multiply X by this to get km.  Used for 1-D data only.\n"
            "  1.0    if X is already in km\n"
            "  0.001  if X is in metres\n"
            "For lat/lon 2-D profiles the haversine formula is used automatically.")

        form.addRow("X -- position (required):", self.cmb_x)
        form.addRow("Y -- position (optional):", self.cmb_y)
        form.addRow("Gravity (required):", self.cmb_gz)
        form.addRow("Uncertainty (optional):", self.cmb_unc)
        form.addRow("X scale → km (1-D only):", self.spin_xscale)

        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._auto_guess()

    def _auto_guess(self):
        """Try to pre-select sensible defaults from column names."""
        col_lower = [c.lower() for c in self._cols]

        def find(keywords, combo, offset=0):
            for kw in keywords:
                for i, cl in enumerate(col_lower):
                    if kw in cl:
                        combo.setCurrentIndex(i + offset)
                        return

        find(['longitude', 'lon', 'mga_east', 'easting', ' x'], self.cmb_x)
        find(['latitude',  'lat', 'mga_north', 'northing', ' y'],
             self.cmb_y, offset=1)   # +1 for <none>
        find(['ba_1984', 'bouguer', 'faa', 'grav', 'anomaly'],
             self.cmb_gz)
        find(['unc', 'err', 'std', 'sigma'], self.cmb_unc, offset=1)

        # Suggest scale for easting/northing
        x_name = self.cmb_x.currentText().lower()
        if 'east' in x_name or 'north' in x_name or 'mga' in x_name:
            self.spin_xscale.setValue(0.001)

    def get_selection(self) -> dict:
        def opt(cmb):
            v = cmb.currentText()
            return None if v == self._NONE else v

        return {
            'x':          self.cmb_x.currentText(),
            'y':          opt(self.cmb_y),
            'gz':         self.cmb_gz.currentText(),
            'gz_unc':     opt(self.cmb_unc),
            'x_scale_km': self.spin_xscale.value(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Profile selector dialog
# ─────────────────────────────────────────────────────────────────────────────

class ProfileSelectorDialog(QDialog):
    """
    Show a map of gravity stations; user clicks two points to define a
    profile line, sets a swath half-width, and the dialog returns which
    stations fall within the swath.

    Features
    --------
    • Navigation toolbar for zoom / pan (profile clicks are only registered
      when the toolbar is in pointer mode, not zoom or pan).
    • Automatic haversine-distance conversion when the X/Y columns look like
      longitude / latitude (both column names contain 'lon' or 'lat').
    """

    def __init__(self, df, x_col: str, y_col: str, gz_col: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Profile")
        self.resize(880, 680)

        self._df     = df
        self._x_col  = x_col
        self._y_col  = y_col
        self._gz_col = gz_col

        self._clicks        = []   # up to 2 (x, y) click coordinates
        self._line_artist   = None
        self._swath_patch   = None
        self._dot_artists   = []
        self._projected_in_km = False  # True when haversine was applied

        # scatter / colorbar state
        self._scatter       = None
        self._colorbar      = None
        self._gz_values     = None

        layout = QVBoxLayout(self)

        self._info_label = QLabel(
            "Zoom/pan first if needed, then click the profile START point.")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._info_label)

        # ── map canvas + toolbar ──────────────────────────────────────────
        self._fig    = Figure(figsize=(8, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._ax     = self._fig.add_subplot(111)

        # Toolbar must be parented to a QWidget, not a QDialog directly,
        # so we embed both in a container widget.
        map_container = QWidget()
        map_vbox      = QVBoxLayout(map_container)
        map_vbox.setContentsMargins(0, 0, 0, 0)
        self._toolbar = NavigationToolbar2QT(self._canvas, map_container)
        map_vbox.addWidget(self._toolbar)
        map_vbox.addWidget(self._canvas)
        layout.addWidget(map_container)

        # ── swath controls ────────────────────────────────────────────────
        swath_row = QHBoxLayout()
        swath_row.addWidget(QLabel("Swath half-width:"))
        self._spin_swath = QDoubleSpinBox()
        self._spin_swath.setRange(1e-6, 1e9)
        self._spin_swath.setSingleStep(1.0)
        self._spin_swath.setDecimals(3)
        self._spin_swath.valueChanged.connect(self._redraw_swath)
        swath_row.addWidget(self._spin_swath)

        self._swath_unit_label = QLabel()
        swath_row.addWidget(self._swath_unit_label)
        swath_row.addStretch()

        self._btn_color = QPushButton("Color by Gravity")
        self._btn_color.setCheckable(True)
        self._btn_color.setChecked(False)
        self._btn_color.setToolTip(
            "Toggle gravity-value colouring of the station scatter plot")
        self._btn_color.toggled.connect(self._toggle_color_by_gravity)
        swath_row.addWidget(self._btn_color)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._reset)
        swath_row.addWidget(self._btn_reset)
        layout.addLayout(swath_row)

        # ── buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_ok = QPushButton("Apply Profile")
        self._btn_ok.setEnabled(False)
        self._btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_ok)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        self._plot_stations()
        self._cid = self._canvas.mpl_connect(
            'button_press_event', self._on_click)

    # ── lat/lon detection ─────────────────────────────────────────────────

    @property
    def _is_latlon(self) -> bool:
        """True when both column names suggest geographic coordinates."""
        combined = (self._x_col + self._y_col).lower()
        return 'lon' in combined and 'lat' in combined

    # ── map drawing ───────────────────────────────────────────────────────

    def _plot_stations(self):
        # Remove existing colorbar before clearing axes (prevents duplication)
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

        self._ax.clear()
        xs = self._df[self._x_col].values.astype(float)
        ys = self._df[self._y_col].values.astype(float)
        self._gz_values = self._df[self._gz_col].values.astype(float)

        # Default: single colour — fast, no colorbar
        self._scatter = self._ax.scatter(
            xs, ys, s=6, color='steelblue', zorder=2, picker=False)

        self._ax.set_xlabel(self._x_col)
        self._ax.set_ylabel(self._y_col)
        self._ax.set_aspect('equal', adjustable='box')

        self._line_artist = None
        self._swath_patch = None
        self._dot_artists.clear()

        # Reset the colour toggle button without re-triggering the slot
        self._btn_color.blockSignals(True)
        self._btn_color.setChecked(False)
        self._btn_color.blockSignals(False)

        self._canvas.draw_idle()

        # Swath unit label and default value
        if self._is_latlon:
            self._swath_unit_label.setText(
                "degrees  (lat/lon detected; profile distances → km via haversine)")
            self._spin_swath.setValue(0.5)   # ~55 km default
        else:
            self._swath_unit_label.setText("(same units as X/Y columns)")
            self._spin_swath.setValue(5.0)

    def _toggle_color_by_gravity(self, checked: bool):
        """Switch scatter between a single colour and gravity-value colouring."""
        if self._scatter is None or self._gz_values is None:
            return

        if checked:
            self._scatter.set_array(self._gz_values)
            self._scatter.set_cmap('RdBu_r')
            self._scatter.autoscale()
            self._colorbar = self._fig.colorbar(
                self._scatter, ax=self._ax,
                label=self._gz_col, shrink=0.8)
        else:
            if self._colorbar is not None:
                self._colorbar.remove()
                self._colorbar = None
            self._scatter.set_array(None)
            self._scatter.set_facecolor('steelblue')

        self._canvas.draw_idle()

    def _on_click(self, event):
        # Only capture clicks when the toolbar is in pointer mode
        if self._toolbar.mode != '':
            return
        if event.inaxes is not self._ax or event.button != 1:
            return
        if len(self._clicks) >= 2:
            return

        cx, cy = event.xdata, event.ydata
        self._clicks.append((cx, cy))
        dot, = self._ax.plot(cx, cy, 'r+', ms=14, mew=2, zorder=6)
        self._dot_artists.append(dot)
        self._canvas.draw_idle()

        if len(self._clicks) == 1:
            self._info_label.setText(
                "Now click the profile END point.  "
                "(Zoom/pan first if needed.)")
        else:
            self._info_label.setText(
                "Profile defined.  Adjust swath width and click Apply Profile.")
            self._draw_line()
            self._redraw_swath()
            self._btn_ok.setEnabled(True)

    def _draw_line(self):
        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except Exception:
                pass
        (x0, y0), (x1, y1) = self._clicks
        self._line_artist, = self._ax.plot(
            [x0, x1], [y0, y1], 'r-', lw=1.5, zorder=5)
        self._canvas.draw_idle()

    def _redraw_swath(self):
        if len(self._clicks) < 2:
            return
        if self._swath_patch is not None:
            try:
                self._swath_patch.remove()
            except Exception:
                pass
            self._swath_patch = None

        (x0, y0), (x1, y1) = self._clicks
        hw = self._spin_swath.value()

        dx, dy  = x1 - x0, y1 - y0
        length  = np.hypot(dx, dy)
        if length < 1e-30:
            return
        px, py  = -dy / length, dx / length    # perpendicular unit vector

        corners = np.array([
            [x0 + px * hw, y0 + py * hw],
            [x1 + px * hw, y1 + py * hw],
            [x1 - px * hw, y1 - py * hw],
            [x0 - px * hw, y0 - py * hw],
        ])
        self._swath_patch = mpatches.Polygon(
            corners, closed=True,
            facecolor='yellow', alpha=0.25,
            edgecolor='darkorange', linewidth=1.2, zorder=3)
        self._ax.add_patch(self._swath_patch)
        self._canvas.draw_idle()

    def _reset(self):
        """Remove profile overlay artists; keep the station scatter intact."""
        self._clicks.clear()
        self._btn_ok.setEnabled(False)
        self._info_label.setText(
            "Zoom/pan first if needed, then click the profile START point.")

        for dot in self._dot_artists:
            try:
                dot.remove()
            except Exception:
                pass
        self._dot_artists.clear()

        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except Exception:
                pass
            self._line_artist = None

        if self._swath_patch is not None:
            try:
                self._swath_patch.remove()
            except Exception:
                pass
            self._swath_patch = None

        self._canvas.draw_idle()

    # ── public result ─────────────────────────────────────────────────────

    def get_profile(self):
        """Return (start_xy, end_xy, swath_half_width) or None."""
        if len(self._clicks) < 2:
            return None
        return self._clicks[0], self._clicks[1], self._spin_swath.value()

    def compute_projected_data(self):
        """
        Project all stations within the swath onto the profile line.

        When X/Y columns are geographic (lat/lon), along-profile distances
        are computed using the haversine formula and returned in km.
        Otherwise they are returned in the original input units.

        Sets ``self._projected_in_km = True`` when haversine was applied so
        the caller knows no further scaling is needed.

        Returns
        -------
        along    : ndarray -- along-profile distances
        in_swath : bool ndarray -- True for stations inside the swath
        """
        self._projected_in_km = False

        if len(self._clicks) < 2:
            return None, None

        (x0, y0), (x1, y1) = self._clicks
        hw = self._spin_swath.value()

        xs = self._df[self._x_col].values.astype(float)
        ys = self._df[self._y_col].values.astype(float)

        dx, dy  = x1 - x0, y1 - y0
        length  = np.hypot(dx, dy)
        if length < 1e-30:
            return np.zeros(len(xs)), np.zeros(len(xs), dtype=bool)

        tx, ty = dx / length, dy / length   # tangent unit vector
        px, py = -ty, tx                     # left-perpendicular unit vector

        rx, ry = xs - x0, ys - y0
        along_deg = rx * tx + ry * ty        # in input-coordinate units
        cross     = rx * px + ry * py

        in_swath = np.abs(cross) <= hw

        if self._is_latlon:
            # Project each station onto the profile line in coordinate space,
            # then compute the signed haversine distance from the start point.
            lon_proj = x0 + along_deg * tx
            lat_proj = y0 + along_deg * ty
            along_km = _haversine_km(x0, y0, lon_proj, lat_proj)
            # Preserve sign (negative = behind the start point)
            along_km *= np.sign(along_deg)
            self._projected_in_km = True
            return along_km, in_swath

        return along_deg, in_swath


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level loader
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_data(path, parent=None) -> Optional[ObservedData]:
    """
    Load gravity data from a CSV file.

    1. Read the file with pandas.
    2. Show ColumnSelectorDialog so the user maps columns.
    3. If a Y column was chosen, show ProfileSelectorDialog for 2-D selection.
    4. Return ObservedData (x always in km) or None if user cancels.
    """
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception as exc:
        QMessageBox.critical(parent, "Error loading CSV", str(exc))
        return None

    if df.empty:
        QMessageBox.warning(parent, "Empty file",
                            "The CSV file contains no data rows.")
        return None

    # ── column selection ──────────────────────────────────────────────────
    col_dlg = ColumnSelectorDialog(list(df.columns), parent)
    if col_dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    sel = col_dlg.get_selection()

    x_col   = sel['x']
    y_col   = sel['y']
    gz_col  = sel['gz']
    unc_col = sel['gz_unc']
    x_scale = sel['x_scale_km']

    # Validate required columns
    for col in (x_col, gz_col):
        if col not in df.columns:
            QMessageBox.critical(parent, "Column error",
                                 f"Column '{col}' not found in the CSV.")
            return None

    # Drop rows with NaN in required columns
    req = [c for c in (x_col, y_col, gz_col) if c is not None]
    df  = df.dropna(subset=req).reset_index(drop=True)
    if df.empty:
        QMessageBox.warning(parent, "No valid rows",
                            "All rows have NaN in required columns.")
        return None

    gz_data = df[gz_col].values.astype(float)
    gz_unc  = (df[unc_col].values.astype(float)
               if unc_col and unc_col in df.columns else None)

    profile_start = profile_end = None
    swath_hw      = 5.0

    # ── 2-D profile selection ─────────────────────────────────────────────
    if y_col is not None:
        prof_dlg = ProfileSelectorDialog(df, x_col, y_col, gz_col, parent)
        if prof_dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        result = prof_dlg.get_profile()
        if result is None:
            QMessageBox.warning(parent, "No profile",
                                "No profile was defined.")
            return None
        profile_start, profile_end, swath_hw = result
        along, in_swath = prof_dlg.compute_projected_data()

        idx = np.where(in_swath)[0]
        if idx.size == 0:
            QMessageBox.warning(parent, "Empty swath",
                                "No stations fall within the selected swath.")
            return None

        gz_data = gz_data[idx]
        gz_unc  = gz_unc[idx]  if gz_unc  is not None else None

        if prof_dlg._projected_in_km:
            # Haversine already returned km; no further scaling
            x_data   = along[idx]
            x_scale  = 1.0
        else:
            x_data   = along[idx] * x_scale

    else:
        # 1-D: use X column directly
        x_data = df[x_col].values.astype(float) * x_scale   # → km

    # Sort by x
    order   = np.argsort(x_data)
    x_data  = x_data[order]
    gz_data = gz_data[order]
    gz_unc  = gz_unc[order] if gz_unc is not None else None

    return ObservedData(
        x                = x_data,
        gz               = gz_data,
        gz_unc           = gz_unc,
        masked           = np.zeros(len(x_data), dtype=bool),
        profile_start    = profile_start,
        profile_end      = profile_end,
        swath_half_width = swath_hw,
        x_col            = x_col,
        y_col            = y_col or "",
        gz_col           = gz_col,
        gz_unc_col       = unc_col or "",
        x_scale_km       = x_scale,
        source_file      = str(path),
    )
