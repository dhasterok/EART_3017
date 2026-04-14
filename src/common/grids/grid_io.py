"""
grid_io.py
----------
Readers for gridded geophysical ASCII formats.

Functions
---------
read_pacs_grd(filepath)
    Read a PACS/NEVCBA-style ASCII grid file (e.g. ``nevboug.grd.txt``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PacsGrid:
    """
    Container returned by :func:`read_pacs_grd`.

    Attributes
    ----------
    grid : ndarray, shape (nrows, ncols)
        Grid values; cells outside the survey have value NaN.
    x : ndarray, shape (ncols,)
        X coordinates in the grid's native units (km for LCC grids).
    y : ndarray, shape (nrows,)
        Y coordinates in the grid's native units (km for LCC grids).
    meta : dict
        Parsed header fields:

        * ``title``     -- descriptive string from line 6
        * ``proj``      -- projection code (4 = Lambert Conformal Conic)
        * ``cmerid``    -- central meridian (degrees)
        * ``baselat``   -- base latitude (degrees)
        * ``ncols``     -- number of grid columns
        * ``nrows``     -- number of grid rows
        * ``x_orig``    -- X origin coordinate
        * ``dx``        -- X cell spacing
        * ``y_orig``    -- Y origin coordinate
        * ``dy``        -- Y cell spacing
        * ``nodata``    -- sentinel value used for missing cells (0.0)
    """
    grid: np.ndarray
    x:    np.ndarray
    y:    np.ndarray
    meta: dict


def read_pacs_grd(filepath) -> PacsGrid:
    """
    Read a PACS/NEVCBA ASCII grid file.

    The 10-line header format is::

        Line 1 : FILETYPE / creation date (ignored)
        Line 2 : internal filename (ignored)
        Line 3 : dataset description (ignored)
        Line 4 : Fortran read format, e.g. ``(5E16.8)`` (ignored)
        Line 5 : line-count metadata (ignored)
        Line 6 : grid title  [56 chars], program [8 chars],
                 central meridian, base latitude
        Line 7 : ncols  nrows  nval  proj
                 x_orig  dx  y_orig  dy
        Lines 8-10 : human-readable column annotations (ignored)

    Data follow from line 11 onward.  Each row of the grid is stored with
    values padded to the next multiple of 5 per the Fortran format; the
    reader discards those padding values automatically.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.grd.txt`` (or ``.grd``) ASCII file.

    Returns
    -------
    PacsGrid
        Named container with ``grid``, ``x``, ``y``, and ``meta``.

    Raises
    ------
    ValueError
        If the header cannot be parsed.
    """
    filepath = Path(filepath)

    with filepath.open() as fh:
        lines = fh.readlines()

    if len(lines) < 11:
        raise ValueError(f"{filepath}: file has fewer than 11 lines; "
                         "not a valid PACS grid.")

    # ── Line 6: title, central meridian, base latitude ────────────────────
    line6 = lines[5]
    title  = line6[:56].strip()
    # Extract all numbers (including negatives) from the tail of line 6
    nums6  = re.findall(r'-?\d+\.?\d*', line6[56:])
    try:
        cmerid  = float(nums6[-2])
        baselat = float(nums6[-1])
    except (IndexError, ValueError):
        cmerid  = float('nan')
        baselat = float('nan')

    # ── Line 7: grid dimensions and spatial parameters ─────────────────────
    parts = lines[6].split()
    if len(parts) < 8:
        raise ValueError(f"{filepath}: line 7 has fewer than 8 fields.")
    try:
        ncols  = int(parts[0])
        nrows  = int(parts[1])
        proj   = int(parts[3])
        x_orig = float(parts[4])
        dx     = float(parts[5])
        y_orig = float(parts[6])
        dy     = float(parts[7])
    except ValueError as exc:
        raise ValueError(f"{filepath}: could not parse grid spec: {exc}") from exc

    # ── Read all data values (lines 11 onward) ─────────────────────────────
    vals = []
    for line in lines[10:]:
        vals.extend(float(v) for v in line.split())

    # Each row is padded to the next multiple of 5 (Fortran format).
    vals_per_row = int(np.ceil(ncols / 5)) * 5
    expected = nrows * vals_per_row
    if len(vals) < expected:
        raise ValueError(
            f"{filepath}: expected {expected} data values "
            f"({nrows} rows × {vals_per_row}), found {len(vals)}."
        )

    # Reshape and trim padding from each row
    raw  = np.array(vals[:expected], dtype=float).reshape(nrows, vals_per_row)
    grid = raw[:, :ncols]

    # Replace nodata sentinel (0.0) with NaN
    nodata = 0.0
    grid[grid == nodata] = np.nan

    # Coordinate arrays
    x = x_orig + np.arange(ncols) * dx
    y = y_orig + np.arange(nrows) * dy

    meta = dict(
        title=title, proj=proj,
        cmerid=cmerid, baselat=baselat,
        ncols=ncols, nrows=nrows,
        x_orig=x_orig, dx=dx,
        y_orig=y_orig, dy=dy,
        nodata=nodata,
    )

    return PacsGrid(grid=grid, x=x, y=y, meta=meta)
