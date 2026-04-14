"""
prism.py
--------
Vertical gravity of a right rectangular prism using the analytical formula
of Nagy et al. (2000) *Journal of Geodesy*, 74, 552-560.

The prism extends from x1 to x2, y1 to y2, z1 (top) to z2 (bottom),
with z positive downward.  The observation point is at the origin (0,0,0)
in the shifted coordinate system.

Public API
----------
gz_prism(x1, x2, y1, y2, z1, z2, drho)
    Vertical gravity (m/s²) of a single prism at the origin.

gz_prism_grid(obs_x, obs_y, prism_cx, prism_cy, dx, dy, h, drho)
    Vectorised: gravity at every (obs_x, obs_y) observation point from
    a 2-D grid of prisms, each with square cross-section dx × dy (m) and
    depth extent h (m), top at surface (z=0).
"""

from __future__ import annotations

import numpy as np

G_SI = 6.67430e-11   # m³ kg⁻¹ s⁻²


# ── core analytical kernel ────────────────────────────────────────────────────

def _f(x: float | np.ndarray,
       y: float | np.ndarray,
       z: float | np.ndarray) -> float | np.ndarray:
    """
    Indefinite integral kernel for gz (Nagy et al. 2000, eq. 6).

        f(x,y,z) = y·ln(x + r) + x·ln(y + r) - z·arctan(x·y / (z·r))

    where r = sqrt(x² + y² + z²).  A small epsilon guards the arctan
    singularity when z ≈ 0 (observation on the face of the prism).
    """
    r = np.sqrt(x*x + y*y + z*z)
    eps = 1e-30
    return (y * np.log(x + r + eps)
            + x * np.log(y + r + eps)
            - z * np.arctan2(x * y, z * r + eps))


def gz_prism(x1: float, x2: float,
             y1: float, y2: float,
             z1: float, z2: float,
             drho: float) -> float:
    """
    Vertical gravity (m/s²) of a right rectangular prism at the origin.

    The prism occupies [x1,x2] × [y1,y2] × [z1,z2] in a coordinate system
    where the observation point is at the origin and z is positive downward.

    Parameters
    ----------
    x1, x2 : float  E–W prism corners (m), x2 > x1
    y1, y2 : float  N–S prism corners (m), y2 > y1
    z1, z2 : float  depth to top and bottom (m), z2 > z1 ≥ 0
    drho   : float  density contrast (kg/m³), positive = denser than host

    Returns
    -------
    float  gz in m/s²
    """
    # Alternating-sign sum over the 8 corners of the prism
    gz = 0.0
    for xi, sx in ((x1, -1), (x2, +1)):
        for yi, sy in ((y1, -1), (y2, +1)):
            for zi, sz in ((z1, +1), (z2, -1)):
                gz += sx * sy * sz * _f(xi, yi, zi)
    return G_SI * drho * gz


# ── vectorised grid forward ───────────────────────────────────────────────────

def gz_prism_grid(obs_x: np.ndarray,
                  obs_y: np.ndarray,
                  prism_cx: np.ndarray,
                  prism_cy: np.ndarray,
                  dx: float,
                  dy: float,
                  h: np.ndarray,
                  drho: float,
                  z_top: float = 0.0) -> np.ndarray:
    """
    Vertical gravity (mGal) at a set of observation points from a 2-D grid
    of vertical rectangular prisms.

    Each prism has a square horizontal cross-section ``dx × dy`` (m), is
    centred at (prism_cx[i], prism_cy[i]), has its top at ``z_top`` (m,
    positive downward) and its base at ``z_top + h[i]`` (m).

    The computation is O(N_obs × N_prisms), so the observation grid and
    prism grid should be the same to keep the problem tractable.  For large
    grids (> 10⁴ cells) use the Parker-Oldenburg spectral inversion instead.

    Parameters
    ----------
    obs_x, obs_y   : 1-D arrays, observation coordinates (m)
    prism_cx/cy    : 1-D arrays, prism centre coordinates (m)
    dx, dy         : prism half-widths (m)  — full width is 2*dx × 2*dy
                     Pass dx = grid_spacing / 2 for touching prisms.
    h              : 1-D array, prism thickness (m), one value per prism
    drho           : scalar density contrast (kg/m³)
    z_top          : depth to prism top (m), default 0 (at surface)

    Returns
    -------
    gz : 1-D array, vertical gravity in mGal at each observation point
    """
    obs_x = np.asarray(obs_x, dtype=float).ravel()
    obs_y = np.asarray(obs_y, dtype=float).ravel()
    prism_cx = np.asarray(prism_cx, dtype=float).ravel()
    prism_cy = np.asarray(prism_cy, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()

    n_obs = len(obs_x)
    gz_total = np.zeros(n_obs)

    # Only compute prisms with non-zero thickness
    active = h > 0.0
    px = prism_cx[active]
    py = prism_cy[active]
    ph = h[active]

    for i, (cx, cy, hi) in enumerate(zip(px, py, ph)):
        z2 = z_top + hi
        for j in range(n_obs):
            # Shift origin to observation point
            x1_ = cx - dx - obs_x[j]
            x2_ = cx + dx - obs_x[j]
            y1_ = cy - dy - obs_y[j]
            y2_ = cy + dy - obs_y[j]
            gz_total[j] += gz_prism(x1_, x2_, y1_, y2_, z_top, z2, drho)

    return gz_total * 1e5  # m/s² → mGal
