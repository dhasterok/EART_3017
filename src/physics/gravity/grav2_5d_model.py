"""
grav2_5d_model.py
-----------------
2-D vertical gravity anomaly (mGal) for a polygonal cross-section,
computed by distributing infinite horizontal cylinders over the cross-section.

Each area element dA at position (x', z') contributes:

    δgz = 2 G ρ (z' − z_obs) / r²  dA        r² = (x_obs−x')² + (z_obs−z')²

Summing over a fine regular grid of source points inside the polygon
recovers the exact 2-D result in the limit of fine grid spacing.

Grid spacing h is chosen adaptively:

    h = min(z_min, body_width, body_height) / N_CELLS

where z_min is the shallowest depth of the polygon and N_CELLS = 20.
This ensures the shallowest part of the body is always resolved with at
least 20 cells, preventing near-surface artefacts.
"""

import numpy as np
from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

G_SI        = 6.674e-11   # m³ kg⁻¹ s⁻²
MGAL_PER_SI = 1.0e5       # 1 m s⁻² = 1e5 mGal
_N_CELLS    = 20          # minimum cells across the shallowest dimension


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _points_in_polygon(xq, zq, vx, vz):
    """
    Vectorised ray-casting point-in-polygon test.

    xq, zq : (N,) arrays of query points
    vx, vz : (V,) arrays of polygon vertices — all in the same units
    """
    n = len(vx)
    inside = np.zeros(len(xq), dtype=bool)
    xj, zj = vx[-1], vz[-1]
    for i in range(n):
        xi, zi = vx[i], vz[i]
        dz = float(zj - zi)
        if abs(dz) < 1e-12:
            xj, zj = xi, zi
            continue
        cross = ((zi > zq) != (zj > zq)) & \
                (xq < (xj - xi) * (zq - zi) / dz + xi)
        inside ^= cross
        xj, zj = xi, zi
    return inside


def _grid_points(vertices_km):
    """
    Regular grid of source points inside the polygon.

    Returns
    -------
    xq, zq : ndarray  -- source positions (m)
    h      : float    -- grid cell side (m); cell area = h²
    """
    vx = np.array([v[0] for v in vertices_km])
    vz = np.array([v[1] for v in vertices_km])

    x0, x1 = vx.min() * 1e3, vx.max() * 1e3   # m
    z0, z1 = vz.min() * 1e3, vz.max() * 1e3
    width  = max(x1 - x0, 1.0)
    height = max(z1 - z0, 1.0)
    z_min  = max(z0, 1.0)   # shallowest depth (m), floor at 1 m

    h = min(z_min, width, height) / _N_CELLS
    h = max(h, 1.0)   # absolute floor at 1 m

    xs = np.arange(x0 + h / 2.0, x1, h)
    zs = np.arange(z0 + h / 2.0, z1, h)
    if xs.size == 0 or zs.size == 0:
        return np.empty(0), np.empty(0), h

    XX, ZZ = np.meshgrid(xs, zs)
    xg, zg = XX.ravel(), ZZ.ravel()

    mask = _points_in_polygon(xg, zg, vx * 1e3, vz * 1e3)
    return xg[mask], zg[mask], h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gz(x_obs, z_obs, vertices, rho: float,
               strike_half_length_km=None, n_strike: int = 201):
    """
    2-D vertical gravity anomaly (mGal) via distributed infinite cylinders.

    Parameters
    ----------
    x_obs    : array-like  -- observation x-positions (km)
    z_obs    : float       -- observation depth (km; 0 = surface)
    vertices : list of [x, z]  -- polygon vertices (km; z ≥ 0 downward)
    rho      : float       -- density contrast (kg/m³)
    strike_half_length_km, n_strike : ignored (API compatibility)
    """
    x_obs = np.asarray(x_obs, dtype=float)
    if len(vertices) < 3:
        return np.zeros(len(x_obs))

    xq, zq, h = _grid_points(vertices)
    if xq.size == 0:
        return np.zeros(len(x_obs))

    z_obs_m   = float(z_obs) * 1e3
    cell_area = h * h

    # Vectorised over all observations at once
    xi   = x_obs[:, np.newaxis] * 1e3 - xq[np.newaxis, :]   # (N_x, N_q)
    zeta = z_obs_m - zq[np.newaxis, :]                        # broadcast (N_q,)
    r2   = xi*xi + zeta*zeta
    r2   = np.where(r2 < 1.0, 1.0, r2)

    # kernel: 2(z' − z_obs) / r² = 2(−ζ) / r²
    gz = G_SI * rho * cell_area * np.sum(2.0 * (-zeta) / r2, axis=1)

    return gz * MGAL_PER_SI


def compute_gz_multi(x_obs, z_obs, bodies,
                     strike_half_length_km=None, n_strike: int = 201):
    """Sum 2-D gravity contributions from multiple polygon bodies."""
    x_obs    = np.asarray(x_obs, dtype=float)
    gz_total = np.zeros(len(x_obs))
    for body in bodies:
        if not getattr(body, 'visible', True) or len(body.vertices) < 3:
            continue
        gz_total += compute_gz(x_obs, z_obs, body.vertices,
                               rho=getattr(body, 'density', 0.0))
    return gz_total
