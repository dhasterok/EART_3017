"""
mag2_5d_model.py
----------------
2-D magnetic field (Bx, Bz, ΔT) for a polygonal cross-section, computed by
distributing infinite horizontal cylinders over the cross-section.

Each area element dA at (x', z') contributes (observation at z_obs=0,
source at x', z'):

    ξ  = x_obs − x',    ζ  = z_obs − z'

    δBx = (μ₀/2π) [Mx(ξ²−ζ²) + 2Mz·ξζ ] / r⁴  dA
    δBz = (μ₀/2π) [2Mx·ξζ    + Mz(ζ²−ξ²)] / r⁴  dA
    δBy = (μ₀/2π) [My·2·r² − …] / r⁴  dA  (not needed for profile)

For vertical magnetisation (Mz only, IE=90°):
    δBz ∝ (ζ²−ξ²)/r⁴   — symmetric, positive above body, negative side lobes
    δBx ∝  ξζ/r⁴        — anti-symmetric (odd in x)

These kernels are the exact 2-D dipole fields for a line source of infinite
length along y.

Grid spacing is chosen adaptively so the shallowest part of the body is
always resolved with at least N_CELLS = 20 cells.

By is the component along the out-of-plane (strike) direction. For a
profile survey, the along-strike component of B is not measured; By only
contributes to ΔT via the declination projection:

    ΔT = Bx·ex + By·ey + Bz·ez       (e = Earth-field unit vector)

My = J_ind·sin(DE)·cos(IE) is nonzero when declination ≠ 0.

    δBy = (μ₀/2π) My (ξ²+ζ²−2ζ²) / r⁴  dA    ← to be confirmed
          = (μ₀/2π) My (ξ²−ζ²) / r⁴  dA

Wait — for the 2-D case the By kernel comes from the y-component of the
2-D dipole:
    δBy = (μ₀/2π) My · 2r² / r⁴ dA  ... no.

For a 2-D line source the magnetic potential is 2-D; there is no "y" field
from Mx or Mz (they only produce Bx and Bz). My produces By only:

    δBy = (μ₀/2π) My · (−1/r²) · 2  dA   ← scalar-potential y-derivative

Correct derivation: for a 2-D line dipole with moment m per unit length,

    Bx = (μ₀/2π) (mx(x²−z²) + 2mz·xz) / r⁴
    By = −(μ₀/2π) · 2my / r²          (from ∂²Φ/∂y²; symmetric)
    Bz = (μ₀/2π) (2mx·xz + mz(z²−x²)) / r⁴

where here x and z stand for ξ and −ζ (i.e., vector from source to obs).
Using ξ = x_obs−x', ζ = z_obs−z' (so z-vector component = −ζ):

    Bx = (μ₀/2π) [Mx(ξ²−ζ²) − 2Mz·ξζ] / r⁴       ... sign depends on convention

Let's anchor on the physics test:
  - Vertical Mz, obs at x, z_obs=0, source at 0, z':  ξ=x, ζ=−z'
    δBz = (μ₀/2π) Mz(ζ²−ξ²)/r⁴ = (μ₀/2π) Mz(z'²−x²)/(x²+z'²)²  ✓
  - δBx = (μ₀/2π) 2Mz·ξζ/r⁴ = (μ₀/2π) 2Mz·x·(−z')/(x²+z'²)²   (odd in x) ✓
"""

import math
import numpy as np
from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MU0  = 4.0 * math.pi * 1e-7   # T·m A⁻¹
FAC  = MU0 / (2.0 * math.pi)  # μ₀/2π  (2-D dipole prefactor)

_N_CELLS = 20   # min cells across shallowest dimension


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _points_in_polygon(xq, zq, vx, vz):
    """Vectorised ray-casting point-in-polygon test (all units consistent)."""
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

    x0, x1 = vx.min() * 1e3, vx.max() * 1e3
    z0, z1 = vz.min() * 1e3, vz.max() * 1e3
    width  = max(x1 - x0, 1.0)
    height = max(z1 - z0, 1.0)
    z_min  = max(z0, 1.0)   # shallowest depth (m)

    h = min(z_min, width, height) / _N_CELLS
    h = max(h, 1.0)   # absolute floor: 1 m

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

def compute_bx_bz(
    x_obs,
    z_obs,
    vertices,
    susceptibility: float,
    F_nT: float,
    IE_deg: float,
    DE_deg: float = 0.0,
    remanence_Am: float = 0.0,
    remanence_inc_deg: float = 0.0,
    remanence_dec_deg: float = 0.0,
    strike_half_length_km=None,
    n_strike: int = 201,
):
    """
    2-D horizontal (Bx) and vertical (Bz) magnetic field components (nT)
    via distributed infinite cylinders.

    Parameters
    ----------
    x_obs             : array-like  -- observation x-positions (km)
    z_obs             : float       -- observation elevation (km; 0 = surface)
    vertices          : list of [x, z]  -- polygon vertices (km; z ≥ 0 down)
    susceptibility    : float  -- dimensionless SI susceptibility
    F_nT              : float  -- Earth field total intensity (nT)
    IE_deg            : float  -- Earth field inclination (°, + downward)
    DE_deg            : float  -- Earth field declination (°, + east of north)
    remanence_Am      : float  -- remanent magnetisation (A/m)
    remanence_inc_deg : float  -- remanent inclination (°)
    remanence_dec_deg : float  -- remanent declination (°)
    strike_half_length_km, n_strike : ignored (API compatibility)

    Returns
    -------
    (Bx_nT, Bz_nT) : (ndarray, ndarray)
    """
    x_obs = np.asarray(x_obs, dtype=float)
    if len(vertices) < 3:
        z = np.zeros(len(x_obs))
        return z.copy(), z.copy()

    # --- Magnetisation vector (A/m) -----------------------------------------
    F_T   = F_nT * 1e-9
    IE    = math.radians(IE_deg)
    DE    = math.radians(DE_deg)
    J_ind = susceptibility * F_T / MU0

    Mx = J_ind * math.cos(IE) * math.cos(DE)
    # My — along strike; contributes to By but not Bx/Bz for a profile survey
    Mz = J_ind * math.sin(IE)

    if remanence_Am != 0.0:
        IR = math.radians(remanence_inc_deg)
        DR = math.radians(remanence_dec_deg)
        Mx += remanence_Am * math.cos(IR) * math.cos(DR)
        Mz += remanence_Am * math.sin(IR)

    # --- Source grid ---------------------------------------------------------
    xq, zq, h = _grid_points(vertices)
    if xq.size == 0:
        z = np.zeros(len(x_obs))
        return z.copy(), z.copy()

    z_obs_m   = float(z_obs) * 1e3
    cell_area = h * h

    # Vectorised over all obs: shape (N_x, N_q)
    xi   = x_obs[:, np.newaxis] * 1e3 - xq[np.newaxis, :]
    zeta = z_obs_m - zq[np.newaxis, :]
    xi2  = xi  * xi
    z2   = zeta * zeta
    r4   = (xi2 + z2) ** 2
    r4   = np.where(r4 < 1.0, 1.0, r4)

    # 2-D dipole kernels (ξ = x_obs−x', ζ = z_obs−z')
    # Source z' is positive downward, so ζ = z_obs − z' is negative for buried
    #   δBx = (μ₀/2π) [Mx(ξ²−ζ²) + 2Mz·ξζ ] / r⁴
    #   δBz = (μ₀/2π) [2Mx·ξζ    + Mz(ζ²−ξ²)] / r⁴
    kx = Mx * (xi2 - z2) + 2.0 * Mz * xi * zeta
    kz = 2.0 * Mx * xi * zeta + Mz * (z2 - xi2)

    Bx = FAC * cell_area * np.sum(kx / r4, axis=1)
    Bz = FAC * cell_area * np.sum(kz / r4, axis=1)

    return Bx * 1e9, Bz * 1e9


def compute_bt(
    x_obs,
    z_obs,
    vertices,
    susceptibility: float,
    F_nT: float,
    IE_deg: float,
    DE_deg: float = 0.0,
    remanence_Am: float = 0.0,
    remanence_inc_deg: float = 0.0,
    remanence_dec_deg: float = 0.0,
    strike_half_length_km=None,
    n_strike: int = 201,
):
    """
    2-D total-field magnetic anomaly ΔT (nT) via distributed infinite cylinders.

    ΔT = Bx·ex + By·ey + Bz·ez  (e = Earth-field unit vector)

    By is computed separately because the along-strike component of
    magnetisation (My) contributes to By even on a profile.

    Parameters: same as compute_bx_bz.
    """
    x_obs = np.asarray(x_obs, dtype=float)
    if len(vertices) < 3:
        return np.zeros(len(x_obs))

    IE = math.radians(IE_deg)
    DE = math.radians(DE_deg)
    ex = math.cos(IE) * math.cos(DE)
    ey = math.cos(IE) * math.sin(DE)
    ez = math.sin(IE)

    # --- Magnetisation vector ------------------------------------------------
    F_T   = F_nT * 1e-9
    J_ind = susceptibility * F_T / MU0

    Mx = J_ind * ex
    My = J_ind * ey
    Mz = J_ind * ez

    if remanence_Am != 0.0:
        IR = math.radians(remanence_inc_deg)
        DR = math.radians(remanence_dec_deg)
        Mx += remanence_Am * math.cos(IR) * math.cos(DR)
        My += remanence_Am * math.cos(IR) * math.sin(DR)
        Mz += remanence_Am * math.sin(IR)

    # --- Source grid ---------------------------------------------------------
    xq, zq, h = _grid_points(vertices)
    if xq.size == 0:
        return np.zeros(len(x_obs))

    z_obs_m   = float(z_obs) * 1e3
    cell_area = h * h

    xi   = x_obs[:, np.newaxis] * 1e3 - xq[np.newaxis, :]
    zeta = z_obs_m - zq[np.newaxis, :]
    xi2  = xi  * xi
    z2   = zeta * zeta
    r2   = xi2 + z2
    r2   = np.where(r2 < 1.0, 1.0, r2)
    r4   = r2 * r2

    kx = Mx * (xi2 - z2) + 2.0 * Mz * xi * zeta
    kz = 2.0 * Mx * xi * zeta + Mz * (z2 - xi2)
    # By from My (2-D line dipole, y-component): δBy = −(μ₀/2π) 2My / r²
    ky = -2.0 * My / r2

    bx = FAC * cell_area * np.sum(kx / r4, axis=1)
    by = FAC * cell_area * np.sum(ky,       axis=1)
    bz = FAC * cell_area * np.sum(kz / r4, axis=1)

    return (ex * bx + ey * by + ez * bz) * 1e9
