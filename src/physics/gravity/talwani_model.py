"""
talwani_model.py
----------------
2D gravity forward model -- Blakely (1995) formulation of the Talwani method.

Computes the vertical gravity anomaly (gz) at surface points due to a
2-D polygonal body of infinite along-strike extent.

Formula
-------
For a polygon with N vertices P_k = (x_k, z_k) relative to the observation
point (shifted so observation is at origin), the vertical gravity is:

    gz = 2 G ρ  Σ_{k=1}^{N}  I_k

where for the edge from P_k to P_{k+1} (indices mod N):

    a    = x_{k+1} - x_k
    b    = z_{k+1} - z_k
    L    = sqrt(a² + b²)          [edge length]
    p    = (x_{k+1} z_k - x_k z_{k+1}) / L   [signed perp. distance]
    cosφ = a / L                  [edge direction cosines]
    sinφ = b / L
    θ_k  = atan2(z_k, x_k)
    r_k  = sqrt(x_k² + z_k²)

    I_k  = p · [ cosφ · (θ_{k+1} − θ_k) − sinφ · ln(r_{k+1}/r_k) ]

Degenerate cases (zero-length edge, or edge through obs. point) return 0.

References
----------
Blakely, R. J. (1995). Potential Theory in Gravity and Magnetic Applications.
    Cambridge University Press, pp. 168--169.
Talwani, M., Worzel, J. L., and Landisman, M. (1959). Rapid gravity
    computations for two-dimensional bodies. J. Geophys. Res., 64, 49--59.

Conventions
-----------
    x   : horizontal distance  (km)
    z   : depth                (km, positive *downward*)
    rho : density contrast     (kg/m³)
    gz  : vertical gravity     (mGal)
"""

import math
import numpy as np


G_SI       = 6.674e-11    # m³ kg⁻¹ s⁻²
M_PER_KM   = 1_000.0
MGAL_PER_SI = 1.0e5       # 1 m/s² = 1e5 mGal


# ---------------------------------------------------------------------------
# Single-edge contribution (all coordinates in metres, obs. at origin)
# ---------------------------------------------------------------------------

def _edge_gz(x1: float, z1: float, x2: float, z2: float) -> float:
    """
    Blakely (1995) contribution of one polygon edge to gz at the origin.

    Parameters
    ----------
    x1, z1 : start vertex (metres, relative to observation point; z ≥ 0)
    x2, z2 : end   vertex (metres, relative to observation point; z ≥ 0)

    Returns
    -------
    float  -- dimensionless; multiply by 2·G·ρ to get gz in m/s².
    """
    EPS = 1.0e-8

    a = x2 - x1
    b = z2 - z1
    L2 = a * a + b * b

    if L2 < EPS:          # zero-length edge
        return 0.0

    L = math.sqrt(L2)

    r1 = math.sqrt(x1 * x1 + z1 * z1)
    r2 = math.sqrt(x2 * x2 + z2 * z2)

    if r1 < EPS or r2 < EPS:   # obs. point at a vertex
        return 0.0

    # Perpendicular distance from origin to the edge line (signed)
    # p = (x2*z1 - x1*z2) / L
    p = (x2 * z1 - x1 * z2) / L

    if abs(p) < EPS:      # edge line passes through obs. point
        return 0.0

    cos_phi = a / L       # edge direction cosines
    sin_phi = b / L

    theta1 = math.atan2(z1, x1)
    theta2 = math.atan2(z2, x2)

    dtheta = theta2 - theta1
    # Wrap to (−π, π]
    if dtheta >  math.pi:  dtheta -= 2.0 * math.pi
    if dtheta < -math.pi:  dtheta += 2.0 * math.pi

    ln_r = math.log(r2 / r1)   # ln(r2/r1)

    return p * (cos_phi * dtheta - sin_phi * ln_r)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gz(x_obs, vertices, rho: float) -> np.ndarray:
    """
    Vertical gravity at surface observation points due to one 2-D polygon.

    Parameters
    ----------
    x_obs    : array-like  -- observation x-positions (km)
    vertices : list of [x, z]  -- polygon vertices (km); z positive downward
    rho      : float  -- density contrast (kg/m³)

    Returns
    -------
    gz : ndarray  -- vertical gravity anomaly (mGal)
    """
    x_obs = np.asarray(x_obs, dtype=float)
    n = len(vertices)

    if n < 3:
        return np.zeros(x_obs.shape)

    # Convert to metres
    vx = np.array([v[0] for v in vertices], dtype=float) * M_PER_KM
    vz = np.array([v[1] for v in vertices], dtype=float) * M_PER_KM

    gz = np.zeros(len(x_obs), dtype=float)

    for k, xp_km in enumerate(x_obs):
        xp_m = xp_km * M_PER_KM
        total = 0.0
        for i in range(n):
            j = (i + 1) % n
            total += _edge_gz(vx[i] - xp_m, vz[i],
                              vx[j] - xp_m, vz[j])
        gz[k] = 2.0 * G_SI * rho * total * MGAL_PER_SI

    return gz


def compute_gz_multi(x_obs, bodies) -> np.ndarray:
    """
    Sum gz contributions from multiple PolygonBody objects.

    Parameters
    ----------
    x_obs  : array-like  -- observation x-positions (km)
    bodies : iterable of objects with .vertices (list) and .density (float)
             and .visible (bool)

    Returns
    -------
    gz_total : ndarray  -- total vertical gravity anomaly (mGal)
    """
    x_obs = np.asarray(x_obs, dtype=float)
    gz_total = np.zeros(len(x_obs), dtype=float)

    for body in bodies:
        if getattr(body, 'visible', True) and len(body.vertices) >= 3:
            gz_total += compute_gz(x_obs, body.vertices, body.density)

    return gz_total
