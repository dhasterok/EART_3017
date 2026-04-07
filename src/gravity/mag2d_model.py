"""
mag2d_model.py
--------------
2-D total-field magnetic anomaly for a polygon of infinite along-strike extent.

Reference
---------
Blakely, R. J. (1995). Potential Theory in Gravity and Magnetic Applications.
    Cambridge University Press, §6.3.
Won, I. J. & Bevis, M. (1987). Computing the gravitational and magnetic
    anomalies due to a polygon.  Geophysics, 52(2), 232–238.

Formula
-------
For each polygon edge from P_k = (x_k, z_k) to P_{k+1} (obs at origin):

    a      = x_{k+1} − x_k
    b      = z_{k+1} − z_k
    L      = sqrt(a² + b²)
    cos φ  = a / L,  sin φ = b / L
    p      = (x_{k+1} z_k − x_k z_{k+1}) / L   [signed perp. distance]
    Δθ     = atan2(z_{k+1}, x_{k+1}) − atan2(z_k, x_k)   [wrapped ±π]
    ln r   = ln(r_{k+1} / r_k)

    F_k  =  p · (cos φ · Δθ − sin φ · ln r)   [= Talwani gravity integral]
    H_k  =  p · (sin φ · Δθ + cos φ · ln r)   [magnetic counterpart]

Anomalous field components for magnetization (Mx, Mz):

    Bx = (μ₀/2π) · (Mx · ΣF_k − Mz · ΣH_k)
    Bz = (μ₀/2π) · (Mz · ΣF_k + Mx · ΣH_k)

Total-field anomaly projected onto Earth's field direction:

    ΔT = Bx cos(IE) + Bz sin(IE)   (T → ×10⁹ → nT)

Induced-magnetisation relations (SI):

    J  = χ · F / μ₀   (A m⁻¹),   F in Tesla
    Mx = J cos(IE),   Mz = J sin(IE)

Conventions
-----------
    x   horizontal distance  (km)
    z   depth                (km, positive downward)
    χ   susceptibility       (SI, dimensionless)
    F   Earth's field        (nT)
    IE  inclination          (degrees, positive downward from horizontal)
    ΔT  total-field anomaly  (nT)
"""

import math
import numpy as np


MU0       = 4.0 * math.pi * 1e-7   # T·m A⁻¹
M_PER_KM  = 1_000.0                # m km⁻¹


# ---------------------------------------------------------------------------
# Single-edge contribution
# ---------------------------------------------------------------------------

def _edge_mag(x1: float, z1: float,
              x2: float, z2: float) -> tuple[float, float]:
    """
    Blakely / Won–Bevis edge integrals F_k and H_k (obs at origin).

    Returns
    -------
    (F_k, H_k) : (float, float)
        F_k is identical to the Talwani gravity integral I_k.
        Degenerate cases return (0, 0).
    """
    EPS = 1.0e-8

    a  = x2 - x1
    b  = z2 - z1
    L2 = a*a + b*b
    if L2 < EPS:
        return 0.0, 0.0

    L  = math.sqrt(L2)
    r1 = math.sqrt(x1*x1 + z1*z1)
    r2 = math.sqrt(x2*x2 + z2*z2)
    if r1 < EPS or r2 < EPS:
        return 0.0, 0.0

    p = (x2*z1 - x1*z2) / L
    if abs(p) < EPS:
        return 0.0, 0.0

    cos_phi = a / L
    sin_phi = b / L

    theta1 = math.atan2(z1, x1)
    theta2 = math.atan2(z2, x2)
    dtheta = theta2 - theta1
    if dtheta >  math.pi:  dtheta -= 2.0 * math.pi
    if dtheta < -math.pi:  dtheta += 2.0 * math.pi

    lnr = math.log(r2 / r1)

    Fk = p * (cos_phi * dtheta - sin_phi * lnr)   # gravity-like
    Hk = p * (sin_phi * dtheta + cos_phi * lnr)   # magnetic counterpart

    return Fk, Hk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_bt(x_obs, vertices, susceptibility: float,
               F_nT: float, IE_deg: float) -> np.ndarray:
    """
    Total-field magnetic anomaly of one 2-D polygon (induced magnetisation).

    Parameters
    ----------
    x_obs         : array-like  -- observation x-positions (km)
    vertices      : list of [x, z]  -- polygon vertices (km); z ≥ 0 downward
    susceptibility: float  -- dimensionless SI susceptibility χ
    F_nT          : float  -- Earth's field total intensity (nT)
    IE_deg        : float  -- Earth's field inclination (degrees, + downward)

    Returns
    -------
    ΔT : ndarray  -- total-field anomaly (nT)
    """
    x_obs = np.asarray(x_obs, dtype=float)
    n = len(vertices)
    if n < 3:
        return np.zeros(len(x_obs))

    # Convert vertices to metres
    vx = np.array([v[0] for v in vertices], dtype=float) * M_PER_KM
    vz = np.array([v[1] for v in vertices], dtype=float) * M_PER_KM

    # Induced magnetisation
    F_T  = F_nT * 1.0e-9                    # nT → T
    J    = susceptibility * F_T / MU0       # A m⁻¹  (valid for χ ≪ 1)
    IE   = math.radians(IE_deg)
    Mx   = J * math.cos(IE)                 # horizontal (along profile)
    Mz   = J * math.sin(IE)                 # vertical (downward)

    # Earth-field unit vector (for total-field projection)
    ex   = math.cos(IE)
    ez   = math.sin(IE)

    dt = np.zeros(len(x_obs))

    for k, xp_km in enumerate(x_obs):
        xp_m  = xp_km * M_PER_KM
        sum_F = 0.0
        sum_H = 0.0
        for i in range(n):
            j = (i + 1) % n
            Fk, Hk = _edge_mag(vx[i] - xp_m, vz[i],
                                vx[j] - xp_m, vz[j])
            sum_F += Fk
            sum_H += Hk

        Bx = (MU0 / (2.0 * math.pi)) * ( Mx * sum_F - Mz * sum_H)
        Bz = (MU0 / (2.0 * math.pi)) * ( Mz * sum_F + Mx * sum_H)

        dt[k] = (Bx * ex + Bz * ez) * 1.0e9    # T → nT

    return dt


def compute_bt_multi(x_obs, bodies, F_nT: float, IE_deg: float) -> np.ndarray:
    """
    Sum ΔT contributions from multiple PolygonBody-like objects.

    Parameters
    ----------
    x_obs   : array-like  -- observation x-positions (km)
    bodies  : iterable of objects with .vertices (list), .susceptibility (float),
              and .visible (bool)
    F_nT    : float  -- Earth's field total intensity (nT)
    IE_deg  : float  -- Earth's field inclination (degrees)

    Returns
    -------
    dt_total : ndarray  -- total-field anomaly (nT)
    """
    x_obs    = np.asarray(x_obs, dtype=float)
    dt_total = np.zeros(len(x_obs))
    for body in bodies:
        if getattr(body, 'visible', True) and len(body.vertices) >= 3:
            if body.susceptibility != 0.0:
                dt_total += compute_bt(
                    x_obs, body.vertices,
                    body.susceptibility, F_nT, IE_deg,
                )
    return dt_total
