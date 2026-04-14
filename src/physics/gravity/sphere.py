from __future__ import annotations
import numpy as np

G_CONST = 6.67430e-11
FOUR_PI_OVER_3 = 4.0 * np.pi / 3.0

"""
Buried sphere forward and Jacobian.
Parameters p = [z, R, drho] with z (depth, m), R (radius, m), drho (kg/m^3).
Forward for vertical gravity at horizontal radius r:
    g(r) = G * (4/3*pi*R^3*drho) * z / (r^2 + z^2)^(3/2)
"""

def forward_sphere_r(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    z, R, drho = p
    M = FOUR_PI_OVER_3 * (R**3) * drho
    rr2 = r*r + z*z
    return G_CONST * M * z / (rr2 ** 1.5)


def forward_profile(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    r = np.abs(x)
    return forward_sphere_r(r, p)


def forward_map(X: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
    r = np.sqrt(X*X + Y*Y)
    return forward_sphere_r(r, p)


def jacobian_r(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    z, R, drho = p
    rr2 = r*r + z*z
    rr2_3_2 = rr2 ** 1.5
    rr2_5_2 = rr2 ** 2.5
    A = G_CONST * FOUR_PI_OVER_3
    B = z / rr2_3_2
    dB_dz = (r*r - 2.0*z*z) / rr2_5_2
    dg_dz    = A * (R**3) * drho * dB_dz
    dg_dR    = 3.0 * A * (R**2) * drho * B
    dg_ddrho = A * (R**3) * B
    return np.column_stack([dg_dz, dg_dR, dg_ddrho])


def add_noise(g: np.ndarray, sigma: float, rng: np.random.Generator|None=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return g + rng.normal(0.0, sigma, size=g.shape)
