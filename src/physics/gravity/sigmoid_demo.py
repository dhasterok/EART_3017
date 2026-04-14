import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0

G = 6.674e-11  # SI units

# Robust trapezoidal integration wrapper with fallback
def trapz_yx(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    elif hasattr(np, "trapz"):
        return np.trapz(y, x, axis=axis)
    else:
        # Minimal fallback
        dx = np.diff(x)
        y_mid = 0.5 * (y[..., 1:] + y[..., :-1])
        return np.sum(y_mid * dx, axis=axis)

def lopolith_gravity(R, w, Atop, Abot, z0, drho, kmax=None, nk=4000):
    """
    Axisymmetric gravity of a sigmoid-lens thin sheet:
    t(r) = (Atop + Abot) * s(r), with s(r) = 0.5 * [1 - tanh(r/w)].
    Returns g_z(R) at the surface (SI units).
    """
    R = np.atleast_1d(R).astype(float)

    # Choose kmax based on decay scales: depth (z0) and width (w).
    # The spectral factors e^{-k z0} and 1/sinh(pi k w / 2) ensure convergence.
    if kmax is None:
        # Include several e-foldings of both filters
        k_d = 8.0 / max(z0, 1e-9)
        k_w = 8.0 / max(w, 1e-9)
        kmax = max(k_d, k_w)

    k = np.linspace(0.0, kmax, nk)
    k[0] = 1e-12  # avoid 0/0

    # Sigmoid Hankel transform approximation:
    # S(k) = \widetilde{s}(k) ≈ (π w^2 / 2) * (k w) / sinh(π k w / 2)
    S = (np.pi * w**2 / 2.0) * (k * w) / np.sinh(np.pi * k * w / 2.0)

    # Areal mass transform
    Mtilde = drho * (Atop + Abot) * S  # units: kg/m^2 in spectral domain

    # Integrand: J0(kR) * e^{-k z0} * k * Mtilde
    J = j0(np.outer(k, R))  # shape (nk, nR)
    decay = np.exp(-k[:, None] * z0)
    integrand = J * decay * (k[:, None] * Mtilde[:, None])

    gz = 2.0 * np.pi * G * trapz_yx(integrand, k, axis=0)
    return gz  # m/s^2

def sphere_gravity(R, a, z0, drho):
    R = np.atleast_1d(R).astype(float)
    return (4.0 * np.pi * G / 3.0) * drho * a**3 * z0 / (R**2 + z0**2)**1.5

if __name__ == "__main__":
    # Spatial sampling
    R = np.linspace(0, 5000, 401)

    # Lopolith parameters
    w = 800.0      # m   (controls radial width/taper)
    Atop = 300.0   # m   (positive for lens)
    Abot = 300.0   # m
    z0 = 1500.0    # m   (reference depth to mid-surface)
    drho = 300.0   # kg/m^3

    # Compute anomalies
    gz_lopo = lopolith_gravity(R, w, Atop, Abot, z0, drho)
    gz_sphere = sphere_gravity(R, a=1000.0, z0=z0, drho=drho)

    # Plot
    plt.figure(figsize=(7,4.5))
    plt.plot(R, gz_lopo*1e5, label="Sigmoid lopolith", lw=2)  # convert to mGal (approx: 1 m/s^2 = 1e5 mGal)
    plt.plot(R, gz_sphere*1e5, label="Sphere", lw=2, ls="--")
    plt.xlabel("Radius (m)")
    plt.ylabel("g$_z$ (mGal)")
    plt.title("Sigmoid Lopolith vs Sphere (no noise)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()