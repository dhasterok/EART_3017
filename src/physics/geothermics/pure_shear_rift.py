"""
pure_shear_rift.py

Analytical temperature and heat-flow for McKenzie (1978) pure shear rifting,
following the Royden et al. (1980) thinning-factor convention:

    gamma = 1 - L'/L   (0 = no rifting, ~1 = complete lithosphere removal)

Working units (match rift.m / terms.m):
  z       km
  t       Ma
  kappa   km² Ma⁻¹
  L       km
  k       W m⁻¹ K⁻¹     — k * ΔT / L_km  gives mW m⁻² directly
  T       °C
  q       mW m⁻²
"""

import numpy as np


def fourier_coefs(n, gamma, Ts, Tm):
    """
    Fourier coefficients a_n for the post-rift temperature anomaly.

        a_n = 2 ΔT (-1)^(n+1) sin(nπγ) / [n²π²(1-γ)]

    Parameters
    ----------
    n     : array-like  term indices (1-based integers)
    gamma : float       thinning factor (0 < γ < 1)
    Ts    : float       surface temperature [°C]
    Tm    : float       mantle temperature  [°C]

    Returns
    -------
    an : ndarray  shape (N,)  [°C]
    """
    n = np.asarray(n, float)
    dT = Tm - Ts
    return (2 * dT * (-1) ** (n + 1) * np.sin(n * np.pi * gamma)
            / (n ** 2 * np.pi ** 2 * (1.0 - gamma)))


def equilibrium_temperature(z, Ts, Tm, L):
    """Linear equilibrium temperature T(z, t→∞) [°C]."""
    return Ts + (Tm - Ts) * np.asarray(z, float) / L


def initial_temperature(z, gamma, Ts, Tm, L):
    """
    Post-rift initial temperature T(z, t=0⁺) [°C].

    Steeper gradient (Tm - Ts) / (L (1-γ)) for z < L(1-γ),
    then T = Tm for z ≥ L(1-γ).
    """
    z = np.asarray(z, float).ravel()
    Ti = np.full_like(z, float(Tm))
    mask = z < L * (1.0 - gamma)
    Ti[mask] = Ts + (Tm - Ts) * z[mask] / (L * (1.0 - gamma))
    Ti[0] = Ts
    return Ti


def temperature(z, t, gamma, kappa, Ts, Tm, L, N=20):
    """
    Temperature field T(z, t) [°C].

        T = Tₑ + Σₙ aₙ sin(nπz/L) exp(−n²π²κt/L²)

    Parameters
    ----------
    z     : array (nz,)  depths [km]
    t     : array (nt,)  times  [Ma]
    gamma : float        thinning factor
    kappa : float        thermal diffusivity [km² Ma⁻¹]
    Ts    : float        surface temperature [°C]
    Tm    : float        mantle temperature  [°C]
    L     : float        lithospheric thickness [km]
    N     : int          number of Fourier terms

    Returns
    -------
    T : ndarray (nz, nt)  [°C]
    """
    z = np.asarray(z, float).ravel()
    t = np.asarray(t, float).ravel()
    n = np.arange(1, N + 1, dtype=float)

    an = fourier_coefs(n, gamma, Ts, Tm)           # (N,)
    Te = equilibrium_temperature(z, Ts, Tm, L)     # (nz,)

    # zterm[k, j] = sin(n_k π z_j / L)
    zterm = np.sin(np.outer(n * np.pi / L, z))     # (N, nz)
    # tterm[k, i] = exp(−n_k² π² κ t_i / L²)
    tterm = np.exp(np.outer(-n ** 2 * np.pi ** 2 * kappa / L ** 2, t))  # (N, nt)

    # series[j, i] = Σ_k  an[k] * zterm[k,j] * tterm[k,i]
    series = (an[:, None, None] * zterm[:, :, None] * tterm[:, None, :]).sum(axis=0)  # (nz, nt)

    T = Te[:, None] + series
    T[0, :] = Ts     # enforce surface BC
    return T


def heat_flow(t, gamma, kappa, Ts, Tm, L, k=2.5, N=20):
    """
    Surface heat flow q(t) [mW m⁻²].

        q = k ΔT / L × (1 + 2 Σₙ nπ âₙ exp(−n²π²κt/L²))

    where âₙ = (-1)^(n+1) sin(nπγ) / (n²π²(1-γ))  (normalised; no 2ΔT factor).
    With L in km and k in W m⁻¹ K⁻¹, the product k ΔT / L gives mW m⁻² directly.

    Parameters
    ----------
    t     : array (nt,)  times [Ma]
    gamma : float        thinning factor
    kappa : float        thermal diffusivity [km² Ma⁻¹]
    Ts    : float        surface temperature [°C]
    Tm    : float        mantle temperature  [°C]
    L     : float        lithospheric thickness [km]
    k     : float        thermal conductivity [W m⁻¹ K⁻¹]
    N     : int          number of Fourier terms

    Returns
    -------
    q : ndarray (nt,)  [mW m⁻²]
    """
    t = np.asarray(t, float).ravel()
    n = np.arange(1, N + 1, dtype=float)
    dT = Tm - Ts

    # Normalised Fourier coefficients (rift.m convention, without 2*dT)
    an_hat = ((-1) ** (n + 1) * np.sin(n * np.pi * gamma)
              / (n ** 2 * np.pi ** 2 * (1.0 - gamma)))
    tterm = np.exp(np.outer(-n ** 2 * np.pi ** 2 * kappa / L ** 2, t))  # (N, nt)

    q = k * dT / L * (1.0 + 2.0 * (n[:, None] * np.pi * an_hat[:, None] * tterm).sum(axis=0))
    return q   # mW m⁻² (k*ΔT/L_km ≡ mW m⁻²)


def decay_curves(t, kappa, L, N=20):
    """
    Amplitude decay factors exp(−n²π²κt/L²) for each mode n.

    Parameters
    ----------
    t     : array (nt,)  times [Ma]
    kappa : float        thermal diffusivity [km² Ma⁻¹]
    L     : float        lithospheric thickness [km]
    N     : int          number of modes

    Returns
    -------
    dec : ndarray (N, nt)
    """
    t = np.asarray(t, float).ravel()
    n = np.arange(1, N + 1, dtype=float)
    return np.exp(np.outer(-n ** 2 * np.pi ** 2 * kappa / L ** 2, t))


def fourier_terms(z, gamma, Ts, Tm, L, N=20):
    """
    Individual Fourier terms and their cumulative sums at t = 0.

    term_k(z) = aₙ sin(nπz/L)

    Parameters
    ----------
    z     : array (nz,)  depths [km]
    gamma : float        thinning factor
    Ts    : float        surface temperature [°C]
    Tm    : float        mantle temperature  [°C]
    L     : float        lithospheric thickness [km]
    N     : int          number of terms

    Returns
    -------
    an       : (N,)    Fourier coefficients [°C]
    terms_z  : (N, nz) individual terms [°C]
    cumsum_z : (N, nz) cumulative sums  [°C]  (cumsum_z[k] = Σ_{j=1}^{k+1} term_j)
    """
    z = np.asarray(z, float).ravel()
    n = np.arange(1, N + 1, dtype=float)
    an = fourier_coefs(n, gamma, Ts, Tm)
    zterm = np.sin(np.outer(n * np.pi / L, z))   # (N, nz)
    terms_z = an[:, None] * zterm                 # (N, nz)
    cumsum_z = np.cumsum(terms_z, axis=0)         # (N, nz)
    return an, terms_z, cumsum_z
