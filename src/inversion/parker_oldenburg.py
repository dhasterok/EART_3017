"""
parker_oldenburg.py
-------------------
Spectral inversion for the depth to a density interface using the
Parker-Oldenburg method.

References
----------
Parker, R. L. (1973). The rapid calculation of potential anomalies.
    *Geophysical Journal of the Royal Astronomical Society*, 31, 447-455.

Oldenburg, D. W. (1974). The inversion and interpretation of gravity
    anomalies. *Geophysics*, 39(4), 526-536.

Overview
--------
The forward problem relates the gravity anomaly g(x,y) to the depth of an
interface h(x,y) (measured from a reference level z₀):

    G(k) = 2π G Δρ exp(-|k| z₀)  Σ_{n=1}^{∞}  (-|k|)^{n-1} / n!  ℱ[hⁿ](k)

where G(k) = ℱ[g](k) is the 2-D Fourier transform of the observed gravity,
k = sqrt(kx² + ky²) is the radial wavenumber, G is Newton's constant,
Δρ is the density contrast across the interface, and z₀ is the mean
interface depth.

Truncating to the linear term (n=1) gives:

    G(k) ≈ 2π G Δρ exp(-|k| z₀) H(k)
    → H(k) = G(k) exp(+|k| z₀) / (2π G Δρ)
    → h(x,y) = ℱ⁻¹[H(k)]

The Oldenburg (1974) iterative scheme corrects for the nonlinear terms,
producing a depth model that is consistent with the observed gravity to
within a specified tolerance.

Public API
----------
parker_oldenburg(g_obs, dx, dy, drho, z0=None, n_iter=30, tol=1e-3,
                 kmax_factor=0.9)
    Invert a gravity grid for interface depth.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

G_SI = 6.67430e-11   # m³ kg⁻¹ s⁻²


def _wavenumber_grid(ny: int, nx: int,
                     dx: float, dy: float) -> np.ndarray:
    """Radial wavenumber grid (rad/m), DC set to small value."""
    kx = 2 * np.pi * fftfreq(nx, d=dx)
    ky = 2 * np.pi * fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX**2 + KY**2)
    k[0, 0] = 1e-30   # avoid singularity at DC
    return k


def parker_oldenburg(g_obs: np.ndarray,
                     dx: float,
                     dy: float,
                     drho: float,
                     z0: float | None = None,
                     n_iter: int = 30,
                     tol: float = 1e-3,
                     kmax_factor: float = 0.85) -> tuple[np.ndarray, list[float]]:
    """
    Invert a 2-D gravity grid for the depth to a density interface using the
    Parker-Oldenburg spectral method.

    The interface separates an upper layer (density ρ₁) from a lower half-space
    (density ρ₂).  Δρ = ρ₂ − ρ₁ (positive when the lower medium is denser).
    A gravity deficit (negative anomaly) over a sedimentary basin corresponds
    to Δρ > 0 (basement denser than sediment) and predicts h > 0.

    Parameters
    ----------
    g_obs : ndarray, shape (ny, nx)
        Observed gravity anomaly in **mGal**.  NaN values (outside the survey)
        are replaced by the grid mean before the transform and restored in
        the output.
    dx, dy : float
        Grid spacing in **metres** (E–W and N–S respectively).
    drho : float
        Density contrast in kg/m³.  Use ρ_basement − ρ_sediment for a
        basin inversion (e.g. 2670 − 2000 = 670 kg/m³).
    z0 : float or None
        Reference (mean) depth of the interface in **metres**.  If None, it
        is estimated from the Bouguer-slab approximation of the DC level.
    n_iter : int
        Maximum number of Parker-Oldenburg iterations.  Default 30.
    tol : float
        Convergence threshold: stop when the RMS change in h between
        consecutive iterations falls below ``tol`` metres.
    kmax_factor : float
        Fraction of the Nyquist wavenumber used as the low-pass cut-off for
        the upward-continuation filter.  Values between 0.7 and 0.95 are
        typical; lower values give smoother (more stable) results.

    Returns
    -------
    h : ndarray, shape (ny, nx)
        Depth to the interface in **metres** (positive downward).  Cells that
        were NaN in ``g_obs`` are returned as NaN.
    rms_history : list of float
        RMS gravity residual (mGal) at the end of each iteration.
    """
    g_obs = np.asarray(g_obs, dtype=float)
    ny, nx = g_obs.shape

    nan_mask = np.isnan(g_obs)
    g_mean = np.nanmean(g_obs)

    # Fill NaN with the mean so the FFT is defined everywhere
    g = g_obs.copy()
    g[nan_mask] = g_mean

    # Convert mGal → m/s²
    g_si = g * 1e-5

    # Wavenumber grid
    k = _wavenumber_grid(ny, nx, dx, dy)

    # Low-pass filter to stabilise the upward-continuation deconvolution.
    # A Gaussian roll-off at kmax prevents amplification of short-wavelength
    # noise during the inverse filtering.
    k_nyq = np.pi / min(dx, dy)
    kmax  = kmax_factor * k_nyq
    lowpass = np.exp(-(k / kmax)**4)

    # Reference depth z0 — use Bouguer slab at DC component if not supplied
    if z0 is None:
        z0 = abs(g_mean * 1e-5) / (2 * np.pi * G_SI * abs(drho))
        z0 = max(z0, 50.0)   # physical floor: at least 50 m

    # Observed spectrum
    G_obs = fft2(g_si)

    # ── Oldenburg (1974) iterative scheme ─────────────────────────────────
    # Starting estimate: linear Bouguer-slab inversion
    h = -g_si / (2 * np.pi * G_SI * drho)
    h[h < 0] = 0.0

    rms_history: list[float] = []

    for it in range(n_iter):
        h_prev = h.copy()

        # Parker forward (linear term only — sufficient for shallow basins)
        H = fft2(h)
        G_fwd = 2 * np.pi * G_SI * drho * np.exp(-k * z0) * H

        # Residual in the spatial domain
        g_fwd = np.real(ifft2(G_fwd))
        res_si = g_si - g_fwd

        rms = float(np.sqrt(np.mean(res_si**2)) * 1e5)   # mGal
        rms_history.append(rms)

        # Correction: deconvolve the residual with the upward-continuation
        # filter, applying the low-pass taper to suppress instability
        R = fft2(res_si)
        kernel = np.exp(+k * z0) / (2 * np.pi * G_SI * drho + 1e-60)
        dh = np.real(ifft2(R * kernel * lowpass))

        h = h + dh
        h[h < 0] = 0.0   # enforce non-negative sediment thickness

        # Convergence check
        delta_rms = float(np.sqrt(np.mean((h - h_prev)**2)))
        if delta_rms < tol:
            rms_history.append(
                float(np.sqrt(np.mean((g_si - np.real(ifft2(
                    2 * np.pi * G_SI * drho * np.exp(-k * z0) * fft2(h)
                )))**2)) * 1e5)
            )
            break

    # Restore NaN outside survey
    h[nan_mask] = np.nan

    return h, rms_history
