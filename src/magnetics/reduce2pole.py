"""
reduce2pole.py  –  spectral phase-transformation filters for gridded magnetics
==============================================================================
Provides:
    reduce_to_pole   – reduce-to-pole (RTP) transform
    pseudo_gravity   – pseudo-gravity transform (Baranov 1957)

Both operate on uniformly-spaced 2-D grids and use the same underlying
spectral phase-transformation engine (``_phase_transform``).

Reference
---------
Baranov, V. (1957).  A new method for interpretation of aeromagnetic maps:
    pseudo-gravimetric anomalies.  *Geophysics*, 22(2), 359–383.

Blakely, R. J. (1995).  *Potential Theory in Gravity and Magnetic
    Applications*.  Cambridge University Press.
"""

import numpy as np


# ── helpers ─────────────────────────────────────────────────────────────────

def _dircos(inc_deg, dec_deg, azimuth_deg=0.0):
    """
    Direction cosines from inclination, declination, and strike azimuth.

    Parameters
    ----------
    inc_deg     : inclination  (degrees, positive down)
    dec_deg     : declination  (degrees, east of north)
    azimuth_deg : strike azimuth of the profile (degrees); usually 0

    Returns
    -------
    mx, my, mz : direction cosines  (floats)
    """
    inc = np.radians(inc_deg)
    dec = np.radians(dec_deg)
    azm = np.radians(azimuth_deg)
    mx = np.cos(inc) * np.cos(dec - azm)
    my = np.cos(inc) * np.sin(dec - azm)
    mz = np.sin(inc)
    return mx, my, mz


def _wavenumber_grids(x, y):
    """
    Build 2-D wavenumber grids (rad / unit) matching the layout used by
    ``fft2`` / ``ifft2`` for a grid with coordinates *x* (columns) and
    *y* (rows).

    Near-zero wavenumbers are nudged to a small value to avoid division by
    zero in filter denominators.
    """
    nx = len(x)
    ny = len(y)
    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])

    dkx = 2.0 * np.pi / (nx * dx)
    dky = 2.0 * np.pi / (ny * dy)

    # Use the same wrap-around convention as the MATLAB original
    jj = np.arange(nx)
    ii = np.arange(ny)
    kx_1d = np.where(jj <= nx // 2, jj * dkx, (jj - nx) * dkx)
    ky_1d = np.where(ii <= ny // 2, ii * dky, (ii - ny) * dky)

    KX, KY = np.meshgrid(kx_1d, ky_1d)

    # Prevent exact zeros in the denominator
    KX[KX == 0.0] = 1e-12
    KY[KY == 0.0] = 1e-12

    K = np.sqrt(KX**2 + KY**2)
    return KX, KY, K


def _upsilon(mx, my, mz, KX, KY, K):
    """
    Complex magnetisation/field direction factor in the wavenumber domain:

        upsilon(k) = mz + i * (kx*mx + ky*my) / |k|
    """
    return mz + 1j * (KX * mx + KY * my) / K


# ── core transform ───────────────────────────────────────────────────────────

def _phase_transform(x, y, mag, f_obs, m_obs, f_new, m_new):
    """
    General spectral phase transformation.

    Converts a gridded total-field magnetic anomaly from one set of
    field/magnetisation directions to another.

    Parameters
    ----------
    x, y    : 1-D coordinate arrays (columns, rows) – any consistent length
              unit (km, m, …)
    mag     : 2-D array, shape (len(y), len(x)), total-field anomaly (nT)
    f_obs   : (inclination, declination) of the *observed* inducing field (°)
    m_obs   : (inclination, declination) of the *observed* remanent / total
              magnetisation direction (°)
    f_new   : (inclination, declination) of the *target* inducing field (°)
    m_new   : (inclination, declination) of the *target* magnetisation (°)

    Returns
    -------
    mag_t        : 2-D array, transformed magnetic anomaly (same units as mag)
    phase_filter : 2-D complex array, the spectral filter that was applied
    """
    mx1, my1, mz1 = _dircos(*m_obs)
    fx1, fy1, fz1 = _dircos(*f_obs)
    mx2, my2, mz2 = _dircos(*m_new)
    fx2, fy2, fz2 = _dircos(*f_new)

    KX, KY, K = _wavenumber_grids(x, y)

    ups_m1 = _upsilon(mx1, my1, mz1, KX, KY, K)
    ups_f1 = _upsilon(fx1, fy1, fz1, KX, KY, K)
    ups_m2 = _upsilon(mx2, my2, mz2, KX, KY, K)
    ups_f2 = _upsilon(fx2, fy2, fz2, KX, KY, K)

    phase_filter = (ups_m2 * ups_f2) / (ups_m1 * ups_f1)

    MAG = np.fft.fft2(mag)
    mag_t = np.real(np.fft.ifft2(MAG * phase_filter))

    return mag_t, phase_filter


# ── public API ───────────────────────────────────────────────────────────────

def reduce_to_pole(x, y, mag, f, m):
    """
    Reduce-to-pole (RTP) transform.

    Maps the total-field magnetic anomaly to what would be observed at the
    magnetic pole, where both the inducing field and magnetisation are
    vertical.  At the pole the anomaly is symmetric over its source and
    purely positive over a susceptibility high.

    Parameters
    ----------
    x, y : 1-D coordinate arrays (columns / rows)
    mag  : 2-D total-field anomaly, shape (len(y), len(x))  [nT]
    f    : (inclination, declination) of the inducing field   [degrees]
    m    : (inclination, declination) of the magnetisation    [degrees]

    Returns
    -------
    mag_rtp      : 2-D RTP anomaly  [nT]
    phase_filter : 2-D complex spectral filter
    """
    return _phase_transform(x, y, mag, f, m, (90.0, 0.0), (90.0, 0.0))


def pseudo_gravity(x, y, mag, f, m):
    """
    Pseudo-gravity transform (Baranov 1957).

    Converts a total-field magnetic anomaly to a pseudo-gravity anomaly —
    the gravity response that would be produced by the same source if
    magnetisation were replaced by proportional density contrast.

    The spectral filter is::

        W(k) = 1 / ( |k| * upsilon_f(k) * upsilon_m(k) )

    where upsilon encodes the field/magnetisation directions and the 1/|k|
    factor spatially integrates the field (giving a gravity-like response
    that is smoother and more symmetric than the raw magnetic anomaly).

    The result is proportional to gravity in arbitrary units; scale by
    ``(rho / kappa) * (G / (2*pi))`` to obtain physical units.

    Parameters
    ----------
    x, y : 1-D coordinate arrays (columns / rows)
    mag  : 2-D total-field anomaly, shape (len(y), len(x))  [nT]
    f    : (inclination, declination) of the inducing field   [degrees]
    m    : (inclination, declination) of the magnetisation    [degrees]

    Returns
    -------
    pg           : 2-D pseudo-gravity anomaly  [arbitrary units]
    pg_filter    : 2-D complex spectral filter
    """
    mx, my, mz = _dircos(*m)
    fx, fy, fz = _dircos(*f)

    KX, KY, K = _wavenumber_grids(x, y)

    ups_m = _upsilon(mx, my, mz, KX, KY, K)
    ups_f = _upsilon(fx, fy, fz, KX, KY, K)

    pg_filter = 1.0 / (K * ups_f * ups_m)

    MAG = np.fft.fft2(mag)
    pg = np.real(np.fft.ifft2(MAG * pg_filter))

    return pg, pg_filter
