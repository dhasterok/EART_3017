"""
hfsurvey.py

Utilities for extracting heat-flow observations from a global .mat dataset
within a geographic polygon and projecting them onto a profile distance axis.
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat
from matplotlib.path import Path as MplPath


def sphangle(lon, lat, lon0, lat0, quad=0):
    """
    Great-circle angle between points (lon, lat) and a reference (lon0, lat0).

    Parameters
    ----------
    lon, lat : array_like, degrees
    lon0, lat0 : float, reference point in degrees
    quad : int
        0 – all angles positive (default)
        1 – positive where lon > lon0
        2 – positive where lon < lon0
        3 – positive where lat > lat0
        4 – positive where lat < lat0

    Returns
    -------
    delta : ndarray, radians
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    th0, ph0 = np.radians(lon0), np.radians(lat0)
    x0 = np.cos(ph0) * np.cos(th0)
    y0 = np.cos(ph0) * np.sin(th0)
    z0 = np.sin(ph0)

    th = np.radians(lon)
    ph = np.radians(lat)
    x  = np.cos(ph) * np.cos(th)
    y  = np.cos(ph) * np.sin(th)
    z  = np.sin(ph)

    dot   = x0*x + y0*y + z0*z
    delta = np.arccos(np.clip(dot, -1.0, 1.0))

    if quad == 0:
        return delta

    sgn = np.ones_like(th)
    if quad == 1:
        sgn[th < th0] = -1.0
    elif quad == 2:
        sgn[th > th0] = -1.0
    elif quad == 3:
        sgn[ph < ph0] = -1.0
    elif quad == 4:
        sgn[ph > ph0] = -1.0

    return delta * sgn


def hfsurvey(v, data_file):
    """
    Extract heat-flow observations from a MATLAB hf_data.mat file within a
    polygon and return distances along a profile.

    Parameters
    ----------
    v : (5, 2) array-like
        Rows 0–3 : polygon vertices [lon, lat]
        Row  4   : reference point [lon0, lat0] for distance calculation
    data_file : path-like
        Path to hf_data.mat.

    Returns
    -------
    x  : (N,) great-circle distance from reference point (km)
    qo : (N,) observed heat flow (mW/m²)
    qe : (N,) heat-flow uncertainty (mW/m²)
    """
    mat = loadmat(str(data_file))
    hf  = mat["hf_data"]

    def _field(name):
        val = hf[name]
        while val.dtype == object:
            val = val.flat[0]
        return np.asarray(val, dtype=float).ravel()

    def _field2d(name):
        val = hf[name]
        while val.dtype == object:
            val = val.flat[0]
        return np.asarray(val, dtype=float)

    lon_hf = _field("lon")
    lat_hf = _field("lat")
    qc     = _field2d("qc")
    qu     = _field2d("qu")

    v          = np.asarray(v, dtype=float)
    poly_verts = v[:4, :]
    ref        = v[4, :]

    poly_path = MplPath(poly_verts)
    inside    = poly_path.contains_points(np.column_stack([lon_hf, lat_hf]))

    R_earth = 6371.0
    dist = R_earth * sphangle(lon_hf[inside], lat_hf[inside],
                               ref[0], ref[1], quad=1)

    q_obs = qc[inside, 0]
    q_unc = qu[inside, 1] if qu.shape[1] > 1 else qc[inside, 1]

    return dist, q_obs, q_unc
