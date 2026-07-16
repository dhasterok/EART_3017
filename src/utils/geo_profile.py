"""
geo_profile.py
--------------
Sample a great-circle profile between two (lon, lat) points from a
geographic grid (NetCDF/xarray-openable raster).

Usage
-----
    from src.utils.geo_profile import great_circle_points, sample_profile

    lons, lats, dist_km = great_circle_points(-130, -72, 50, -65, n=500)
    dist_km, values = sample_profile("data/topography/ETOPO_2022_v1_30s_N90W180_surface.nc",
                                      "z", -130, -72, 50, -65, n=500)
"""

from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")


def great_circle_points(lon0, lat0, lon1, lat1, n=500):
    """Return `n` equally spaced points along the great-circle path from
    (lon0, lat0) to (lon1, lat1), plus cumulative distance along the path.

    Parameters
    ----------
    lon0, lat0, lon1, lat1 : float
        Endpoint coordinates in degrees.
    n : int
        Number of points along the path, including both endpoints.

    Returns
    -------
    lons, lats : ndarray, shape (n,)
    dist_km    : ndarray, shape (n,)
        Cumulative distance from the first point, in kilometres.
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    _, _, dist_m = _GEOD.inv(lon0, lat0, lon1, lat1)

    if n == 2:
        lons = np.array([lon0, lon1])
        lats = np.array([lat0, lat1])
    else:
        pts = _GEOD.npts(lon0, lat0, lon1, lat1, n - 2)
        lons = np.array([lon0] + [p[0] for p in pts] + [lon1])
        lats = np.array([lat0] + [p[1] for p in pts] + [lat1])

    dist_km = np.linspace(0.0, dist_m / 1000.0, n)
    return lons, lats, dist_km


def sample_profile(grid, variable, lon0, lat0, lon1, lat1, n=500, method="bilinear"):
    """Sample a gridded variable along a great-circle profile.

    Parameters
    ----------
    grid : str, Path, xarray.Dataset, or xarray.DataArray
        A NetCDF file path (lazily opened) or an already-loaded xarray
        object with "lon"/"lat" coordinates.
    variable : str or None
        Name of the data variable to sample. Ignored if `grid` is already
        a DataArray.
    lon0, lat0, lon1, lat1 : float
        Profile endpoints in degrees.
    n : int
        Number of points along the profile.
    method : str
        Interpolation method passed to `xarray.DataArray.sel`/`.interp`
        ("nearest" for a fast lookup, "linear" for bilinear interpolation).

    Returns
    -------
    dist_km, lons, lats, values : ndarray
    """
    if isinstance(grid, (str, Path)):
        da = xr.open_dataset(grid)[variable]
    elif isinstance(grid, xr.Dataset):
        da = grid[variable]
    else:
        da = grid

    lons, lats, dist_km = great_circle_points(lon0, lat0, lon1, lat1, n=n)
    lon_pts = xr.DataArray(lons, dims="points")
    lat_pts = xr.DataArray(lats, dims="points")

    if method == "nearest":
        values = da.sel(lon=lon_pts, lat=lat_pts, method="nearest").values
    else:
        values = da.interp(lon=lon_pts, lat=lat_pts, method=method).values

    return dist_km, lons, lats, values


def load_crustal_thickness(txt_path, variable="Hc", cache_path=None):
    """Build a regular lon/lat grid DataArray from the ECM1-format crustal
    thickness table (see data/crustal_thickness/README_ECM1.txt).

    Parameters
    ----------
    txt_path : str or Path
        Path to the tab-separated ECM1 table (e.g. ECM1.txt).
    variable : str
        Column to grid (e.g. "Hc" total crustal thickness, "Hcc"
        crystalline crust thickness).
    cache_path : str or Path, optional
        If given, save the gridded result as NetCDF here for faster reuse.

    Returns
    -------
    xarray.DataArray with dims ("lat", "lon").
    """
    import pandas as pd

    df = pd.read_csv(txt_path, sep="\t")
    pivot = df.pivot(index="Lat", columns="Lon", values=variable)
    da = xr.DataArray(
        pivot.values.astype(np.float32),
        coords={"lat": pivot.index.values, "lon": pivot.columns.values},
        dims=["lat", "lon"],
        name=variable,
        attrs={"source": str(txt_path)},
    )

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        da.to_dataset(name=variable).to_netcdf(cache_path)

    return da
