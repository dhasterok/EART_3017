from .registry import DATASETS
from .global_grid import GlobalGrid

def worldgrid(name, latlim=None, lonlim=None):
    """
    Python equivalent of worldgrid.m
    """
    cfg = DATASETS[name]
    grid = GlobalGrid(cfg["path"], cfg["var"])

    latlim = latlim or (-90, 90)
    lonlim = lonlim or (-180, 180)

    subset = grid.subset_bbox(
        lonlim[0], lonlim[1], latlim[0], latlim[1]
    )

    data = subset[cfg["var"]]
    lat = subset[grid.lat_name]
    lon = subset[grid.lon_name]

    return data, lat, lon

def downsample(ds, factor_lat=2, factor_lon=2, method="mean"):
    """
    Downsample by integer factors.
    """
    reducers = {
        "mean": xr.core.rolling.DataArray.mean,
        "median": xr.core.rolling.DataArray.median,
        "min": xr.core.rolling.DataArray.min,
        "max": xr.core.rolling.DataArray.max,
    }

    if method not in reducers:
        raise ValueError(f"Unknown method {method}")

    return ds.coarsen(
        lat=factor_lat,
        lon=factor_lon,
        boundary="trim",
    ).reduce(getattr(ds, method))


def upsample_to(ds, target_grid, method="linear"):
    """
    Upsample ds to match target_grid coordinates.
    """

    return ds.interp(
        lat=target_grid.lat,
        lon=target_grid.lon,
        method=method,
    )