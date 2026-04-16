import xarray as xr
import numpy as np
from pathlib import Path

class GlobalGrid:
    """
    Generic handler for global regular lat–lon grids.

    Assumes:
    - regular spacing
    - monotonic lat/lon
    - longitude in [-180, 180] or [0, 360]
    """

    def __init__(self, nc_path, var_name=None):
        self.path = Path(nc_path)
        self.ds = xr.open_dataset(self.path, chunks="auto")

        self.lon_name = self._find_coord(["lon", "longitude", "x"])
        self.lat_name = self._find_coord(["lat", "latitude", "y"])

        self.var_name = var_name or list(self.ds.data_vars)[0]

        self.lon = self.ds[self.lon_name]
        self.lat = self.ds[self.lat_name]
        self.data = self.ds[self.var_name]

        self._check_regular()

    def _find_coord(self, names):
        for n in names:
            if n in self.ds.coords:
                return n
        raise KeyError(f"Expected one of {names} in dataset")

    def _check_regular(self):
        dlon = np.diff(self.lon.values)
        dlat = np.diff(self.lat.values)
        if not (np.allclose(dlon, dlon[0]) and np.allclose(dlat, dlat[0])):
            raise ValueError("Grid is not regularly spaced")

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------
    def subset_bbox(self, lon_min, lon_max, lat_min, lat_max):
        lat_slice = slice(lat_min, lat_max)

        if lon_min <= lon_max:
            return self.ds.sel(
                {
                    self.lon_name: slice(lon_min, lon_max),
                    self.lat_name: lat_slice,
                }
            )

        # wraparound
        left = self.ds.sel(
            {
                self.lon_name: slice(lon_min, self.lon.max()),
                self.lat_name: lat_slice,
            }
        )
        right = self.ds.sel(
            {
                self.lon_name: slice(self.lon.min(), lon_max),
                self.lat_name: lat_slice,
            }
        )

        return xr.concat([left, right], dim=self.lon_name)

    # ------------------------------------------------------------------
    # Polygon mask (optional dependency)
    # ------------------------------------------------------------------
    def mask_polygon(self, subset, polygon):
        """
        polygon: shapely.geometry.Polygon in lon/lat
        """
        import shapely.vectorized as sv

        lon2d, lat2d = np.meshgrid(
            subset[self.lon_name], subset[self.lat_name]
        )
        mask = sv.contains(polygon, lon2d, lat2d)
        return subset.where(mask)

    # ------------------------------------------------------------------
    # Memory awareness
    # ------------------------------------------------------------------
    def estimate_bytes(self, subset):
        ny = subset.dims[self.lat_name]
        nx = subset.dims[self.lon_name]
        dtype = subset[self.var_name].dtype
        return ny * nx * dtype.itemsize