"""
make_topo_grid.py
-----------------
Subset SRTM15_V2.7 onto the SA magnetics grid (GDA2020 geographic,
EPSG:7844) and save as a student-friendly NetCDF.

The output grid matches SA_WDMAM_adjusted_UTM.nc exactly:
  - same x/y axes (lon/lat in degrees)
  - same crs coordinate variable and spatial_ref WKT

Usage
-----
    python src/utils/make_topo_grid.py
"""

from pathlib import Path
import sys

import numpy as np
import xarray as xr

_HERE   = Path(__file__).resolve().parent
_COURSE = _HERE.parent.parent
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATAPATH    = _COURSE / "data"
DEM_PATH    = DATAPATH / "topography/SRTM15_V2.7.nc"
REF_PATH    = DATAPATH / "SA_TMI_GDA2020/SA_WDMAM_adjusted_UTM.nc"
OUTPUT_PATH = DATAPATH / "topography/SA_DEM_GDA2020.nc"

# ── Load reference grid axes and CRS ─────────────────────────────────────────
print("Loading reference grid …")
ref     = xr.open_dataset(REF_PATH)
x_ref   = ref.x.values   # longitudes (axis direction matches reference grid)
y_ref   = ref.y.values   # latitudes  (axis direction matches reference grid)
crs_wkt = ref["crs"].attrs["spatial_ref"]
ref.close()

lon_min, lon_max = float(x_ref.min()), float(x_ref.max())
lat_min, lat_max = float(y_ref.min()), float(y_ref.max())
print(f"  Reference extent : {lon_min:.4f}–{lon_max:.4f} E,  "
      f"{lat_min:.4f}–{lat_max:.4f} N")
print(f"  Reference size   : {len(y_ref)} rows × {len(x_ref)} cols")

# ── Load SRTM15 subset (add small margin to avoid edge effects) ───────────────
margin = 0.05
print("Loading SRTM15 subset …")
with xr.open_dataset(DEM_PATH) as ds:
    sub = ds["z"].sel(
        lon=slice(lon_min - margin, lon_max + margin),
        lat=slice(lat_min - margin, lat_max + margin),
    ).load()

print(f"  SRTM15 subset    : {sub.shape}  "
      f"z = {float(sub.min()):.0f} – {float(sub.max()):.0f} m")
print(f"  lat axis         : {float(sub.lat[0]):.4f} → {float(sub.lat[-1]):.4f}  "
      f"({'increasing' if sub.lat[-1] > sub.lat[0] else 'decreasing'})")
print(f"  lon axis         : {float(sub.lon[0]):.4f} → {float(sub.lon[-1]):.4f}")

# ── Interpolate onto reference grid using xarray (handles axis order safely) ──
print("Interpolating onto reference grid …")
topo_da = sub.interp(
    lat=xr.DataArray(y_ref, dims="y"),
    lon=xr.DataArray(x_ref, dims="x"),
    method="linear",
)
topo = topo_da.values.astype(np.float32)

# Ensure y is decreasing (north at row 0) regardless of reference grid orientation
if y_ref[-1] > y_ref[0]:
    y_ref = y_ref[::-1]
    topo  = topo[::-1, :]

valid = np.isfinite(topo)
print(f"  Output shape     : {topo.shape}")
print(f"  Elevation range  : {np.nanmin(topo):.1f} – {np.nanmax(topo):.1f} m")
print(f"  Valid pixels     : {valid.sum()} / {topo.size}  "
      f"({valid.mean()*100:.1f} %)")

# Spot-check a known land point (Adelaide ~138.6°E, 34.9°S)
iy = np.argmin(np.abs(y_ref - (-34.9)))
ix = np.argmin(np.abs(x_ref - 138.6))
print(f"\n  Spot check Adelaide ({y_ref[iy]:.3f}°N, {x_ref[ix]:.3f}°E): "
      f"{topo[iy, ix]:.1f} m  (expected ~50–200 m)")

# ── Build output Dataset ──────────────────────────────────────────────────────
ds_out = xr.Dataset(
    {
        "elevation": xr.DataArray(
            topo,
            dims=["y", "x"],
            attrs={
                "long_name"    : "Elevation above sea level",
                "units"        : "m",
                "source"       : "SRTM15+ V2.7",
                "grid_mapping" : "crs",
            },
        ),
        "crs": xr.DataArray(
            np.int64(0),
            attrs={
                "spatial_ref"       : crs_wkt,
                "grid_mapping_name" : "latitude_longitude",
            },
        ),
    },
    coords={
        "x": xr.DataArray(x_ref, dims=["x"],
                          attrs={"units": "degrees_east",  "long_name": "Longitude"}),
        "y": xr.DataArray(y_ref, dims=["y"],
                          attrs={"units": "degrees_north", "long_name": "Latitude"}),
    },
    attrs={
        "title"               : "South Australia DEM — GDA2020 geographic grid",
        "institution"         : "University teaching dataset",
        "source_dem"          : str(DEM_PATH),
        "reference_grid"      : str(REF_PATH),
        "geographical_extent" : (f"{lon_min:.4f}E–{lon_max:.4f}E, "
                                 f"{abs(lat_min):.4f}S–{abs(lat_max):.4f}S"),
        "crs_epsg"            : "7844",
    },
)

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
encoding = {
    "elevation": {
        "zlib"      : True,
        "complevel" : 4,
        "dtype"     : "float32",
        "_FillValue": float("nan"),
    }
}

ds_out.to_netcdf(OUTPUT_PATH, encoding=encoding)
print(f"\nSaved → {OUTPUT_PATH.resolve()}")
print(f"Grid size : {topo.shape[0]} rows × {topo.shape[1]} cols")
