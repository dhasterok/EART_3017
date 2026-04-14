import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import xarray as xr
import matplotlib.pyplot as plt
from affine import Affine

TMI_FILE = "data/SA_TMI_GDA2020/SA_TMI.ers"

with rasterio.open(TMI_FILE) as ds:
    tmi = ds.read(1).astype(float)
    tmi_transform = ds.transform
    tmi_crs = ds.crs
    tmi_nodata = ds.nodata
    ny, nx = ds.shape

print(ds)

# Handle nodata
tmi[tmi == tmi_nodata] = np.nan

WDMAM_FILE = "data/wdmam.nc"

ds_wdmam = xr.open_dataset(WDMAM_FILE)
wdmam_crs = ds_wdmam.crs

print(tmi_crs)

# Latitude:  -40° ≤ lat ≤ -28°
# Longitude: 128° ≤ lon ≤ 142°

# Define bounds
lat_min, lat_max = -40.0, -25.0
lon_min, lon_max = 128.0, 142.0

ds_wdmam_sub = ds_wdmam.sel(
    lat=slice(lat_min, lat_max),
    lon=slice(lon_min, lon_max)
)

wdmam = ds_wdmam_sub["Band1"].values
lat = ds_wdmam_sub["lat"].values
lon = ds_wdmam_sub["lon"].values

dlon = lon[1] - lon[0]
dlat = lat[1] - lat[0]

wdmam_transform = (
    Affine.translation(lon.min() - dlon/2, lat.max() + dlat/2)
    * Affine.scale(dlon, -dlat)
)

wdmam_crs = "EPSG:4326"

wdmam_on_tmi = np.zeros((ny, nx), dtype=np.float32)

reproject(
    source=wdmam.astype(np.float32),
    destination=wdmam_on_tmi,
    src_transform=wdmam_transform,
    src_crs=wdmam_crs,
    dst_transform=tmi_transform,
    dst_crs=tmi_crs,
    resampling=Resampling.bilinear
)

mask = np.isfinite(tmi) & np.isfinite(wdmam_on_tmi)

tmi_vals = tmi[mask]
wdmam_vals = wdmam_on_tmi[mask]

NMAX = 300_000
if len(tmi_vals) > NMAX:
    idx = np.random.choice(len(tmi_vals), NMAX, replace=False)
    tmi_vals = tmi_vals[idx]
    wdmam_vals = wdmam_vals[idx]

plt.figure(figsize=(6,6))
plt.scatter(wdmam_vals, tmi_vals, s=1, alpha=0.3)
plt.xlabel("WDMAM Magnetic Anomaly (nT)")
plt.ylabel("SA TMI (nT)")
plt.title("SA TMI vs WDMAM (Common Area)")
plt.grid(True)
plt.show()

print("SA TMI mean:", np.nanmean(tmi_vals))
print("WDMAM mean:", np.nanmean(wdmam_vals))
print("SA TMI std:", np.nanstd(tmi_vals))
print("WDMAM std:", np.nanstd(wdmam_vals))

coeff = np.polyfit(wdmam_vals, tmi_vals, 1)
print("Approximate linear relation: TMI ≈ a*WDMAM + b")
print("a =", coeff[0], "b =", coeff[1])

offset = np.nanmean(tmi_vals - wdmam_vals)
print("Offset to apply to WDMAM:", offset)


# Offset‑adjusted WDMAM
wdmam_adj = wdmam_on_tmi + offset

print("Adjusted WDMAM mean:", np.nanmean(wdmam_adj))
print("Adjusted WDMAM std :", np.nanstd(wdmam_adj))

plt.figure(figsize=(7,6))
im = plt.imshow(
    wdmam_adj,
    cmap="seismic",
    origin="lower"
)
plt.colorbar(im, label="Magnetic anomaly (nT)")
plt.title("WDMAM (offset‑adjusted, reprojected to SA TMI grid)")
plt.xlabel("Grid column")
plt.ylabel("Grid row")
plt.show()


# --- build coordinate vectors from the SA_TMI affine transform ---
x0, y0 = tmi_transform * (0, 0)
dx = tmi_transform.a
dy = tmi_transform.e   # will be negative for north-up grids

x = x0 + dx * np.arange(nx)
y = y0 + dy * np.arange(ny)

# Reverse y to ensure north-up orientation
y = y[::-1]
wdmam_adj = wdmam_adj[::-1, :]

# --- build xarray Dataset ---
ds_out = xr.Dataset(
    data_vars=dict(
        magnetic_anomaly=(("y", "x"), wdmam_adj.astype(np.float32))
    ),
    coords=dict(
        x=("x", x),
        y=("y", y)
    ),
    attrs=dict(
        title="WDMAM magnetic anomaly (offset-adjusted)",
        description=(
            "Global WDMAM magnetic anomaly reprojected to SA TMI grid "
            "and offset-adjusted to match regional mean. "
            "Intended as long-wavelength background for FFT stabilization."
        ),
        units="nT",
        source_wdmam="World Digital Magnetic Anomaly Map (WDMAM)",
        reference_grid="SA_TMI",
        processing_step="Offset adjustment only; no low-pass filtering yet",
        note=(
            "This dataset is not intended for geological interpretation. "
            "Used solely to stabilize FFT-based magnetic field processing."
        )
    )
)

ds_out.coords["crs"] = xr.DataArray(
    0,
    attrs={
        "spatial_ref": tmi_crs.to_wkt(),
        "grid_mapping_name": "transverse_mercator"
    }
)

# --- write file ---
OUT_FILE = "data/SA_TMI_GDA2020/SA_WDMAM_adjusted_UTM.nc"
ds_out.to_netcdf(OUT_FILE)

print(f"NetCDF written: {OUT_FILE}")
