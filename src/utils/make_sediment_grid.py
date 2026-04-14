"""
Create a sediment thickness NetCDF for OZSEEBASE using contour constraints
and basement outcrop masking.

Inputs:
  - Geognostics_OZSEEBASE_2021.tif          (RGB reference grid only)
  - ozseebase_2021_contours_500m.shp       (Contours, depth in meters, negative values)
  - ozseebase_2021_basement_outcrops.shp  (Basement outcrops, thickness = 0)
  - SA_TMI.tif                             (target grid)

Output:
  - SA_OZSEEBASE_SED.nc  (Sediment thickness in meters, positive)

Author: D. Hasterok (workflow), Python implementation by Copilot
"""

import sys
from pathlib import Path

_HERE   = Path(__file__).resolve().parent
_COURSE = _HERE.parent.parent
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import geopandas as gpd
from scipy.interpolate import griddata
from tqdm import tqdm
import xarray as xr

# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
REF_TIF = "data/OZSEEBASE_sedthick/Geognostics_OZSEEBASE_2021.tif"
CONTOURS_SHP = "data/OZSEEBASE_sedthick/ozseebase_2021_contours_500m.shp"
OUTCROPS_SHP = "data/OZSEEBASE_sedthick/ozseebase_2021_basement_outcrops.shp"
TMI_TIF = "data/SA_TMI_GDA2020/SA_TMI.tif"
OUT_NC = "data/OZSEEBASE_sedthick/SA_OZSEEBASE_SED.nc"

# ---------------------------------------------------------------------
# 1. Read OZSEEBASE reference grid (geometry only)
# ---------------------------------------------------------------------
with rasterio.open(REF_TIF) as src:
    transform = src.transform
    crs = src.crs
    width = src.width
    height = src.height

x0, y0 = transform * (0, 0)
dx = transform.a
dy = -transform.e

x = x0 + dx * np.arange(width)
y = y0 - dy * np.arange(height)
X, Y = np.meshgrid(x, y)

# ---------------------------------------------------------------------
# 2. Load contours and extract control points
# ---------------------------------------------------------------------
contours = gpd.read_file(CONTOURS_SHP).to_crs(crs)

if "Contour" not in contours.columns:
    raise ValueError("Contour shapefile must contain a 'Contour' attribute")

px, py, pz = [], [], []

print("Extracting contour points...")
for _, row in tqdm(contours.iterrows(), total=len(contours)):
    depth = -row["Contour"]  # positive thickness
    if np.isnan(depth):
        continue

    line = row.geometry
    for d in np.linspace(0, line.length, max(int(line.length / 1000), 5)):
        p = line.interpolate(d)
        px.append(p.x)
        py.append(p.y)
        pz.append(depth)

px = np.asarray(px)
py = np.asarray(py)
pz = np.asarray(pz)

# ---------------------------------------------------------------------
# 3. Add basement outcrop constraints (thickness = 0)
# ---------------------------------------------------------------------
outcrops = gpd.read_file(OUTCROPS_SHP).to_crs(crs)

print("Adding basement outcrop control points...")
for geom in tqdm(outcrops.geometry):
    x_c, y_c = geom.centroid.coords[0]
    px = np.append(px, x_c)
    py = np.append(py, y_c)
    pz = np.append(pz, 0.0)

# ---------------------------------------------------------------------
# 4. Interpolate sediment thickness
# ---------------------------------------------------------------------
print("Interpolating sediment thickness...")
Z = griddata(
    points=(px, py),
    values=pz,
    xi=(X, Y),
    method="linear"
)

Z = np.nan_to_num(Z, nan=0.0)

# ---------------------------------------------------------------------
# 5. Enforce outcrop mask exactly
# ---------------------------------------------------------------------
mask = rasterize(
    [(geom, 1) for geom in outcrops.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

Z[mask == 1] = 0.0
Z[Z < 0] = 0.0

# ---------------------------------------------------------------------
# 6. Reproject sediment thickness to SA_TMI grid
# ---------------------------------------------------------------------
with rasterio.open(TMI_TIF) as tmi:
    sed_on_tmi = np.zeros((tmi.height, tmi.width), dtype=np.float32)

    reproject(
        source=Z.astype(np.float32),
        destination=sed_on_tmi,
        src_transform=transform,
        src_crs=crs,
        dst_transform=tmi.transform,
        dst_crs=tmi.crs,
        resampling=Resampling.bilinear
    )

    # Build coordinates for NetCDF
    x0, y0 = tmi.transform * (0, 0)
    dx = tmi.transform.a
    dy = -tmi.transform.e

    x_tmi = x0 + dx * np.arange(tmi.width)
    y_tmi = y0 - dy * np.arange(tmi.height)

# ---------------------------------------------------------------------
# 7. Write NetCDF using xarray (with CRS)
# ---------------------------------------------------------------------

# Explicitly define CRS: GDA2020 / MGA Zone 53
sed_crs = CRS.from_epsg(7845)

ds = xr.Dataset(
    data_vars=dict(
        sediment_thickness=(("y", "x"), sed_on_tmi)
    ),
    coords=dict(
        x=("x", x_tmi),
        y=("y", y_tmi)
    ),
    attrs=dict(
        title="OZSEEBASE sediment thickness (interpolated)",
        units="meters",
        positive="down",
        method="Interpolated from 500 m contours with basement outcrop constraints",
        note=(
            "Sediment thickness grid reprojected to SA_TMI grid. "
            "Intended for joint analysis with gravity and magnetics."
        )
    )
)

# Add CF-compliant grid mapping
ds.coords["crs"] = xr.DataArray(
    0,
    attrs={
        "grid_mapping_name": "transverse_mercator",
        "spatial_ref": sed_crs.to_wkt()
    }
)

# Link data variable to grid mapping
ds["sediment_thickness"].attrs["grid_mapping"] = "crs"
ds["sediment_thickness"].attrs["units"] = "m"
ds["sediment_thickness"].attrs["long_name"] = "Sediment thickness"

ds.to_netcdf(OUT_NC)

print(f"Sediment thickness NetCDF written with CRS:\n   {OUT_NC}")