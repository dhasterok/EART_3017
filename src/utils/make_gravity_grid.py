"""
Reconstruct gravity values from an RGB-encoded gravity map using
monotonic, non-linear inversion of the colour ramp via isotonic
regression on HSV hue, with explicit handling of the red–white
saturation tail.

Author: D. Hasterok (workflow)
Implementation: Python
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from scipy.ndimage import label
import xarray as xr
import matplotlib.pyplot as plt
import colorsys
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator   # monotonic cubic

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def rgb_to_hsv_array(rgb):
    """
    Convert an (N,3) array of RGB values [0–255] to HSV [0–1].
    """
    rgb = rgb.astype(np.float32) / 255.0
    hsv = np.zeros_like(rgb)
    for i in range(rgb.shape[0]):
        hsv[i] = colorsys.rgb_to_hsv(*rgb[i])
    return hsv


# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
GRAV_FILE = (
    "data/sa_grav_tif_gda2020/"
    "SA_Grav_2024_BA_2007_UMS2_SCAP_100m_GDA2020_GEODETIC.ers"
)

RGB_FILE = (
    "data/sa_grav_tif_gda2020/TIF/"
    "sa_grav_2024_ba_2007_ums2_scap_100m_gda2020_geodetic_noclip.tif"
)

OUT_NC = "data/sa_grav_tif_gda2020/SA_GRAV_HUE_ISOTONIC_RECONSTRUCTED.nc"


# ---------------------------------------------------------------------
# 1. Load gravity (reference grid)
# ---------------------------------------------------------------------
with rasterio.open(GRAV_FILE) as ds:
    grav = ds.read(1).astype(float)
    transform = ds.transform
    crs = ds.crs
    nodata = ds.nodata
    ny, nx = ds.shape

grav[grav == nodata] = np.nan


# ---------------------------------------------------------------------
# 2. Load and reproject RGB → gravity grid
# ---------------------------------------------------------------------
with rasterio.open(RGB_FILE) as ds:
    rgb_src = ds.read().astype(np.float32)  # (3, ny, nx)
    rgb_transform = ds.transform
    rgb_crs = ds.crs

rgb = np.zeros((3, ny, nx), dtype=np.float32)

for b in range(3):
    reproject(
        source=rgb_src[b],
        destination=rgb[b],
        src_transform=rgb_transform,
        src_crs=rgb_crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.nearest
    )

# reshape to (ny, nx, 3)
rgb = np.moveaxis(rgb, 0, -1)

# ------------------------------------------------------------
# Convert observed RGB → HSV
# ------------------------------------------------------------
mask_obs = np.isfinite(grav)

rgb_obs = rgb[mask_obs].reshape(-1, 3).astype(np.float32) / 255.0
g_obs   = grav[mask_obs]

hsv_obs = np.array([colorsys.rgb_to_hsv(*c) for c in rgb_obs])

H = hsv_obs[:, 0]   # Hue ∈ [0,1)
S = hsv_obs[:, 1]   # Saturation
V = hsv_obs[:, 2]   # Brightness (Value)

# ------------------------------------------------------------
# Exclude low-saturation (whiteward) pixels from mapping
# ------------------------------------------------------------
SAT_THRESH = 0.25
use = S > SAT_THRESH

H_use = H[use]
g_use = g_obs[use]

# ------------------------------------------------------------
# Bin hue and take median gravity per bin
# ------------------------------------------------------------
NBINS = 512   # sufficiently fine for smooth ramp

bins = np.linspace(H_use.min(), H_use.max(), NBINS + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

g_bin = np.full(NBINS, np.nan)

for i in range(NBINS):
    idx = (H_use >= bins[i]) & (H_use < bins[i + 1])
    if np.any(idx):
        g_bin[i] = np.median(g_use[idx])

# Drop empty bins
valid_bins = np.isfinite(g_bin)

H_bin = bin_centers[valid_bins]
g_bin = g_bin[valid_bins]

# ------------------------------------------------------------
# Construct monotonic interpolant
# ------------------------------------------------------------
# PCHIP preserves monotonicity and curvature (no oscillations)
f_hue_to_grav = PchipInterpolator(H_bin, g_bin, extrapolate=False)


# import numpy as np
# import matplotlib.pyplot as plt
# import colorsys

# ------------------------------------------------------------
# Prepare observed data (robust & consistent)
# ------------------------------------------------------------

# Optional subsampling to keep plots readable & fast
NMAX = 300_000
if len(H) > NMAX:
    idx = np.random.choice(len(H), NMAX, replace=False)
    H, S, V, g_obs = H[idx], S[idx], V[idx], g_obs[idx]

# ------------------------------------------------------------
# Create multi-panel diagnostic plot
# ------------------------------------------------------------
fig, axs = plt.subplots(
    2, 2,
    figsize=(12, 10),
    constrained_layout=True
)

# ------------------------------------------------------------
# (1) Hue vs Gravity
# ------------------------------------------------------------
axs[0, 0].scatter(H, g_obs, s=1, alpha=0.3)
axs[0, 0].set_xlabel("Hue (0–1)")
axs[0, 0].set_ylabel("Gravity (mGal)")
axs[0, 0].set_title("Hue vs Gravity")

# ------------------------------------------------------------
# (2) Hue vs Gravity, coloured by Saturation
# ------------------------------------------------------------
sc = axs[0, 1].scatter(
    H, g_obs,
    c=S,
    s=1,
    alpha=0.4,
    cmap="viridis"
)
axs[0, 1].set_xlabel("Hue (0–1)")
axs[0, 1].set_ylabel("Gravity (mGal)")
axs[0, 1].set_title("Hue vs Gravity (colour = Saturation)")
cbar = fig.colorbar(sc, ax=axs[0, 1])
cbar.set_label("Saturation")

# ------------------------------------------------------------
# (3) Unwrapped Hue vs Gravity
# ------------------------------------------------------------
H_unwrap = np.concatenate([H, H + 1.0])
g_unwrap = np.concatenate([g_obs, g_obs])

axs[1, 0].scatter(H_unwrap, g_unwrap, s=1, alpha=0.3)
axs[1, 0].set_xlabel("Hue (unwrapped)")
axs[1, 0].set_ylabel("Gravity (mGal)")
axs[1, 0].set_title("Hue vs Gravity (Unwrapped)")

# ------------------------------------------------------------
# (4) Value (brightness) vs Gravity
# ------------------------------------------------------------
axs[1, 1].scatter(V, g_obs, s=1, alpha=0.3)
axs[1, 1].set_xlabel("Value (Brightness)")
axs[1, 1].set_ylabel("Gravity (mGal)")
axs[1, 1].set_title("Brightness vs Gravity")

plt.show()

# ------------------------------------------------------------
# Convert ALL RGB → HSV
# ------------------------------------------------------------
rgb_all = rgb.reshape(-1, 3).astype(np.float32) / 255.0
hsv_all = np.array([colorsys.rgb_to_hsv(*c) for c in rgb_all])

H_all = hsv_all[:, 0]
S_all = hsv_all[:, 1]
V_all = hsv_all[:, 2]

# Clamp hue to observed range (no extrapolation!)
H_min, H_max = H_bin.min(), H_bin.max()
H_all_clamped = np.clip(H_all, H_min, H_max)

grav_est = f_hue_to_grav(H_all_clamped).reshape(grav.shape)

# ------------------------------------------------------------
# Brightness-based extension for red→white tail
# ------------------------------------------------------------
# Perceptual luminance
L = (
    0.2126 * rgb[..., 0] +
    0.7152 * rgb[..., 1] +
    0.0722 * rgb[..., 2]
)

# Identify red pixels WITH observed gravity
is_red_obs = (
    (rgb[..., 0] > 200) &
    (rgb[..., 1] < 80) &
    (rgb[..., 2] < 80) &
    mask_obs
)

if np.any(is_red_obs):
    L_red = np.nanmedian(L[is_red_obs])
    L_white = np.nanmax(L)

    g_max_obs = np.nanmax(g_obs)
    g_max_ext = g_max_obs + 25.0   # explicit assumption

    white_tail = (
        ~mask_obs &
        (S_all.reshape(grav.shape) <= SAT_THRESH) &
        (L > L_red)
    )

    grav_est[white_tail] = (
        g_max_obs
        + (L[white_tail] - L_red) / (L_white - L_red)
        * (g_max_ext - g_max_obs)
    )

grav_filled = grav.copy()
grav_filled[~mask_obs] = grav_est[~mask_obs]


# ---------------------------------------------------------------------
# 9. Build coordinates and export NetCDF
# ---------------------------------------------------------------------
x0, y0 = transform * (0, 0)
dx = transform.a
dy = -transform.e

x = x0 + dx * np.arange(nx)
y = y0 - dy * np.arange(ny)

ds_out = xr.Dataset(
    data_vars=dict(
        gravity=(("y", "x"), grav_filled)
    ),
    coords=dict(
        x=("x", x),
        y=("y", y)
    ),
    attrs=dict(
        title="Gravity reconstructed from RGB using isotonic hue inversion",
        units="mGal",
        fill_method="Isotonic regression on HSV hue with red–white brightness extension",
        note="Filled values are estimates used solely to stabilise Fourier transforms"
    )
)


# Define CRS explicitly
grav_crs = CRS.from_epsg(7845)  # GDA2020 / MGA zone 53

# Add CF-compliant grid mapping variable
ds_out.coords["crs"] = xr.DataArray(
    0,
    attrs={
        "grid_mapping_name": "transverse_mercator",
        "spatial_ref": grav_crs.to_wkt()
    }
)

# Tell the data variable which grid mapping it uses
ds_out["gravity"].attrs["grid_mapping"] = "crs"


ds_out.to_netcdf(OUT_NC)
print(f"NetCDF written: {OUT_NC}")


# ---------------------------------------------------------------------
# 10. Diagnostics
# ---------------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.imshow(grav_filled, cmap="seismic")
plt.colorbar(label="mGal")
plt.title("Final Gravity Field (FFT‑ready)")
plt.show()