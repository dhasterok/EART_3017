# %% [markdown]
# ---
# # Case Study: Nevada Basin Thickness from FFTs
#
# ## Learning Objectives:
# - See how FFTs can be used to analyze real world geophysical data
# - Apply spectral high-pass filtering to isolate shallow (basin) gravity signal from the regional trend
# - Use inversion to estimate basin thickness
# - See how aliasing affects basin model

# %%
from pathlib import Path
import sys

_ROOT    = Path(__file__).resolve().parent.parent.parent.parent   # src/demos/fft/ -> project root
DATAPATH = _ROOT / "data"

FIGDIR = Path(__file__).resolve().parent / "tmp_figures"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from src.common.grids.grid_io import read_pacs_grd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS

from ipywidgets import interact, FloatSlider, IntSlider
from src.inversion.parker_oldenburg import parker_oldenburg

from scipy.fft import fft2 as _fft2, ifft2 as _ifft2, fftfreq as _fftfreq
from scipy.fft import rfft as _rfft2, rfftfreq as _rfftfreq2
from scipy.fft import fft2, ifft2, fftfreq, rfft, rfftfreq

from src.utils.figure_utils import figutils as fu
fu = fu(FIGDIR)

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.grid": True,
})


# %% [markdown]
# ## Part 1 — Nevada Bouguer Gravity Anomaly
#
# We now apply the same concepts to the **Nevada Complete Bouguer Gravity** dataset
# (NEVCBA, USGS/PACES), a gridded compilation of ground gravity measurements across
# the Basin-and-Range Province.
#
# - Grid spacing: 2 km × 2 km (Lambert Conformal Conic projection)
# - Anomaly type: Complete Bouguer (terrain-corrected)
# - Nyquist wavelength: **4 km**
# - Spatial extent: ~564 km E–W × 794 km N–S

# %% [markdown]
# ### 1.1 Load and inspect the gravity grid
#
# **Objective:** load the gridded ASCII file and confirm the spatial metadata
# before plotting.

# %%
crs_lcc = CRS.from_proj4(
    "+proj=lcc "
    "+lat_1=33 +lat_2=45 "
    "+lat_0=0 "
    "+lon_0=-117 "
    "+x_0=0 +y_0=0 "
    "+units=km +datum=WGS84"
)
print(crs_lcc)

grd     = read_pacs_grd(DATAPATH / "nevboug.grd.txt")
gravity = grd.grid   # shape (nrows, ncols)
x       = grd.x      # km, west–east  (LCC projection)
y       = grd.y      # km, south–north

print(f'Grid shape : {gravity.shape}  ({grd.meta["nrows"]} rows × {grd.meta["ncols"]} cols)')
print(f'X range    : {x.min():.1f} – {x.max():.1f} km')
print(f'Y range    : {y.min():.1f} – {y.max():.1f} km')
print(f'Gravity    : {np.nanmin(gravity):.2f} – {np.nanmax(gravity):.2f} mGal')

# %% [markdown]
# ### 1.2 Background layers — hillshade and state boundaries
#
# Before plotting the gravity we prepare two orientation layers:
#
# 1. **Hillshade** — ETOPO 30-arcsecond topography reprojected to LCC and illuminated
#    from the NW (azimuth 315°, altitude 45°).  Basins appear dark (low elevation);
#    ranges appear bright.  Adding hillshade to a potential-field map helps the eye
#    correlate gravity anomalies with topography.
#
# 2. **State boundaries** — clipped to the gravity grid extent and reprojected to LCC.
#
# Both are computed once here and reused on all subsequent maps.

# %%
import netCDF4
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from src.common.plotting.hillshade import hillshade

# Load ETOPO and reproject to LCC grid
ds = netCDF4.Dataset(DATAPATH / "topography" / "ETOPO_2022_v1_30s_N90W180_surface.nc")
lat_all = ds.variables['lat'][:]
lon_all = ds.variables['lon'][:]

lat_mask = (lat_all >= 34.5) & (lat_all <= 42.5)
lon_mask = (lon_all >= -121.0) & (lon_all <= -113.5)
lat_idx  = np.where(lat_mask)[0]
lon_idx  = np.where(lon_mask)[0]
z_etopo  = np.array(ds.variables['z'][lat_idx[0]:lat_idx[-1]+1,
                                       lon_idx[0]:lon_idx[-1]+1], dtype=float)
lat_sub = np.array(lat_all[lat_idx])
lon_sub = np.array(lon_all[lon_idx])
ds.close()

to_lonlat = Transformer.from_crs(crs_lcc, 'EPSG:4326', always_xy=True)
XX, YY    = np.meshgrid(x, y)
LON_grd, LAT_grd = to_lonlat.transform(XX.ravel(), YY.ravel())
LON_grd = LON_grd.reshape(XX.shape)
LAT_grd = LAT_grd.reshape(XX.shape)

interp_topo = RegularGridInterpolator(
    (lat_sub, lon_sub), z_etopo,
    method='linear', bounds_error=False, fill_value=np.nan
)
topo = interp_topo(
    np.stack([LAT_grd.ravel(), LON_grd.ravel()], axis=1)
).reshape(XX.shape)

hs = hillshade(np.where(np.isnan(topo), 0.0, topo),
               dx_m=grd.meta['dx'] * 3e2,
               dy_m=grd.meta['dy'] * 3e2)

states      = gpd.read_file(DATAPATH / "political" / "tl_2025_us_state.shp")
states_lcc  = states.to_crs(crs_lcc)
bbox        = box(x.min(), y.min(), x.max(), y.max())
states_clip = states_lcc.clip(bbox)

# %%
def plot_gravity_profiles(x_km, y_km, datasets, y_targets=(4600, 4700), savename=None):
    """Plot E-W gravity profiles at several y-values as tightly stacked,
    small-multiple panels (one row per y_target) sharing the x-axis, each
    labeled with its location in the lower-left corner -- rather than
    overlaying the profiles (which made same-colored lines hard to tell
    apart even with different linestyles).
    """
    n = len(y_targets)
    fig = plt.figure(figsize=(6, 1.35 * n))
    gs = fig.add_gridspec(nrows=n, ncols=1, hspace=0.08)

    axes = []
    for i in range(n):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)

    for i, (ax, y_target) in enumerate(zip(axes, y_targets)):
        row = int(np.argmin(np.abs(y_km - y_target)))
        y_actual = y_km[row]
        for label, data, color in datasets:
            ax.plot(x_km, data[row, :], color=color, lw=1.1)
        ax.set_ylabel('Gravity\n(mGal)', fontsize=8)
        ax.grid(False)
        ax.tick_params(axis='both', which='both', direction='out',
                        top=True, right=True, labelsize=8)
        if i < n - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel('Easting (km, LCC)', fontsize=8)

        ax.text(0.02, 0.05, f'y = {y_actual:.0f} km', fontsize=8,
                va='bottom', ha='left', transform=ax.transAxes)

    handles = [Line2D([0], [0], color=color, lw=1.1) for _, _, color in datasets]
    labels  = [label for label, _, _ in datasets]
    fig.legend(handles, labels, fontsize=7, loc='upper center',
               ncol=len(datasets), frameon=False, bbox_to_anchor=(0.42, 1.02))

    if savename is not None:
        fu.savefig(fig, savename)
    plt.show()

# %% [markdown]
# ### 1.3 Nevada Complete Bouguer Gravity map
#
# **Objective:** familiarise yourself with the spatial pattern of the Bouguer anomaly
# before any corrections are applied.
#
# The **Complete Bouguer anomaly** corrects for:
# - the free-air effect of elevation,
# - the gravitational attraction of the rock slab between the station and the geoid,
# - terrain corrections for nearby topographic relief.
#
# What remains reflects lateral density variations in the crust and upper mantle.

# %%
fig, ax = plt.subplots(figsize=(7, 9))

img = ax.pcolormesh(x, y, gravity, cmap='RdBu_r', shading='auto')
ax.pcolormesh(x, y, hs, cmap='gray', shading='auto', vmin=0, vmax=1, alpha=0.45)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.autoscale(False)
states_clip.boundary.plot(ax=ax, color='black', linewidth=0.5)
plt.colorbar(img, ax=ax, label='Bouguer anomaly (mGal)', shrink=0.7)
ax.set_xlabel('Easting (km, LCC)')
ax.set_ylabel('Northing (km, LCC)')
ax.set_title('Nevada Complete Bouguer Gravity')
ax.set_aspect('equal')
plt.grid(False)
plt.tight_layout()
fu.savefig(fig, "01_nevada_bouguer_map")
plt.show()

# %% [markdown]
# > **Key observations:**
# > - The overall trend is **negative** across Nevada (−250 to −100 mGal), reflecting
# >   thick, low-density crust typical of the extended Basin-and-Range crust.
# > - There is a pronounced **NW–SE gradient**: gravity increases toward the eastern
# >   edge (Colorado Plateau) and decreases toward the Sierra Nevada.  This long-wavelength
# >   signal is dominated by crustal thickness variations, not by individual basins.
# > - Individual basin–range pairs produce gravity "stripes" with wavelengths of 20–50 km,
# >   but they are hard to see against the regional background — the same problem we
# >   observed in the synthetic model.

# %% [markdown]
# ### 1.4 Aliasing on real data — the effect of station spacing
#
# **Objective:** demonstrate how coarsening the sampling of the real Nevada gravity
# grid produces aliasing of the basin-scale signal.
#
# The slider controls the effective station spacing (in km).  At each setting the
# notebook subsamples the full-resolution 2 km grid, plots the resulting map, and
# shows the E-W power spectrum with the new Nyquist wavelength marked.
#
# > **Questions to consider:**
# > - At what spacing do the individual NV basins (~20–50 km wide) begin to alias?
# > - Where does the aliased energy appear in the spectrum?
# > - How does the apparent sediment thickness from the inversion change with spacing?

# %%
from scipy.fft import rfft as _rfft, rfftfreq as _rfftfreq

def _ew_spectrum(grid, dx_km):
    """Mean E-W power spectrum of a 2-D grid."""
    ny_g, nx_g = grid.shape
    k  = _rfftfreq(nx_g, d=dx_km); k[0] = np.nan
    wl = 1.0 / k
    ps = np.zeros(len(k))
    n  = 0
    for row in grid:
        if np.isnan(row).all(): continue
        r = row.copy(); r[np.isnan(r)] = np.nanmean(r)
        r -= np.polyval(np.polyfit(np.arange(nx_g), r, 1), np.arange(nx_g))
        ps += np.abs(_rfft(r))**2; n += 1
    return wl, ps / max(n, 1)


def show_aliasing(spacing_km=2):
    step      = max(1, int(round(spacing_km / grd.meta['dx'])))
    actual_dx = step * grd.meta['dx']
    nyquist   = 2 * actual_dx

    g_sub  = gravity[::step, ::step]
    x_sub  = x[::step]
    y_sub  = y[::step]
    hs_sub = hs[::step, ::step]

    wl_full, pw_full = _ew_spectrum(gravity, grd.meta['dx'])
    wl_sub,  pw_sub  = _ew_spectrum(g_sub,   actual_dx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                              gridspec_kw={'width_ratios': [1, 1.3]})

    ax = axes[0]
    vmin, vmax = np.nanpercentile(gravity, [2, 98])
    ax.pcolormesh(x_sub, y_sub, g_sub, cmap='RdBu_r', shading='auto',
                  vmin=vmin, vmax=vmax)
    ax.pcolormesh(x_sub, y_sub, hs_sub, cmap='gray', shading='auto',
                  vmin=0, vmax=1, alpha=0.35)
    states_clip.boundary.plot(ax=ax, color='black', linewidth=0.5)
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max())
    ax.set_aspect('equal')
    ax.set_title(f'Bouguer gravity\n(spacing = {actual_dx:.0f} km, '
                 f'Nyquist = {nyquist:.0f} km)')
    ax.set_xlabel('Easting (km, LCC)')
    ax.set_ylabel('Northing (km, LCC)')

    ax = axes[1]
    ax.loglog(wl_full, pw_full, 'k',  lw=0.8, alpha=0.5, label='Full res (2 km)')
    ax.loglog(wl_sub,  pw_sub,  'C1', lw=1.0,             label=f'Subsampled ({actual_dx:.0f} km)')
    ax.axvline(nyquist, color='C1', ls='--', lw=1.2,
               label=f'Nyquist = {nyquist:.0f} km')
    ax.axvline(2 * grd.meta['dx'], color='k', ls=':', lw=0.8,
               label=f'Full-res Nyquist = {2*grd.meta["dx"]:.0f} km')
    ax.axvspan(20, 60, alpha=0.10, color='steelblue',
               label='Typical basin width (20–60 km)')
    ax.set_xlim(3, 600); ax.set_ylim(1e2, None)
    ax.set_xlabel('Wavelength (km)')
    ax.set_ylabel('Power (mGal² km)')
    ax.set_title('E–W power spectrum (Bouguer gravity)')
    ax.legend(fontsize=8)

    plt.suptitle(f'Aliasing demo — station spacing {actual_dx:.0f} km',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    # fu.savefig(fig, "01_nevada_bouguer_map")
    plt.show()

    print(f'Spacing: {actual_dx:.0f} km  |  Nyquist: {nyquist:.0f} km  |  '
          f'Grid size: {g_sub.shape[0]}×{g_sub.shape[1]}')
    if nyquist > 20:
        print(f'⚠  Nyquist ({nyquist:.0f} km) > 20 km: basin-scale signal is aliased.')
    else:
        print('✓  Nyquist < 20 km: basin-scale basins are resolved.')


interact(
    show_aliasing,
    spacing_km=FloatSlider(
        min=2, max=30, step=2, value=2,
        description='Spacing (km)',
        style={'description_width': 'initial'},
    ),
);

# %%
# E–W gravity profiles at y = 4600 and 4700 km — Bouguer gravity
plot_gravity_profiles(x, y, [('Bouguer gravity', gravity, 'firebrick')])

# %% [markdown]
# > **Key observations:**
# > - At 2–4 km spacing the individual graben signatures are clearly resolved.
# > - At ~10 km spacing (Nyquist 20 km) the smallest basins begin to blur; the
# >   power spectrum shows the Nyquist cut-off approaching the basin-scale band.
# > - At 16 km spacing (Nyquist 32 km) the 20 km basins alias: the power spectrum
# >   develops spurious peaks at longer wavelengths and the map shows "phantom"
# >   basin-scale structure that does not exist in the geology.
# > - This is exactly the aliasing we diagnosed in the synthetic model — the same
# >   physics, but now visible in real data with real consequences for interpretation.

# %% [markdown]
# ---
# ## Part 2 — Profile and Spectral Analysis
#
# We now extract a 1-D east–west profile through the centre of the grid and
# compute the power spectrum of the Complete Bouguer gravity to quantify the
# dominant basin wavelengths (superimposed on the broader regional trend).

# %% [markdown]
# ### 2.1 E–W gravity profile
#
# **Objective:** look at the raw Bouguer gravity profile before any filtering, to
# see the scale of the regional trend that the high-pass filter (Part 4) will need
# to remove.

# %%
plot_gravity_profiles(
    x, y,
    [('Bouguer gravity', gravity, 'C0')],
    savename="03_bouguer_profile_raw"
)

# %% [markdown]
# > **Key observations:**
# > - The Bouguer gravity profile has a large-amplitude (> 100 mGal) regional ramp.
# > - Superimposed on that ramp are ±20–50 mGal oscillations at the 20–60 km
# >   wavelength of the individual basins — visible, but not yet isolated from
# >   the regional trend.

# %% [markdown]
# ### 2.2 E–W and N–S power spectra
#
# **Objective:** identify the dominant wavelengths of the Complete Bouguer gravity in
# both orientations and compare them to the basin geometry visible in the map.
#
# Each row (E–W) and column (N–S) of the grid is Fourier-transformed independently.
# The mean power spectrum across all rows/columns suppresses row-to-row noise while
# preserving any spatially coherent periodicities.

# %%
dx_km  = grd.meta['dx']
dy_km  = grd.meta['dy']
ny_g, nx_g = gravity.shape

k_ew = rfftfreq(nx_g, d=dx_km); k_ew[0] = np.nan
wavelength_ew = 1.0 / k_ew

power_sum_ew = np.zeros(len(k_ew))
n_rows = 0
for row in gravity:
    if np.isnan(row).all(): continue
    row_filled = row.copy()
    row_filled[np.isnan(row_filled)] = np.nanmean(row_filled)
    row_filled -= np.polyval(np.polyfit(np.arange(nx_g), row_filled, 1), np.arange(nx_g))
    power_sum_ew += np.abs(rfft(row_filled))**2
    n_rows += 1
power_ew = power_sum_ew / n_rows

k_ns = rfftfreq(ny_g, d=dy_km); k_ns[0] = np.nan
wavelength_ns = 1.0 / k_ns

power_sum_ns = np.zeros(len(k_ns))
n_cols = 0
for col in gravity.T:
    if np.isnan(col).all(): continue
    col_filled = col.copy()
    col_filled[np.isnan(col_filled)] = np.nanmean(col_filled)
    col_filled -= np.polyval(np.polyfit(np.arange(ny_g), col_filled, 1), np.arange(ny_g))
    power_sum_ns += np.abs(rfft(col_filled))**2
    n_cols += 1
power_ns = power_sum_ns / n_cols

power_mean = power_ew
wavelength  = wavelength_ew

fig, ax = plt.subplots(figsize=(10, 5))
ax.loglog(wavelength_ew, power_ew, 'firebrick',  lw=0.9, label='E–W')
ax.loglog(wavelength_ns, power_ns, 'C0', lw=0.9, label='N–S', alpha=0.7)
ax.axvspan(20, 60, alpha=0.10, color='steelblue', label='Typical basin width (20–60 km)')
ax.axvline(2 * dx_km, color='k', ls='--', lw=0.8, label=f'Nyquist ({2*dx_km:.0f} km)')
ax.set_xlim(4, 300)
ax.set_xlabel('Wavelength (km)')
ax.set_ylabel('Power (mGal² km)')
ax.set_title('E–W and N–S power spectra of Bouguer gravity')
ax.legend(fontsize=8)
plt.grid(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# > **Key observations:**
# > - Both spectra show a **red-noise background** (power increasing with wavelength),
# >   typical of geophysical potential fields — here dominated by the long-wavelength
# >   regional trend rather than any basin/Moho separation.
# > - The E–W spectrum has elevated power in the 20–60 km band relative to the
# >   smooth power-law background — this is the basin signal.
# > - The N–S spectrum has higher power at longer wavelengths (60–120 km), consistent
# >   with NNE-trending basins that are longer in the N–S direction than the E–W direction.
# > - Both spectra flatten to a noise floor near 10³ mGal² km at wavelengths < 6 km.

# %% [markdown]
# ### 2.3 Statistical significance of spectral peaks
#
# **Objective:** determine which spectral peaks are real signals rather than noise,
# accounting for the spatial correlation of the gravity field (which reduces the
# effective number of independent observations).
#
# The approach:
# 1. Fit a **power-law background** anchored to wavelength ranges away from the
#    basin-scale signal (6–10 km and 80–120 km).
# 2. **Prewhiten** the spectrum by dividing by the background — this removes the
#    red-noise ramp so the significance threshold is a flat horizontal line.
# 3. Apply a **chi-squared test** ($\chi^2$ with DOF degrees of freedom) based on
#    the effective number of independent rows, estimated from the N–S autocorrelation
#    length of the gravity field.

# %%
from scipy.stats import chi2

col_means = np.nanmean(gravity, axis=1)
col_means -= col_means[~np.isnan(col_means)].mean()
col_valid  = col_means.copy()
col_valid[np.isnan(col_valid)] = 0.0
acf = np.correlate(col_valid, col_valid, mode='full')
acf = acf[len(acf) // 2:]
acf /= acf[0]
e_fold_lag  = np.argmax(acf < 1.0 / np.e)
corr_len_km = e_fold_lag * dy_km
n_eff = max(1, int(ny_g * dy_km / corr_len_km))
dof   = 2 * n_eff

print(f'N–S e-folding correlation length : {corr_len_km:.0f} km')
print(f'Effective independent rows (N_eff): {n_eff}')
print(f'Degrees of freedom per spectral bin: {dof}')

anchor_mask = (
    ((wavelength_ew >= 6.2) & (wavelength_ew <= 10.0)) |
    ((wavelength_ew >= 80.0) & (wavelength_ew <= 120.0))
)
log_wl_a = np.log10(wavelength_ew[anchor_mask])
log_pw_a = np.log10(power_ew[anchor_mask])
coeffs   = np.polyfit(log_wl_a, log_pw_a, 1)

background_ew = 10 ** np.polyval(coeffs, np.log10(wavelength_ew))
background_ns = 10 ** np.polyval(coeffs, np.log10(wavelength_ns))

print(f'Power-law fit: P ∝ λ^{coeffs[0]:.2f}'
      f'  (anchored at 6.2–10 km and 80–120 km)')

power_white_ew = power_ew / background_ew
power_white_ns = power_ns / background_ns

thresh_95 = chi2.ppf(0.95, dof) / dof
thresh_99 = chi2.ppf(0.99, dof) / dof


fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.loglog(wavelength_ew, power_ew, 'firebrick',  lw=0.9, label='E–W')
ax.loglog(wavelength_ns, power_ns, 'C0', lw=0.9, label='N–S', alpha=0.7)
ax.loglog(wavelength_ew, background_ew, 'k--', lw=1.4,
          label=f'Power-law background  (P ∝ λ$^{{{coeffs[0]:.2f}}}$)')
ax.axvspan(20, 60, alpha=0.10, color='steelblue')
ax.set_xlim(4, 300)
ax.set_xlabel('Wavelength (km)')
ax.set_ylabel('Power (mGal² km)')
ax.set_title('E–W and N–S power spectra with power-law background')
ax.legend(fontsize=8)
plt.grid(False)
plt.tight_layout()
fu.savefig(fig, "04_power_spectra_ew_ns")
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.loglog(wavelength_ew, power_white_ew, 'firebrick',  lw=0.9, label='E–W prewhitened')
ax.loglog(wavelength_ns, power_white_ns, 'C0', lw=0.9, label='N–S prewhitened', alpha=0.7)
ax.axhline(thresh_95, color='gray',  lw=1.0, ls='-.', label=f'95 % (χ², DOF={dof})')
ax.axhline(thresh_99, color='black', lw=1.0, ls=':',  label=f'99 % (χ², DOF={dof})')
ax.axvspan(20, 60, alpha=0.10, color='steelblue', label='Typical basin width (20–60 km)')

sig_mask_ew = (power_white_ew > thresh_95) & (wavelength_ew >= 6)
ax.plot(wavelength_ew[sig_mask_ew], power_white_ew[sig_mask_ew],
        'o', ms=5, color='firebrick', zorder=5, label='E–W peaks > 95 %')

sig_mask_ns = (power_white_ns > thresh_95) & (wavelength_ns >= 6)
ax.plot(wavelength_ns[sig_mask_ns], power_white_ns[sig_mask_ns],
        's', ms=5, color='C0', zorder=5, alpha=0.8, label='N–S peaks > 95 %')

ax.set_xlim(4, 300)
ax.set_xlabel('Wavelength (km)')
ax.set_ylabel('Power / background')
ax.set_title(
    f'Prewhitened spectra  (N_eff={n_eff}, DOF={dof}, '
    f'corr. length≈{corr_len_km:.0f} km)'
)
ax.legend(fontsize=8)

plt.grid(False)
plt.tight_layout()
fu.savefig(fig, "05_power_spectra_background")
plt.show()

sig_wl_ew = np.sort(wavelength_ew[sig_mask_ew])
if len(sig_wl_ew) > 0:
    gaps     = np.where(np.diff(1.0 / sig_wl_ew) > 0.05)[0] + 1
    clusters = np.split(sig_wl_ew, gaps)
    print('E–W significant clusters (>95 %):')
    for c in clusters:
        ratios  = [power_white_ew[wavelength_ew == w][0] for w in c]
        peak_wl = c[np.argmax(ratios)]
        print(f'  λ = {c[0]:.1f}–{c[-1]:.1f} km  (peak {peak_wl:.1f} km, ratio={max(ratios):.2f})')
else:
    print('E–W: no peaks exceed 95 % threshold.')

sig_wl_ns = np.sort(wavelength_ns[sig_mask_ns])
if len(sig_wl_ns) > 0:
    gaps     = np.where(np.diff(1.0 / sig_wl_ns) > 0.05)[0] + 1
    clusters = np.split(sig_wl_ns, gaps)
    print('N–S significant clusters (>95 %):')
    for c in clusters:
        ratios  = [power_white_ns[wavelength_ns == w][0] for w in c]
        peak_wl = c[np.argmax(ratios)]
        print(f'  λ = {c[0]:.1f}–{c[-1]:.1f} km  (peak {peak_wl:.1f} km, ratio={max(ratios):.2f})')
else:
    print('N–S: no peaks exceed 95 % threshold.')

# %% [markdown]
# > **Key observations:**
# > - After prewhitening, the E–W spectrum shows a significant cluster in the
# >   **16–47 km** wavelength range (peak near 22 km) — the dominant basin spacing.
# > - The N–S spectrum peaks at longer wavelengths (17–79 km, peak ~44 km),
# >   confirming that basins are elongated N–S.
# > - The wide DOF confidence intervals (DOF = 16) reflect the strong N–S spatial
# >   correlation: adjacent rows sample essentially the same geology, so only ~8
# >   truly independent E–W profiles contribute.

# %% [markdown]
# ---
# ## Part 3 — High-Pass Filtering and Basin Inversion
#
# The spectral analysis confirms that basin-scale power sits at wavelengths
# shorter than ~60 km.  Before inverting for sediment thickness we apply a
# spectral high-pass filter directly to the Complete Bouguer gravity to remove
# the regional trend that would otherwise produce unrealistically thick
# "sediment" in broad regional lows.

# %% [markdown]
# ### 3.1 Raised-cosine high-pass filter
#
# **Objective:** apply a spectral filter that:
# - Blocks wavelengths $>$ 100 km (regional/crustal-thickness signal, flexure)
# - Fully passes wavelengths $<$ 60 km (individual basin signal)
# - Uses a smooth raised-cosine taper in between to prevent **Gibbs ringing**
#
# > **Note:** NaN values at survey edges are filled with the grid mean before the FFT
# > (NaNs propagate through the transform and poison all output), then restored
# > in the filtered result.

# %%
dx_km_hp = grd.meta['dx']
dy_km_hp = grd.meta['dy']
ny_g, nx_g = gravity.shape

kx_ck = fftfreq(nx_g, d=dx_km_hp)
ky_ck = fftfreq(ny_g, d=dy_km_hp)
KX_ck, KY_ck = np.meshgrid(kx_ck, ky_ck)
k_ck  = np.sqrt(KX_ck**2 + KY_ck**2)

k_lo = 0.01          # cycles/km  — fully blocked below (λ > 100 km)
k_hi = 1.0 / 60.0   # cycles/km  — fully passed above  (λ <  60 km)

# %% [markdown]
# **Filter response.** The raised-cosine taper as a function of wavelength --
# fully blocked beyond 100 km, fully passed below 60 km, with a smooth
# transition in between rather than a sharp cutoff (which would ring).

# %%
wl_plot = np.geomspace(4, 300, 400)   # km
k_plot  = 1.0 / wl_plot               # cycles/km
h_plot  = np.where(
    k_plot <= k_lo, 0.0,
    np.where(
        k_plot >= k_hi, 1.0,
        0.5 * (1 - np.cos(np.pi * (k_plot - k_lo) / (k_hi - k_lo)))
    )
)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(wl_plot, h_plot, color='k', lw=1.5)
ax.axvspan(1 / k_hi, 1 / k_lo, color='steelblue', alpha=0.12, label='Transition band')
ax.axvline(1 / k_lo, color='C3', ls='--', lw=1, label=f'Stop  (λ > {1/k_lo:.0f} km)')
ax.axvline(1 / k_hi, color='C2', ls='--', lw=1, label=f'Pass  (λ < {1/k_hi:.0f} km)')
ax.set_xscale('log')
ax.set_xlim(4, 300)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('Wavelength (km)')
ax.set_ylabel('Filter gain $H(k)$')
ax.set_title('Raised-cosine high-pass filter response')
ax.legend(fontsize=8)
plt.grid(False)
plt.tight_layout()
fu.savefig(fig, "05b_highpass_filter_curve")
plt.show()

# %%
hp_filter = np.where(
    k_ck <= k_lo, 0.0,
    np.where(
        k_ck >= k_hi, 1.0,
        0.5 * (1 - np.cos(np.pi * (k_ck - k_lo) / (k_hi - k_lo)))
    )
)

nan_mask = np.isnan(gravity)
gravity_filled = gravity.copy()
gravity_filled[nan_mask] = np.nanmean(gravity)
g_hp = np.real(ifft2(fft2(gravity_filled) * hp_filter))
g_hp[nan_mask] = np.nan

print(f'Pass band  : k > {k_hi:.4f} cycles/km  (λ < {1/k_hi:.0f} km)')
print(f'Stop band  : k < {k_lo:.4f} cycles/km  (λ > {1/k_lo:.0f} km)')
print(f'gravity std : {np.nanstd(gravity):.2f} mGal')
print(f'g_hp     std : {np.nanstd(g_hp):.2f} mGal')

fig, ax = plt.subplots(figsize=(7, 9))
img = ax.pcolormesh(x, y, g_hp, cmap='RdBu_r', shading='auto', vmin=-60, vmax=60)
ax.pcolormesh(x, y, hs, cmap='gray', shading='auto', vmin=0, vmax=1, alpha=0.45)
states_clip.boundary.plot(ax=ax, color='black', linewidth=0.5)
ax.set_title('High-pass filtered  (λ < 60 km passed)')
ax.set_aspect('equal')
ax.set_xlabel('Easting (km, LCC)')
ax.set_ylabel('Northing (km, LCC)')
plt.colorbar(img, ax=ax, label='mGal', shrink=0.7)
plt.grid(False)
plt.tight_layout()
fu.savefig(fig, "06_highpass_filtered_map")
plt.show()

# %%
plot_gravity_profiles(
    x, y, #[('Bouguer gravity', gravity, 'firebrick'),
    [('HP filtered',     g_hp,    'darkorange')], #],
    savename="07_bouguer_vs_hp_profile",
)

# %% [markdown]
# > **Key observations:**
# > - The filtered map shows the individual basin lows and highs but with the broad
# >   regional trend removed.
# > - The amplitude range has narrowed substantially relative to the raw Bouguer
# >   gravity (roughly ±30 mGal vs > 100 mGal regional ramp before filtering).
# > - Basin outlines are sharper and align more closely with the topographic basins
# >   visible in the hillshade — the filter has improved the spatial resolution of
# >   the shallow source.

# %% [markdown]
# ### 3.2 Parker–Oldenburg sediment thickness inversion
#
# **Objective:** convert the high-pass filtered gravity residual into an estimate
# of **sediment thickness** $h(x,y)$ by inverting the linear Bouguer-slab relation
# $$g(x,y) = 2\pi G\,\Delta\rho\, h(x,y)$$
# using the spectral Parker–Oldenburg iterative method.
#
# **Parameters**
#
# | Parameter | Value | Rationale |
# |-----------|-------|-----------|
# | $\Delta\rho$ | −170 kg m$^{-3}$ | basement (2670) minus sediment (2500) |
# | $k_\text{max}$ | 85 % Nyquist | low-pass to stabilise the deconvolution |
# | Convergence | 0.1 m change in $h$ | ~30 iterations sufficient |

# %%
drho_sed = 2500 - 2670   # kg/m³, basement minus sediment
dx_m = grd.meta['dx'] * 1e3
dy_m = grd.meta['dy'] * 1e3

h_sed, rms_hist = parker_oldenburg(
    g_hp,
    dx=dx_m, dy=dy_m,
    drho=drho_sed,
    n_iter=30, tol=0.1, kmax_factor=0.85,
)

print(f'Iterations run : {len(rms_hist)}')
print(f'Initial RMS    : {rms_hist[0]:.2f} mGal')
print(f'Final RMS      : {rms_hist[-1]:.2f} mGal')
print(f'Sediment depth : {np.nanmin(h_sed)/1e3:.2f} – {np.nanmax(h_sed)/1e3:.2f} km')

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(range(1, len(rms_hist)+1), rms_hist, 'k.-')
ax.set_xlabel('Iteration')
ax.set_ylabel('RMS residual (mGal)')
ax.set_title('Parker-Oldenburg convergence')
plt.grid(False)
plt.tight_layout()
plt.show()

cities = {
    'Las Vegas':   (-115.1398, 36.1699, 's'),
    'Reno':        (-119.8138, 39.5296, 's'),
    'Elko':        (-115.7631, 40.8324, 's'),
    'Eureka':      (-115.9631, 39.5077, 's'),
    'Ely':         (-114.8891, 39.2477, 's'),
    'Winnemucca':  (-117.7357, 40.9730, 's'),
    'Carson City': (-119.7674, 39.1638, '*'),
}
to_lcc = Transformer.from_crs('EPSG:4326', crs_lcc, always_xy=True)

fig, ax = plt.subplots(figsize=(7, 9))
img = ax.pcolormesh(x, y, h_sed / 1e3, cmap='CMRmap_r', shading='auto',
                    vmin=0, vmax=np.nanpercentile(h_sed / 1e3, 98))
ax.pcolormesh(x, y, hs, cmap='gray', shading='auto', vmin=0, vmax=1, alpha=0.45)
plt.colorbar(img, ax=ax, label='Sediment thickness (km)', shrink=0.7)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.autoscale(False)
states_clip.boundary.plot(ax=ax, color='white', linewidth=0.8)
for name, (lon_c, lat_c, marker) in cities.items():
    x_c, y_c = to_lcc.transform(lon_c, lat_c)
    size = 90 if marker == '*' else 45
    ax.scatter(x_c, y_c, marker=marker, s=size, facecolor='white',
               edgecolor='black', linewidth=0.8, zorder=6)
    ax.annotate(name, (x_c, y_c), xytext=(4, 4), textcoords='offset points',
                fontsize=7, color='white',
                fontweight='bold',
                path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])
ax.set_xlabel('Easting (km, LCC)')
ax.set_ylabel('Northing (km, LCC)')
ax.set_title('Basin sediment thickness from high-pass filtered Bouguer gravity\n'
             f'(Parker-Oldenburg, $\\Delta\\rho$ = {drho_sed} kg/m³)')
ax.set_aspect('equal')
plt.grid(False)
plt.tight_layout()
fu.savefig(fig, "08_sediment_thickness_map")
plt.show()

# %% [markdown]
# ### 3.3 Alternative inversion: exponential compaction model
#
# **Objective:** the Parker–Oldenburg inversion above assumes a single, constant
# density contrast between sediment and basement. Real basin fill compacts with
# depth — porosity (and hence the density deficit relative to basement) is
# highest near the surface and decays as grains are compressed together. We
# model this with Athy's law:
# $$\phi(z) = \phi_0\, e^{-z/z_c}, \qquad \phi_0 = 0.20,\ \ z_c = 1.5\ \text{km}$$
#
# The sediment density at depth is $\rho_\text{sed}(z) = \rho_\text{grain} -
# (\rho_\text{grain}-\rho_\text{fluid})\,\phi(z)$, so its contrast with the
# basement is
# $$\Delta\rho(z) = (\rho_\text{grain}-\rho_\text{basement})
#   - (\rho_\text{grain}-\rho_\text{fluid})\,\phi(z)$$
#
# The first term is the **residual contrast that survives full compaction** —
# it is only zero if the sediment grain density happens to equal the basement
# density. Here we keep them distinct ($\rho_\text{grain}=2550$,
# $\rho_\text{basement}=2670\ \text{kg/m}^3$), so a fully compacted column
# still carries a permanent $-120\ \text{kg/m}^3$ deficit — compaction makes
# the *rate* of mass deficit accumulation decay with depth, but never drives it
# to zero, so there is **no saturation**: thickness can still grow without
# bound for a large enough anomaly, just more slowly than the constant-$\Delta
# \rho$ model would predict.
#
# Integrating down to a basin thickness $h$ gives the "Bouguer-slab" gravity
# anomaly:
# $$g(h) = 2\pi G\left[(\rho_\text{grain}-\rho_\text{basement})\,h -
#   (\rho_\text{grain}-\rho_\text{fluid})\,\phi_0\, z_c
#   \left(1 - e^{-h/z_c}\right)\right]$$
#
# This is transcendental in $h$ (a linear term plus a decaying exponential), so
# we solve it per pixel with a **vectorized Newton–Raphson** iteration (the
# same $\,\delta p = r / J\,$ update used by `gauss_newton`
# ([src/inversion/gauss_newton.py](../../../src/inversion/gauss_newton.py)) in
# the scalar-parameter case — applied elementwise across the whole grid at
# once rather than looping the generic multi-parameter solver over ~10⁵
# independent single-pixel problems, which would be far slower for no benefit
# here since each pixel is an independent 1-parameter root find):
# $$J(h) = \frac{dg}{dh} = 2\pi G\,\Delta\rho(h), \qquad
#   h_{k+1} = h_k + \frac{g_\text{hp} - g(h_k)}{J(h_k)}$$

# %%
G_const      = 6.67430e-11   # m^3 kg^-1 s^-2
phi0         = 0.20          # surface porosity
z_c          = 1.5e3         # compaction (e-folding) depth, m
rho_grain    = 2550.0        # kg/m^3, sediment grain density
rho_fluid    = 1100.0        # kg/m^3, pore fluid (water)
rho_basement = 2670.0        # kg/m^3, basement density (matches Parker–Oldenburg above)

drho_far = rho_grain - rho_basement   # kg/m^3, residual contrast at full compaction
print(f'Residual density contrast at full compaction : {drho_far:.1f} kg/m³')
print(f'Surface density contrast (z=0)                : '
      f'{drho_far - (rho_grain - rho_fluid) * phi0:.1f} kg/m³')

def _fwd_g(h):
    """Forward gravity anomaly (SI, m/s^2) for basin thickness h (m)."""
    return 2 * np.pi * G_const * (
        drho_far * h - (rho_grain - rho_fluid) * phi0 * z_c * (1 - np.exp(-h / z_c))
    )

def _dgdh(h):
    """Jacobian dg/dh = 2*pi*G*drho(h)."""
    return 2 * np.pi * G_const * (
        drho_far - (rho_grain - rho_fluid) * phi0 * np.exp(-h / z_c)
    )

g_hp_si  = g_hp * 1e-5                 # mGal -> SI (m/s^2)
nan_edge = np.isnan(g_hp)
basin    = (g_hp_si < 0) & ~nan_edge   # only invert where gravity is low

h = np.zeros_like(g_hp)
h[basin] = 500.0                       # initial guess, m

for _ in range(40):
    resid = g_hp_si - _fwd_g(h)
    jac   = _dgdh(h)
    dp = np.zeros_like(h)
    dp[basin] = resid[basin] / jac[basin]
    h[basin] = np.clip(h[basin] + dp[basin], 0.0, None)

h_comp = h.copy()
h_comp[nan_edge] = np.nan

final_resid_mgal = np.abs(g_hp_si - _fwd_g(h))[basin] * 1e5
print(f'Newton max residual         : {np.nanmax(final_resid_mgal):.2e} mGal')
print(f'Compaction thickness range : {np.nanmin(h_comp)/1e3:.2f} '
      f'– {np.nanmax(h_comp)/1e3:.2f} km')

fig, ax = plt.subplots(figsize=(7, 9))
img = ax.pcolormesh(x, y, h_comp / 1e3, cmap='CMRmap_r', shading='auto',
                    vmin=0, vmax=np.nanpercentile(h_comp / 1e3, 98))
ax.pcolormesh(x, y, hs, cmap='gray', shading='auto', vmin=0, vmax=1, alpha=0.45)
plt.colorbar(img, ax=ax, label='Sediment thickness (km)', shrink=0.7)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.autoscale(False)
states_clip.boundary.plot(ax=ax, color='white', linewidth=0.8)
for name, (lon_c, lat_c, marker) in cities.items():
    x_c, y_c = to_lcc.transform(lon_c, lat_c)
    size = 90 if marker == '*' else 45
    ax.scatter(x_c, y_c, marker=marker, s=size, facecolor='white',
               edgecolor='black', linewidth=0.8, zorder=6)
    ax.annotate(name, (x_c, y_c), xytext=(4, 4), textcoords='offset points',
                fontsize=7, color='white',
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])
ax.set_xlabel('Easting (km, LCC)')
ax.set_ylabel('Northing (km, LCC)')
ax.set_title('Basin sediment thickness from exponential compaction model\n'
             r'($\phi_0$ = 0.20, $z_c$ = 1.5 km, $\rho_\mathrm{grain}$ = 2550 kg/m³)')
ax.set_aspect('equal')
plt.grid(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.4 Comparing the two thickness estimates
#
# **Objective:** see where the constant-$\Delta\rho$ and compaction models
# agree, and where they diverge.

# %%
mask_cmp = np.isfinite(h_sed) & np.isfinite(h_comp)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(h_sed[mask_cmp] / 1e3, h_comp[mask_cmp] / 1e3,
           s=4, color='C2', alpha=0.15, edgecolor='none')
lims = (0, max(np.nanpercentile(h_sed[mask_cmp] / 1e3, 99.5),
               np.nanpercentile(h_comp[mask_cmp] / 1e3, 99.5)))
ax.plot(lims, lims, 'k--', lw=1.0, label='1:1')
ax.set_xlim(*lims)
ax.set_ylim(*lims)
ax.set_xlabel('Parker–Oldenburg thickness (km)')
ax.set_ylabel('Compaction-model thickness (km)')
ax.set_title('Constant-$\\Delta\\rho$ vs. exponential-compaction thickness')
ax.set_aspect('equal')
ax.legend(fontsize=8)
plt.grid(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# > **Key observations:**
# > - At the surface the compaction model's density deficit (~−430 kg/m³, i.e.
# >   $\Delta\rho(0) = (\rho_\text{grain}-\rho_\text{basement}) -
# >   \phi_0(\rho_\text{grain}-\rho_\text{fluid})$) is much larger in magnitude
# >   than the constant −170 kg/m³ used by Parker–Oldenburg here, so a thin
# >   compacting column already produces the same gravity signal as a thicker
# >   constant-density one — the compaction curve sits *below* the 1:1 line for
# >   shallow basins.
# > - As the modelled column gets deeper, its density deficit relaxes toward
# >   the residual full-compaction value (~−120 kg/m³, nonzero because
# >   $\rho_\text{grain}\ne\rho_\text{basement}$), which is close in magnitude
# >   to Parker–Oldenburg's constant −170 kg/m³ — so the compaction/PO
# >   thickness ratio climbs from ~0.4 for shallow basins (< 1 km) toward
# >   ~0.7 for the deepest ones (5–8 km), converging toward 1 without actually
# >   reaching it over the range sampled here. (The two never fully agree
# >   because Parker–Oldenburg is a spectral inversion — accounting for how
# >   gravity from a source is attenuated with wavelength/depth — while this
# >   compaction model is solved pointwise in real space; a literal 1:1 match
# >   isn't expected even at the same asymptotic density contrast.)
# > - Because the residual contrast never reaches zero, thickness is not
# >   capped the way it was when grain and basement density were assumed
# >   equal — there is no saturation limit and no masked-out cells here, only
# >   the pre-existing edges of the survey.
# > - Because the two models use different (and not directly comparable)
# >   density contrasts, this plot is best read as a shape comparison — do the
# >   basins line up, does the ranking of "deep" vs "shallow" basins agree —
# >   rather than a literal thickness-for-thickness match.

# %% [markdown]
# ### 3.5 Distribution of inverted sediment thickness
#
# **Objective:** check whether the deeper (~8 km) sediment estimates are common
# or rare outliers, and compare against a known reference point — Las Vegas
# Valley basin fill is documented at roughly 5 km.

# %%
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

h_km_valid = (h_sed / 1e3)[np.isfinite(h_sed)]
h_km_valid = h_km_valid[h_km_valid > 0]

xlim = (1e-2, 1e1)
bins = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 60)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(h_km_valid, bins=bins, color='lightblue', edgecolor='none')
ax.axvline(5.0, color='k', ls='--', lw=1.2, label='Las Vegas basin fill (~5 km)')
ax.set_xscale('log')
ax.set_xlim(*xlim)
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Sediment thickness (km)')
ax.set_ylabel('Grid cell count')
ax.set_title('Distribution of inverted basin sediment thickness')
ax.legend(fontsize=8)
plt.grid(False)
plt.tight_layout()
fu.savefig(fig, "09_sediment_thickness_distribution")
plt.show()

frac_gt5 = (h_km_valid > 5.0).mean() * 100
frac_gt8 = (h_km_valid > 8.0).mean() * 100
print(f'Median thickness  : {np.median(h_km_valid):.2f} km')
print(f'95th percentile   : {np.percentile(h_km_valid, 95):.2f} km')
print(f'Cells > 5 km      : {frac_gt5:.2f} %')
print(f'Cells > 8 km      : {frac_gt8:.2f} %')

# %% [markdown]
# > **Key observations:**
# > - Most grid cells cluster at modest thickness, with the bulk of the
# >   distribution well under 5 km — consistent with typical Basin-and-Range
# >   basin fill (Las Vegas Valley itself reaches ~5 km).
# > - The handful of cells approaching 8 km sit in the extreme tail of the
# >   distribution, not the bulk of the map — they are most likely the
# >   leftover-regional-signal artefact described above, concentrated in a few
# >   broad low-gravity areas rather than spread across many basins.

# %%
plot_gravity_profiles(
    x, y, #[('Bouguer gravity', gravity, 'firebrick'),
    [('HP filtered',     g_hp,    'darkorange')], #],
)

# %% [markdown]
# > **Key observations:**
# > - The inversion converges in ~30 iterations, reducing the RMS residual from
# >   ~15 mGal to ~7 mGal.
# > - Sediment thickness reaches up to ~8 km in places — noticeably deeper than the
# >   ~3 km seismic/well constraints in the major NV basins.  This is the direct
# >   consequence of relying on the HP filter alone: some long-wavelength regional
# >   signal (crustal-thickness variation) leaks through the 60–100 km transition
# >   band and is misread by the inversion as extra sediment.
# > - Thick sediment still broadly coincides spatially with topographic basins in
# >   the hillshade, but the amplitude is less trustworthy than in a two-step
# >   isostatic + high-pass workflow — a tradeoff worth keeping in mind when using
# >   these HP-only depths quantitatively.

# %% [markdown]
# ---
# ## Part 4 — Impact of Aliasing on the Basin Inversion
#
# The previous widget showed how coarser sampling moves the Nyquist wavelength into
# the basin-scale band.  This widget goes further: it **reruns the complete processing
# pipeline** (high-pass filter → Parker–Oldenburg inversion) on the decimated grid so
# you can see directly how the sediment-thickness model degrades when the data are
# aliased.

# %% [markdown]
# ### 4.1 Full-pipeline aliasing demonstration
#
# **Objective:** understand why station spacing is a fundamental limit on inversion
# quality, independent of the sophistication of the inversion algorithm.
#
# Use the slider to decimate the gravity grid — keeping every Nth point in both
# directions — then observe:
#
# 1. How the Nyquist wavelength (= 2 × station spacing) moves across the power
#    spectrum relative to the 20–60 km basin band.
# 2. How the sediment-thickness map produced by the Parker–Oldenburg inversion
#    degrades as individual basins become unresolvable.

# %%
import matplotlib.gridspec as _mgs

_DRHO_SED = 2500 - 2670
_K_STOP   = 0.01
_K_PASS   = 1.0 / 60.0


def _pipeline(step):
    """Decimate gravity by factor `step` then run HP→PO pipeline."""
    g_s   = gravity[::step, ::step].copy()
    x_s   = x[::step]; y_s = y[::step]
    dx_s  = grd.meta['dx'] * step
    dy_s  = grd.meta['dy'] * step
    ny_s, nx_s = g_s.shape
    nan_s = np.isnan(g_s)

    kx_k = _fftfreq(nx_s, d=dx_s)
    ky_k = _fftfreq(ny_s, d=dy_s)
    KX_k, KY_k = np.meshgrid(kx_k, ky_k)
    k_ck2 = np.sqrt(KX_k**2 + KY_k**2)
    taper = np.where(
        k_ck2 <= _K_STOP, 0.0,
        np.where(k_ck2 >= _K_PASS, 1.0,
                 0.5*(1 - np.cos(np.pi*(k_ck2 - _K_STOP)/(_K_PASS - _K_STOP))))
    )
    gf     = g_s.copy(); gf[nan_s] = np.nanmean(g_s)
    g_hp_s = np.real(_ifft2(taper * _fft2(gf)))
    g_hp_s[nan_s] = np.nan

    h_s, rms_s = parker_oldenburg(
        g_hp_s,
        dx=dx_s*1e3, dy=dy_s*1e3,
        drho=_DRHO_SED,
        n_iter=30, tol=0.1, kmax_factor=0.85,
    )

    kf = _rfftfreq2(nx_s, d=dx_s); kf[0] = np.nan
    wl = 1.0 / kf
    ps = np.zeros(len(kf)); nrow = 0
    for row in g_s:
        if np.isnan(row).all(): continue
        r = row.copy(); r[np.isnan(r)] = np.nanmean(r)
        r -= np.polyval(np.polyfit(np.arange(nx_s), r, 1), np.arange(nx_s))
        ps += np.abs(_rfft2(r))**2; nrow += 1
    ps /= max(nrow, 1)

    return x_s, y_s, h_s, rms_s, wl, ps, dx_s, g_s, g_hp_s


def _ordinal(n):
    return ("every point" if n == 1
            else "every 2nd point" if n == 2
            else "every 3rd point" if n == 3
            else f"every {n}th point")


def show_pipeline_aliasing(decimation=1):
    step = int(decimation)
    x_s, y_s, h_s, rms_s, wl, ps, dx_s, g_s, g_hp_s = _pipeline(step)
    nyquist = 2 * dx_s
    aliased = nyquist > 30

    h_km = h_s / 1e3
    pos  = h_km[h_km > 0]
    vmax_h = max(float(np.nanpercentile(pos, 95)) if pos.size else 3.0, 0.1)

    fig  = plt.figure(figsize=(14, 13))
    spec = _mgs.GridSpec(3, 2, figure=fig, height_ratios=[4, 1.5, 1.5],
                         hspace=0.45, wspace=0.3)
    axes    = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1])]
    ax_p4600 = fig.add_subplot(spec[1, :])
    ax_p4700 = fig.add_subplot(spec[2, :])

    ax = axes[0]
    im = ax.pcolormesh(x_s, y_s, h_km, cmap='CMRmap_r', shading='auto',
                       vmin=0, vmax=vmax_h)
    ax.pcolormesh(x, y, hs, cmap='gray', shading='auto', vmin=0, vmax=1, alpha=0.45)
    plt.colorbar(im, ax=ax, label='Sediment thickness (km)', shrink=0.85)
    states_clip.boundary.plot(ax=ax, color='white', linewidth=0.8)
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max())
    ax.set_aspect('equal')
    status = '⚠ ALIASED' if aliased else '✓ Resolved'
    ax.set_title(
        f'Sediment thickness — {_ordinal(step)}\n'
        f'Δx = {dx_s:.0f} km  |  Nyquist = {nyquist:.0f} km  [{status}]'
    )
    ax.set_xlabel('Easting (km, LCC)')
    ax.set_ylabel('Northing (km, LCC)')

    ax = axes[1]
    ax.loglog(wl, ps, 'firebrick', lw=0.8)
    ax.axvline(nyquist, color='C1', ls='--', lw=1.8,
               label=f'Nyquist = {nyquist:.0f} km')
    ax.axvspan(20, 60, alpha=0.12, color='steelblue',
               label='Typical basin width (20–60 km)')
    ax.set_xlim(3, 600)
    ax.set_xlabel('Wavelength (km)')
    ax.set_ylabel('Power (mGal² km)')
    ax.set_title('E–W Bouguer gravity power spectrum')
    ax.legend(fontsize=9)

    for ax_p, y_target in [(ax_p4600, 4600), (ax_p4700, 4700)]:
        row_p    = int(np.argmin(np.abs(y_s - y_target)))
        y_act    = y_s[row_p]
        row_orig = int(np.argmin(np.abs(y - y_target)))
        ax_p.plot(x,   gravity[row_orig, :],  color='C3', lw=1, ls='--', alpha=0.5,
                  label='Bouguer gravity (original)')
        ax_p.plot(x,   g_hp[row_orig, :],     color='C2', lw=1, ls='--', alpha=0.5,
                  label='HP filtered (original)')
        ax_p.plot(x_s, g_s[row_p, :],         color='C3', lw=1.5,
                  label='Bouguer gravity (decimated)')
        ax_p.plot(x_s, g_hp_s[row_p, :],      color='C2', lw=1.5,
                  label='HP filtered (decimated)')
        ax_p.set_xlim(x.min(), x.max())
        ax_p.set_ylabel('Gravity (mGal)')
        ax_p.set_xlabel('Easting (km, LCC)')
        ax_p.set_title(f'E–W profile at y ≈ {y_act:.0f} km')
        ax_p.legend(fontsize=8)
        ax_p.grid(True)

    plt.suptitle(
        f'Full-pipeline aliasing — {_ordinal(step)} '
        f'({y_s.size}×{x_s.size} grid, {len(rms_s)} iter)',
        fontsize=11, fontweight='bold',
        color='firebrick' if aliased else 'black'
    )
    plt.tight_layout()
    plt.show()

    print(f'Spacing: {dx_s:.0f} km  |  Nyquist: {nyquist:.0f} km  |  '
          f'Grid: {y_s.size}×{x_s.size}  |  Iterations: {len(rms_s)}')
    if aliased:
        print(f'  ⚠  Nyquist ({nyquist:.0f} km) > 30 km — basin geometry is NOT resolved.')
    else:
        print(f'  ✓  Nyquist ({nyquist:.0f} km) ≤ 30 km — basin-scale structure is resolved.')


interact(
    show_pipeline_aliasing,
    decimation=IntSlider(
        min=1, max=20, step=1, value=1,
        description='Decimation',
        style={'description_width': 'initial'},
    ),
);

# %% [markdown]
# > **Key observations:**
# > - **Decimation = 1** (2-km spacing, Nyquist = 4 km): the Nyquist line is well to
# >   the left of the basin band.  The sediment map reproduces individual basin troughs.
# > - **Decimation = 3–4** (6–8 km, Nyquist = 12–16 km): smaller sub-basins start to
# >   merge in the sediment map; peak depths shift slightly.
# > - **Decimation = 5** (10 km, Nyquist = 20 km): the inversion still converges but
# >   the sediment map has lost fine structure.
# > - **Decimation ≥ 7** (≥ 14 km, Nyquist ≥ 28 km): basin-scale energy is now aliased;
# >   the inversion produces artefacts that bear no physical relationship to real basins.
# > - **Take-away:** no amount of sophisticated processing can recover wavelengths
# >   shorter than the Nyquist.  The station spacing is the hard limit on resolution.
