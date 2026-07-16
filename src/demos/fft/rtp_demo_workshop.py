# %% [markdown]
# # Reduction to the Pole and Magnetic Anomalies: A Maar Case Study
#
# Magnetic anomalies depend on geometry, magnetisation direction, and observation
# latitude.  Unlike gravity, magnetic anomalies are inherently asymmetric.
# Reduction to the pole (RTP) is a Fourier-domain operation that simplifies
# interpretation.  RTP is powerful — but fragile.
#
# This notebook applies the Fourier ideas developed earlier to magnetic
# interpretation, where phase matters critically.

# %% [markdown]
# ## Learning objectives
#
# By the end of this notebook, students should be able to:
# - Explain why magnetic anomalies are asymmetric away from the pole
# - Describe RTP as a phase correction, not a filter
# - Predict when RTP will work well and when it will fail
# - Interpret RTP results for a simple geological body
# - Recognize artefacts caused by sampling, noise, and remanence

# %% [markdown]
# ### Geological motivation: Why maars?
#
# **Purpose:**
# We'll be looking at an idealized magnetic anomaly for a volcanic maar.
#
# A *maar* is a volcanic explosion crater with infilled sediments.  The (generally)
# mafic magmas are saturated with volatiles.  As they rise, the volatiles exsolve
# rapidly from the magma creating an explosion underground as they rapidly rise to
# the surface.  They are often roughly conical in cross section, have a strong
# magnetic contrast with surrounding rocks, and produce compact, high-amplitude
# magnetic anomalies.
#
# If you flew a magnetic survey over a maar at mid-latitude, where would the anomaly
# peak be relative to the vent?

# %% [markdown]
# ### Imports & Setup

# %%
from pathlib import Path
import sys

_ROOT    = Path(__file__).parent.parent.parent.parent   # src/demos/fft/ -> project root
DATAPATH = _ROOT / "data"
FIGDIR = Path(__file__).resolve().parent / "tmp_figures"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as _gs

from scipy.fft import fft2, ifft2, fftfreq
import cmcrameri.cm as cmc

from src.physics.magnetics.reduce2pole import reduce_to_pole, pseudo_gravity

from src.utils.figure_utils import figutils as fu
fu = fu(FIGDIR)


plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.grid": True,
})

# %% [markdown]
# ## Coordinate system & assumptions
#
# **Objective:** Make assumptions explicit.
#
# State clearly:
# - Cartesian coordinates *x*, *y*, *z*
# - Observation at the surface
# - Induced magnetisation only (for now)
# - Uniform susceptibility
#
# Known inducing field:
# - Inclination, *I*
# - Declination, *D*

# %%
# Spatial grid (meters)
L  = 10_000      # half-width of domain (m)
dx = 50.0        # grid spacing (m)

x = np.arange(-L, L + dx, dx)
y = np.arange(-L, L + dx, dx)
X, Y = np.meshgrid(x, y)

ny, nx = X.shape

# %% [markdown]
# ## PART A — FORWARD MAGNETIC MODEL
#
# ### Maar / cone geometry
#
# **Objective:** Define the source geometry conceptually.
#
# Include:
# - Cone radius
# - Cone height
# - Burial depth
# - Susceptibility contrast
#
# Before executing: Sketch the body in cross-section and predict the anomaly shape.

# %%
def cone_geometry(X, Y, radius=1500, height=800):
    """Cone-shaped body representing a maar / diatreme. Returns thickness (m)."""
    R = np.sqrt(X**2 + Y**2)
    h = height * (1 - R / radius)
    h[h < 0] = 0.0
    return h


def plot_mag_map(ax, X, Y, data, title, cmap='RdBu_r', label='nT'):
    """Consistently-styled pcolormesh map on a given axes."""
    im = ax.pcolormesh(X / 1000, Y / 1000, data, cmap=cmap, shading='auto')
    plt.colorbar(im, ax=ax, label=label, shrink=0.85)
    ax.set_title(title)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_aspect('equal')


def plot_mag_profile(ax, coords, datasets, title, direction='EW'):
    """Plot profiles through the centre of the grid."""
    mid = len(coords) // 2
    for entry in datasets:
        label, data, color = entry[:3]
        ls = entry[3] if len(entry) > 3 else '-'
        profile = data[mid, :] if direction == 'EW' else data[:, mid]
        ax.plot(coords / 1000, profile, color=color, ls=ls, label=label)
    xlabel = 'x (km)' if direction == 'EW' else 'y (km)'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True)


h_cone = cone_geometry(X, Y)

# %% [markdown]
# ### Define cone geometry (subsurface model)
#
# Purpose:
# - Define a simple axisymmetric body (cone or truncated cone).
#
# No magnetics yet — just geometry.

# %%
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

plot_mag_map(axs[0], X, Y, h_cone,
             title='Maar / Cone Geometry', cmap='YlOrBr', label='Thickness (m)')

plot_mag_profile(axs[1], x, [('Depth', h_cone, 'C1')],
                 title='Cone geometry — E–W (y = 0)')
axs[1].set_ylabel("Depth (m)")
axs[1].invert_yaxis()
plt.tight_layout()
fu.savefig(fig, "rtp_cone_geometry")
plt.show()

# %% [markdown]
# ### From geometry to magnetisation
#
# Explain briefly:
#
# - Induced magnetisation $\mathbf{M} = \chi \mathbf{H}$
# - Direction of $\mathbf{M}$ equals inducing field
#
# Magnetisation direction controls anomaly asymmetry.

# %%
def magnetic_anomaly(h, inc_deg, dec_deg, depth=1000.0, chi=0.05):
    """
    Total-field magnetic anomaly for induced magnetisation only.
    Thin-sheet approximation in the Fourier domain.
    """
    inc = np.deg2rad(inc_deg)
    dec = np.deg2rad(dec_deg)

    Fx = np.cos(inc) * np.cos(dec)
    Fy = np.cos(inc) * np.sin(dec)
    Fz = np.sin(inc)

    kx_1d = 2 * np.pi * fftfreq(nx, dx)
    ky_1d = 2 * np.pi * fftfreq(ny, dx)
    KX_f, KY_f = np.meshgrid(kx_1d, ky_1d)

    k = np.sqrt(KX_f**2 + KY_f**2)
    k[0, 0] = 1e-10

    upsilon = Fz + 1j * (Fx * KX_f + Fy * KY_f) / k

    H = fft2(h)
    T = np.real(ifft2(k * np.exp(-k * depth) * upsilon**2 * H))
    return chi * T

# %% [markdown]
# ### Forward magnetic anomaly (space domain)
#
# **Purpose:**
# - Compute the magnetic anomaly for the cone at a specified latitude.
# - 2-D map view shows a clearly asymmetric anomaly with peak offset from the source.

# %%
T_raw  = magnetic_anomaly(h_cone, inc_deg=60, dec_deg=15)
T_vert = magnetic_anomaly(h_cone, inc_deg=90, dec_deg=0)

def show_anomaly(inc=60, dec=15, savename=None):
    T_raw = magnetic_anomaly(h_cone, inc_deg=inc, dec_deg=dec)
    mid   = len(x) // 2

    fig, (ax_map, ax_prof) = plt.subplots(1, 2, figsize=(14, 5))

    plot_mag_map(ax_map, X, Y, T_raw,
                 title=f'Magnetic anomaly (inc={inc}°, dec={dec}°)')

    ax_prof.plot(x / 1000, T_raw[mid, :],  color='C0', ls='-',  label='Raw E–W')
    ax_prof.plot(x / 1000, T_raw[:, mid],  color='C0', ls=':',  label='Raw N–S')
    ax_prof.plot(x / 1000, T_vert[mid, :], color='k',  ls='--', label='Vertical field')
    ax_prof.set_xlabel('Distance (km)')
    ax_prof.set_ylabel('Anomaly (arb. units)')
    ax_prof.set_title(f'Cross-sections through body centre  —  I={inc}°, D={dec}°')
    ax_prof.legend(fontsize=8)
    ax_prof.grid(True)

    plt.tight_layout()
    if savename is not None:
        fu.savefig(fig, savename)
    plt.show()


# Static figure for the "set" field geometry used throughout this demo
# (I = 60°, D = 15°, matching T_raw above) -- ipywidgets sliders don't
# render outside a Jupyter front-end, so a fixed, representative anomaly
# is shown instead of an interactive sweep.
show_anomaly(inc=60, dec=15, savename="rtp_forward_anomaly")

# %% [markdown]
# ### Interpretation checkpoint
#
# Questions:
#
# - Where is the maximum relative to the cone?
# - Is this offset geological or geometric?
# - Would gravity behave this way?

# %% [markdown]
# ## PART B — FREQUENCY DOMAIN VIEW (PHASE MATTERS)
#
# ### Why look in the Fourier domain?
#
# Key statements:
#
# - RTP does not change wavelengths
# - RTP modifies phase
# - Short wavelengths are amplified
#
# ### FFT of the magnetic anomaly — phase as geometry
#
# Tie back to Demo 1:
# - Phase slope ↔ spatial shift
# - Phase structure ↔ asymmetry
#
# RTP works by undoing the phase shift caused by the inclined inducing field.

# %%
def show_fourier_view(inc=60, dec=15, savename=None):
    """Show TMI and RTP in space and Fourier domains."""
    T     = magnetic_anomaly(h_cone, inc_deg=inc, dec_deg=dec)
    T_rtp, _ = reduce_to_pole(x, y, T, f=(inc, dec), m=(inc, dec))

    F     = np.fft.fftshift(np.fft.fft2(T))
    F_rtp = np.fft.fftshift(np.fft.fft2(T_rtp))

    kx_s = np.fft.fftshift(np.fft.fftfreq(nx, dx)) * 1e3
    ky_s = np.fft.fftshift(np.fft.fftfreq(ny, dx)) * 1e3

    kmax = 0.40   # cy/km
    iy = np.where(np.abs(ky_s) <= kmax)[0]
    ix = np.where(np.abs(kx_s) <= kmax)[0]
    F_c     = F    [iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]
    F_rtp_c = F_rtp[iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]

    amp_tmi   = np.log10(np.abs(F_c)     + 1e-10)
    amp_rtp   = np.log10(np.abs(F_rtp_c) + 1e-10)
    phase_tmi = np.degrees(np.angle(F_c))

    H_rtp   = F_rtp_c / (F_c + 1e-4 * np.max(np.abs(F_c)))
    phase_H = np.degrees(np.angle(H_rtp))

    kext = [-kmax, kmax, -kmax, kmax]
    kw_f = dict(extent=kext, origin='lower', aspect='equal')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    ext_km = [x.min()/1e3, x.max()/1e3, y.min()/1e3, y.max()/1e3]
    vt = np.percentile(np.abs(T), 99)

    im00 = axes[0, 0].imshow(T,     origin='lower', extent=ext_km,
                              cmap='RdBu_r', vmin=-vt, vmax=vt)
    im01 = axes[0, 1].imshow(T_rtp, origin='lower', extent=ext_km,
                              cmap='RdBu_r', vmin=-vt, vmax=vt)
    for ax, title in zip(axes[0, :2],
                         [f'TMI  (I = {inc}°, D = {dec}°)', 'RTP anomaly']):
        ax.contour(X/1e3, Y/1e3, h_cone, levels=[1],
                   colors='k', linewidths=0.7, linestyles='--')
        ax.set(title=title, xlabel='x (km)', ylabel='y (km)')
    plt.colorbar(im00, ax=axes[0, 0], label='arb. units', shrink=0.85)
    plt.colorbar(im01, ax=axes[0, 1], label='arb. units', shrink=0.85)

    mid = nx // 2
    axes[0, 2].plot(x/1e3, T[mid, :],      'C0', lw=1.5,         label='TMI E–W')
    axes[0, 2].plot(x/1e3, T[:, mid],      'C0', lw=1.5, ls=':',  label='TMI N–S')
    axes[0, 2].plot(x/1e3, T_rtp[mid, :],  'C3', lw=1.5,         label='RTP E–W')
    axes[0, 2].plot(x/1e3, T_rtp[:, mid],  'C3', lw=1.5, ls=':',  label='RTP N–S')
    axes[0, 2].plot(x/1e3, T_vert[mid, :], 'k',  lw=1.2, ls='--', label='Vertical field')
    axes[0, 2].axvline(0, color='0.6', lw=0.8, ls=':')
    axes[0, 2].set(title='Profiles through source centre',
                   xlabel='Distance (km)', ylabel='Anomaly (arb. units)')
    axes[0, 2].legend(fontsize=8)

    vlog = [min(amp_tmi.min(), amp_rtp.min()), max(amp_tmi.max(), amp_rtp.max())]

    im10 = axes[1, 0].imshow(amp_tmi, **kw_f, cmap='inferno',
                              vmin=vlog[0], vmax=vlog[1])
    im11 = axes[1, 1].imshow(amp_rtp, **kw_f, cmap='inferno',
                              vmin=vlog[0], vmax=vlog[1])
    for ax, title in zip(axes[1, :2], [
            'log₁₀ |F(TMI)| — amplitude', 'log₁₀ |F(RTP)| — amplitude']):
        ax.set(title=title, xlabel='kx (cy/km)', ylabel='ky (cy/km)')
    plt.colorbar(im10, ax=axes[1, 0], label='log₁₀ amplitude', shrink=0.85)
    plt.colorbar(im11, ax=axes[1, 1], label='log₁₀ amplitude', shrink=0.85)

    im12 = axes[1, 2].imshow(phase_tmi, **kw_f, cmap=cmc.romaO, vmin=-180, vmax=180)
    axes[1, 2].set(title='∠F(TMI) — phase of TMI spectrum',
                   xlabel='kx (cy/km)', ylabel='ky (cy/km)')
    plt.colorbar(im12, ax=axes[1, 2], label='Phase (°)', shrink=0.85)

    fig.suptitle(
        f'Fourier-domain view  —  I = {inc}°, D = {dec}°\n'
        'Key observation: amplitude spectra are nearly identical '
        '(RTP is a phase filter, not amplitude).'
    )
    if savename is not None:
        fu.savefig(fig, savename)
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    im_ph = axes2[0].imshow(phase_tmi, **kw_f, cmap=cmc.romaO, vmin=-180, vmax=180)
    axes2[0].set(title=f'∠F(TMI)  —  phase before RTP  (I = {inc}°)',
                 xlabel='kx (cy/km)', ylabel='ky (cy/km)')
    plt.colorbar(im_ph, ax=axes2[0], label='Phase (°)')

    im_hf = axes2[1].imshow(phase_H, **kw_f, cmap=cmc.romaO, vmin=-180, vmax=180)
    axes2[1].set(title='∠H_RTP  —  phase correction applied by RTP filter',
                 xlabel='kx (cy/km)', ylabel='ky (cy/km)')
    plt.colorbar(im_hf, ax=axes2[1], label='Phase correction (°)')

    fig2.suptitle(
        'The RTP filter rotates the phase at every wavenumber.\n'
        'The correction varies with direction in k-space because the field '
        'inclination breaks N–S symmetry.'
    )
    if savename is not None:
        fu.savefig(fig2, f"{savename}_phase")
    plt.show()


# Static figure for the same set field geometry (I = 60°, D = 15°) used above.
show_fourier_view(inc=60, dec=15, savename="rtp_fourier_view")

# %% [markdown]
# ## PART C — REDUCTION TO THE POLE
#
# ### What RTP does (conceptual)
#
# At the magnetic pole:
# - Inducing field is vertical
# - Magnetic anomalies are symmetric
#
# RTP asks: *What would this anomaly look like if the inducing field were vertical?*
# No new information is added.  RTP is a re-expression of existing data.
#
# ### RTP operator (Fourier domain)
#
# **Purpose:** Apply RTP to the anomaly.

# %%
def explore_rtp(inc=60, dec=15, noise=0.0, savename=None):
    T = magnetic_anomaly(h_cone, inc_deg=inc, dec_deg=dec)

    if noise > 0:
        T = T + noise * np.std(T) * np.random.randn(*T.shape)

    T_rtp, _ = reduce_to_pole(x, y, T, f=(inc, dec), m=(inc, dec))

    mid = len(x) // 2

    fig  = plt.figure(figsize=(18, 8))
    spec = _gs.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_geo  = fig.add_subplot(spec[0, 0])
    ax_raw  = fig.add_subplot(spec[0, 1])
    ax_rtp  = fig.add_subplot(spec[0, 2])
    ax_prof = fig.add_subplot(spec[1, :])

    plot_mag_map(ax_geo, X, Y, h_cone,
                 title='Cone geometry', cmap='YlOrBr', label='Thickness (m)')
    plot_mag_map(ax_raw, X, Y, T,
                 title=f'Raw anomaly (I={inc}°, D={dec}°)')
    plot_mag_map(ax_rtp, X, Y, T_rtp,
                 title='RTP anomaly')

    ax_prof.plot(x / 1000, T[mid, :],      color='C0', ls='-',  label='Raw E–W')
    ax_prof.plot(x / 1000, T[:, mid],      color='C0', ls=':',  label='Raw N–S')
    ax_prof.plot(x / 1000, T_rtp[mid, :],  color='C3', ls='-',  label='RTP E–W')
    ax_prof.plot(x / 1000, T_rtp[:, mid],  color='C3', ls=':',  label='RTP N–S')
    ax_prof.plot(x / 1000, T_vert[mid, :], color='k',  ls='--', label='Vertical field')
    ax_prof.set_xlabel('Distance (km)')
    ax_prof.set_ylabel('Anomaly (arb. units)')
    ax_prof.set_title(f'Cross-sections through body centre  —  I={inc}°, D={dec}°')
    ax_prof.legend(fontsize=8)
    ax_prof.grid(True)

    if savename is not None:
        fu.savefig(fig, savename)
    plt.show()


# Static figure for the same set field geometry (I = 60°, D = 15°) used above,
# with no added noise.
explore_rtp(inc=60, dec=15, noise=0.0, savename="rtp_pole_reduction")

# %% [markdown]
# ### Instability near the magnetic equator
#
# The activity text claims RTP becomes unstable and produces ringing
# artefacts below about |I| ~ 15-20 deg. That claim needs a figure to back
# it up rather than sitting as an assertion -- run the same pipeline at a
# low inclination, still using the water-level-stabilised `reduce_to_pole`,
# so the instability (or its absence) is visible directly rather than
# taken on faith.

# %%
# Same forward geometry and declination as above, but at I = 8 deg -- well
# below the stability threshold quoted in the activity text.
explore_rtp(inc=8, dec=15, noise=0.0, savename="rtp_low_inclination_instability")

# %% [markdown]
# ### RTP result: interpretation
#
# - Is the anomaly more symmetric?
# - Is the maximum now over the source?
# - What changed — amplitude, phase, or both?
#
# RTP primarily rotates phase; amplitude changes are secondary and wavelength-dependent.
#
# - At what inclination does RTP become unstable?
# - What happens to short wavelengths?
# - Does RTP still "work" when noise is added?

# %% [markdown]
# ## Magnetics of a real Maar (Tower Hill Volcanic Complex, Victoria)
#
# ### Earth's Field
#
# Earth's magnetic field at the site of the Tower Hill Volcanics is approximately
# inclination −70° and declination 10.5°.

# %%
df = pd.read_csv(DATAPATH / "tower_hill_mag.csv")

lon_th = np.sort(df['lon'].unique())
lat_th = np.sort(df['lat'].unique())
LON_th, LAT_th = np.meshgrid(lon_th, lat_th)

anom_th = np.full(LON_th.shape, np.nan)
for _, row in df.iterrows():
    ri = np.searchsorted(lat_th, row['lat'])
    ci = np.searchsorted(lon_th, row['lon'])
    anom_th[ri, ci] = row['mag_anom']

dlat_km = np.mean(np.diff(lat_th)) * 111.0
dlon_km = np.mean(np.diff(lon_th)) * 111.0 * np.cos(np.radians(lat_th.mean()))

x_th = lon_th * dlon_km / np.mean(np.diff(lon_th))
y_th = lat_th * dlat_km / np.mean(np.diff(lat_th))

anom_filled = anom_th.copy()
anom_filled[np.isnan(anom_th)] = np.nanmean(anom_th)

rtp_th, _ = reduce_to_pole(x_th, y_th, anom_filled, f=(-70, 10.5), m=(-70, 10.5))
rtp_th[np.isnan(anom_th)] = np.nan

prof_lon  = 142.355
col       = int(np.argmin(np.abs(lon_th - prof_lon)))
lon_actual = lon_th[col]

vmax = np.nanpercentile(np.abs(anom_th), 98)

fig  = plt.figure(figsize=(16, 10))
spec = _gs.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
ax_raw  = fig.add_subplot(spec[0, 0])
ax_rtp  = fig.add_subplot(spec[0, 1])
ax_prof = fig.add_subplot(spec[1, :])

im0 = ax_raw.pcolormesh(LON_th, LAT_th, anom_th,
                         cmap='RdBu_r', shading='auto', vmin=-vmax, vmax=vmax)
plt.colorbar(im0, ax=ax_raw, label='Anomaly (nT)', shrink=0.85)
ax_raw.axvline(lon_actual, color='k', lw=0.8, ls='--')
ax_raw.set_title('Tower Hill — Magnetic anomaly (IGRF removed)')
ax_raw.set_xlabel('Longitude (°E)')
ax_raw.set_ylabel('Latitude (°N)')
ax_raw.set_aspect('equal')

im1 = ax_rtp.pcolormesh(LON_th, LAT_th, rtp_th,
                          cmap='RdBu_r', shading='auto', vmin=-vmax, vmax=vmax)
plt.colorbar(im1, ax=ax_rtp, label='Anomaly (nT)', shrink=0.85)
ax_rtp.axvline(lon_actual, color='k', lw=0.8, ls='--')
ax_rtp.set_title('Tower Hill — RTP (Inc=−70°, Dec=10.5°)')
ax_rtp.set_xlabel('Longitude (°E)')
ax_rtp.set_ylabel('Latitude (°N)')
ax_rtp.set_aspect('equal')

ax_prof.plot(lat_th, anom_th[:, col], color='C0', label='Raw anomaly')
ax_prof.plot(lat_th, rtp_th[:, col],  color='C3', label='RTP anomaly')
ax_prof.set_xlabel('Latitude (°N)')
ax_prof.set_ylabel('Anomaly (nT)')
ax_prof.set_title(f'N–S cross-section at lon ≈ {lon_actual:.4f}°E')
ax_prof.legend(fontsize=9)
ax_prof.grid(True)

plt.suptitle('Tower Hill Volcanic Complex — Magnetic anomaly and RTP',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fu.savefig(fig, "rtp_tower_hill_real_data")
plt.show()

# %% [markdown]
# ## Key takeaways
#
# | Aspect           | Gravity        | Magnetics       |
# |------------------|----------------|-----------------|
# | Field type       | Scalar         | Vector          |
# | Polarity         | Always positive | Positive/negative |
# | Phase sensitivity | Low           | High            |
# | Robustness       | High           | Fragile         |
# | RTP equivalent   | Not needed     | Often required  |
#
# - Magnetic anomalies are asymmetric away from the pole
# - RTP is a Fourier-domain phase correction
# - RTP simplifies interpretation but amplifies problems
# - Sampling and aliasing matter more for magnetics than gravity
# - Geological assumptions must be stated explicitly
