"""
victoria_magnetics.py  –  load, IGRF-correct, and plot the Victoria TMI grid

Survey: GSV Colac West VIMP, acquired 1999, 80 m terrain clearance, 200 m line spacing.
IGRF evaluated at mid-survey epoch (1999-06-01) and 80 m altitude.
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import ppigrf
import pandas as pd

# ── Load TMI grid ──────────────────────────────────────────────────────────
nc_file = 'data/P761-grid-tmi.nc'

ds = netCDF4.Dataset(nc_file)
lat = np.array(ds.variables['lat'][:])
lon = np.array(ds.variables['lon'][:])
tmi = np.array(ds.variables['Band1'][:], dtype=float)
fill = ds.variables['Band1']._FillValue
ds.close()

tmi[tmi == fill] = np.nan

LON, LAT = np.meshgrid(lon, lat)

# ── Compute IGRF total field across the grid ───────────────────────────────
# Survey acquired 1999; terrain clearance 80 m = 0.08 km.
# IGRF is smooth on this scale so evaluate on a coarse subgrid (every 50th
# point, ~2.5 km) then interpolate back to the full resolution.
survey_date = pd.Timestamp('1999-06-01')
alt_km = 0.08

step = 50
lat_c = lat[::step]
lon_c = lon[::step]
LON_c, LAT_c = np.meshgrid(lon_c, lat_c)

Be_c, Bn_c, Bu_c = ppigrf.igrf(LON_c, LAT_c, alt_km, survey_date)
# ppigrf returns shape (1, nlat, nlon) — squeeze the leading time dimension
Be_c = np.squeeze(Be_c); Bn_c = np.squeeze(Bn_c); Bu_c = np.squeeze(Bu_c)
F_c = np.sqrt(Be_c**2 + Bn_c**2 + Bu_c**2)

# Bilinear interpolation back to full grid
from scipy.interpolate import RegularGridInterpolator
interp = RegularGridInterpolator((lat_c, lon_c), F_c, method='linear',
                                  bounds_error=False, fill_value=None)
F_igrf = interp(np.stack([LAT.ravel(), LON.ravel()], axis=1)).reshape(LAT.shape)

# Mean inclination and declination (for reference / RTP)
inc_mean = float(np.degrees(np.arctan2(-Bu_c, np.sqrt(Be_c**2 + Bn_c**2))).mean())
dec_mean = float(np.degrees(np.arctan2(Be_c, Bn_c)).mean())
print(f'IGRF  F  : {F_igrf.mean():.1f} nT')
print(f'IGRF Inc : {inc_mean:.1f}°')
print(f'IGRF Dec : {dec_mean:.1f}°')

# ── Subtract IGRF → magnetic anomaly ──────────────────────────────────────
mag_anom = tmi - F_igrf
mag_anom[np.isnan(tmi)] = np.nan

# ── Plot ───────────────────────────────────────────────────────────────────
vmax = np.nanpercentile(np.abs(mag_anom), 98)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# TMI
im0 = axes[0].pcolormesh(LON, LAT, tmi, cmap='RdBu_r', shading='auto')
plt.colorbar(im0, ax=axes[0], label='TMI (nT)', shrink=0.85)
axes[0].set_title('Total Magnetic Intensity (raw)')
axes[0].set_xlabel('Longitude (°E)')
axes[0].set_ylabel('Latitude (°N)')
axes[0].set_aspect('equal')

# Magnetic anomaly
im1 = axes[1].pcolormesh(LON, LAT, mag_anom, cmap='RdBu_r', shading='auto',
                          vmin=-vmax, vmax=vmax)
plt.colorbar(im1, ax=axes[1], label='Magnetic anomaly (nT)', shrink=0.85)
axes[1].set_title(f'Magnetic anomaly (TMI \u2212 IGRF)\nIGRF: F={F_igrf.mean():.0f} nT, '
                  f'Inc={inc_mean:.1f}\u00b0, Dec={dec_mean:.1f}\u00b0')
axes[1].set_xlabel('Longitude (°E)')
axes[1].set_ylabel('Latitude (°N)')
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()

# ── Extract Tower Hill region and save ─────────────────────────────────────
lon_min, lon_max =  142.30, 142.42
lat_min, lat_max = -38.38, -38.26

li = (lon >= lon_min) & (lon <= lon_max)
lj = (lat >= lat_min) & (lat <= lat_max)
lon_r = lon[li]
lat_r = lat[lj]
anom_r = mag_anom[np.ix_(lj, li)]

LON_r, LAT_r = np.meshgrid(lon_r, lat_r)
mask = ~np.isnan(anom_r.ravel())
df = pd.DataFrame({
    'lon': LON_r.ravel()[mask],
    'lat': LAT_r.ravel()[mask],
    'mag_anom': anom_r.ravel()[mask],
})
out_file = 'tower_hill_mag.csv'
df.to_csv(out_file, index=False)
print(f'Saved {len(df)} points to {out_file}')
print(f'  lon: {lon_r.min():.4f} – {lon_r.max():.4f}')
print(f'  lat: {lat_r.min():.4f} – {lat_r.max():.4f}')
print(f'  anomaly: {df.mag_anom.min():.1f} – {df.mag_anom.max():.1f} nT')
