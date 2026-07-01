"""
elevfield.py  --  Elevation field analysis for the gradient workshop activity.

Loads a region from ETOPO 2022, projects to a local UTM grid, computes
partial derivatives and the gradient magnitude/direction, then produces:

  Figure 1  --  2D contour maps: h, dh/dy, dh/dx, |∇h| + quiver + streamlines
  Figure 2  --  1D elevation and x-derivative profile across the middle row
  Figure 3  --  3D surface plots of the same four fields

Available regions (set REGION below):
    shasta, marble, richat, arkaroola, greenland
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers projection)
import netCDF4
from scipy.interpolate import RegularGridInterpolator
from pyproj import CRS, Transformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGION = 'fiordland'     # change to select region
PROFILE_ROW = None    # None → middle row; integer → specific row index
NLEVELS = 20

REGIONS = {
    'shasta':    dict(lat=(41.25, 41.55), lon=(-122.4, -122.0)),
    'marble':    dict(lat=(36.50, 36.80), lon=(-112.0, -111.5)),
    'richat':    dict(lat=(20.90, 21.30), lon=(-11.7,  -11.1)),
    'arkaroola': dict(lat=(-31.9,-29.5), lon=(138.2,  139.7)),
    'greenland': dict(lat=(67.00, 67.50), lon=(-38.5,  -37.0)),
    'fiordland':  dict(lat=(-46.3, -44.0), lon=(166.0,  168.0)),
}

TOPO_FILE = (
    Path(__file__).parents[2]
    / 'data' / 'topography' / 'ETOPO_2022_v1_30s_N90W180_bed.nc'
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def midpt(v):
    """Midpoints between successive elements of a 1-D array."""
    return (v[:-1] + v[1:]) / 2.0


def utm_transformer(lat_c, lon_c):
    """Return a WGS-84 geographic → UTM Transformer for the given centre point."""
    zone = int((lon_c + 180) / 6) + 1
    crs = CRS.from_dict({
        'proj': 'utm', 'zone': zone,
        'north': lat_c >= 0, 'datum': 'WGS84',
    })
    return Transformer.from_crs('EPSG:4326', crs, always_xy=True)


# ---------------------------------------------------------------------------
# Data loading and reprojection
# ---------------------------------------------------------------------------

def load_elevation(topo_file, lat_bounds, lon_bounds):
    """Extract a lat/lon window from the ETOPO NetCDF file."""
    with netCDF4.Dataset(topo_file) as ds:
        lat_all = ds.variables['lat'][:]
        lon_all = ds.variables['lon'][:]

        lat_idx = np.where(
            (lat_all >= lat_bounds[0]) & (lat_all <= lat_bounds[1])
        )[0]
        lon_idx = np.where(
            (lon_all >= lon_bounds[0]) & (lon_all <= lon_bounds[1])
        )[0]

        lat = lat_all[lat_idx]
        lon = lon_all[lon_idx]
        z = ds.variables['z'][
            lat_idx[0]:lat_idx[-1] + 1,
            lon_idx[0]:lon_idx[-1] + 1,
        ].astype(float)

    return lat, lon, z


def to_utm_grid(lat, lon, z):
    """
    Reproject elevation onto a regular UTM grid (km), centred at the origin.

    Returns
    -------
    easting, northing : 1-D arrays (km, zero-centred)
    elev              : 2-D array of elevation on the UTM grid
    dx, dy            : grid spacing (km)
    """
    lat_c = float((lat[0] + lat[-1]) / 2)
    lon_c = float((lon[0] + lon[-1]) / 2)
    tfm = utm_transformer(lat_c, lon_c)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    east_m, north_m = tfm.transform(lon_grid, lat_grid)
    east_km  = east_m  / 1e3
    north_km = north_m / 1e3

    dx = float(np.abs(east_km[0, 0]  - east_km[0, 1]))
    dy = float(np.abs(north_km[0, 0] - north_km[1, 0]))

    # Regular output grid (trim one point each side, matching MATLAB's inner crop)
    easting  = np.arange(east_km[0, 1],   east_km[0, -2]  + dx * 0.5, dx)
    northing = np.arange(north_km[1, 0],  north_km[-2, 0] + dy * 0.5, dy)

    # Evaluate target UTM coords in geographic space for the interpolator
    east_out, north_out = np.meshgrid(easting * 1e3, northing * 1e3)
    lon_out, lat_out = tfm.transform(east_out, north_out, direction='INVERSE')

    interp = RegularGridInterpolator(
        (lat, lon), z, method='linear', bounds_error=False
    )
    elev = interp((lat_out, lon_out))

    easting  -= easting.mean()
    northing -= northing.mean()

    return easting, northing, elev, dx, dy


# ---------------------------------------------------------------------------
# Compute fields
# ---------------------------------------------------------------------------

region = REGIONS[REGION]
lat, lon, z = load_elevation(TOPO_FILE, region['lat'], region['lon'])
easting, northing, Elev, dx, dy = to_utm_grid(lat, lon, z)

# Forward differences (same as MATLAB diff / manual subtraction)
delev_dx = np.diff(Elev, axis=1) / dx   # shape (ny,   nx-1); x-derivative
delev_dy = np.diff(Elev, axis=0) / dy   # shape (ny-1, nx  ); y-derivative

ex     = midpt(easting)   # x-coords of delev_dx columns
ny_mid = midpt(northing)  # y-coords of delev_dy rows

# Average both derivatives onto the common (ex, ny_mid) grid for vector plots
u = (delev_dx[:-1, :] + delev_dx[1:, :]) / 2   # shape (ny-1, nx-1)
v = (delev_dy[:, :-1] + delev_dy[:, 1:]) / 2   # shape (ny-1, nx-1)
grad_mag = np.sqrt(u**2 + v**2)

# 5-point stencil Laplacian on the interior of the UTM grid
#   ∇²h ≈ (h[i,j+1] - 2h[i,j] + h[i,j-1]) / dx²
#         + (h[i+1,j] - 2h[i,j] + h[i-1,j]) / dy²
laplacian = (
    (Elev[1:-1, 2:] - 2*Elev[1:-1, 1:-1] + Elev[1:-1, :-2]) / dx**2
    + (Elev[2:, 1:-1] - 2*Elev[1:-1, 1:-1] + Elev[:-2, 1:-1]) / dy**2
)
lap_x = easting[1:-1]    # x-coords of interior columns
lap_y = northing[1:-1]   # y-coords of interior rows

# Profile row
row = Elev.shape[0] // 2 if PROFILE_ROW is None else PROFILE_ROW

# ---------------------------------------------------------------------------
# Figure 1 -- 2D contour maps (2 rows × 3 cols)
# ---------------------------------------------------------------------------

fig1, axes = plt.subplots(2, 3, figsize=(14, 9))

ax = axes[0, 0]
ax.contour(easting, northing, Elev, NLEVELS)
ax.streamplot(ex, ny_mid, u, v, color='r', linewidth=0.7, density=1)
ax.set_title(r'elevation, $h(x,y)$')
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$y$ (km)')
ax.set_aspect('equal')

ax = axes[0, 1]
ax.contour(ex, northing, delev_dx, NLEVELS)
ax.set_title(r'$\partial h/\partial x$')
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$y$ (km)')
ax.set_aspect('equal')

ax = axes[0, 2]
ax.contour(easting, ny_mid, delev_dy, NLEVELS)
ax.set_title(r'$\partial h/\partial y$')
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$y$ (km)')
ax.set_aspect('equal')

ax = axes[1, 0]
ax.contour(ex, ny_mid, grad_mag, NLEVELS)
ax.quiver(ex, ny_mid, u, v, color='r')
ax.set_title(
    r'$\nabla h = \dfrac{\partial h}{\partial x}\,\hat{x}'
    r'+ \dfrac{\partial h}{\partial y}\,\hat{y}$'
)
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$y$ (km)')
ax.set_aspect('equal')

ax = axes[1, 1]
ax.contour(lap_x, lap_y, laplacian, NLEVELS)
ax.set_title(r'$\nabla^2 h$')
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$y$ (km)')
ax.set_aspect('equal')

axes[1, 2].set_visible(False)

fig1.suptitle(REGION.replace('_', ' ').title(), fontsize=13)
fig1.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2 -- 1D profile
# ---------------------------------------------------------------------------

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

ax1.plot(easting, Elev[row, :])
ax1.set_ylabel('Elevation (m)')
ax1.set_title(rf'Profile along $y = {northing[row]:.1f}$ km')
ax1.set_xlim(easting[[0, -1]])

ax2.axhline(0, color='k', linewidth=0.8)
ax2.plot(ex, delev_dx[row, :])
ax2.set_ylabel(r'$\partial h/\partial x$ (m km$^{-1}$)')
ax2.set_xlim(easting[[0, -1]])

# Laplacian profile at the same northing; interior grid shifts row index by 1
ax3.axhline(0, color='k', linewidth=0.8)
ax3.plot(lap_x, laplacian[row - 1, :])
ax3.set_xlabel(r'$x$ (km)')
ax3.set_ylabel(r'$\nabla^2 h$ (m km$^{-2}$)')
ax3.set_xlim(easting[[0, -1]])

fig2.tight_layout()

# ---------------------------------------------------------------------------
# Figure 3 -- 3D surface plots (2 rows × 3 cols)
# ---------------------------------------------------------------------------

fig3 = plt.figure(figsize=(16, 10))

def add_surf(fig, pos, x, y, Z, title, cmap='viridis'):
    X, Y = np.meshgrid(x, y)
    ax = fig.add_subplot(pos, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cmap, shade=True, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$y$ (km)')

add_surf(fig3, 231, easting, northing, Elev,      r'elevation, $h(x,y)$')
add_surf(fig3, 232, ex,      northing, delev_dx,  r'$\partial h/\partial x$', 'RdBu_r')
add_surf(fig3, 233, easting, ny_mid,   delev_dy,  r'$\partial h/\partial y$', 'RdBu_r')
add_surf(fig3, 234, ex,      ny_mid,   grad_mag,  r'$|\nabla h|$')
add_surf(fig3, 235, lap_x,   lap_y,    laplacian, r'$\nabla^2 h$', 'RdBu_r')

fig3.suptitle(REGION.replace('_', ' ').title(), fontsize=13)
fig3.tight_layout()

# ---------------------------------------------------------------------------
# Figure 4 -- 2D pseudocolor maps
# ---------------------------------------------------------------------------

fig4, axes4 = plt.subplots(2, 3, figsize=(14, 9))

def pcolor_panel(ax, x, y, Z, title, cmap='viridis', symmetric=False):
    if symmetric:
        vmax = np.nanmax(np.abs(Z))
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    pcm = ax.pcolormesh(x, y, Z, cmap=cmap, norm=norm, shading='auto')
    fig4.colorbar(pcm, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$y$ (km)')
    ax.set_aspect('equal')

pcolor_panel(axes4[0, 0], easting, northing, Elev,
             r'elevation, $h(x,y)$', cmap='viridis')
pcolor_panel(axes4[0, 1], ex, northing, delev_dx,
             r'$\partial h/\partial x$', cmap='RdBu_r', symmetric=True)
pcolor_panel(axes4[0, 2], easting, ny_mid, delev_dy,
             r'$\partial h/\partial y$', cmap='RdBu_r', symmetric=True)
pcolor_panel(axes4[1, 0], ex, ny_mid, grad_mag,
             r'$|\nabla h|$', cmap='viridis')
pcolor_panel(axes4[1, 1], lap_x, lap_y, laplacian,
             r'$\nabla^2 h$', cmap='RdBu_r', symmetric=True)

axes4[1, 2].set_visible(False)

fig4.suptitle(REGION.replace('_', ' ').title(), fontsize=13)
fig4.tight_layout()

plt.show()
