"""
**Week 1 Practical** — Explicit Laplacian Smoothing on ETOPO2

**Aim**: Implement the 2-D five-point Laplacian *explicitly*, apply iterative smoothing, and relate *sharp vs smooth* to *distance from sources*. No FFTs.

**What you'll do**
1. Load and plot elevation for a selected region.
2. Compute the gradient of topography.
3. Write your own 5-point Laplacian on a unit grid.
4. Run iterative smoothing (diffusion-like update).
5. Locate linear features
6. *(Optional)* Use the **lat--lon weighted Laplacian** with per-row $\Delta x(\phi)$ and scalar $\Delta y$.

> Tip: Keep regions modest so it runs fast. Use consistent colour scales across maps.
"""
# %% 0. Imports & plotting defaults
import numpy as np
import xarray as xr # used for opening netCDF files
import matplotlib.pyplot as plt # for plotting
import matplotlib.colors as mcolors

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150})

# %%
"""
## 1. Load and Plot Elevation (ETOPO2 NetCDF)
### 1.1. Load Elevation and filter for a rectangular region
We won't plot the entire world, only a subset of it.

You'll need to start by inputting the correct filename/path below.  Then choose a regional subset so figures render quickly.
"""
# >>> YOUR CODE HERE (1 line) >>>
### Correct the path to the netCDF file on your system
# --- Open the topography file ---
nc_path = '../../data/topography/ETOPO_2022_v1_30s_N90W180_bed.nc'

try:
    ds = xr.open_dataset(nc_path)
except FileNotFoundError:
    print(f"File not found: {nc_path}. Please check the path and filename.")
except Exception as e:
    print(f"An error occurred while opening the dataset: {e}")
# <<< END YOUR CODE <<<

lon = ds['lon'].values
lat = ds['lat'].values
Z   = ds['z'].values.astype(float)   # shape (lat, lon)

# %%
"""
Create a regional subset by defining the minimum and maximum longitude and latitude extents.  These extents will be used to create a rectangular map.
"""
# >>> YOUR CODE HERE (1 line) >>>
# Regional subset (example: 130--155ºE, 45--25ºS)
lon_range = [130, 155]### enter the min and max longitude
lat_range = [-45, -25]### enter the min and max latitude
# <<< END YOUR CODE <<<

lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])
lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
lon_r = lon[lon_mask]
lat_r = lat[lat_mask]

H0 = Z[np.ix_(lat_mask, lon_mask)]
H0.shape

"""1.2. Quick look at the regional map"""
def make_cpt_colormap(segments, name):
    """
    segments: list of (x0, (r,g,b), x1, (r,g,b))
    RGB must be 0--255 integers.
    """

    cdict = {"red": [], "green": [], "blue": []}

    for x0, c0, x1, c1 in segments:
        r0, g0, b0 = [v/255 for v in c0]
        r1, g1, b1 = [v/255 for v in c1]

        cdict["red"].append((x0, r0, r0))
        cdict["red"].append((x1, r1, r1))

        cdict["green"].append((x0, g0, g0))
        cdict["green"].append((x1, g1, g1))

        cdict["blue"].append((x0, b0, b0))
        cdict["blue"].append((x1, b1, b1))

    return mcolors.LinearSegmentedColormap(name, cdict)

bathy_segments = [
    (0.0,      (0, 0, 0),        0.125,    (31, 40, 79)),
    (0.125,    (31, 40, 79),     0.25,     (38, 60, 106)),
    (0.25,     (38, 60, 106),    0.5,      (53, 99, 160)),
    (0.5,      (53, 99, 160),    0.5625,   (72, 151, 211)),
    (0.5625,   (72, 151, 211),   0.6875,   (102, 205, 170)),  # mediumaquamarine
    (0.6875,   (102, 205, 170),  0.875,    (141, 210, 239)),
    (0.875,    (141, 210, 239),  1.0,      (245, 255, 255)),
]

dem2_segments = [
    (0.0,          (0, 97, 71),        0.0102040816327, (16, 122, 47)),
    (0.0102040816327, (16, 122, 47),   0.102040816327,  (232, 215, 125)),
    (0.102040816327,  (232, 215, 125), 0.244897959184,  (161, 67, 0)),
    (0.244897959184,  (161, 67, 0),    0.34693877551,   (100, 50, 25)),
    (0.34693877551,   (100, 50, 25),   0.571428571429,  (110, 109, 108)),
    (0.571428571429,  (110, 109, 108), 0.816326530612,  (255, 254, 253)),
    (0.816326530612,  (255, 254, 253), 1.0,             (255, 254, 253)),
]

bathy_cmap = make_cpt_colormap(bathy_segments, "gmt_bathy")
dem2_cmap  = make_cpt_colormap(dem2_segments,  "gmt_dem2")

def topo_cmap(vmin, vmax):
    ocean = bathy_cmap(np.linspace(0, 1, 256))
    land  = dem2_cmap(np.linspace(0, 1, 256))

    colors = np.vstack((ocean, land))
    cmap = mcolors.LinearSegmentedColormap.from_list("gmt_topo", colors)

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    return cmap, norm
#vmin = np.nanmin(H0)
#vmax = np.nanmax(H0)
vmin = -6000
vmax = 6000

cmap, norm = topo_cmap(vmin, vmax)
fig, ax = plt.subplots(figsize=(6.5,4.2))
### plot the topography using ax.imshow(), including a colorbar, title, and axes labels
im = ax.imshow(H0, extent=[lon_r.min(), lon_r.max(), lat_r.min(), lat_r.max()],
               origin='lower', cmap=cmap, norm=norm)
cb = plt.colorbar(im, ax=ax)
cb.set_label('Elevation (m)')
ax.set_title('Regional Topography')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()


# %%
""" **2. Compute and Plot Elevation Gradient**
Let's create some maps of the gradient.  Create 4 subplots showing
1. partial derivative in x-direction, $\frac{\partial h}{\partial x}$
2. partial derivative in y-direction, $\frac{\partial h}{\partial y}$
3. gradient magnitude, $|\nabla h| = [(\frac{\partial h}{\partial x})^2 + (\frac{\partial h}{\partial y})^2]^{1/2}$, and
4. gradient angle, $\theta = \tan^{-1}(\frac{\partial h/\partial x}{\partial h/\partial y})$.

For simplicity, we'll ignore the differences in magnitude between the $dx$ and $dy$ and treat both as equal to one (individual pixels).  Python has some useful functions for these calculations `np.gradient`, `np.hypot`, and `np.arctan2`.  These will be similar to the types of figures we looked at in the gradient workshop.
"""

### Compute the gradients dHx and dHy, the gradient magnitude, and the gradient angle in degrees
dHy, dHx = np.gradient(H0) # returns gradients in y and x directions, respectively
grad_mag = np.hypot(dHx, dHy) # computes sqrt(dHx^2 + dHy^2) for each point

grad_angle_rad = np.arctan2(dHy, dHx)  # angle in radians
grad_angle_deg = np.degrees(grad_angle_rad)  # convert to degrees

fig, axs = plt.subplots(2, 2, figsize=(7,6),
                        sharex=True, sharey=True)

extent = [lon_r.min(), lon_r.max(),
          lat_r.min(), lat_r.max()]

# --- dH/dx ---
im0 = axs[0,0].imshow(dHx, extent=extent, origin='lower',
                      cmap='seismic',
                      vmin=-np.percentile(abs(dHx), 98),
                      vmax= np.percentile(abs(dHx), 98))
axs[0,0].set_title('∂H/∂x')
fig.colorbar(im0, ax=axs[0,0], shrink=0.8)

# --- dH/dy ---
im1 = axs[0,1].imshow(dHy, extent=extent, origin='lower',
                      cmap='seismic',
                      vmin=-np.percentile(abs(dHy), 98),
                      vmax= np.percentile(abs(dHy), 98))
axs[0,1].set_title('∂H/∂y')
fig.colorbar(im1, ax=axs[0,1], shrink=0.8)

# --- magnitude ---
im2 = axs[1,0].imshow(grad_mag, extent=extent, origin='lower',
                      cmap='viridis',
                      vmax=np.percentile(grad_mag, 98))
axs[1,0].set_title('Gradient Magnitude')
fig.colorbar(im2, ax=axs[1,0], shrink=0.8)

# --- angle ---
im3 = axs[1,1].imshow(grad_angle_deg,
                      extent=extent,
                      origin='lower',
                      cmap='twilight',
                      vmin=-180, vmax=180)

axs[1,1].set_title('Gradient Angle (degrees)')
cb_angle = fig.colorbar(im3, ax=axs[1,1], shrink=0.8)
for ax in axs.flat:
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')

# Set ticks at -180, -90, 0, 90, 180
cb_angle.set_ticks([-180, -90, 0, 90, 180])
cb_angle.set_label('Degrees')
plt.suptitle('Topographic Gradient Components', y=0.98)
plt.tight_layout()
plt.show()

# %%
"""
**3. Compute and Plot the Laplacian of Elevation**
Compute the Laplacian of the original elevation map using a 5-point stencil. We'll use *Neumann edges* (replicate borders) so no artificial sinks/sources appear at the boundary during smoothing.  We'll work up to the code, starting with the math.

If we wanted to take a derivative with discrete points, we can approximate it in one dimension by
$$\frac{\partial^2 h}{\partial x^2} \approx \frac{(h_{i+1} - 2 h_{i} + h_{i-1})}{\Delta x^2}$$
where $\Delta x$ is the spacing between the points. We'll revisit this approximation in the final week of the course; it is called a finite difference approximation and is derived from Taylor series approximations of derivatives.  In two dimensions, the result is similar,
$$\nabla^2 h = \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} \approx \frac{(h_{i+1,j} - 2 h_{i,j} + h_{i-1,j})}{\Delta x^2} + \frac{(h_{i,j+1} - 2 h_{i,j} + h_{i,j-1})}{\Delta y^2}$$

For images and grids where the pixel spacing is equal in x and y, we can simplify the Laplacian stencil as:
$$\nabla^2 h \approx (h_{i+1,j} + h_{i-1,j} + h_{i,j+1} + h_{i,j-1}) - 4 h_{i,j}$$

Use this to create the stencil below in the form $(N + S + E + W) - 4.0*C$, where N, S, E, W are the pixels above, below, left and right of a central pixel (C), respectively.
"""

def laplacian_2d(H, bc='neumann'):
    """
    Explicit 5-point Laplacian with unit spacing.
    bc: 'neumann' (replicate) or 'dirichlet' (zeros outside).
    Returns array with same shape as H.
    """
    # start by padding H with a 1-cell border according to the specified bc
    if bc == 'neumann':
        # replicates the edge values (zero-gradient)
        Hn = np.pad(H, ((1,1),(1,1)), mode='edge')
    elif bc == 'dirichlet':
        # pads with zeros (fixed-value)
        Hn = np.pad(H, ((1,1),(1,1)), mode='constant', constant_values=0)
    else:
        raise ValueError("bc must be 'neumann' or 'dirichlet'")

    # compute the Laplacian using the 5-point stencil
    ### create arrays C, N, S, E, W for the center, north, south, east, and west points of the stencil
    C = Hn[1:-1, 1:-1]  # center
    N = Hn[2:, 1:-1]    # north (down)
    S = Hn[:-2, 1:-1]   # south (up)
    E = Hn[1:-1, 2:]    # east (right)
    W = Hn[1:-1, :-2]   # west (left)

    # return the Laplacian (same shape as H)
    return (N + S + E + W) - 4.0*C

L0 = laplacian_2d(H0)

v = np.percentile(np.abs(L0), 95)
fig, ax = plt.subplots(figsize=(6.5,4.2))
im = ax.imshow(L0, extent=[lon_r.min(), lon_r.max(), lat_r.min(), lat_r.max()],
                 origin='lower', cmap='seismic', vmin=-v, vmax=v)
cb_lap = plt.colorbar(im, ax=ax); cb_lap.set_label('∇²h (arb.)')
ax.set(xlabel='Longitude', ylabel='Latitude', title='Discrete Laplacian (unit grid)')
plt.tight_layout()

# %%
"""
**4. Iterative smoothing**
*4.1. Smooth the map*
We update $H$ by $H^{k+1} = H^{k} + \alpha \nabla^2 H^{k}$. Start with $\alpha=0.1$ and steps $K = 5,10,20,40$.

> **Stability tip**: with unit spacing and the 5-pt stencil, $\alpha$ in $[0.05,0.2]$ is typically safe. If you see oscillations, reduce $\alpha$.
"""
def diffuse_iter(H, alpha=0.1, steps=10, lap_fn=laplacian_2d, **lap_kwargs):
    Hs = H.copy()
    for _ in range(steps):
        Hs = Hs + alpha * lap_fn(Hs, **lap_kwargs)
    return Hs

alpha = 0.1
Ks    = [5, 10, 20, 40]
Hs    = [diffuse_iter(H0, alpha=alpha, steps=K) for K in Ks]

# Compare maps with identical colour limits
vmin, vmax = np.percentile(H0, [1,99])
fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.5), sharex=True, sharey=True)

axs = axs.ravel()  # flatten 2x3 into 1D array

# --- Original ---
im = axs[0].imshow(
    H0,
    extent=[lon_r.min(), lon_r.max(), lat_r.min(), lat_r.max()],
    origin='lower', cmap=cmap, norm=norm
)
axs[0].set_title('Original')

# --- Smoothed ---
for ax, K, Hk in zip(axs[1:], Ks, Hs):
    ax.imshow(
        Hk,
        extent=[lon_r.min(), lon_r.max(), lat_r.min(), lat_r.max()],
        origin='lower', cmap=cmap, norm=norm
    )
    ax.set_title(f'K={K}')

# Remove unused subplot (bottom right)
fig.delaxes(axs[-1])

# --- Single colorbar for all ---
fig.subplots_adjust(right=0.88)  # leave room on right

cax = fig.add_axes([0.8, 0.125, 0.02, 0.3])
# [left, bottom, width, height] in figure fraction

cb_topo = fig.colorbar(im, cax=cax)
cb_topo.set_label('Elevation (m)')
#cb_topo.set_ticks(np.linspace(vmin, vmax, 2000))

fig.suptitle('Iterative Laplacian Smoothing (larger K ⇒ smoother)')
plt.tight_layout()

# %%
"""
*4.2 Smoothing profile*
Pick a row/column crossing an interesting topographic feature.  Then plot the profile and various smoothed profiles.
"""
# --- find latitude index closest to -34 ---
### Select a profile latitude
ilat = np.argmin(np.abs(lat_r - (-34.0)))

# --- restrict longitude window ---
### Limit the size to a smaller region for better visualization (a couple degrees is sufficient)
mask = (lon_r >= 137.5) & (lon_r <= 139.5)

x1d = lon_r[mask]
y_orig = H0[ilat, mask]
y_s20  = Hs[2][ilat, mask]  # K=20

fig, ax = plt.subplots(figsize=(6.5,4.2))
ax.plot(x1d, y_orig, 'k', lw=1.4, label='Original')
for K, Hk in zip(Ks, Hs):
    ax.plot(x1d, Hk[ilat, mask], lw=1.0, label=f'K={K}')

ax.set(
    xlabel='Longitude',
    ylabel='Elevation (m)',
    title='Cross-section at 34°S (137.5--139.5°E)'
)
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()

# %%
"""
**5. Identify Lineaments**
Now that we can produce maps of both the gradient and Laplacian of elevation, produce a map that identifies linear features on the map.  We can identify these by determining where the gradient and or Laplacian change.  Let's compare the results by identifing points of change by using:
- the gradient only
- the Laplacian only
- combined gradient and Laplacian

Try also with smoothing before computing the gradient and Laplacian.
"""

lon_mask = (lon_r >= 134) & (lon_r <= 141)
lat_mask = (lat_r >= -36.5) & (lat_r <= -29)

lon_sub = lon_r[lon_mask]
lat_sub = lat_r[lat_mask]
H_sub   = H0[np.ix_(lat_mask, lon_mask)]

H_sub_smooth = diffuse_iter(H_sub, alpha=0.1, steps=5)

# --- Derivatives ---
dHy, dHx = np.gradient(H_sub_smooth)
grad_mag = np.hypot(dHx, dHy)

L_sub = laplacian_2d(H_sub_smooth)

# --- Thresholds ---
grad_thresh = np.percentile(grad_mag, 92)
lap_thresh  = np.percentile(np.abs(L_sub), 90)

mask_grad = grad_mag > grad_thresh
mask_lap  = np.abs(L_sub) > lap_thresh
mask_both = mask_grad & mask_lap

# Mask Laplacian for zero-crossing contours
L_grad  = np.where(mask_grad, L_sub, np.nan)
L_lap   = np.where(mask_lap,  L_sub, np.nan)
L_both  = np.where(mask_both, L_sub, np.nan)

extent = [lon_sub.min(), lon_sub.max(),
          lat_sub.min(), lat_sub.max()]

fig, axs = plt.subplots(1, 3, figsize=(6.5,3),
                        sharex=True, sharey=True)

# --- Gradient only ---
axs[0].imshow(H_sub, extent=extent,
              origin='lower', cmap=cmap, norm=norm)
axs[0].contour(lon_sub, lat_sub, L_grad,
               levels=[0], colors='black', linewidths=0.7)
axs[0].set_title('Gradient Threshold')

# --- Laplacian only ---
axs[1].imshow(H_sub, extent=extent,
              origin='lower', cmap=cmap, norm=norm)
axs[1].contour(lon_sub, lat_sub, L_lap,
               levels=[0], colors='black', linewidths=0.7)
axs[1].set_title('Laplacian Threshold')

# --- Both ---
axs[2].imshow(H_sub, extent=extent,
              origin='lower', cmap=cmap, norm=norm)
axs[2].contour(lon_sub, lat_sub, L_both,
               levels=[0], colors='black', linewidths=0.7)
axs[2].set_title('Gradient + Laplacian')

for ax in axs:
    ax.set_xlabel('Lon')
axs[0].set_ylabel('Lat')

plt.suptitle('Comparison of Lineament Detection Criteria', y=0.97)
plt.tight_layout()

# %%
"""
**6. (Optional) Lat--Lon Weighted Laplacian**
On a lat--lon grid with angular steps $\Delta\phi, \Delta\lambda$, physical spacings are $\Delta y\approx R\Delta\phi$ and $\Delta x(\phi)\approx R\cos\phi\,\Delta\lambda$. Implement the weighted Laplacian and test with a smaller $\alpha$.
"""

def latlon_dx_dy(lat_deg, dlat_deg, dlon_deg, R=6371000.0):
    phi  = np.deg2rad(lat_deg)
    dphi = np.deg2rad(dlat_deg)
    dlmb = np.deg2rad(dlon_deg)
    dy   = R * dphi
    dx_row = R * np.cos(phi) * dlmb
    return dx_row, dy

def laplacian_2d_latlon(H, dx_row, dy, bc='neumann'):
    ny, nx = H.shape
    if bc == 'neumann':
        Hn = np.pad(H, ((1,1),(1,1)), mode='edge')
    else:
        Hn = np.pad(H, ((1,1),(1,1)), mode='constant', constant_values=0)
    C = Hn[1:-1,1:-1]; N = Hn[2:,1:-1]; S = Hn[:-2,1:-1]; E = Hn[1:-1,2:]; W = Hn[1:-1,:-2]
    inv_dy2 = 1.0/(dy*dy)
    inv_dx2 = (1.0/(dx_row*dx_row))[:, None]  # broadcast along columns
    return inv_dy2*(N - 2.0*C + S) + inv_dx2*(E - 2.0*C + W)

# Example use (uncomment to try):
dlat = float(np.abs(lat_r[1] - lat_r[0]))
dlon = float(np.abs(lon_r[1] - lon_r[0]))
dx_row, dy = latlon_dx_dy(lat_r, dlat, dlon)
Hw20 = diffuse_iter(H0, alpha=0.1, steps=20, lap_fn=laplacian_2d_latlon, dx_row=dx_row, dy=dy)
fig, ax = plt.subplots(figsize=(6.5,4.2))
im = ax.imshow(Hw20, extent=[lon_r.min(), lon_r.max(), lat_r.min(), lat_r.max()],
                 origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax, label='Elevation (m)'); ax.set_title('Weighted Laplacian, K=20, α=0.05')
plt.tight_layout()

dHy, dHx = np.gradient(H_sub)

Hyy, Hyx = np.gradient(dHy)
Hxy, Hxx = np.gradient(dHx)

trace = Hxx + Hyy
det_term = np.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)

lambda1 = 0.5 * (trace + det_term)
lambda2 = 0.5 * (trace - det_term)

lambda_large = np.maximum(np.abs(lambda1), np.abs(lambda2))
lambda_small = np.minimum(np.abs(lambda1), np.abs(lambda2))

ridge_measure = lambda_large - lambda_small

ridge_thresh = np.percentile(ridge_measure, 95)
ridge_mask = ridge_measure > ridge_thresh

dHy, dHx = np.gradient(H_sub)
grad_mag = np.hypot(dHx, dHy)

grad_thresh = np.percentile(grad_mag, 90)

fault_mask = ridge_mask & (grad_mag > grad_thresh)

plt.figure(figsize=(6.5,5))

plt.imshow(H_sub,
           extent=[lon_sub.min(), lon_sub.max(),
                   lat_sub.min(), lat_sub.max()],
           origin='lower', cmap=cmap, norm=norm)

plt.contour(lon_sub, lat_sub, fault_mask,
            levels=[0.5], colors='red', linewidths=0.8)

plt.title("Fault Lineaments (Hessian Eigenvalue Method)")
plt.tight_layout()