"""
iberia2d.py

2-D steady-state thermal model of the Iberian Abyssal Plain.
After Louden et al. [EPSL, 1997].

Translation of iberia2d.m / ssthermal.m / T1d.m

Units
-----
Depths / distances : km
Temperatures       : °C
Heat flow          : mW/m²
Conductivity       : W/m/K
Heat production    : µW/m³
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE.parents[2] / "data" / "geothermics" / "iberia"

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# -----------------

def t1d(z, Ts, Tb, qs, H, k):
    """
    1-D steady-state temperature profile (downward integration).

    Parameters
    ----------
    z  : 1-D array, depth nodes (km)
    Ts : float, surface temperature (°C)
    Tb : float, basal temperature (°C)
    qs : float, surface heat flow (mW/m²)
    H  : 1-D array, heat production at cell centres, length len(z)-1
    k  : 1-D array, conductivity at cell centres, length len(z)-1

    Returns
    -------
    T  : 1-D array, temperature at z nodes (°C)
    """
    z = np.asarray(z, dtype=float).ravel()
    H = np.asarray(H, dtype=float).ravel()
    k = np.asarray(k, dtype=float).ravel()
    n  = len(z)
    dz = np.diff(z)

    q = qs - np.cumsum(H * dz)   # heat flow at lower face of each cell
    q = np.concatenate(([qs], q))

    T = np.zeros(n)
    T[0] = Ts
    flag = False
    for i in range(n - 1):
        if flag:
            T[i + 1] = Tb
            continue
        T[i + 1] = T[i] + q[i] / k[i] * dz[i] - 0.5 * H[i] / k[i] * dz[i]**2
        if T[i + 1] >= Tb:
            T[i + 1] = Tb
            flag = True
    return T


def _tsolve(To, z, iz, ix, zb, K, H_node, q0=None):
    """
    One Jacobi sweep of the 2-D steady-state heat equation.

    Parameters
    ----------
    To     : (nz, nx) temperature from previous iteration
    z      : (nz,) depth nodes
    iz     : (nz-1,) = 1/dz²
    ix     : (nx-1,) = 1/dx²
    zb     : (nx,)   depth to the lower boundary at each column
    K      : (nz, nx, 4) node-centred conductivities [up, down, left, right]
    H_node : (nz, nx) node-centred heat production
    q0     : (nx,) optional surface heat-flow for Neumann top BC

    Returns
    -------
    Tn : (nz, nx) updated temperature
    """
    nz, nx = To.shape
    Tn = To.copy()

    istart = 1   # Python 0-indexed first interior row (= MATLAB row 2, qflag=0)

    if q0 is not None:
        # Neumann top boundary: prescribe surface heat flow
        dz = np.sqrt(1.0 / iz[0])
        Tn[1, :] = Tn[0, :] + dz / K[1, :, 0] * (q0 - 0.5 * H_node[1, :] * dz)
        istart = 2

    # Interior row/col slices
    # Depth:  i in [istart, nz-1)  — bottom row held fixed via zb mask / Dirichlet Tb
    # Lateral: j in [1, nx-1)      — columns 0 and nx-1 are ghost nodes for Neumann BCs
    #
    # Note: the original MATLAB used j=2:nx_local-1 (nx_local = nx-1), which left the
    # second-to-last column unupdated and broke the right insulating BC. Corrected here
    # to j=2:nx_local (i.e. j_sl = slice(1, nx-1)) so both lateral BCs are symmetric.
    i_sl = slice(istart, nz - 1)
    j_sl = slice(1,      nx - 1)

    # 1/dz² weights for each interior row
    iz_up = iz[istart - 1 : nz - 2]   # length nz-1-istart
    iz_dn = iz[istart     : nz - 1]

    # 1/dx² weights for each interior column (length nx-2)
    ix_lt = ix[0 : nx - 2]
    ix_rt = ix[1 : nx - 1]

    K_up = K[i_sl, j_sl, 0] * iz_up[:, np.newaxis]
    K_dn = K[i_sl, j_sl, 1] * iz_dn[:, np.newaxis]
    K_lt = K[i_sl, j_sl, 2] * ix_lt[np.newaxis, :]
    K_rt = K[i_sl, j_sl, 3] * ix_rt[np.newaxis, :]

    To_up = To[istart - 1 : nz - 2,  1 : nx - 1]
    To_dn = To[istart + 1 : nz,       1 : nx - 1]
    To_lt = To[istart     : nz - 1,   0 : nx - 2]
    To_rt = To[istart     : nz - 1,   2 : nx    ]

    numer = (H_node[i_sl, j_sl]
             + To_up * K_up
             + To_dn * K_dn
             + To_lt * K_lt
             + To_rt * K_rt)
    denom = K_up + K_dn + K_lt + K_rt

    # Depth mask: below zb → hold temperature fixed at Tb (applied via To)
    z_2d  = z[i_sl, np.newaxis]
    zb_2d = zb[np.newaxis, j_sl]
    interior_mask = z_2d < zb_2d   # strict < so the node AT zb is held at Tb, not solved

    with np.errstate(invalid='ignore', divide='ignore'):
        Tn_new = np.where(interior_mask & (denom > 0),
                          numer / denom,
                          To[i_sl, j_sl])

    Tn[i_sl, j_sl] = Tn_new

    # Insulating (Neumann) lateral boundary conditions
    Tn[:, 0]  = Tn[:, 1]
    Tn[:, -1] = Tn[:, -2]

    return Tn


def ssthermal(x, z, Ts, Tb, qs, qflag, zb, bflag, k_cell, h_cell,
              tol=2e-4, max_iter=500_000, verbose=True,
              plot_interval=0, plot_pause=1.0):
    """
    2-D steady-state heat conduction solver (Jacobi iteration).

    Parameters
    ----------
    x, z          : 1-D coordinate arrays (km)
    Ts            : float or (nx,) surface temperature (°C)
    Tb            : float or (nx,) basal temperature (°C)
    qs            : float or (nx,) surface heat flow (mW/m²)
    qflag         : 0 = Dirichlet top BC, 1 = Neumann top BC
    zb            : (nx,) depth to lower thermal boundary (km)
    bflag         : 1 = use provided zb; 0 = use bottom of domain
    k_cell        : (nz-1, nx-1) cell-centred thermal conductivity (W/m/K)
    h_cell        : (nz-1, nx-1) cell-centred heat production (µW/m³)
    tol           : convergence tolerance (°C)
    max_iter      : maximum number of Jacobi iterations
    verbose       : print convergence progress
    plot_interval : update live convergence figure every this many iterations
                    (0 = disabled)
    plot_pause    : seconds to pause after each live figure update (default 1.0)

    Returns
    -------
    T  : (nz, nx) temperature field (°C)
    q0 : (nx,) surface heat flow at convergence (mW/m²)
    """
    z = np.asarray(z, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    nz = len(z)
    nx = len(x)

    # Broadcast scalar BCs to full arrays
    if np.isscalar(qs):
        qs = np.full(nx, float(qs))
    if np.isscalar(Tb):
        Tb = np.full(nx, float(Tb))
    if np.isscalar(Ts):
        Ts = np.full(nx, float(Ts))

    if not bflag:
        zb = np.full(nx, z[-1])

    # ── Build node-centred conductivity K[nz, nx, 4] ─────────────────────────
    # Directions: 0=up, 1=down, 2=left, 3=right
    K = np.zeros((nz, nx, 4))

    K[1:nz,  1:nx-1, 0] = 0.5 * (k_cell[:, :-1] + k_cell[:, 1:])
    K[1:nz,  0,      0] = k_cell[:, 0]
    K[1:nz,  -1,     0] = k_cell[:, -1]

    K[:nz-1, 1:nx-1, 1] = K[1:nz, 1:nx-1, 0]
    K[:nz-1, 0,      1] = k_cell[:, 0]
    K[:nz-1, -1,     1] = k_cell[:, -1]

    K[1:nz-1, 1:nx,  2] = 0.5 * (k_cell[:-1, :] + k_cell[1:, :])
    K[1:nz-1, :nx-1, 3] = K[1:nz-1, 1:nx, 2]

    # ── Build node-centred heat production H[nz, nx] ─────────────────────────
    H_node = np.zeros((nz, nx))
    H_node[1:nz-1, 1:nx-1] = 0.25 * (h_cell[:-1, :-1] + h_cell[:-1, 1:]
                                       + h_cell[1:, :-1] + h_cell[1:, 1:])
    H_node[1:nz-1, 0]  = 0.5 * (h_cell[:-1, 0] + h_cell[1:, 0])
    H_node[1:nz-1, -1] = 0.5 * (h_cell[:-1, -1] + h_cell[1:, -1])

    # ── Initialise with column-wise 1-D geotherms ────────────────────────────
    T = np.zeros((nz, nx))
    T[0, :] = Ts
    for j in range(nx):
        ind_b = int(np.argmax(z >= zb[j]))
        T[:, j] = t1d(z, Ts[j], Tb[j], qs[j],
                      H_node[:-1, j], K[:, j, 1])
        T[ind_b:, j] = Tb[j]

    # ── Inverse grid-spacing squared ─────────────────────────────────────────
    dz_step = z[1] - z[0]
    iz = 1.0 / np.diff(z)**2
    ix = 1.0 / np.diff(x)**2

    # ── Optional live convergence figure ──────────────────────────────────────
    if plot_interval > 0:
        plt.ion()
        fig_live, (ax_lhf, ax_lT) = plt.subplots(2, 1, figsize=(10, 8))
        fig_live.suptitle('Convergence monitor')

        _im = ax_lT.imshow(T, extent=[x[0], x[-1], z[-1], z[0]],
                           aspect='auto', cmap='RdYlBu_r', origin='upper',
                           vmin=Ts.min(), vmax=Tb.max())
        ax_lT.plot(x, zb, 'k-', linewidth=1, label='Sediment base')
        #plt.colorbar(_im, ax=ax_lT, label='Temperature (°C)')
        plt.colorbar(_im, ax=ax_lT, orientation='horizontal', label='Temperature (°C)',
             shrink=0.7, pad=0.18)
        ax_lT.set_xlabel('Distance (km)')
        ax_lT.set_ylabel('Depth (km)')
        ax_lT.legend(loc='lower left')
        ax_lT.set_aspect(20)

        _q0_init = K[2, :, 0] * (T[2, :] - T[1, :]) / dz_step
        ax_lhf.plot(x, qs, 'r--', label='Reference $q_s$')
        ax_lhf.plot(x, _q0_init, color='steelblue', alpha=0.35, linewidth=0.8)
        ax_lhf.set_xlabel('Distance (km)')
        ax_lhf.set_ylabel('Heat flow (mW/m²)')
        ax_lhf.legend()
        ax_lhf.grid(True)
        _ttl = ax_lhf.set_title('Iteration 0')

        plt.tight_layout()
        plt.pause(plot_pause)

    # ── Jacobi iteration ─────────────────────────────────────────────────────
    q0_arg = qs if qflag == 1 else None

    for c in range(1, max_iter + 1):
        Told = T.copy()
        T    = _tsolve(Told, z, iz, ix, zb, K, H_node, q0=q0_arg)

        change = np.max(np.abs(Told - T))
        if verbose and (c == 1 or c % 1000 == 0):
            print(f"  iter {c:5d}  max ΔT = {change:.6f} °C")

        if plot_interval > 0 and c % plot_interval == 0:
            _q0_now = K[2, :, 0] * (T[2, :] - T[1, :]) / dz_step
            ax_lhf.plot(x, _q0_now, color='steelblue', alpha=0.35, linewidth=0.8)
            ax_lhf.relim()
            ax_lhf.autoscale_view()
            _ttl.set_text(f'Iteration {c},  max ΔT = {change:.4f} °C')
            _im.set_data(T)
            plt.draw()
            plt.pause(plot_pause)

        if change < tol:
            if verbose:
                print(f"  Converged at iter {c}  (max ΔT = {change:.6f} °C)")
            break
    else:
        print(f"  Warning: did not converge after {max_iter} iterations.")

    if plot_interval > 0:
        _ttl.set_text(f'Converged — iteration {c},  max ΔT = {change:.6f} °C')
        _im.set_data(T)
        ax_lhf.plot(x, K[2, :, 0] * (T[2, :] - T[1, :]) / dz_step,
                    color='navy', linewidth=1.5, label='Computed $q_0$ (final)')
        ax_lhf.legend()
        ax_lhf.relim()
        ax_lhf.autoscale_view()
        plt.draw()
        plt.pause(plot_pause)
        plt.ioff()

    # Heat flow from the gradient between nodes 1 and 2 (first two nodes below
    # the surface). Using a fixed pair of interior nodes avoids the column-by-column
    # depth variation that makes the base-gradient estimate noisy.
    q0_out = K[2, :, 0] * (T[2, :] - T[1, :]) / dz_step

    return T, q0_out


# ──────────────────────────────────────────────────────────────────────────────
# Main script
# ──────────────────────────────────────────────────────────────────────────────

# ── Horizontal grid ───────────────────────────────────────────────────────────
dx = 0.2
x  = np.arange(0, 240 + dx, dx)          # km

# ── Load depth interfaces from Louden et al. [1997] ──────────────────────────
d = np.loadtxt(DATA / "louden97_depths.dat")

# The file contains two segments separated by a NaN row:
#   segment 1 → seafloor depth vs distance
#   segment 2 → igneous basement depth vs distance
nan_rows = np.where(np.isnan(d[:, 0]))[0]
split    = nan_rows[0]

zsf_data = d[:split, :]
zig_data = d[split + 1:, :]

zsf = np.interp(x, zsf_data[:, 0], zsf_data[:, 1])   # seafloor depth (km)
zig = np.interp(x, zig_data[:, 0], zig_data[:, 1])   # igneous basement depth (km)
h   = zig - zsf                                        # sediment thickness (km)

# ── Vertical grid (discretised to dz) ────────────────────────────────────────
dz  = 0.05
dig = np.round(h / dz) * dz                          # snapped sediment thickness
z   = np.arange(0, dig.max() + dz, dz)               # depth nodes (km)

nc = len(x)
nr = len(z)

# ── Boundary conditions ───────────────────────────────────────────────────────
Ts = 2.5                                              # surface temperature (°C)
Tb = 40 + (65 - 40) / 240 * x                        # basal temperature, linear ramp

# ── Material properties (exponential increase with depth) ─────────────────────
k0     = 2.25    # W/m/K
kw     = 0.6     # W/m/K conductivity of pore water (assumed constant)
A0     = 1.5     # µW/m³
phi0   = 0.25    # porosity fraction
lam    = 0.43    # 1/km decay constant

z_mid  = z[:-1] + dz / 2                             # cell-centre depths (km)
phi    = phi0 * np.exp(-lam * z_mid)                 # porosity profile
ks_1d  = 1.0 / ((1 - phi) / k0 + phi / kw)          # harmonic mean conductivity
hs_1d  = A0 * (1 - phi)                              # heat production (solid fraction)

K_cell = np.tile(ks_1d[:, np.newaxis], (1, nc - 1))  # (nr-1, nc-1)
H_cell = np.tile(hs_1d[:, np.newaxis], (1, nc - 1))  # (nr-1, nc-1)

# ── Material properties figure ────────────────────────────────────────────────
fig_mat, (ax_phi, ax_k, ax_A) = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

ax_phi.plot(phi, z_mid, 'b-')
ax_phi.set_xlabel('Porosity')
ax_phi.set_ylabel('Depth (km)')
ax_phi.set_title('Porosity')
ax_phi.grid(True)

ax_k.plot(ks_1d, z_mid, 'b-')
ax_k.axvline(k0, color='k',       linestyle='--', label=f'$k_0$ = {k0} W m$^{{-1}}$ K$^{{-1}}$')
ax_k.axvline(kw, color='steelblue', linestyle='--', label=f'$k_w$ = {kw} W m$^{{-1}}$ K$^{{-1}}$')
ax_k.set_xlabel('Thermal conductivity (W m$^{-1}$ K$^{-1}$)')
ax_k.set_title('Thermal Conductivity')
ax_k.legend(fontsize=8)
ax_k.grid(True)

ax_A.plot(hs_1d, z_mid, 'b-')
ax_A.axvline(A0, color='k', linestyle='--', label=f'$A_0$ = {A0} µW m$^{{-3}}$')
ax_A.set_xlabel('Heat production (µW m$^{-3}$)')
ax_A.set_title('Heat Production')
ax_A.legend(fontsize=8)
ax_A.grid(True)

ax_phi.set_ylim(bottom=0)
ax_phi.invert_yaxis()   # shared y — inverts all three panels
plt.tight_layout()

# ── Observed heat flow (Iberian Abyssal Plain, Louden et al. 1997) ────────────
hf = np.loadtxt(DATA / "Iberia_HF.csv", delimiter=',', skiprows=1)
xq = hf[:, 0]   # distance along profile (km)
qo = hf[:, 1]   # heat flow (mW/m²)
qe = hf[:, 2]   # uncertainty (mW/m²)

qs_interp = np.interp(x, xq, qo, left=np.nan, right=np.nan)
qs_fill   = np.where(np.isnan(qs_interp),
                     np.nanmedian(qs_interp),
                     qs_interp)

# ── 2-D thermal model ─────────────────────────────────────────────────────────
print("Running 2-D steady-state thermal solver …")
T, q2d = ssthermal(x, z, Ts, Tb, qs=30.0, qflag=0,
                   zb=dig, bflag=1,
                   k_cell=K_cell, h_cell=H_cell,
                   tol=5e-3, verbose=True, plot_interval=100, plot_pause=1.5)

# ── 1-D heat-flow estimate ────────────────────────────────────────────────────
# Thermal resistance integral q = ΔT / ∫₀ʰ dz/k(z).  No closed form exists for
# the harmonic-mean k(z), so integrate numerically over the cell-centred grid.
q1d = np.zeros(nc)
for j in range(nc):
    n_cells = int(round(dig[j] / dz))
    if n_cells > 0:
        R = np.sum(dz / ks_1d[:n_cells])   # thermal resistance (m²·K/W)
        q1d[j] = (Tb[j] - Ts) / R

print(f"q1d  min={q1d.min():.2f}  max={q1d.max():.2f}  mean={q1d.mean():.2f}  mW/m²")
print(f"q2d  min={q2d.min():.2f}  max={q2d.max():.2f}  mean={q2d.mean():.2f}  mW/m²")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(x, q1d, 'b-', label='1-D analytical')
ax1.plot(x, q2d, 'r-', label='2-D numerical')
ax1.errorbar(xq, qo, yerr=qe, fmt='ko', label='Observed')
ax1.set_xlim(0, 240)
ax1.set_ylim(20, 120)
ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Heat flow (mW/m²)')
ax1.legend()
ax1.grid(True)

im2 = ax2.imshow(T, extent=[x[0], x[-1], z[-1], z[0]],
                 aspect='auto', cmap='RdYlBu_r', origin='upper')
ax2.plot(x, dig, 'k-', linewidth=1, label='Sediment base')

isotherms = np.linspace(Ts, Tb.mean(), 5)[1:-1]   # 3 intermediate values
cs = ax2.contour(x, z, T, levels=isotherms, colors='k',
                 linestyles='--', linewidths=0.8)
ax2.clabel(cs, fmt='%.0f °C', fontsize=7, inline=True)

ax2.set_xlabel('Distance (km)')
ax2.set_ylabel('Depth (km)')
ax2.set_aspect(20)
ax2.legend(loc='lower left')
fig.colorbar(im2, ax=ax2, orientation='horizontal', label='Temperature (°C)',
             shrink=0.7, pad=0.18)

plt.tight_layout()

# ── Heat-flow residual figure ─────────────────────────────────────────────────
q2d_obs = np.interp(xq, x, q2d)          # model q at each observation site
res     = qo - q2d_obs                    # observed − modelled

mad  = np.mean(np.abs(res))                         # L1: mean absolute deviation (mW/m²)

valid = np.isfinite(qe) & (qe > 0)                 # exclude zero/NaN uncertainties
if valid.any():
    wmad = np.mean(np.abs(res[valid]) / qe[valid])  # L1: weighted MAD (σ units)
else:
    wmad = np.nan

fig_res, (ax_rx, ax_rh) = plt.subplots(1, 2, figsize=(11, 5))

# Left: residual vs distance with observation uncertainties
ax_rx.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax_rx.errorbar(xq, res, yerr=qe, fmt='o', color='steelblue',
               ecolor='steelblue', elinewidth=1, capsize=3,
               label='Residual ± 1σ')
ax_rx.set_xlabel('Distance (km)')
ax_rx.set_ylabel('Residual (mW m$^{-2}$)')
ax_rx.set_title('Heat flow residual  (observed − 2-D model)')
ax_rx.set_xlim(0, 240)
ax_rx.legend()
ax_rx.grid(True)
ax_rx.text(0.05, 0.2,
           f'MAD = {mad:.1f} mW m$^{{-2}}$\n'
           f'WMAD = {wmad:.2f} σ',
           transform=ax_rx.transAxes, ha='left', va='top', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

# Right: histogram of residuals
nbins = max(25, int(np.ceil(len(res) / 3)))
ax_rh.hist(res, bins=nbins, color='steelblue', edgecolor='white', alpha=0.8)
ax_rh.axvline(0,          color='k', linewidth=0.8, linestyle='--', label='Zero')
ax_rh.axvline(np.median(res), color='r', linewidth=1.2, linestyle='-',
              label=f'Median = {np.median(res):.1f} mW m$^{{-2}}$')
ax_rh.set_xlabel('Residual (mW m$^{-2}$)')
ax_rh.set_ylabel('Count')
ax_rh.set_title('Residual distribution')
ax_rh.legend(fontsize=8)
ax_rh.grid(True, axis='y')

plt.tight_layout()
plt.show()
