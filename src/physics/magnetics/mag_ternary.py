import numpy as np
import matplotlib.pyplot as plt
import importlib.util, os

def _load_module(name, rel_path):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

TernaryPlot = _load_module('ternary_plot',
    '../../common/plotting/ternary_plot.py').ternary

# ── Temperature anchor data ───────────────────────────────────────────────────
# Titanomagnetite Fe(3-x)TixO4: x=0 (magnetite, 580°C) → x=1 (ulvöspinel, -153°C)
# Curie temperatures — Nagata (1961)
x_tm = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
T_tm = np.array([580., 430., 280., 130., -20., -153.])

def tm_curie(X_Usp):
    """Curie Temperature of Titanomagnetite

    Lattard et al. (2006) provide a quadratic fit to the Curie temperature of titanomagnetite as a function of the ulvöspinel fraction 
    https://doi.org/10.1029/2006JB004591
    
    Parameters
    ----------
    X_Usp: 
        Fraction ulvospinel
    
    Returns
    -------
    Curie Temperature (°C)
    """
    return -400.062*X_Usp*X_Usp	- 414.955*X_Usp + 593.471
T_tm = 

# Titanohematite Fe(2-x)TixO3: x=0 (hematite, 675°C) → x=1 (ilmenite, -218°C)
# Ordering temperatures — Ishikawa & Syono (1963)
x_th = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
T_th = np.array([675., 535., 390., 238., 80., -60., -120., -170., -195., -210., -218.])

# ── Cation-fraction compositions ──────────────────────────────────────────────
# Coordinates are (TiO2, FeO, Fe2O3) cation fractions (a=top, b=left, c=right).
#
# Titanomagnetite Fe(3-x)TixO4:
#   Cations per formula unit: x Ti4+, (1+x) Fe2+, 2(1-x) Fe3+  → sum = 3
#   → (TiO2, FeO, Fe2O3) = (x/3, (1+x)/3, 2(1-x)/3)
def tm_tern(x):
    return x/3, (1+x)/3, 2*(1-x)/3

# Titanohematite Fe(2-x)TixO3:
#   Cations per formula unit: x Ti4+, x Fe2+, 2(1-x) Fe3+  → sum = 2
#   → (TiO2, FeO, Fe2O3) = (x/2, x/2, 1-x)
def th_tern(x):
    return x/2, x/2, 1-x

# ── Set up figure and ternary axes ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))
# Initialise without labels so we control placement and z-order later
tern = TernaryPlot(ax, labels=None, style='ternary')

h_tri = 0.5 / np.tan(np.pi / 6)   # triangle height ≈ 0.866

# ── Analytical interpolation on a grid ───────────────────────────────────────
# Avoids triangulation of collinear points entirely.
# For each grid point the temperature is a linear blend between the two series
# at the same TiO2 cation fraction t:
#   TM series: x_TM = 3t  →  h_TM(t) = 2(1-3t)/3,  T_TM = interp(x_TM, ...)
#   TH series: x_TH = 2t  →  h_TH(t) = 1 - 2t,      T_TH = interp(x_TH, ...)
# Blend factor s = 0 on TM join, s = 1 on TH join.

nx = ny = 600
xg = np.linspace(-0.5, 0.5, nx)
yg = np.linspace(0, h_tri, ny)
X, Y = np.meshgrid(xg, yg)

T_frac, W_frac, H_frac = tern.xy2tern(X, Y)   # (TiO2, FeO, Fe2O3) fractions

eps = 1e-6

# Series temperatures and Fe2O3 fractions at each grid point's TiO2 level
T_TM_grid = np.interp(np.clip(3*T_frac, 0, 1).ravel(), x_tm, T_tm).reshape(T_frac.shape)
T_TH_grid = np.interp(np.clip(2*T_frac, 0, 1).ravel(), x_th, T_th).reshape(T_frac.shape)
h_TM_grid = 2*(1 - 3*T_frac) / 3   # TM join Fe2O3 fraction
h_TH_grid = 1 - 2*T_frac           # TH join Fe2O3 fraction

Z = np.full(X.shape, np.nan)

# Region A (t ∈ [0, 1/3]): blend between TM join and TH join
dh_A = h_TH_grid - h_TM_grid
s_A  = np.where(dh_A > eps, (H_frac - h_TM_grid) / dh_A, 0.0)
in_A = ((T_frac >= -eps) & (T_frac <= 1/3 + eps) &
        (H_frac >= h_TM_grid - eps) & (H_frac <= h_TH_grid + eps) &
        (W_frac >= -eps))
Z[in_A] = ((1 - s_A)*T_TM_grid + s_A*T_TH_grid)[in_A]

# Region B (t ∈ (1/3, 1/2]): blend between ulvöspinel–ilmenite edge and TH join
T_lo_B = np.interp(np.clip(T_frac, 1/3, 0.5).ravel(),
                   [1/3, 0.5], [-153., -218.]).reshape(T_frac.shape)
s_B  = np.where(h_TH_grid > eps, H_frac / h_TH_grid, 0.0)
in_B = ((T_frac > 1/3 - eps) & (T_frac <= 0.5 + eps) &
        (H_frac >= -eps) & (H_frac <= h_TH_grid + eps) &
        (W_frac >= -eps))
Z[in_B] = ((1 - s_B)*T_lo_B + s_B*T_TH_grid)[in_B]

# ── Contour fill and temperature lines ───────────────────────────────────────
levels = [-200, -150, -100, -50, 0, 100, 200, 300, 400, 500, 600, 675]

cf = ax.contourf(X, Y, Z, levels=levels, cmap='plasma', extend='both', zorder=1)
cs = ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.7,
                alpha=0.5, zorder=2)
ax.clabel(cs, fmt='%d °C', fontsize=8, inline=True)

# ── Solid-solution series lines (plotted on top of contours) ─────────────────
xs = np.linspace(0, 1, 500)
tm_x, tm_y = tern.tern2xy(xs/3, (1+xs)/3, 2*(1-xs)/3)
th_x, th_y = tern.tern2xy(xs/2, xs/2, 1-xs)
ax.plot(tm_x, tm_y, 'w-',  lw=2.5, label='Titanomagnetite', zorder=5)
ax.plot(th_x, th_y, 'w--', lw=2.5, label='Titanohematite',  zorder=5)

# ── Key Fe-Ti oxide phases ────────────────────────────────────────────────────
# (TiO2, FeO, Fe2O3, label, (dx, dy), ha)
phases = [
    (0,   1/3, 2/3, 'Magnetite\nFe$_3$O$_4$ (580 °C)',              ( 0.02,  0.01), 'left'),
    (1/3, 2/3,  0,  'Ulvöspinel\nFe$_2$TiO$_4$ (−153 °C)',          (-0.03,  0.02), 'right'),
    (1/2, 1/2,  0,  'Ilmenite\nFeTiO$_3$ (−218 °C)',                 ( 0.02,  0.01), 'left'),
    (0,    0,   1,  'Hematite\nFe$_2$O$_3$ (675 °C)',                ( 0.02, -0.04), 'left'),
    (1/3,  0,  2/3, 'Pseudobrookite\nFe$_2$TiO$_5$',                 ( 0.02,  0.02), 'left'),
    (2/3, 1/3,  0,  'Ferropseudobrookite\nFeTi$_2$O$_5$',           (-0.03,  0.01), 'right'),
]

for t_f, w_f, h_f, label, (dx, dy), ha in phases:
    px, py = tern.tern2xy(t_f, w_f, h_f)
    ax.scatter(px, py, s=50, c='white', edgecolors='k',
               linewidths=1.2, zorder=8)
    ax.text(px + dx, py + dy, label, fontsize=8.5, ha=ha, va='center',
            zorder=9,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      alpha=0.75, edgecolor='none'))

# ── Average oceanic titanomagnetite Fe2.4Ti0.6O4  (x = 0.6) ─────────────────
px_oc, py_oc = tern.tern2xy(*tm_tern(0.6))
ax.scatter(px_oc, py_oc, s=120, c='cyan', marker='*',
           edgecolors='k', linewidths=0.8, zorder=9)
ax.text(px_oc - 0.02, py_oc - 0.03,
        'Avg. oceanic TM:\nFe$_{2.4}$Ti$_{0.6}$O$_4$ (130 °C)',
        fontsize=8.5, ha='right', va='bottom', zorder=10)

# ── Triangle outline and vertex labels (on top of everything) ────────────────
ax.plot([-0.5, 0, 0.5, -0.5], [0, h_tri, 0, 0], 'k-', lw=1.5, zorder=10)

lkw = dict(fontsize=13, fontweight='bold', zorder=11)
ax.text( 0,           h_tri + 0.04, 'TiO$_2$',      ha='center', va='bottom', **lkw)
ax.text(-0.5 - 0.05,  0,            'FeO',           ha='right',  va='center', **lkw)
ax.text( 0.5 + 0.05,  0,            'Fe$_2$O$_3$',  ha='left',   va='center', **lkw)

# ── Colorbar, legend, title ───────────────────────────────────────────────────
cbar = fig.colorbar(cf, ax=ax, shrink=0.65, pad=0.03)
cbar.set_label('Curie Temperature (°C)', fontsize=11)

ax.legend(loc='upper right', fontsize=10, framealpha=0.85)
ax.set_title('Fe–Ti Oxide Ternary: Curie Temperatures', fontsize=13)

plt.rcParams['svg.fonttype'] = 'none'   # keep text as text, not outlines

plt.tight_layout()
plt.savefig('Fe_Ti_oxide_ternary.png', dpi=200)
plt.savefig('Fe_Ti_oxide_ternary.svg')
plt.show()
