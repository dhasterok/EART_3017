"""
buried_sphere.py

Analytical temperature anomaly and surface heat flow due to a sphere
buried in a conducting half-space that differs in heat production and
thermal conductivity from the host rock (z positive downward).

Background geotherm
-------------------
The host-rock temperature field T_l is the 1-D steady-state conductive
geotherm in a homogeneous medium:

    T_l(z) = T_0 + (q_l/k_l) z - (A_l / 2 k_l) z²

with surface temperature T_0, surface heat flow q_l, thermal conductivity
k_l, and heat production A_l.  The gradient evaluated at the sphere centre
is used in the anomaly expressions:

    G = dT_l/dz |_{z = z_c} = q_l/k_l - A_l z_c / k_l

Temperature anomaly
-------------------
Define Δ = A_s/k_s - A_l/k_l  (effective source contrast).

Inside  (R ≤ a):
    T' = α + β (z - z_c) - (Δ/6) R²
    α = Δ a² (k_l + 2 k_s) / (6 k_l)
    β = -(k_s - k_l) G / (k_s + 2 k_l)

Outside (R > a):
    T' = γ/R + δ (z - z_c) / R³
    γ = k_s Δ a³ / (3 k_l)
    δ = -(k_s - k_l) G a³ / (k_s + 2 k_l)

where R = sqrt(x² + (z - z_c)²).

These coefficients satisfy both boundary conditions at R = a:
  (1) continuity of T'
  (2) continuity of k ∂T_total/∂R  (= k ∂T_bg/∂R + k ∂T'/∂R)

Surface heat flow anomaly (z = 0, R_s = sqrt(x² + z_c²) at y = 0)
-------------------------------------------------------------------
    q'(x) = k_s Δ a³ z_c / (3 R_s³)
             - k_l (k_s - k_l)/(k_s + 2 k_l) · G a³ (1/R_s³ - 3 z_c²/R_s⁵)

q' is upward-positive (conventional geoscience sign).

Units: SI (m, K, W m⁻¹ K⁻¹, W m⁻³) except where labelled km / mW m⁻² / °C.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Parameters  (edit here)
# ---------------------------------------------------------------------------
a   = 5.0e3     # sphere radius                          [m]
z_c = 8.0e3    # depth to sphere centre (z positive ↓)  [m]

A_s = 4.0e-6    # heat production, sphere                [W m⁻³]
A_l = 1.5e-6    # heat production, host rock             [W m⁻³]
k_s = 3.0       # thermal conductivity, sphere           [W m⁻¹ K⁻¹]
k_l = 2.5       # thermal conductivity, host rock        [W m⁻¹ K⁻¹]

T_0 = 15.0      # surface temperature                    [°C]
q_l = 65.0e-3   # surface heat flow                      [W m⁻²]

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


# Background gradient at sphere centre
G = q_l / k_l - A_l * z_c / k_l       # [K m⁻¹]

# Effective source contrast
Delta = A_s / k_s - A_l / k_l         # [K m⁻²]

# Interior coefficients  T'_in = alpha + beta_in*(z-z_c) - (Delta/6)*R²
alpha   = Delta * a**2 * (k_l + 2.0*k_s) / (6.0 * k_l)
beta_in = -(k_s - k_l) * G / (k_s + 2.0*k_l)

# Exterior coefficients  T'_out = gamma/R + delta_out*(z-z_c)/R³
gamma     = k_s * Delta * a**3 / (3.0 * k_l)
delta_out = -(k_s - k_l) * G * a**3 / (k_s + 2.0*k_l)

# Dipole prefactor for surface heat flow
cfac = k_l * (k_s - k_l) / (k_s + 2.0 * k_l)


# ---------------------------------------------------------------------------
# Background geotherm T_l(z)
# ---------------------------------------------------------------------------
z_lim_km = 35.0
z_geo_m  = np.linspace(0.0, z_lim_km * 1e3, 5000)
T_l      = T_0 + (q_l / k_l) * z_geo_m - (A_l / (2.0 * k_l)) * z_geo_m**2


# ---------------------------------------------------------------------------
# Cross-section grid (x–z plane, y = 0)
# ---------------------------------------------------------------------------
n_cross  = 500
x_lim_km = 35.0

x_km = np.linspace(-x_lim_km, x_lim_km, n_cross)
z_km = np.linspace(0.0,        z_lim_km, n_cross)
X_km, Z_km = np.meshgrid(x_km, z_km)
X, Z = X_km * 1e3, Z_km * 1e3

R = np.hypot(X, Z - z_c)
R = np.where(R == 0.0, 1e-9, R)        # guard at sphere centre
cos_theta = (Z - z_c) / R

T_prime = np.where(
    R <= a,
    alpha + beta_in * (Z - z_c) - (Delta / 6.0) * R**2,
    gamma / R + delta_out * (Z - z_c) / R**3,
)

# ---------------------------------------------------------------------------
# Surface heat flow profile (y = 0, z = 0)
# ---------------------------------------------------------------------------
x_prof_m = np.linspace(-x_lim_km, x_lim_km, 600) * 1e3
Rs_prof  = np.sqrt(x_prof_m**2 + z_c**2)

q_prime_1d = (
    k_s * Delta * a**3 * z_c / (3.0 * Rs_prof**3)
    - cfac * G * a**3 * (1.0 / Rs_prof**3 - 3.0 * z_c**2 / Rs_prof**5)
)
q_prof_mW = q_prime_1d * 1.0e3   # upward positive
x_prof_km = x_prof_m / 1e3

# ---------------------------------------------------------------------------
# Total temperature at x = 0  (background + anomaly through sphere centre)
# ---------------------------------------------------------------------------
idx_x0   = np.argmin(np.abs(x_km))                          # column closest to x=0
T_l_xz   = T_0 + (q_l / k_l) * (z_km * 1e3) - (A_l / (2.0 * k_l)) * (z_km * 1e3)**2
T_total  = T_l_xz + T_prime[:, idx_x0]                     # T_l + T' at x=0

# ---------------------------------------------------------------------------
# Figure layout  (manual axes positions for exact alignment)
#
#   ax_geo  : geotherm profile  — left of cross-section, same y-extent
#   ax_xz   : T' cross-section — 1:1 data aspect (z_lim / 2·x_lim)
#   ax_prof : q' profile       — above cross-section, same x-extent
#   ax_cbar : colorbar          — right of cross-section
#
# Because positions are computed from the cross-section dimensions,
# left/right edges of ax_prof and ax_xz are identical, and
# top/bottom edges of ax_geo and ax_xz are identical.
# ---------------------------------------------------------------------------
fig_w, fig_h = 10.0, 7.0          # figure size (inches)

# Margin fractions (adjust if labels are clipped)
m_left   = 0.10   # space for ax_geo y-axis labels
m_right  = 0.04   # space right of colorbar
m_bottom = 0.09   # space for x-axis labels
m_top    = 0.14   # space for title text
gap_h    = 0.015  # horizontal gap between geotherm and cross-section
gap_v    = 0.025  # vertical gap between profile and cross-section
geo_frac = 0.13   # geotherm axes width (figure fraction)
prof_frac = 0.20  # profile axes height (figure fraction)
cbar_frac = 0.020 # colorbar width
cbar_gap  = 0.012 # gap between cross-section right edge and colorbar

# Cross-section extents
xz_left   = m_left + geo_frac + gap_h
xz_width  = 1.0 - xz_left - m_right - cbar_gap - cbar_frac
# Equal data aspect: axes_height / axes_width = z_lim / (2·x_lim)
xz_height = xz_width * fig_w * (z_lim_km / (2.0 * x_lim_km)) / fig_h
xz_bottom = m_bottom

# Geotherm axes  — same vertical extent as cross-section
geo_left   = m_left
geo_bottom = xz_bottom
geo_height = xz_height          # ← aligned

# Profile axes  — same horizontal extent as cross-section
prof_left   = xz_left
prof_bottom = xz_bottom + xz_height + gap_v
prof_width  = xz_width           # ← aligned

# Colorbar
cbar_left   = xz_left + xz_width + cbar_gap
cbar_bottom = xz_bottom
cbar_height = xz_height

fig    = plt.figure(figsize=(fig_w, fig_h))
ax_xz  = fig.add_axes([xz_left,   xz_bottom,   xz_width,   xz_height])
ax_geo = fig.add_axes([geo_left,  geo_bottom,  geo_frac,   geo_height],
                       sharey=ax_xz)
ax_prof = fig.add_axes([prof_left, prof_bottom, prof_width, prof_frac])
ax_cbar = fig.add_axes([cbar_left, cbar_bottom, cbar_frac,  cbar_height])

# ---------------------------------------------------------------------------
# Surface heat flow profile
# ---------------------------------------------------------------------------
ax_prof.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax_prof.plot(x_prof_km, q_prof_mW, color='C0', linewidth=2.0)
ax_prof.fill_between(x_prof_km, 0, q_prof_mW,
                      where=q_prof_mW >= 0, alpha=0.20, color='C3')
ax_prof.fill_between(x_prof_km, 0, q_prof_mW,
                      where=q_prof_mW < 0,  alpha=0.20, color='C0')

ax_prof.set_xlim(-x_lim_km, x_lim_km)
ax_prof.set_xlabel('x  (km)')
ax_prof.set_ylabel("q'  (mW m⁻²)")
ax_prof.set_title(
    "Surface heat flow anomaly  q'(x)  [upward positive, y = 0]\n"
    f'$T_0$ = {T_0:.0f} °C   '
    f'$q_\\ell$ = {q_l*1e3:.0f} mW m⁻²   '
    f'$k_s$ = {k_s} W m⁻¹K⁻¹   $k_\\ell$ = {k_l} W m⁻¹K⁻¹   '
    f'$A_s$ = {A_s*1e6:.0f} μW m⁻³   $A_\\ell$ = {A_l*1e6:.0f} μW m⁻³   '
    f'$a$ = {a/1e3:.0f} km   $z_c$ = {z_c/1e3:.0f} km   '
    f'$G(z_c)$ = {G*1e3:.1f} mK m⁻¹',
    fontsize=8,
)
ax_prof.xaxis.set_minor_locator(mticker.AutoMinorLocator())
ax_prof.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax_prof.grid(True, alpha=0.30)

# ---------------------------------------------------------------------------
# Geotherm panel: background T_l(z) and total T at x = 0
# ---------------------------------------------------------------------------
ax_geo.plot(T_l,      z_geo_m / 1e3, color='C1', linewidth=2.0,
            label='$T_\\ell(z)$')
ax_geo.plot(T_total,  z_km,          color='C0', linewidth=1.5,
            linestyle='--', label='$T_\\ell + T\'$  (x=0)')

ax_geo.set_ylim(z_lim_km, 0)           # depth increases downward
ax_geo.set_xlabel('Temperature  (°C)')
ax_geo.set_ylabel('Depth  (km)')
ax_geo.set_title('Geotherm')
ax_geo.xaxis.set_minor_locator(mticker.AutoMinorLocator())
ax_geo.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax_geo.legend(fontsize=7, loc='lower right')
ax_geo.grid(True, alpha=0.30)

# ---------------------------------------------------------------------------
# T' cross-section
# ---------------------------------------------------------------------------
exterior = np.abs(T_prime[R > a])
vmax = (np.percentile(exterior, 99.5) if exterior.size
        else np.percentile(np.abs(T_prime), 99.5))
levels_T = np.linspace(-vmax, vmax, 25)

cf = ax_xz.contourf(X_km, Z_km, T_prime, levels=levels_T,
                     cmap='RdBu_r', extend='both')
cs = ax_xz.contour(X_km, Z_km, T_prime, levels=levels_T[::4],
                    colors='k', linewidths=0.4, alpha=0.5)
ax_xz.clabel(cs, fmt='%.2f K', fontsize=6)

fig.colorbar(cf, cax=ax_cbar, label="T'  (K)")

th = np.linspace(0, 2 * np.pi, 360)
ax_xz.plot(a / 1e3 * np.cos(th),
           z_c / 1e3 + a / 1e3 * np.sin(th),
           'w-', linewidth=1.8,
           label=f'Sphere boundary  (a = {a/1e3:.0f} km)')
ax_xz.plot(0, z_c / 1e3, 'w+', markersize=9, markeredgewidth=1.8)

ax_xz.set_xlim(-x_lim_km, x_lim_km)
ax_xz.set_ylim(z_lim_km, 0)
ax_xz.set_xlabel('x  (km)')
ax_xz.set_title("Temperature anomaly  T'")
ax_xz.xaxis.set_minor_locator(mticker.AutoMinorLocator())
ax_xz.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax_xz.legend(loc='lower right', fontsize=8, framealpha=0.6)
plt.setp(ax_xz.get_yticklabels(), visible=False)   # depth labels on ax_geo

# ---------------------------------------------------------------------------
# Save and display
# ---------------------------------------------------------------------------
out = 'buried_sphere.png'
plt.savefig(out, dpi=150)
print(f'Saved → {out}')
print(f'G at z_c = {G*1e3:.2f} mK/m  '
      f'(q_l = {q_l*1e3:.0f} mW/m², k_l = {k_l}, A_l = {A_l*1e6:.0f} μW/m³, z_c = {z_c/1e3:.0f} km)')
plt.show()
