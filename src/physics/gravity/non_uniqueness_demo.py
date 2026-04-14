#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sphere vs. 3D sigmoids-of-revolution (gravity, depth positive downward)

Features
--------
- Reference sphere gz along a surface profile.
- 3D radially-symmetric sigmoids (symmetric lens or flat-bottom cap), with
  t(r) = t_max * sech^2(r/w), formed by revolution.
- Surface guard: top never breaches the surface (z_top >= z_min_top).
- Volume-matched defaults (same volume & Δρ as sphere) and Option C search
  (peak enforced via Δρ; rank by half-width).
- Parameter-driven mode: enter sigmoid parameters; returns misfit vs. sphere
  (peak error, half-width error, RMSE) and plots anomalies + geometry.

Usage examples
--------------
# Parameter mode (symmetric lens)
python sphere_vs_sigmoid_of_revolution.py \
  --mode param --shape symmetric --zc 350 --w 600 --tmax 260 --drho 300

# Parameter mode (flat cap)
python sphere_vs_sigmoid_of_revolution.py \
  --mode param --shape flat --zflat 380 --w 520 --tmax 240 --drho 300

# Option C (search seeded around volume-matched defaults)
python sphere_vs_sigmoid_of_revolution.py --mode optionC
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import argparse
import sys

# =========================================================
# Constants / Units (SI)
# =========================================================
G    = 6.67430e-11   # m^3 kg^-1 s^-2
UGAL = 1e8           # 1 µGal = 1e-8 m/s^2
LN2  = np.log(2.0)   # ln(2) used in closed-form volume integral

# =========================================================
# Reference Sphere: gz at z=0 (depth positive downward)
# =========================================================
def sphere_gz_profile(x, a, drho, z0):
    """
    Vertical gravity (gz, downward-positive) from a buried sphere, observed at z=0.
    x   : array of horizontal positions (m)
    a   : sphere radius (m)
    drho: density contrast (kg/m^3)
    z0  : center depth (m, positive downward)
    Returns: gz(x) in m/s^2 (downward-positive)
    """
    return (4.0*np.pi*G/3.0) * drho * a**3 * z0 / (x**2 + z0**2)**1.5

# =========================================================
# Sigmoidal lens-of-revolution shapes (depth positive downward)
# t(r) = t_max * sech^2(r/w)
# =========================================================
def sech2(u):
    return 1.0 / np.cosh(u)**2

def lens3d_bounds_symmetric(r, zc, t_max, w, z_min_top=1.0):
    """
    Symmetric lens centered at (0, 0, zc).
    Thickness t(r) = t_max * sech^2(r/w).
    top  z_top(r) = zc - 0.5*t(r)
    bot  z_bot(r) = zc + 0.5*t(r)
    Enforce z_top >= z_min_top to avoid breaching the surface.
    """
    t = t_max * sech2(r / w)
    z_top = zc - 0.5 * t
    z_bot = zc + 0.5 * t
    z_top = np.maximum(z_top, z_min_top)
    z_bot = np.maximum(z_bot, z_top)
    return z_top, z_bot

def lens3d_bounds_flat(r, z_flat, t_max, w, z_min_top=1.0):
    """
    Flat-bottom cap: bottom z = z_flat (constant), top z_top(r) = z_flat - t(r).
    Enforce z_top >= z_min_top to stay below surface.
    """
    t = t_max * sech2(r / w)
    z_top = z_flat - t
    z_top = np.maximum(z_top, z_min_top)
    z_bot = np.maximum(z_flat, z_top)
    return z_top, z_bot

def recommended_rmax(w, x_max, tail_mult=9.0, pad_w=8.0):
    """
    Choose a radial truncation large enough to suppress sech^2 tails and
    avoid edge artifacts inside |x|<=x_max.
    tail_mult*w handles the analytic tail; pad_w*w ensures margin beyond x_max.
    """
    return max(tail_mult*w, x_max + pad_w*w)

# =========================================================
# 3-D forward model (axisymmetric numeric quadrature)
# =========================================================
def gz_axisymmetric_lens(
    x_obs, drho, ztop_zbot_fn, r_max,
    dr=25.0, dz=25.0, ntheta=24
):
    """
    Compute downward-positive gz at z=0 from an axisymmetric body by summing
    contributions of annular bricks (r*dr*dtheta*dz). Depth positive downward.

    Returns: gz(x_obs) in m/s^2 (downward-positive).
    """
    x_obs = np.asarray(x_obs)
    gz = np.zeros_like(x_obs, dtype=float)

    # Grids
    r_edges = np.arange(0.0, r_max + dr, dr)
    if len(r_edges) < 2:
        r_edges = np.array([0.0, dr])
    r_cent  = 0.5 * (r_edges[:-1] + r_edges[1:])
    dtheta  = 2.0 * np.pi / ntheta
    thetas  = np.linspace(0.0, 2.0*np.pi - dtheta, ntheta)

    # Shape bounds
    z_top, z_bot = ztop_zbot_fn(r_cent)

    # Integrate ring-by-ring
    cth = np.cos(thetas)
    for i, r in enumerate(r_cent):
        zt, zb = z_top[i], z_bot[i]
        if zt >= zb:
            continue

        # Vertical cells
        z_edges = np.arange(zt, zb + dz, dz)
        if len(z_edges) < 2:
            z_edges = np.array([zt, zb])
        z_cent  = 0.5 * (z_edges[:-1] + z_edges[1:])

        # Volume of an annular sector at (r, z): dV = r * dtheta * dr * dz
        dV = r * dtheta * dr * dz
        dm = drho * dV

        for zc in z_cent:
            # Sum over azimuth samples; vectorize over x for speed
            for k in range(ntheta):
                r3 = (x_obs**2 + r**2 - 2.0*x_obs*r*cth[k] + zc**2)**1.5 + 1e-30
                gz += G * dm * zc / r3

    return gz

# =========================================================
# Utilities: half-width and geometry polygons for plotting
# =========================================================
def peak_and_halfwidth(x, g):
    """
    Return (g_peak, half_width) using linear interpolation at half-maximum.
    Assumes unimodal positive curve. Half-width = (x_right - x_left)/2.
    """
    i0 = np.argmax(g)
    g0 = g[i0]
    target = 0.5 * g0

    # Right crossing
    xr = np.nan
    for j in range(i0, len(g)-1):
        if (g[j] - target) * (g[j+1] - target) <= 0.0:
            t = (target - g[j]) / (g[j+1] - g[j] + 1e-30)
            xr = x[j] + t * (x[j+1] - x[j])
            break

    # Left crossing
    xl = np.nan
    for j in range(i0, 0, -1):
        if (g[j] - target) * (g[j-1] - target) <= 0.0:
            t = (target - g[j]) / (g[j-1] - g[j] + 1e-30)
            xl = x[j] + t * (x[j-1] - x[j])
            break

    if np.isnan(xl) or np.isnan(xr):
        return g0, np.nan
    return g0, 0.5 * (xr - xl)

def make_axisymmetric_polygon_xz(x_profile, z_top_of_absx, z_bot_of_absx):
    """
    Build a closed polygon for plotting the revolution body's cross-section along x (y=0).
    z_top_of_absx, z_bot_of_absx are functions of |x|.
    """
    x_half = x_profile[x_profile >= 0.0]
    zt = z_top_of_absx(x_half)
    zb = z_bot_of_absx(x_half)

    # Mirror to x<0
    x_right, zt_right, zb_right = x_half, zt, zb
    x_left,  zt_left,  zb_left  = -x_half[::-1], zt[::-1], zb[::-1]

    # Top from left->right, bottom from right->left
    xs = np.concatenate([x_left, x_right])
    zs = np.concatenate([zt_left, zt_right])
    xs2 = np.concatenate([x_right[::-1], x_left[::-1]])
    zs2 = np.concatenate([zb_right[::-1], zb_left[::-1]])

    xs_full = np.concatenate([xs, xs2])
    zs_full = np.concatenate([zs, zs2])
    return np.column_stack([xs_full, zs_full])

# =========================================================
# Volume relations and volume-match initializer (closed-form)
# V_lens = 2π * t_max * ∫_0^∞ r * sech^2(r/w) dr = 2π * t_max * w^2 * ln(2)
# -> t_max = V_sphere / (2π * w^2 * ln(2))
# =========================================================
def tmax_for_volume_closed_form(V_target, w):
    return V_target / (2.0 * np.pi * (w**2) * LN2)

def minimal_w_for_guard_symmetric(V_target, zc, z_min_top):
    """
    Guard at r=0: z_top(0) = zc - 0.5*t_max >= z_min_top  ->  t_max <= 2(zc - z_min_top).
    With t_max = V/(2π w^2 ln2), solve for minimal w satisfying the inequality.
    """
    tmax_max = max(1e-6, 2.0 * (zc - z_min_top))
    w_min = np.sqrt(V_target / (2.0 * np.pi * tmax_max * LN2))
    return w_min, tmax_for_volume_closed_form(V_target, w_min)

def minimal_w_for_guard_flat(V_target, z_flat, z_min_top):
    """
    Guard at r=0: z_top(0) = z_flat - t_max >= z_min_top  ->  t_max <= z_flat - z_min_top.
    """
    tmax_max = max(1e-6, (z_flat - z_min_top))
    w_min = np.sqrt(V_target / (2.0 * np.pi * tmax_max * LN2))
    return w_min, tmax_for_volume_closed_form(V_target, w_min)

def build_default_symmetric_lens(a_sph, drho_sph, z0_sph, z_min_top=1.0):
    """
    DEFAULT symmetric lens:
      - same volume as sphere
      - same density contrast as sphere
      - centered at zc = z0_sph
      - choose minimal w that satisfies the surface guard; compute tmax accordingly.
    """
    V_sphere = (4.0/3.0) * np.pi * a_sph**3
    zc = z0_sph
    w_star, t_star = minimal_w_for_guard_symmetric(V_sphere, zc, z_min_top)

    def ztop_bot_fn(r):
        return lens3d_bounds_symmetric(r, zc, t_star, w_star, z_min_top=z_min_top)

    return dict(kind="SYM-DEFAULT", zc=zc, w=w_star, tmax=t_star, drho=drho_sph,
                ztop_bot_fn=ztop_bot_fn, r_max=8.0*w_star)

def build_default_flatcap_lens(a_sph, drho_sph, z0_sph, z_min_top=1.0):
    """
    DEFAULT flat-cap:
      - same volume as sphere
      - same density contrast as sphere
      - base depth z_flat = z0_sph
      - choose minimal w that satisfies the surface guard; compute tmax accordingly.
    """
    V_sphere = (4.0/3.0) * np.pi * a_sph**3
    z_flat   = z0_sph
    w_star, t_star = minimal_w_for_guard_flat(V_sphere, z_flat, z_min_top)

    def ztop_bot_fn(r):
        return lens3d_bounds_flat(r, z_flat, t_star, w_star, z_min_top=z_min_top)

    return dict(kind="FLAT-DEFAULT", z_flat=z_flat, w=w_star, tmax=t_star, drho=drho_sph,
                ztop_bot_fn=ztop_bot_fn, r_max=8.0*w_star)

# =========================================================
# Search seeded around defaults: enforce peak via Δρ, rank by half-width
# =========================================================
def find_axisymmetric_matches_to_sphere(
    x, g_sphere,
    default_model,             # dict from builder above
    shape="symmetric",         # 'symmetric' or 'flat'
    peak_tol=0.02,             # 2% peak tolerance
    drho_bounds=(100.0, 1200.0),
    frac_span_w=0.25,          # ±25% around default w
    frac_span_depth=0.20,      # ±20% around depth (zc or z_flat)
    frac_span_tmax=0.25,       # ±25% around default tmax
    z_min_top=1.0,
    r_extent_mult=5.0,
    dr=25.0, dz=25.0, ntheta=24,
    topN=3
):
    """
    For candidates in a neighborhood around the default, compute base field with Δρ=drho0,
    rescale Δρ to match the sphere peak (within bounds/tolerance), then rank by half-width.
    Returns topN matches (list of dicts) and the reference (g0, hw0).
    """
    g0_sph, hw0_sph = peak_and_halfwidth(x, g_sphere)
    if np.isnan(hw0_sph):
        raise RuntimeError("Sphere half-width not found.")

    # Unpack default
    results = []
    if shape == "symmetric":
        zc0, w0, t0 = default_model["zc"], default_model["w"], default_model["tmax"]
        drho0 = default_model["drho"]
        def ztop_bot_fn_factory(zc, w, tmax):
            return lambda r: lens3d_bounds_symmetric(r, zc, tmax, w, z_min_top=z_min_top)
        # grids
        zc_list   = np.linspace((1.0 - frac_span_depth)*zc0, (1.0 + frac_span_depth)*zc0, 5)
        w_list    = np.linspace((1.0 - frac_span_w)*w0,     (1.0 + frac_span_w)*w0,     5)
        tmax_list = np.linspace((1.0 - frac_span_tmax)*t0,  (1.0 + frac_span_tmax)*t0,  5)
        label = "SYM"

        for zc in zc_list:
            tmax_max = 2.0 * (zc - z_min_top)
            for w in w_list:
                for tm in tmax_list:
                    if tm <= 0 or tm > tmax_max:
                        continue
                    ztop_bot_fn = ztop_bot_fn_factory(zc, w, tm)
                    #r_max = max(r_extent_mult*w, 3.0*w)
                    r_max = recommended_rmax(w, x.max(), tail_mult=9.0, pad_w=8.0)
                    g_base = gz_axisymmetric_lens(x, drho0, ztop_bot_fn, r_max, dr=dr, dz=dz, ntheta=ntheta) * UGAL
                    gp, hw = peak_and_halfwidth(x, g_base)
                    if gp <= 0 or np.isnan(hw):
                        continue
                    # Enforce peak via Δρ scaling
                    scale = g0_sph / gp
                    drho_eff = drho0 * scale
                    if not (drho_bounds[0] <= drho_eff <= drho_bounds[1]):
                        continue
                    g_scaled = g_base * scale
                    gp2, hw2 = peak_and_halfwidth(x, g_scaled)
                    if abs(gp2 - g0_sph) > peak_tol * abs(g0_sph):
                        continue
                    results.append(dict(kind=label, zc=zc, w=w, tmax=tm, drho_eff=drho_eff,
                                        g_peak=gp2, hw=hw2, e_hw=abs(hw2 - hw0_sph)/max(hw0_sph,1e-12),
                                        ztop_bot_fn=ztop_bot_fn, r_max=r_max))
    else:
        zf0, w0, t0 = default_model["z_flat"], default_model["w"], default_model["tmax"]
        drho0 = default_model["drho"]
        def ztop_bot_fn_factory(z_flat, w, tmax):
            return lambda r: lens3d_bounds_flat(r, z_flat, tmax, w, z_min_top=z_min_top)
        zf_list   = np.linspace((1.0 - frac_span_depth)*zf0, (1.0 + frac_span_depth)*zf0, 5)
        w_list    = np.linspace((1.0 - frac_span_w)*w0,      (1.0 + frac_span_w)*w0,     5)
        tmax_list = np.linspace((1.0 - frac_span_tmax)*t0,   (1.0 + frac_span_tmax)*t0,  5)
        label = "FLAT"

        for zf in zf_list:
            tmax_max = (zf - z_min_top)
            if tmax_max <= 0:
                continue
            for w in w_list:
                for tm in tmax_list:
                    if tm <= 0 or tm > tmax_max:
                        continue
                    ztop_bot_fn = ztop_bot_fn_factory(zf, w, tm)
                    #r_max = max(r_extent_mult*w, 3.0*w)
                    r_max = recommended_rmax(w, x.max(), tail_mult=9.0, pad_w=8.0)
                    g_base = gz_axisymmetric_lens(x, drho0, ztop_bot_fn, r_max, dr=dr, dz=dz, ntheta=ntheta) * UGAL
                    gp, hw = peak_and_halfwidth(x, g_base)
                    if gp <= 0 or np.isnan(hw):
                        continue
                    scale = g0_sph / gp
                    drho_eff = drho0 * scale
                    if not (drho_bounds[0] <= drho_eff <= drho_bounds[1]):
                        continue
                    g_scaled = g_base * scale
                    gp2, hw2 = peak_and_halfwidth(x, g_scaled)
                    if abs(gp2 - g0_sph) > peak_tol * abs(g0_sph):
                        continue
                    results.append(dict(kind=label, z_flat=zf, w=w, tmax=tm, drho_eff=drho_eff,
                                        g_peak=gp2, hw=hw2, e_hw=abs(hw2 - hw0_sph)/max(hw0_sph,1e-12),
                                        ztop_bot_fn=ztop_bot_fn, r_max=r_max))

    results.sort(key=lambda d: d["e_hw"])
    return results[:topN], (g0_sph, hw0_sph)

# =========================================================
# Misfit evaluator (callable from notebooks if desired)
# =========================================================
def evaluate_sigmoid_misfit(
    x, g_sphere_ugal,
    shape="symmetric", zc=None, zflat=None, w=600.0, tmax=260.0, drho=300.0,
    z_min_top=1.0, dr=25.0, dz=25.0, ntheta=24
):
    """
    Build a sigmoid-of-revolution with user parameters, compute gz profile (µGal),
    and return misfit metrics vs. g_sphere_ugal.
    """
    if w <= 0 or tmax <= 0 or drho <= 0:
        raise ValueError("Parameters w, tmax, drho must be positive.")

    if shape == "symmetric":
        if zc is None:
            raise ValueError("For shape='symmetric' you must provide zc.")
        if tmax > 2.0 * (zc - z_min_top):
            # We don't silently clamp here—warn the caller
            print("Warning: tmax would breach surface at r=0; the plotted body will be clamped.", file=sys.stderr)
        def ztop_bot_fn(r):
            return lens3d_bounds_symmetric(r, zc, tmax, w, z_min_top=z_min_top)
        family = f"Sigmoid (sym): zc={zc:.0f}, w={w:.0f}, tmax={tmax:.0f}, Δρ={drho:.0f}"
    else:
        if zflat is None:
            raise ValueError("For shape='flat' you must provide zflat.")
        if tmax > (zflat - z_min_top):
            print("Warning: tmax would breach surface at r=0; the plotted body will be clamped.", file=sys.stderr)
        def ztop_bot_fn(r):
            return lens3d_bounds_flat(r, zflat, tmax, w, z_min_top=z_min_top)
        family = f"Sigmoid (flat): z_flat={zflat:.0f}, w={w:.0f}, tmax={tmax:.0f}, Δρ={drho:.0f}"

    #r_max = max(5.0*w, 3.0*w)
    r_max = recommended_rmax(w, x.max(), tail_mult=9.0, pad_w=8.0)
    g_lens = gz_axisymmetric_lens(x, drho, ztop_bot_fn, r_max, dr=dr, dz=dz, ntheta=ntheta) * UGAL

    # Misfit metrics
    def rmse(a, b): return np.sqrt(np.mean((a - b)**2))
    gp_lens, hw_lens = peak_and_halfwidth(x, g_lens)
    gp_sph,  hw_sph  = peak_and_halfwidth(x, g_sphere_ugal)
    rmse_val = rmse(g_lens, g_sphere_ugal)

    return dict(
        label=family, g_lens=g_lens,
        peak=gp_lens, hw=hw_lens, rmse=rmse_val,
        d_peak=gp_lens - gp_sph,
        d_peak_pct=(gp_lens - gp_sph)/max(1e-12, gp_sph)*100.0,
        d_hw=(np.nan if np.isnan(hw_lens) else (hw_lens - hw_sph)),
        d_hw_pct=(np.nan if np.isnan(hw_lens) else (hw_lens - hw_sph)/max(1e-12, hw_sph)*100.0),
    ), ztop_bot_fn, r_max

# =========================================================
# CLI / main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Sphere vs. 3D sigmoids-of-revolution. "
                    "Use --mode=param to enter lens parameters and get misfit."
    )
    parser.add_argument("--mode", choices=["optionC", "param"], default="param",
                        help="optionC = search seeded around defaults; "
                             "param = user-specified sigmoid parameters with misfit report.")
    parser.add_argument("--shape", choices=["symmetric", "flat"], default="symmetric",
                        help="Sigmoid type for --mode=param.")
    # Reference sphere controls
    parser.add_argument("--a_sphere", type=float, default=200.0, help="Sphere radius (m).")
    parser.add_argument("--z0_sphere", type=float, default=800.0, help="Sphere center depth (m, down positive).")
    parser.add_argument("--drho_sphere", type=float, default=300.0, help="Sphere density contrast (kg/m^3).")

    # Profile extents
    parser.add_argument("--xmax", type=float, default=3000.0, help="Half-length of profile (m).")
    parser.add_argument("--nx", type=int, default=801, help="Number of profile samples.")

    # Resolution for forward model
    parser.add_argument("--dr", type=float, default=20.0, help="Radial step (m).")
    parser.add_argument("--dz", type=float, default=20.0, help="Vertical step (m).")
    parser.add_argument("--ntheta", type=int, default=48, help="Azimuth samples per ring.")
    parser.add_argument("--zmin_top", type=float, default=1.0, help="Surface guard: top >= zmin_top (m).")

    # Parameters for --shape symmetric (param mode)
    parser.add_argument("--zc", type=float, default=350.0, help="Symmetric lens center depth (m).")
    parser.add_argument("--w", type=float, default=600.0, help="Lateral scale w (m).")
    parser.add_argument("--tmax", type=float, default=260.0, help="Max thickness t_max (m).")
    parser.add_argument("--drho", type=float, default=300.0, help="Lens density contrast (kg/m^3).")

    # Parameters for --shape flat (param mode)
    parser.add_argument("--zflat", type=float, default=380.0, help="Flat-cap bottom depth (m).")

    # Search/matching (Option C) toggles
    parser.add_argument("--peak_tol", type=float, default=0.02, help="Peak tolerance for Option C.")
    parser.add_argument("--drho_min", type=float, default=100.0, help="Lower bound for Δρ in matching.")
    parser.add_argument("--drho_max", type=float, default=1200.0, help="Upper bound for Δρ in matching.")

    args = parser.parse_args()

    # ------------------------
    # Reference sphere
    # ------------------------
    a_sphere  = args.a_sphere
    drho_sph  = args.drho_sphere
    z0_sph    = args.z0_sphere

    x = np.linspace(-abs(args.xmax), abs(args.xmax), args.nx)
    g_sphere = sphere_gz_profile(x, a_sphere, drho_sph, z0_sph) * UGAL
    g0_sph, hw0_sph = peak_and_halfwidth(x, g_sphere)

    print("\nReference sphere:")
    print(f"  a={a_sphere:.1f} m, Δρ={drho_sph:.1f} kg/m^3, z0={z0_sph:.1f} m")
    print(f"  Peak ≈ {g0_sph:.2f} µGal, Half-width ≈ {hw0_sph:.1f} m\n")

    if args.mode == "optionC":
        # ---- Option C: volume-matched defaults + search ----
        z_min_top = args.zmin_top

        sym_def = build_default_symmetric_lens(a_sphere, drho_sph, z0_sph, z_min_top=z_min_top)
        flat_def = build_default_flatcap_lens(a_sphere, drho_sph, z0_sph, z_min_top=z_min_top)

        # Compute defaults' fields (with Δρ equal to sphere)
        g_sym_def = gz_axisymmetric_lens(
            x, sym_def["drho"], sym_def["ztop_bot_fn"], sym_def["r_max"],
            dr=args.dr, dz=args.dz, ntheta=args.ntheta
        ) * UGAL
        g_flat_def = gz_axisymmetric_lens(
            x, flat_def["drho"], flat_def["ztop_bot_fn"], flat_def["r_max"],
            dr=args.dr, dz=args.dz, ntheta=args.ntheta
        ) * UGAL

        gp_sym_def, hw_sym_def   = peak_and_halfwidth(x, g_sym_def)
        gp_flat_def, hw_flat_def = peak_and_halfwidth(x, g_flat_def)

        print("Default (volume-matched) symmetric lens:")
        print(f"  zc={sym_def['zc']:.1f} m, w={sym_def['w']:.1f} m, t_max={sym_def['tmax']:.1f} m, Δρ={sym_def['drho']:.1f} kg/m^3")
        print(f"  Peak ≈ {gp_sym_def:.2f} µGal, Half-width ≈ {hw_sym_def:.1f} m")
        print("\nDefault (volume-matched) flat cap:")
        print(f"  z_flat={flat_def['z_flat']:.1f} m, w={flat_def['w']:.1f} m, t_max={flat_def['tmax']:.1f} m, Δρ={flat_def['drho']:.1f} kg/m^3")
        print(f"  Peak ≈ {gp_flat_def:.2f} µGal, Half-width ≈ {hw_flat_def:.1f} m\n")

        # Search around defaults
        sym_matches, _ = find_axisymmetric_matches_to_sphere(
            x, g_sphere, default_model=sym_def, shape="symmetric",
            peak_tol=args.peak_tol, drho_bounds=(args.drho_min, args.drho_max),
            frac_span_w=0.25, frac_span_depth=0.20, frac_span_tmax=0.25,
            z_min_top=z_min_top, r_extent_mult=5.0,
            dr=args.dr, dz=args.dz, ntheta=args.ntheta, topN=3
        )
        flat_matches, _ = find_axisymmetric_matches_to_sphere(
            x, g_sphere, default_model=flat_def, shape="flat",
            peak_tol=args.peak_tol, drho_bounds=(args.drho_min, args.drho_max),
            frac_span_w=0.25, frac_span_depth=0.20, frac_span_tmax=0.25,
            z_min_top=z_min_top, r_extent_mult=5.0,
            dr=args.dr, dz=args.dz, ntheta=args.ntheta, topN=3
        )

        # Build curves to plot (best matches or defaults)
        g_series = [("Sphere", x, g_sphere, "tab:blue")]
        polygons = []

        x_poly = np.linspace(-abs(args.xmax), abs(args.xmax), 401)

        def build_curve_from_match_or_default(match, default, label_kind):
            if match is None:
                ztop_bot_fn = default["ztop_bot_fn"]; r_max = default["r_max"]
                g = gz_axisymmetric_lens(x, default["drho"], ztop_bot_fn, r_max,
                                         dr=args.dr, dz=args.dz, ntheta=args.ntheta) * UGAL
                tag = f"{label_kind} (default)"
            else:
                ztop_bot_fn = match["ztop_bot_fn"]; r_max = match["r_max"]
                g_base = gz_axisymmetric_lens(x, 1.0, ztop_bot_fn, r_max,
                                              dr=args.dr, dz=args.dz, ntheta=args.ntheta) * UGAL
                g = g_base * match["drho_eff"]
                tag = f"{label_kind} (match)"
            def zt_absx(xx):
                rr = np.abs(xx)
                zt, _ = ztop_bot_fn(rr)
                return zt
            def zb_absx(xx):
                rr = np.abs(xx)
                _, zb = ztop_bot_fn(rr)
                return zb
            poly = make_axisymmetric_polygon_xz(x_poly, zt_absx, zb_absx)
            return g, poly, tag

        m_sym  = sym_matches[0]  if sym_matches else None
        m_flat = flat_matches[0] if flat_matches else None

        g_sym,  poly_sym,  tag_sym  = build_curve_from_match_or_default(m_sym,  sym_def,  "Sigmoid (sym)")
        g_flat, poly_flat, tag_flat = build_curve_from_match_or_default(m_flat, flat_def, "Sigmoid (flat)")

        g_series.append((tag_sym,  x, g_sym,  "tab:orange"))
        g_series.append((tag_flat, x, g_flat, "tab:green"))
        polygons.append((tag_sym,  poly_sym,  "tab:orange"))
        polygons.append((tag_flat, poly_flat, "tab:green"))

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 8))
        ax = axes[0]
        for name, xx, gg, col in g_series:
            ax.plot(xx, gg, label=name, lw=2, c=col)
        ax.set_xlabel("Distance x (m)")
        ax.set_ylabel("Gravity $g_z$ ($\\mu$Gal)")
        ax.set_title("Sphere vs. 3D sigmoids (volume-matched default; peak-matched search)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        axg = axes[1]
        axg.set_aspect('equal')
        axg.plot([x.min(), x.max()], [0, 0], 'k-', lw=1)
        sph = Circle((0, z0_sph), radius=a_sphere, facecolor='tab:blue', alpha=0.35, edgecolor='k')
        axg.add_patch(sph)
        axg.text(0, z0_sph, "sphere", ha='center', va='center', fontsize=9)
        for label, poly, col in polygons:
            axg.add_patch(Polygon(poly, closed=True, facecolor=col, alpha=0.35, edgecolor='k'))
            cx = np.mean(poly[:,0]); cz = np.mean(poly[:,1])
            axg.text(cx, cz, label, ha='center', va='center', fontsize=9)
        axg.set_xlim(-abs(args.xmax), abs(args.xmax))
        axg.set_ylim(0, 1500)
        axg.invert_yaxis()
        axg.set_xlabel("Distance x (m)")
        axg.set_ylabel("Depth z (m, positive downward)")
        axg.set_title("Geometry")
        axg.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return

    # ------------------------
    # PARAMETRIC MODE (your request): user-entered sigmoid, misfit report
    # ------------------------
    z_min_top = args.zmin_top

    # Build and evaluate lens
    if args.shape == "symmetric":
        metrics, ztop_bot_fn, r_max = evaluate_sigmoid_misfit(
            x, g_sphere, shape="symmetric",
            zc=args.zc, w=args.w, tmax=args.tmax, drho=args.drho,
            z_min_top=z_min_top, dr=args.dr, dz=args.dz, ntheta=args.ntheta
        )
    else:
        metrics, ztop_bot_fn, r_max = evaluate_sigmoid_misfit(
            x, g_sphere, shape="flat",
            zflat=args.zflat, w=args.w, tmax=args.tmax, drho=args.drho,
            z_min_top=z_min_top, dr=args.dr, dz=args.dz, ntheta=args.ntheta
        )

    # Print misfit
    print("User-specified lens:")
    print(f"  {metrics['label']}")
    print(f"  Peak ≈ {metrics['peak']:.2f} µGal | Δpeak = {metrics['d_peak']:+.2f} µGal "
          f"({metrics['d_peak_pct']:+.2f}%)")
    if not np.isnan(metrics['hw']):
        print(f"  Half-width ≈ {metrics['hw']:.1f} m | ΔHW = {metrics['d_hw']:+.1f} m "
              f"({metrics['d_hw_pct']:+.2f}%)")
    else:
        print("  Half-width: could not be determined (non-unimodal or noisy).")
    print(f"  RMSE (profile) = {metrics['rmse']:.2f} µGal\n")

    # Prepare geometry polygon
    x_poly = np.linspace(-abs(args.xmax), abs(args.xmax), 401)
    def zt_absx(xx):
        rr = np.abs(xx)
        zt, _ = ztop_bot_fn(rr)
        return zt
    def zb_absx(xx):
        rr = np.abs(xx)
        _, zb = ztop_bot_fn(rr)
        return zb
    poly = make_axisymmetric_polygon_xz(x_poly, zt_absx, zb_absx)

    # Plot sphere vs specified lens
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 8))

    ax = axes[0]
    ax.plot(x, g_sphere, label="Sphere", lw=2, c="tab:blue")
    ax.plot(x, metrics['g_lens'], label=metrics['label'], lw=2, c="tab:orange")
    ax.set_xlabel("Distance x (m)")
    ax.set_ylabel("Gravity $g_z$ ($\\mu$Gal)")
    ax.set_title("Sphere vs. User-specified 3D sigmoid")
    ax.grid(True, alpha=0.3)
    ax.legend()

    axg = axes[1]
    axg.set_aspect('equal')
    axg.plot([x.min(), x.max()], [0, 0], 'k-', lw=1)
    sph = Circle((0, z0_sph), radius=a_sphere, facecolor='tab:blue', alpha=0.35, edgecolor='k')
    axg.add_patch(sph)
    axg.text(0, z0_sph, "sphere", ha='center', va='center', fontsize=9)
    axg.add_patch(Polygon(poly, closed=True, facecolor="tab:orange", alpha=0.35, edgecolor='k'))
    axg.text(0, np.mean(zb_absx(np.array([0]))), "sigmoid", ha='center', va='center', fontsize=9)
    axg.set_xlim(-abs(args.xmax), abs(args.xmax))
    axg.set_ylim(0, 1500)
    axg.invert_yaxis()
    axg.set_xlabel("Distance x (m)")
    axg.set_ylabel("Depth z (m, positive downward)")
    axg.set_title("Geometry")
    axg.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()