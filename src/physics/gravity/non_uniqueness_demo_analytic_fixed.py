#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sphere vs. axisymmetric sigmoid (lopolith) using analytic Hankel integration
and comparison to a buried sphere (gravity, depth positive downward).

This version fixes structural issues in optionC by scoping all logic inside
functions (no bare 'return' or 'args' at module level).

Key features:
- Analytic Hankel forward model for a symmetric sigmoid (Atop = Abot = A)
- OptionC uses the analytic symmetric family; flat-cap family remains numeric
- CLI convenience: --Aequal sets Atop = Abot
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import argparse

# Try SciPy for accurate J0; fall back to a piecewise approximation
try:
    from scipy.special import j0 as scipy_j0  # type: ignore
except Exception:  # pragma: no cover
    scipy_j0 = None

# =========================================================
# Constants / Units (SI)
# =========================================================
G = 6.67430e-11  # m^3 kg^-1 s^-2
UGAL = 1e8       # 1 µGal = 1e-8 m/s^2
LN2 = np.log(2.0)

# =========================================================
# Bessel J0: robust implementation with fallback
# =========================================================

def j0(x):
    x = np.asarray(x, dtype=float)
    if scipy_j0 is not None:
        return scipy_j0(x)
    ax = np.abs(x)
    out = np.empty_like(x)
    mask_small = ax <= 1.0
    if np.any(mask_small):
        xs = x[mask_small]
        term = np.ones_like(xs)
        s = term.copy()
        m = 1
        z = (xs*0.5)**2
        while m <= 30:
            term *= (-1.0) * z / (m*m)
            s += term
            if np.max(np.abs(term)) < 1e-16:
                break
            m += 1
        out[mask_small] = s
    mask_large = ~mask_small
    if np.any(mask_large):
        xl = ax[mask_large]
        phase = xl - 0.25*np.pi
        amp = np.sqrt(2.0/(np.pi*xl + 1e-300))
        approx = amp * (np.cos(phase) - (1.0/(8.0*xl+1e-300))*np.sin(phase))
        out[mask_large] = approx
    return out

# =========================================================
# Sphere model
# =========================================================

def sphere_gz_profile(x, a, drho, z0):
    x = np.asarray(x, dtype=float)
    return (4.0*np.pi*G/3.0) * drho * a**3 * z0 / (x**2 + z0**2)**1.5

# =========================================================
# Numeric lens utilities (unchanged; used for flat-cap family)
# =========================================================

def sech2(u):
    return 1.0 / np.cosh(u)**2

def lens3d_bounds_symmetric(r, zc, t_max, w, z_min_top=1.0):
    t = t_max * sech2(r / w)
    z_top = np.maximum(zc - 0.5*t, z_min_top)
    z_bot = np.maximum(zc + 0.5*t, z_top)
    return z_top, z_bot

def lens3d_bounds_flat(r, z_flat, t_max, w, z_min_top=1.0):
    t = t_max * sech2(r / w)
    z_top = np.maximum(z_flat - t, z_min_top)
    z_bot = np.maximum(z_flat, z_top)
    return z_top, z_bot

def gz_axisymmetric_lens(x_obs, drho, ztop_zbot_fn, r_max, dr=25.0, dz=25.0, ntheta=24):
    x_obs = np.asarray(x_obs)
    gz = np.zeros_like(x_obs, dtype=float)
    r_edges = np.arange(0.0, r_max + dr, dr)
    if len(r_edges) < 2:
        r_edges = np.array([0.0, dr])
    r_cent = 0.5*(r_edges[:-1] + r_edges[1:])
    dtheta = 2.0*np.pi / ntheta
    thetas = np.linspace(0.0, 2.0*np.pi - dtheta, ntheta)
    z_top, z_bot = ztop_zbot_fn(r_cent)
    cth = np.cos(thetas)
    for i, r in enumerate(r_cent):
        zt, zb = z_top[i], z_bot[i]
        if zt >= zb:
            continue
        z_edges = np.arange(zt, zb + dz, dz)
        if len(z_edges) < 2:
            z_edges = np.array([zt, zb])
        z_cent = 0.5*(z_edges[:-1] + z_edges[1:])
        dV = r * dtheta * dr * dz
        dm = drho * dV
        for zc in z_cent:
            for k in range(ntheta):
                r3 = (x_obs**2 + r**2 - 2.0*x_obs*r*cth[k] + zc**2)**1.5 + 1e-30
                gz += G * dm * zc / r3
    return gz

# =========================================================
# Analytic Hankel-based sigmoid
# =========================================================

def trapz_yx(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    elif hasattr(np, "trapz"):
        return np.trapz(y, x, axis=axis)
    else:
        dx = np.diff(x)
        y_mid = 0.5 * (y[..., 1:] + y[..., :-1])
        return np.sum(y_mid * dx, axis=axis)

def s_of_r(r, w):
    return 0.5 * (1.0 - np.tanh(r / max(w, 1e-12)))

def S_of_k(k, w):
    kw = k * w
    denom = np.sinh(0.5 * np.pi * kw) + 1e-300
    return (0.5 * np.pi * w**2) * (kw / denom)

def lopolith_gravity_spectral(R, w, Atop, Abot, z0, drho, kmax=None, nk=4096):
    R = np.atleast_1d(np.abs(R)).astype(float)
    if kmax is None:
        k_d = 8.0 / max(z0, 1e-9)
        k_w = 8.0 / max(w, 1e-9)
        kmax = max(k_d, k_w)
    k = np.linspace(0.0, kmax, nk)
    k[0] = 1e-12
    S = S_of_k(k, w)
    Mtilde = drho * (Atop + Abot) * S
    try:
        from scipy.special import j0 as _j0
        J = _j0(np.outer(k, R))
    except Exception:
        J = j0(np.outer(k, R))
    decay = np.exp(-k[:, None] * z0)
    integrand = J * decay * (k[:, None] * Mtilde[:, None])
    gz = 2.0 * np.pi * G * trapz_yx(integrand, k, axis=0)
    return gz

# Effective-depth proxy

def z_eff_from_wz0(w, z0):
    return z0 + 0.5*np.pi*w

def fwhm_from_zeff(z_eff):
    return 1.532 * z_eff

def estimate_z0_from_fwhm(FWHM, w):
    return (FWHM / 1.532) - 0.5*np.pi*w

# =========================================================
# Utilities
# =========================================================

def peak_and_halfwidth(x, g):
    i0 = np.argmax(g)
    g0 = g[i0]
    target = 0.5 * g0
    xr = np.nan
    for j in range(i0, len(g)-1):
        if (g[j] - target) * (g[j+1] - target) <= 0.0:
            t = (target - g[j]) / (g[j+1] - g[j] + 1e-30)
            xr = x[j] + t * (x[j+1] - x[j])
            break
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
    x_half = x_profile[x_profile >= 0.0]
    zt = z_top_of_absx(x_half)
    zb = z_bot_of_absx(x_half)
    x_right, zt_right, zb_right = x_half, zt, zb
    x_left, zt_left, zb_left = -x_half[::-1], zt[::-1], zb[::-1]
    xs = np.concatenate([x_left, x_right])
    zs = np.concatenate([zt_left, zt_right])
    xs2 = np.concatenate([x_right[::-1], x_left[::-1]])
    zs2 = np.concatenate([zb_right[::-1], zb_left[::-1]])
    xs_full = np.concatenate([xs, xs2])
    zs_full = np.concatenate([zs, zs2])
    return np.column_stack([xs_full, zs_full])

# =========================================================
# Defaults and search helpers
# =========================================================

def tmax_for_volume_closed_form(V_target, w):
    return V_target / (2.0 * np.pi * (w**2) * LN2)

def minimal_w_for_guard_symmetric(V_target, zc, z_min_top):
    tmax_max = max(1e-6, 2.0 * (zc - z_min_top))
    w_min = np.sqrt(V_target / (2.0 * np.pi * tmax_max * LN2))
    return w_min, tmax_for_volume_closed_form(V_target, w_min)

def minimal_w_for_guard_flat(V_target, z_flat, z_min_top):
    tmax_max = max(1e-6, (z_flat - z_min_top))
    w_min = np.sqrt(V_target / (2.0 * np.pi * tmax_max * LN2))
    return w_min, tmax_for_volume_closed_form(V_target, w_min)

def build_default_symmetric_lens(a_sph, drho_sph, z0_sph, z_min_top=1.0):
    V_sphere = (4.0/3.0) * np.pi * a_sph**3
    zc = z0_sph
    w_star, t_star = minimal_w_for_guard_symmetric(V_sphere, zc, z_min_top)
    def ztop_bot_fn(r):
        return lens3d_bounds_symmetric(r, zc, t_star, w_star, z_min_top=z_min_top)
    return dict(kind="SYM-DEFAULT", zc=zc, w=w_star, tmax=t_star, drho=drho_sph,
                ztop_bot_fn=ztop_bot_fn, r_max=8.0*w_star)

def build_default_flatcap_lens(a_sph, drho_sph, z0_sph, z_min_top=1.0):
    V_sphere = (4.0/3.0) * np.pi * a_sph**3
    z_flat = z0_sph
    w_star, t_star = minimal_w_for_guard_flat(V_sphere, z_flat, z_min_top)
    def ztop_bot_fn(r):
        return lens3d_bounds_flat(r, z_flat, t_star, w_star, z_min_top=z_min_top)
    return dict(kind="FLAT-DEFAULT", z_flat=z_flat, w=w_star, tmax=t_star, drho=drho_sph,
                ztop_bot_fn=ztop_bot_fn, r_max=8.0*w_star)

# Numeric search for flat family (unchanged)

def find_axisymmetric_matches_to_sphere(
    x, g_sphere,
    default_model,
    shape="symmetric",
    peak_tol=0.02,
    drho_bounds=(100.0, 1200.0),
    frac_span_w=0.25,
    frac_span_depth=0.20,
    frac_span_tmax=0.25,
    z_min_top=1.0,
    r_extent_mult=5.0,
    dr=25.0, dz=25.0, ntheta=24,
    topN=3
):
    g0_sph, hw0_sph = peak_and_halfwidth(x, g_sphere)
    if np.isnan(hw0_sph):
        raise RuntimeError("Sphere half-width not found.")
    results = []
    if shape == "symmetric":
        zc0, w0, t0 = default_model["zc"], default_model["w"], default_model["tmax"]
        drho0 = default_model["drho"]
        def ztop_bot_fn_factory(zc, w, tmax):
            return lambda r: lens3d_bounds_symmetric(r, zc, tmax, w, z_min_top=z_min_top)
        zc_list = np.linspace((1.0 - frac_span_depth)*zc0, (1.0 + frac_span_depth)*zc0, 5)
        w_list  = np.linspace((1.0 - frac_span_w)*w0,  (1.0 + frac_span_w)*w0,  5)
        t_list  = np.linspace((1.0 - frac_span_tmax)*t0,(1.0 + frac_span_tmax)*t0,5)
        label = "SYM"
        for zc in zc_list:
            tmax_max = 2.0 * (zc - z_min_top)
            for w in w_list:
                for tm in t_list:
                    if tm <= 0 or tm > tmax_max:
                        continue
                    ztop_bot_fn = ztop_bot_fn_factory(zc, w, tm)
                    r_max = max(9.0*w, x.max() + 8.0*w)
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
                    results.append(dict(kind=label, zc=zc, w=w, tmax=tm, drho_eff=drho_eff,
                                        g_peak=gp2, hw=hw2, e_hw=abs(hw2 - hw0_sph)/max(hw0_sph,1e-12),
                                        ztop_bot_fn=ztop_bot_fn, r_max=r_max))
    else:
        zf0, w0, t0 = default_model["z_flat"], default_model["w"], default_model["tmax"]
        drho0 = default_model["drho"]
        def ztop_bot_fn_factory(z_flat, w, tmax):
            return lambda r: lens3d_bounds_flat(r, z_flat, tmax, w, z_min_top=z_min_top)
        zf_list = np.linspace((1.0 - frac_span_depth)*zf0, (1.0 + frac_span_depth)*zf0, 5)
        w_list  = np.linspace((1.0 - frac_span_w)*w0,  (1.0 + frac_span_w)*w0,  5)
        t_list  = np.linspace((1.0 - frac_span_tmax)*t0,(1.0 + frac_span_tmax)*t0,5)
        label = "FLAT"
        for zf in zf_list:
            tmax_max = (zf - z_min_top)
            if tmax_max <= 0:
                continue
            for w in w_list:
                for tm in t_list:
                    if tm <= 0 or tm > tmax_max:
                        continue
                    ztop_bot_fn = ztop_bot_fn_factory(zf, w, tm)
                    r_max = max(9.0*w, x.max() + 8.0*w)
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

# Analytic symmetric search

def find_analytic_sigmoid_matches_to_sphere(
    x, g_sphere,
    z0_seed, w_seed, A_seed, drho_seed,
    peak_tol=0.02,
    drho_bounds=(100.0, 1200.0),
    frac_span_w=0.25,
    frac_span_depth=0.20,
    frac_span_A=0.25,
    z_min_top=1.0,
    topN=3
):
    g0_sph, hw0_sph = peak_and_halfwidth(x, g_sphere)
    if np.isnan(hw0_sph):
        raise RuntimeError("Sphere half-width not found.")
    z0_list = np.linspace((1.0 - frac_span_depth)*z0_seed, (1.0 + frac_span_depth)*z0_seed, 5)
    w_list  = np.linspace((1.0 - frac_span_w)*w_seed,      (1.0 + frac_span_w)*w_seed,      5)
    A_list  = np.linspace((1.0 - frac_span_A)*A_seed,      (1.0 + frac_span_A)*A_seed,      5)
    results = []
    for z0 in z0_list:
        Amax = max(1e-6, 2.0*(z0 - z_min_top))  # z_top(0) = z0 + A/2 >= z_min_top
        for w in w_list:
            for A in A_list:
                if A <= 0 or A > Amax:
                    continue
                g_base = lopolith_gravity_spectral(np.abs(x), w, A, A, z0, drho_seed) * UGAL
                gp, hw = peak_and_halfwidth(x, g_base)
                if gp <= 0 or np.isnan(hw):
                    continue
                scale = g0_sph / gp
                drho_eff = drho_seed * scale
                if not (drho_bounds[0] <= drho_eff <= drho_bounds[1]):
                    continue
                g_scaled = g_base * scale
                gp2, hw2 = peak_and_halfwidth(x, g_scaled)
                if abs(gp2 - g0_sph) > peak_tol * abs(g0_sph):
                    continue
                results.append(dict(kind="ANALYTIC-SYM", z0=z0, w=w, A=A, drho_eff=drho_eff,
                                    g_peak=gp2, hw=hw2, e_hw=abs(hw2 - hw0_sph)/max(hw0_sph,1e-12)))
    results.sort(key=lambda d: d["e_hw"])
    return results[:topN], (g0_sph, hw0_sph)

# =========================================================
# OptionC runner (scoped inside a function to avoid bare returns at module level)
# =========================================================

def run_optionC(args, x, g_sphere, a_sphere, z0_sph, drho_sph):
    z_min_top = args.zmin_top
    # Symmetric: use analytic seeds from numeric volume-matched default
    sym_def  = build_default_symmetric_lens(a_sphere, drho_sph, z0_sph, z_min_top=z_min_top)
    z0_seed = sym_def["zc"]; w_seed = sym_def["w"]; A_seed = sym_def["tmax"]; drho_seed = sym_def["drho"]
    # Flat-cap: numeric default retained
    flat_def = build_default_flatcap_lens(a_sphere, drho_sph, z0_sph, z_min_top=z_min_top)

    # Defaults
    g_sym_def  = lopolith_gravity_spectral(np.abs(x), w_seed, A_seed, A_seed, z0_seed, drho_seed) * UGAL
    g_flat_def = gz_axisymmetric_lens(x, flat_def["drho"], flat_def["ztop_bot_fn"], flat_def["r_max"],
                                      dr=args.dr, dz=args.dz, ntheta=args.ntheta) * UGAL
    gp_sym_def, hw_sym_def   = peak_and_halfwidth(x, g_sym_def)
    gp_flat_def, hw_flat_def = peak_and_halfwidth(x, g_flat_def)
    print("Default (analytic) symmetric sigmoid:")
    print(f" z0={z0_seed:.1f} m, w={w_seed:.1f} m, A={A_seed:.1f} m, Δρ={drho_seed:.1f} kg/m^3")
    print(f" Peak ≈ {gp_sym_def:.2f} µGal, Half-width ≈ {hw_sym_def:.1f} m")
    print("\nDefault (volume-matched) flat cap (numeric):")
    print(f" z_flat={flat_def['z_flat']:.1f} m, w={flat_def['w']:.1f} m, t_max={flat_def['tmax']:.1f} m, Δρ={flat_def['drho']:.1f} kg/m^3")
    print(f" Peak ≈ {gp_flat_def:.2f} µGal, Half-width ≈ {hw_flat_def:.1f} m\n")

    # Searches
    sym_matches, _ = find_analytic_sigmoid_matches_to_sphere(
        x, g_sphere,
        z0_seed=z0_seed, w_seed=w_seed, A_seed=A_seed, drho_seed=drho_seed,
        peak_tol=args.peak_tol, drho_bounds=(args.drho_min, args.drho_max),
        frac_span_w=0.25, frac_span_depth=0.20, frac_span_A=0.25,
        z_min_top=z_min_top, topN=3
    )
    flat_matches, _ = find_axisymmetric_matches_to_sphere(
        x, g_sphere, default_model=flat_def, shape="flat",
        peak_tol=args.peak_tol, drho_bounds=(args.drho_min, args.drho_max),
        frac_span_w=0.25, frac_span_depth=0.20, frac_span_tmax=0.25,
        z_min_top=z_min_top, r_extent_mult=5.0,
        dr=args.dr, dz=args.dz, ntheta=args.ntheta, topN=3
    )

    # Build curves
    g_series = [("Sphere", x, g_sphere, "tab:blue")]
    polygons = []
    x_poly = np.linspace(-abs(args.xmax), abs(args.xmax), 401)

    def build_curve_sym_analytic(match_present):
        if match_present is None:
            z0=z0_seed; w=w_seed; A=A_seed; drho=drho_seed
            g = lopolith_gravity_spectral(np.abs(x), w, A, A, z0, drho) * UGAL
            tag = "Sigmoid (analytic sym) (default)"
        else:
            z0=match_present["z0"]; w=match_present["w"]; A=match_present["A"]; drho=match_present["drho_eff"]
            g = lopolith_gravity_spectral(np.abs(x), w, A, A, z0, drho) * UGAL
            tag = "Sigmoid (analytic sym) (match)"
        def zt_absx(xx):
            rr = np.abs(xx); return z0 + A * s_of_r(rr, w)
        def zb_absx(xx):
            rr = np.abs(xx); return z0 - A * s_of_r(rr, w)
        poly = make_axisymmetric_polygon_xz(x_poly, zt_absx, zb_absx)
        return g, poly, tag

    def build_curve_flat_numeric(match_present, default):
        if match_present is None:
            ztop_bot_fn = default["ztop_bot_fn"]; r_max = default["r_max"]
            g = gz_axisymmetric_lens(x, default["drho"], ztop_bot_fn, r_max, dr=args.dr, dz=args.dz, ntheta=args.ntheta) * UGAL
            tag = "Sigmoid (flat) (default)"
        else:
            ztop_bot_fn = match_present["ztop_bot_fn"]; r_max = match_present["r_max"]
            g_base = gz_axisymmetric_lens(x, 1.0, ztop_bot_fn, r_max, dr=args.dr, dz=args.dz, ntheta=args.ntheta) * UGAL
            g = g_base * match_present["drho_eff"]
            tag = "Sigmoid (flat) (match)"
        def zt_absx(xx): rr = np.abs(xx); zt,_ = ztop_bot_fn(rr); return zt
        def zb_absx(xx): rr = np.abs(xx); _,zb = ztop_bot_fn(rr); return zb
        poly = make_axisymmetric_polygon_xz(x_poly, zt_absx, zb_absx)
        return g, poly, tag

    m_sym  = sym_matches[0]  if sym_matches else None
    m_flat = flat_matches[0] if flat_matches else None
    g_sym,  poly_sym,  tag_sym  = build_curve_sym_analytic(m_sym)
    g_flat, poly_flat, tag_flat = build_curve_flat_numeric(m_flat, flat_def)

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
    ax.set_title("Sphere vs. analytic symmetric + numeric flat (defaults; peak-matched search)")
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
    axg.set_title("Geometry (analytic symmetric + numeric flat)")
    axg.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =========================================================
# CLI / main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sphere vs. axisymmetric sigmoids. Use --mode=param to enter parameters; "
            "--shape can be 'sigmoid' (analytic), 'symmetric' (numeric), or 'flat' (numeric)."
        )
    )
    parser.add_argument("--mode", choices=["optionC", "param"], default="param",
                        help="optionC = search; param = user parameters.")
    parser.add_argument("--shape", choices=["sigmoid", "symmetric", "flat"], default="sigmoid",
                        help="Sigmoid type for --mode=param.")

    # Sphere
    parser.add_argument("--a_sphere", type=float, default=200.0, help="Sphere radius (m).")
    parser.add_argument("--z0_sphere", type=float, default=800.0, help="Sphere center depth (m).")
    parser.add_argument("--drho_sphere", type=float, default=300.0, help="Sphere density contrast (kg/m^3).")

    # Profile
    parser.add_argument("--xmax", type=float, default=3000.0, help="Half-length of profile (m).")
    parser.add_argument("--nx", type=int, default=801, help="Number of profile samples.")

    # Numeric resolution (legacy)
    parser.add_argument("--dr", type=float, default=20.0, help="Radial step (m) for numeric model.")
    parser.add_argument("--dz", type=float, default=20.0, help="Vertical step (m) for numeric model.")
    parser.add_argument("--ntheta", type=int, default=48, help="Azimuth samples per ring (numeric model).")
    parser.add_argument("--zmin_top", type=float, default=1.0, help="Surface guard: top ≥ zmin_top (m).")

    # Numeric symmetric/flat params
    parser.add_argument("--zc", type=float, default=350.0, help="Symmetric lens center depth (m).")
    parser.add_argument("--w", type=float, default=600.0, help="Lateral scale w (m).")
    parser.add_argument("--tmax", type=float, default=260.0, help="Max thickness t_max (m).")
    parser.add_argument("--drho", type=float, default=300.0, help="Lens density contrast (kg/m^3).")
    parser.add_argument("--zflat", type=float, default=380.0, help="Flat-cap bottom depth (m).")

    # Analytic sigmoid params
    parser.add_argument("--z0", type=float, default=1500.0, help="Sigmoid reference depth z0 (m).")
    parser.add_argument("--Atop", type=float, default=300.0, help="Top amplitude (m).")
    parser.add_argument("--Abot", type=float, default=300.0, help="Bottom amplitude (m).")
    parser.add_argument("--Aequal", type=float, default=None, help="If provided, sets Atop = Abot = Aequal.")

    # OptionC controls
    parser.add_argument("--peak_tol", type=float, default=0.02, help="Peak tolerance for OptionC.")
    parser.add_argument("--drho_min", type=float, default=100.0, help="Lower bound for Δρ in matching.")
    parser.add_argument("--drho_max", type=float, default=1200.0, help="Upper bound for Δρ in matching.")

    args = parser.parse_args()

    # Aequal convenience
    if args.Aequal is not None:
        args.Atop = float(args.Aequal)
        args.Abot = float(args.Aequal)
        print(f"Using Aequal: setting Atop = Abot = {args.Aequal:.3f} m")

    # Sphere reference
    a_sphere = args.a_sphere
    drho_sph = args.drho_sphere
    z0_sph = args.z0_sphere
    x = np.linspace(-abs(args.xmax), abs(args.xmax), args.nx)
    g_sphere = sphere_gz_profile(x, a_sphere, drho_sph, z0_sph) * UGAL
    g0_sph, hw0_sph = peak_and_halfwidth(x, g_sphere)
    print("\nReference sphere:")
    print(f" a={a_sphere:.1f} m, Δρ={drho_sph:.1f} kg/m^3, z0={z0_sph:.1f} m")
    print(f" Peak ≈ {g0_sph:.2f} µGal, Half-width ≈ {hw0_sph:.1f} m\n")

    if args.mode == "optionC":
        run_optionC(args, x, g_sphere, a_sphere, z0_sph, drho_sph)
        return

    # PARAMETRIC MODE
    if args.shape == "sigmoid":
        # Analytic sigmoid using Atop/Abot
        R = np.abs(x)
        gz_sig = lopolith_gravity_spectral(R, args.w, args.Atop, args.Abot, args.z0, args.drho) * UGAL
        def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
        gp_sig, hw_sig = peak_and_halfwidth(x, gz_sig)
        print("User-specified sigmoid (analytic):")
        print(f" z0={args.z0:.0f}, w={args.w:.0f}, A_top={args.Atop:.0f}, A_bot={args.Abot:.0f}, Δρ={args.drho:.0f}")
        print(f" Peak ≈ {gp_sig:.2f} µGal  Δpeak = {gp_sig-g0_sph:+.2f} µGal ({(gp_sig-g0_sph)/max(1e-12,g0_sph)*100.0:+.2f}%)")
        if not np.isnan(hw_sig):
            print(f" Half-width ≈ {hw_sig:.1f} m  ΔHW = {hw_sig-hw0_sph:+.1f} m ({(hw_sig-hw0_sph)/max(1e-12,hw0_sph)*100.0:+.2f}%)")
        else:
            print(" Half-width: could not be determined.")
        print(f" RMSE (profile) = {rmse(gz_sig, g_sphere):.2f} µGal")
        # Geometry
        x_poly = np.linspace(-abs(args.xmax), abs(args.xmax), 401)
        def zt_absx(xx): rr = np.abs(xx); return args.z0 + args.Atop * s_of_r(rr, args.w)
        def zb_absx(xx): rr = np.abs(xx); return args.z0 - args.Abot * s_of_r(rr, args.w)
        poly = make_axisymmetric_polygon_xz(x_poly, zt_absx, zb_absx)
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 8))
        ax = axes[0]
        ax.plot(x, g_sphere, label="Sphere", lw=2, c="tab:blue")
        ax.plot(x, gz_sig,    label="Sigmoid (analytic)", lw=2, c="tab:orange")
        ax.set_xlabel("Distance x (m)")
        ax.set_ylabel("Gravity $g_z$ ($\\mu$Gal)")
        ax.set_title("Sphere vs. analytic sigmoid lopolith")
        ax.grid(True, alpha=0.3)
        ax.legend()
        axg = axes[1]
        axg.set_aspect('equal')
        axg.plot([x.min(), x.max()], [0, 0], 'k-', lw=1)
        sph = Circle((0, z0_sph), radius=a_sphere, facecolor='tab:blue', alpha=0.35, edgecolor='k')
        axg.add_patch(sph)
        axg.text(0, z0_sph, "sphere", ha='center', va='center', fontsize=9)
        axg.add_patch(Polygon(poly, closed=True, facecolor="tab:orange", alpha=0.35, edgecolor='k'))
        axg.text(0, args.z0, "sigmoid", ha='center', va='center', fontsize=9)
        axg.set_xlim(-abs(args.xmax), abs(args.xmax))
        axg.set_ylim(0, 1500)
        axg.invert_yaxis()
        axg.set_xlabel("Distance x (m)")
        axg.set_ylabel("Depth z (m, positive downward)")
        axg.set_title("Geometry (analytic model)")
        axg.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    # Legacy numeric parametric mode for 'symmetric' or 'flat'
    z_min_top = args.zmin_top
    if args.shape == "symmetric":
        def ztop_bot_fn(r):
            return lens3d_bounds_symmetric(r, args.zc, args.tmax, args.w, z_min_top=z_min_top)
        family = f"Sigmoid (sym): zc={args.zc:.0f}, w={args.w:.0f}, tmax={args.tmax:.0f}, Δρ={args.drho:.0f}"
    else:
        def ztop_bot_fn(r):
            return lens3d_bounds_flat(r, args.zflat, args.tmax, args.w, z_min_top=z_min_top)
        family = f"Sigmoid (flat): z_flat={args.zflat:.0f}, w={args.w:.0f}, tmax={args.tmax:.0f}, Δρ={args.drho:.0f}"
    r_max = max(9.0*args.w, x.max() + 8.0*args.w)
    g_lens = gz_axisymmetric_lens(x, args.drho, ztop_bot_fn, r_max, dr=args.dr, dz=args.dz, ntheta=args.ntheta) * UGAL
    def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
    gp_len, hw_len = peak_and_halfwidth(x, g_lens)
    print("User-specified lens (numeric):")
    print(f" {family}")
    print(f" Peak ≈ {gp_len:.2f} µGal  Δpeak = {gp_len-g0_sph:+.2f} µGal ({(gp_len-g0_sph)/max(1e-12, g0_sph)*100.0:+.2f}%)")
    if not np.isnan(hw_len):
        print(f" Half-width ≈ {hw_len:.1f} m  ΔHW = {hw_len-hw0_sph:+.1f} m ({(hw_len-hw0_sph)/max(1e-12, hw0_sph)*100.0:+.2f}%)")
    else:
        print(" Half-width: could not be determined (non-unimodal or noisy).")
    print(f" RMSE (profile) = {rmse(g_lens, g_sphere):.2f} µGal\n")
    # Geometry
    x_poly = np.linspace(-abs(args.xmax), abs(args.xmax), 401)
    def zt_absx(xx): rr = np.abs(xx); zt,_ = ztop_bot_fn(rr); return zt
    def zb_absx(xx): rr = np.abs(xx); _,zb = ztop_bot_fn(rr); return zb
    poly = make_axisymmetric_polygon_xz(x_poly, zt_absx, zb_absx)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 8))
    ax = axes[0]
    ax.plot(x, g_sphere, label="Sphere", lw=2, c="tab:blue")
    ax.plot(x, g_lens,  label=family,     lw=2, c="tab:orange")
    ax.set_xlabel("Distance x (m)")
    ax.set_ylabel("Gravity $g_z$ ($\\mu$Gal)")
    ax.set_title("Sphere vs. user-specified numeric 3D sigmoid")
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
