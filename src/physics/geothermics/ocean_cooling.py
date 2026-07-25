"""
Activity 9.C -- Oceanic half-space and plate cooling vs. real data
-------------------------------------------------------------------
Loads real sediment-corrected heat flow (hfmix_sed.dat) and bathymetry
(bath_nolips.tz) compilations, binned by seafloor age, and plots each
in both raw and linearized form against the half-space and plate
cooling model predictions:

    q(t)^-2  is linear in t        (half-space)
    s(t)     is linear in sqrt(t)  (half-space)

Both linearizations follow directly from the closed-form solutions in
the geothermics chapter; verify_derivations() checks them symbolically
against that chapter's own equations before any plotting happens.

Unlike an earlier version of this script, the half-space curves are
NOT drawn from assumed literature constants (T_m, k, alpha, ...) --
they are FIT directly to the real data, restricted to the age range
before it visibly departs from linear in the transformed variable
(t < Q_FIT_MAX_MYR for q^-2 vs t; t < S_FIT_MAX_MYR for s vs sqrt(t)).
This means the half-space curve is anchored to this dataset specifically,
not to a generic textbook parameter choice. kappa and L are still
needed to draw the plate-model curve's shape (its Fourier sum depends
on them explicitly, not just on the fitted early-time amplitude), and
are set to course-wide default values -- edit KAPPA_KM2MY / L_PLATE
below if you want the plate curve's decay behaviour to differ.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).resolve().parent
DATADIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "geothermics" / "oceanic"

# ----------------------------------------------------------------------
# Fit ranges: the age below which each linearized quantity is treated
# as pre-deviation and used to fit the half-space amplitude.
# ----------------------------------------------------------------------
Q_FIT_MAX_MYR = 75.0   # heat flow: fit q^-2 vs t using t < this
S_FIT_MAX_MYR = 60.0   # subsidence: fit s vs sqrt(t) using t < this

# ----------------------------------------------------------------------
# Constants used only for the plate model's decay shape (not for the
# half-space amplitude, which is now fit directly to the data below).
# ----------------------------------------------------------------------
KAPPA_KM2MY = 32.0      # km^2/My, thermal diffusivity (course-wide default)
L_PLATE = 125.0         # km, plate thickness (course-wide default)
N_TERMS = 50            # terms in the plate-cooling Fourier sum (converged)
RIDGE_DEPTH_M = 2500.0  # m, reference depth at t=0 (added to s(t), which
                          # gives subsidence relative to the ridge, not
                          # absolute depth)

# ----------------------------------------------------------------------
# Unit conversions: work in SI internally, convert to geological units
# for plotting/display only.
# ----------------------------------------------------------------------
MYR_TO_S = 1e6 * 365.25 * 24 * 3600.0
KM2_TO_M2 = 1e6
KAPPA_SI = KAPPA_KM2MY * KM2_TO_M2 / MYR_TO_S   # m^2/s
L_SI = L_PLATE * 1000.0                          # m


def fit_amplitude(x_lin, y_lin):
    """
    Least-squares slope of y_lin = slope * x_lin, forced through the
    origin (appropriate here since both half-space forms genuinely
    vanish at t=0). Returns the fitted slope.
    """
    x_lin = np.asarray(x_lin, dtype=float)
    y_lin = np.asarray(y_lin, dtype=float)
    return np.sum(x_lin * y_lin) / np.sum(x_lin * x_lin)


def fit_q_amplitude(hf, t_max=Q_FIT_MAX_MYR):
    """
    Fit q_0(t) = A_q / sqrt(t) to heat flow data with t < t_max, via
    linear regression of q^-2 against t (forced through the origin).
    Returns A_q in mW/m^2 * Myr^0.5.
    """
    young = hf[hf['t_mid'] < t_max]
    slope = fit_amplitude(young['t_mid'], 1.0 / young['q_mean']**2)
    return 1.0 / np.sqrt(slope)


def fit_s_amplitude(bath, t_max=S_FIT_MAX_MYR):
    """
    Fit s(t) = B_s * sqrt(t) to bathymetry data (relative to ridge
    depth) with t < t_max, via linear regression forced through the
    origin. Returns B_s in m / Myr^0.5.
    """
    young = bath[bath['t_mid'] < t_max]
    return fit_amplitude(np.sqrt(young['t_mid']), young['mean'] - RIDGE_DEPTH_M)


def q_halfspace(t_myr, A_q):
    """Half-space cooling heat flow, mW/m^2, t in Myr, given fitted A_q."""
    return A_q / np.sqrt(np.asarray(t_myr, dtype=float))


def q_plate(t_myr, A_q, n_terms=N_TERMS):
    """
    Plate cooling heat flow, mW/m^2, t in Myr. The prefactor
    k(T_m-T_0)/L is written in terms of the fitted half-space
    amplitude A_q via k(T_m-T_0) = A_q*sqrt(pi*kappa) -- the same
    physical combination appears in both forms (see Question 1 of the
    activity), so this substitution is exact, not an approximation.
    All time-dependence is kept in Myr throughout to avoid mixed-unit
    bugs; the prefactor absorbs the necessary unit conversions once.
    """
    t_myr = np.asarray(t_myr, dtype=float)
    n = np.arange(1, n_terms + 1).reshape(-1, 1)
    tau_n_myr = (n * np.pi / L_SI) ** 2 * KAPPA_SI * MYR_TO_S
    terms = np.exp(-tau_n_myr * t_myr[None, :])
    total = 1 + 2 * np.sum(terms, axis=0)
    prefactor = A_q * np.sqrt(np.pi * KAPPA_SI * MYR_TO_S) / L_SI
    return prefactor * total


def s_halfspace(t_myr, B_s):
    """Half-space subsidence, m, t in Myr, given fitted B_s."""
    return B_s * np.sqrt(np.asarray(t_myr, dtype=float))


def s_plate(t_myr, B_s, n_terms=N_TERMS):
    """
    Plate cooling subsidence, m, t in Myr. The prefactor
    rho_m*alpha*(T_m-T_0)*L/(2*(rho_m-rho_w)) is written in terms of
    the fitted half-space amplitude B_s, by the same substitution
    logic as q_plate (see Question 2-3 of the activity).
    """
    t_myr = np.asarray(t_myr, dtype=float)
    n = np.arange(0, n_terms).reshape(-1, 1)
    m = 2 * n + 1
    tau_m_myr = (m * np.pi / L_SI) ** 2 * KAPPA_SI * MYR_TO_S
    terms = np.exp(-tau_m_myr * t_myr[None, :]) / m**2
    total = 1 - (8 / np.pi**2) * np.sum(terms, axis=0)
    prefactor = B_s * L_SI * np.sqrt(np.pi / (KAPPA_SI * MYR_TO_S)) / 4
    return prefactor * total


def verify_derivations():
    """
    Symbolically confirm q_halfspace and s_halfspace reproduce the
    chapter's closed-form equations exactly, before any data is
    plotted against them.
    """
    import sympy as sp
    t, kappa, k, Tm, T0, rho_m, rho_w, alpha = sp.symbols(
        't kappa k T_m T_0 rho_m rho_w alpha', positive=True)
    z, eta = sp.symbols('z eta', positive=True)

    T = T0 + (Tm - T0) * sp.erf(z / (2 * sp.sqrt(kappa * t)))
    q_derived = sp.simplify(k * sp.diff(T, z).subs(z, 0))
    q_chapter = k * (Tm - T0) / sp.sqrt(sp.pi * kappa * t)
    assert sp.simplify(q_derived - q_chapter) == 0, "heat flow mismatch!"

    integral = sp.integrate(sp.erfc(eta), (eta, 0, sp.oo))
    assert integral == 1 / sp.sqrt(sp.pi), "erfc integral identity mismatch!"

    s_derived = sp.simplify(
        (rho_m / (rho_m - rho_w)) * alpha * 2 * (Tm - T0) * sp.sqrt(kappa * t) * integral
    )
    s_chapter = (2 * rho_m * alpha * (Tm - T0) / (rho_m - rho_w)) * sp.sqrt(kappa * t / sp.pi)
    assert sp.simplify(s_derived - s_chapter) == 0, "subsidence mismatch!"
    print("Derivations verified against chapter closed forms: OK")


def load_heat_flow(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip().startswith('%') or not line.strip():
                continue
            parts = line.split('%')[0].split()
            if len(parts) < 10:
                continue
            rows.append(dict(tmin=float(parts[0]), tmax=float(parts[1]), N=int(parts[2]),
                              q_mean=float(parts[3]), q_sd=float(parts[4]), use=int(parts[9])))
    df = pd.DataFrame(rows)
    df = df[df['use'] == 1].copy()
    df['t_mid'] = (df['tmin'] + df['tmax']) / 2
    return df


def load_bathymetry(path):
    df = pd.read_csv(path, sep=r'\s+', header=None,
                      names=['tmin', 'tmax', 'mean', 'std', 'Q1', 'Q2', 'Q3', 'use'])
    df = df[df['use'] == 1].copy()
    df['t_mid'] = (df['tmin'] + df['tmax']) / 2
    return df


def plot_heat_flow(hf):
    A_q = fit_q_amplitude(hf)
    print(f"Fitted heat flow amplitude A_q = {A_q:.2f} mW/m^2 * Myr^0.5 "
          f"(from t < {Q_FIT_MAX_MYR} Ma)")

    t_model = np.linspace(0.05, hf['t_mid'].max() * 1.05, 400)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.axvspan(0, Q_FIT_MAX_MYR, color='0.9', zorder=0)
    ax.errorbar(hf['t_mid'], hf['q_mean'], yerr=hf['q_sd'], fmt='o', ms=3,
                color='0.35', ecolor='0.75', elinewidth=1, capsize=0, zorder=2)
    ax.plot(t_model, q_halfspace(t_model, A_q), color='tab:orange', lw=2, label='Half-space (fit)')
    ax.plot(t_model, q_plate(t_model, A_q), color='tab:red', lw=2, label='Plate')
    ax.set_xlabel('Age (Ma)')
    ax.set_ylabel(r'Heat flow, $q$ (mW m$^{-2}$)')
    ax.set_title('Heat flow vs. age')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.axvspan(0, Q_FIT_MAX_MYR, color='0.9', zorder=0)
    q_err_lin = 2 * hf['q_sd'] / hf['q_mean']**3  # propagated error on q^-2
    ax.errorbar(hf['t_mid'], 1 / hf['q_mean']**2, yerr=q_err_lin, fmt='o', ms=3,
                color='0.35', ecolor='0.75', elinewidth=1, capsize=0, zorder=2)
    ax.plot(t_model, 1 / q_halfspace(t_model, A_q)**2, color='tab:orange', lw=2, label='Half-space (fit)')
    ax.plot(t_model, 1 / q_plate(t_model, A_q)**2, color='tab:red', lw=2, label='Plate')
    ax.set_xlabel('Age (Ma)')
    ax.set_ylabel(r'$q^{-2}$ (mW m$^{-2}$)$^{-2}$')
    ax.set_title(r'Linearized: $q^{-2}$ vs. age')
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGDIR / '9c_heatflow_age.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / '9c_heatflow_age.png', dpi=200, bbox_inches='tight')


def plot_bathymetry(bath):
    B_s = fit_s_amplitude(bath)
    print(f"Fitted subsidence amplitude B_s = {B_s:.2f} m / Myr^0.5 "
          f"(from t < {S_FIT_MAX_MYR} Ma)")

    t_model = np.linspace(0.05, bath['t_mid'].max() * 1.05, 400)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.axvspan(0, S_FIT_MAX_MYR, color='0.9', zorder=0)
    ax.errorbar(bath['t_mid'], bath['mean'], yerr=bath['std'], fmt='o', ms=3,
                color='0.35', ecolor='0.75', elinewidth=1, capsize=0, zorder=2)
    ax.plot(t_model, RIDGE_DEPTH_M + s_halfspace(t_model, B_s), color='tab:blue', lw=2, label='Half-space (fit)')
    ax.plot(t_model, RIDGE_DEPTH_M + s_plate(t_model, B_s), color='tab:purple', lw=2, label='Plate')
    ax.set_xlabel('Age (Ma)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Bathymetry vs. age')
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.axvspan(0, np.sqrt(S_FIT_MAX_MYR), color='0.9', zorder=0)
    ax.errorbar(np.sqrt(bath['t_mid']), bath['mean'], yerr=bath['std'], fmt='o', ms=3,
                color='0.35', ecolor='0.75', elinewidth=1, capsize=0, zorder=2)
    ax.plot(np.sqrt(t_model), RIDGE_DEPTH_M + s_halfspace(t_model, B_s), color='tab:blue', lw=2, label='Half-space (fit)')
    ax.plot(np.sqrt(t_model), RIDGE_DEPTH_M + s_plate(t_model, B_s), color='tab:purple', lw=2, label='Plate')
    ax.set_xlabel(r'$\sqrt{\mathrm{Age}}$ (Ma$^{1/2}$)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(r'Linearized: depth vs. $\sqrt{\mathrm{age}}$')
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGDIR / '9c_bathymetry_age.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / '9c_bathymetry_age.png', dpi=200, bbox_inches='tight')


def verify_plate_reduces_to_halfspace():
    """
    Sanity check, not just algebra: confirm q_plate and s_plate
    numerically approach q_halfspace and s_halfspace as t -> 0, since
    that reduction is asserted in the activity (Question 6a) and
    silently relied on by the prefactor substitution above.
    """
    A_q_test, B_s_test = 500.0, 300.0
    for t in [0.5, 1.0, 2.0]:
        qh, qp = q_halfspace(t, A_q_test), q_plate(np.array([t]), A_q_test)[0]
        sh, sp = s_halfspace(t, B_s_test), s_plate(np.array([t]), B_s_test)[0]
        assert abs(qp - qh) / qh < 0.01, f"q_plate does not reduce to q_halfspace at t={t}"
        assert abs(sp - sh) / sh < 0.01, f"s_plate does not reduce to s_halfspace at t={t}"
    print("Plate -> half-space reduction verified at small t: OK")


if __name__ == '__main__':
    verify_derivations()
    verify_plate_reduces_to_halfspace()

    hf = load_heat_flow(DATADIR / 'hfmix_sed.dat')
    bath = load_bathymetry(DATADIR / 'bath_nolips.tz')
    print(f"Heat flow bins (use=1): {len(hf)}, age range {hf['t_mid'].min():.1f}-{hf['t_mid'].max():.1f} Ma")
    print(f"Bathymetry bins (use=1): {len(bath)}, age range {bath['t_mid'].min():.1f}-{bath['t_mid'].max():.1f} Ma")

    plot_heat_flow(hf)
    plot_bathymetry(bath)
    print("Wrote 9c_heatflow_age.{pdf,png}, 9c_bathymetry_age.{pdf,png}")