"""
Collisional activity -- Phase 2: Fischer et al.'s later-stage cooling
model, kept as a SEPARATE model from the sawtooth/thrust-stack
warming model in collisional_geotherms.py.

These are not competing descriptions of the same phase of orogenic
history -- they describe different parts of it:

  - collisional_geotherms.py (the sawtooth model) represents the
    EARLIEST phase: immediately after thrust stacking, a cold,
    non-equilibrium juxtaposition of a thrust sheet over a footwall,
    warming toward a NEW, HIGHER equilibrium as the thickened,
    more-radiogenic column traps heat it hasn't yet accumulated. This
    warming could transiently raise elevation (and hence R) without
    changing root thickness, if post-orogenic heating/melting occurs.
    Real R-vs-age data barely samples this regime (very few points at
    0-200 Ma), so this part of the story is largely unconstrained by
    the data used in this activity.

  - THIS file (Fischer's model) represents a LATER, much longer
    process: a thickened root cooling from a high initial q0 toward a
    lower final qf, held-fixed A, driven by some combination of
    thermal relaxation and (per discussion) erosional removal of
    radiogenic upper crust. Most of the real R-vs-age data (mostly
    older than 200 Ma) falls in the regime this model addresses, which
    is why it is the better match to the OBSERVED data trend, even
    though it is not the first thing that happens after collision.

Fischer's caption (quoted directly, for reference):
  "Analytical calculations cool from a geotherm parameterized by
  initial surface heat flow (q0) and crustal heat production (A) to
  an infinite time geotherm with lower heat flow (qf) but the same A
  value... Initial thermal lithospheric thicknesses are 60-90 km and
  final thicknesses are 190-230 km... Conductivity in all cases is
  2.6 W/m/K, and mantle potential temperature is 1300 C with an
  adiabatic gradient of 0.3 C/km."

The radiogenic layer thickness (45 km) is NOT stated in her caption --
it was found here by testing a range of values against her stated
lithospheric thickness ranges, not derived independently. If you have
the actual value from her methods, replace RADIOGENIC_LAYER_KM below.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).resolve().parent.parent.parent
FIGDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Fischer's stated parameters
# ----------------------------------------------------------------------
K_COND = 2.6           # W/(m K)
A_HP = 0.7             # uW/m^3, held fixed between initial and final states
Q0_INITIAL = 70.0       # mW/m^2 ("hotter" initial case; she also gives 65 for "cooler")
QF_FINAL = 45.0          # mW/m^2 (middle of her three qf cases: 40, 45, 50)
T0 = 0.0                 # deg C, surface temperature (not stated by Fischer; standard choice)
ADIABAT_T0, ADIABAT_GRAD = 1300.0, 0.3

# ----------------------------------------------------------------------
# Reverse-engineered (not stated by Fischer) -- see module docstring.
# ----------------------------------------------------------------------
RADIOGENIC_LAYER_KM = 45.0


def steady_state_T(z, q0, A=A_HP, k=K_COND, zr=RADIOGENIC_LAYER_KM, T0=T0):
    """
    Steady-state T(z) for a layer of constant A from 0 to zr, then A=0
    below zr (radiogenic crust over an effectively non-radiogenic
    lithospheric mantle).
    """
    z = np.asarray(z, dtype=float)
    q_at_zr = q0 - A * zr
    T_at_zr = T0 + (q0 * zr - 0.5 * A * zr**2) / k
    above = T0 + (q0 * z - 0.5 * A * z**2) / k
    below = T_at_zr + (q_at_zr * (z - zr)) / k
    return np.where(z <= zr, above, below)


def find_adiabat_crossing(q0, A=A_HP, k=K_COND, zr=RADIOGENIC_LAYER_KM, z_max=500):
    z = np.linspace(0.1, z_max, 200000)
    T = steady_state_T(z, q0, A, k, zr)
    diff = T - (ADIABAT_T0 + ADIABAT_GRAD * z)
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    return z[idx[0]] if len(idx) else None


def make_figure():
    z_init = find_adiabat_crossing(Q0_INITIAL)
    z_final = find_adiabat_crossing(QF_FINAL)
    print(f"Initial state (q0={Q0_INITIAL}): lithosphere thickness = {z_init:.0f} km "
          f"(Fischer: 60-90 km)")
    print(f"Final state (qf={QF_FINAL}):   lithosphere thickness = {z_final:.0f} km "
          f"(Fischer: 190-230 km)")

    z = np.linspace(0, z_final + 20, 400)
    T_init = steady_state_T(z, Q0_INITIAL)
    T_final = steady_state_T(z, QF_FINAL)
    adiabat = ADIABAT_T0 + ADIABAT_GRAD * z

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.plot(T_init, z, color='tab:red', lw=2,
            label=f'Initial ($q_0$={Q0_INITIAL:.0f} mW/m$^2$)')
    ax.plot(T_final, z, color='tab:blue', lw=2,
            label=f'Final ($q_f$={QF_FINAL:.0f} mW/m$^2$)')
    ax.plot(adiabat, z, color='0.5', lw=1, ls='--', label='Mantle adiabat')
    ax.axhline(RADIOGENIC_LAYER_KM, color='0.85', lw=0.6, ls=':')
    ax.text(5, RADIOGENIC_LAYER_KM - 3, 'base of radiogenic layer', fontsize=7, color='0.6')
    ax.set_xlabel('Temperature ($^\\circ$C)')
    ax.set_ylabel('Depth (km)')
    ax.set_xlim(0, 1600)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fischer_geotherms.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fischer_geotherms.png', dpi=200, bbox_inches='tight')
    print("Wrote fischer_geotherms.{pdf,png}")


if __name__ == '__main__':
    make_figure()