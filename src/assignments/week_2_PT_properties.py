"""
**Week 2 Practical** — Mineral Physical Properties on a P–T Grid

**Aim**: Turn `mineral_properties.csv` into a lookup table of empirical constants, then
compute pressure- and temperature-dependent mineral properties (density, expansivity,
bulk/shear moduli, seismic velocities, heat capacity) on a regular P–T grid, and combine
them with modal mineralogy (Voigt–Reuss–Hill) to estimate bulk-rock properties.

Formulae follow *Properties of Earth Materials* (course notes, Ch. 2) and the constants
in Table 2.1 (`mineral_properties.csv`).
"""
# %% 0. Imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

T0 = 298.0    # reference temperature, K
P0 = 1.0e-4   # reference pressure, GPa (0.1 MPa)

DATA_PATH = Path(__file__).resolve().parents[2] / "week_2_physical_properties" / "assignment" / "mineral_properties.csv"

# %% 1. Load the mineral constants table
"""
`mineral_properties.csv` is transposed relative to a typical table: each row is a
parameter (identified by the short code in column 1) and each column (from column 2
onward) is a mineral. This maps each row to a constant, applying the column-header
scale factors (e.g. `a0 x 10^5` means the tabulated value must be divided by 1e5).
"""
_ROW_MAP = {
    2:  ("M",        1.0),     # formula weight, g/mol
    3:  ("V0",       1.0),     # molar volume, cm^3/mol
    4:  ("rho0",     1.0),     # density @ 298 K, 0.1 MPa, kg/m^3
    5:  ("wtH2O",    1.0),     # wt% H2O
    6:  ("a0",       1e-5),    # expansivity constants, K^-1, K^-2, K
    7:  ("a1",       1e-8),
    8:  ("a2",       1.0),
    9:  ("c0",       1e3),     # specific heat constants, kJ/kg/K -> J/kg/K
    10: ("c1",       1e3),
    11: ("c2",       1e3),
    12: ("c3",       1e3),
    13: ("KT0",      1.0),     # isothermal bulk modulus @ 298 K, 0.1 MPa, GPa
    14: ("KTp",      1.0),     # dKT/dP
    15: ("mu0",      1.0),     # shear modulus @ 298 K, 0.1 MPa, GPa
    16: ("Gamma",    1.0),     # dln(mu)/dln(rho)
    17: ("mup",      1.0),     # dmu/dP
    18: ("gamma_th", 1.0),     # (first) Grüneisen parameter
    19: ("delta_T",  1.0),     # Anderson-Grüneisen parameter
}


def load_mineral_table(path=DATA_PATH):
    """Read mineral_properties.csv into a DataFrame indexed by mineral name."""
    raw = pd.read_csv(path, header=None)
    minerals = raw.iloc[0, 2:].tolist()
    formulas = raw.iloc[1, 2:].tolist()

    records = {}
    for i, mineral in enumerate(minerals):
        col = 2 + i
        rec = {"formula": formulas[i]}
        for row_idx, (key, scale) in _ROW_MAP.items():
            rec[key] = float(raw.iat[row_idx, col]) * scale
        records[mineral] = rec

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "mineral"
    return df


minerals_df = load_mineral_table()

# %% 2. Build the P-T-mineral property grid
"""
Every property below follows directly from Ch. 2 of the course notes:
expansivity (2.3-2.6), density (2.1-2.2), bulk modulus (2.8-2.10), shear modulus
(2.11-2.12), seismic velocity (2.13-2.14) and specific heat (2.15-2.16). Pressure is
carried in GPa throughout (matching the units of KT, KT', mu, mu') and converted to/from
MPa only at the grid boundary and to Pa where an SI pressure derivative is required.
"""


def build_property_grid(df, P_MPa, T_K):
    """Compute mineral properties on a (P, T, mineral) grid.

    Returns an xarray.Dataset with dims (P, T, mineral) holding rho [kg/m^3],
    alpha [K^-1], KT & KS & mu [GPa], Cp [J/kg/K], Vp & Vs [km/s], poisson [-].
    """
    P_MPa = np.asarray(P_MPa, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    P_GPa = P_MPa * 1e-3

    Pb = P_GPa.reshape(-1, 1, 1)
    Tb = T_K.reshape(1, -1, 1)

    def c(name):
        return df[name].to_numpy().reshape(1, 1, -1)

    a0, a1, a2 = c("a0"), c("a1"), c("a2")
    rho0, KT0, KTp = c("rho0"), c("KT0"), c("KTp")
    mu0, mup, Gamma = c("mu0"), c("mup"), c("Gamma")
    gamma_th, delta_T = c("gamma_th"), c("delta_T")
    c0, c1, c2, c3 = c("c0"), c("c1"), c("c2"), c("c3")

    # --- temperature-only terms ---
    A = a0 * (Tb - T0) + 0.5 * a1 * (Tb**2 - T0**2) - a2 * (1.0 / Tb - 1.0 / T0)  # eq. 2.6
    alpha_T = a0 + a1 * Tb + a2 * Tb**-2                                          # eq. 2.3
    rho_T = rho0 * np.exp(-A)                                                     # eq. 2.1
    KT_T = KT0 * np.exp(-delta_T * A)                                             # eq. 2.8
    mu_T = mu0 * np.exp(-Gamma * A)                                               # eq. 2.11

    # --- pressure corrections ---
    KT_PT = KT_T + KTp * (Pb - P0)                                                # eq. 2.9
    mu_PT = mu_T + mup * (Pb - P0)                                                # eq. 2.12
    rho_PT = rho_T * (1 + (Pb - P0) / KT_PT)                                      # eq. 2.2

    # density at pressure P but reference temperature, used only to correct expansivity
    rho_P_ref = rho0 * (1 + (Pb - P0) / KT0)
    alpha_PT = alpha_T * (rho_P_ref / rho0) ** (-delta_T)                        # eq. 2.4

    KS_PT = KT_PT * (1 + Tb * gamma_th * alpha_PT)                               # eq. 2.10

    Vp = np.sqrt((KS_PT * 1e9 + 4 / 3 * mu_PT * 1e9) / rho_PT) / 1e3             # eq. 2.13
    Vs = np.sqrt(mu_PT * 1e9 / rho_PT) / 1e3                                     # eq. 2.14
    poisson = (Vp**2 - 2 * Vs**2) / (2 * (Vp**2 - Vs**2))

    Cp_T = c0 + c1 * Tb**-0.5 + c2 * Tb**-2 + c3 * Tb**-3                        # eq. 2.15
    dalpha_dT = a1 - 2 * a2 * Tb**-3                                             # eq. 2.7
    dCp_dP = -(Tb / rho_PT) * (alpha_PT**2 + dalpha_dT)                          # eq. 2.16, J/kg/K/Pa
    Cp_PT = Cp_T + dCp_dP * ((Pb - P0) * 1e9)

    dims = ("P", "T", "mineral")
    coords = {"P": P_MPa, "T": T_K, "mineral": df.index.to_numpy()}
    shape = (len(P_MPa), len(T_K), len(df))

    def full(arr):
        return np.broadcast_to(arr, shape)

    return xr.Dataset(
        {
            "rho": (dims, full(rho_PT)),
            "alpha": (dims, full(alpha_PT)),
            "KT": (dims, full(KT_PT)),
            "KS": (dims, full(KS_PT)),
            "mu": (dims, full(mu_PT)),
            "Cp": (dims, full(Cp_PT)),
            "Vp": (dims, full(Vp)),
            "Vs": (dims, full(Vs)),
            "poisson": (dims, full(poisson)),
        },
        coords=coords,
        attrs={
            "units": {
                "P": "MPa", "T": "K", "rho": "kg/m^3", "alpha": "1/K",
                "KT": "GPa", "KS": "GPa", "mu": "GPa", "Cp": "J/kg/K",
                "Vp": "km/s", "Vs": "km/s", "poisson": "-",
            }
        },
    )


# Problem-set grid: P = 0.1 to 1711.1 MPa in 570 MPa steps, T = 298 to 948 K in 50 K steps
P_grid_MPa = np.arange(0.1, 1711.1 + 1, 570.0)
T_grid_K = np.arange(298.0, 948.0 + 1, 50.0)

mineral_grid = build_property_grid(minerals_df, P_grid_MPa, T_grid_K)

# %% 3. Mixing formula: bulk-rock properties from modal mineralogy
"""
Density and heat capacity are volume-weighted arithmetic means; the elastic moduli use
the Voigt-Reuss-Hill average (eq. 2.28-2.29); Vp, Vs and Poisson's ratio are recomputed
from the mixed moduli and density rather than averaged directly.
"""


def mix_rock(ds, modes):
    """Combine per-mineral properties into bulk-rock properties given modal fractions.

    modes: dict of {mineral_name: volume_fraction} (need not be pre-normalized).
    Returns an xarray.Dataset with dims (P, T).
    """
    names = list(modes.keys())
    v = np.array([modes[m] for m in names], dtype=float)
    v = v / v.sum()

    sub = ds.sel(mineral=names)
    v_da = xr.DataArray(v, dims="mineral", coords={"mineral": names})

    def vrh(prop):
        voigt = (sub[prop] * v_da).sum("mineral")
        reuss = 1.0 / ((v_da / sub[prop]).sum("mineral"))
        return 0.5 * (voigt + reuss)

    rho = (sub["rho"] * v_da).sum("mineral")
    Cp = (sub["Cp"] * v_da).sum("mineral")
    KT = vrh("KT")
    KS = vrh("KS")
    mu = vrh("mu")

    Vp = np.sqrt((KS * 1e9 + 4 / 3 * mu * 1e9) / rho) / 1e3
    Vs = np.sqrt(mu * 1e9 / rho) / 1e3
    poisson = (Vp**2 - 2 * Vs**2) / (2 * (Vp**2 - Vs**2))

    return xr.Dataset({
        "rho": rho, "Cp": Cp, "KT": KT, "KS": KS, "mu": mu,
        "Vp": Vp, "Vs": Vs, "poisson": poisson, "VpVs": Vp / Vs,
    })


# Illustrative modal mineralogy for upper crust / lower crust / upper mantle.
# Adjust these to match the assignment's actual rock-type compositions.
rock_modes = {
    "granite": {
        "a-quartz": 0.25, "orthoclase": 0.25, "albite": 0.30,
        "anorthite": 0.05, "muscovite": 0.05, "annite": 0.10,
    },
    "mafic granulite": {
        "anorthite": 0.35, "albite": 0.15, "diopside": 0.20,
        "enstatite": 0.15, "hornblende": 0.10, "pyrope": 0.05,
    },
    "lherzolite": {
        "forsterite": 0.60, "fayalite": 0.05, "enstatite": 0.20,
        "diopside": 0.10, "spinel": 0.05,
    },
}

rock_grids = {name: mix_rock(mineral_grid, modes) for name, modes in rock_modes.items()}

# %% 4. Quick look: property vs temperature at each pressure, for each rock type
fig, axs = plt.subplots(1, 3, figsize=(11, 4), sharex=True)
for ax, (name, ds) in zip(axs, rock_grids.items()):
    for P in ds["P"].values:
        ax.plot(ds["T"], ds["Vp"].sel(P=P), label=f"{P:.0f} MPa")
    ax.set_title(name)
    ax.set_xlabel("Temperature (K)")
axs[0].set_ylabel("Vp (km/s)")
axs[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
