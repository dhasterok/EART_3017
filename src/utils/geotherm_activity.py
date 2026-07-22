"""
Activity 8.D -- Building a Geotherm
-------------------------------------------------------------------
Two-stage piecewise steady-state geotherm calculator: q(z) is built
first, entirely independent of k and T, then T(z) is built from the
completed q(z) using Fourier's law. Reproduces every number used in
the activity's guided tables, the given adiabat-crossing point, and
the k/A perturbation reveal table -- so any of it can be checked or
retuned by changing LAYERS_BASE / Q0 / T0 below and rerunning.

Units: z in km, k in W/(m K), A in uW/m^3, q in mW/m^2, T in deg C.
(A [uW/m^3] * z [km] = mW/m^2 with no conversion factor -- see note
in nevada_fft.py / heatflow_from_em1.py, same unit identity used
throughout this course's geothermics activities.)
"""

import numpy as np

ADIABAT_T0 = 1300.0   # deg C, mantle adiabat intercept
ADIABAT_GRAD = 0.3    # deg C / km, mantle adiabat gradient

Q0 = 60.0   # mW/m^2, surface heat flow
T0 = 15.0    # deg C, surface temperature
MOHO = 40.0 # km to Moho, for reference in the worksheet table

# Base-case layer structure, as used in the activity.
LAYERS_BASE = [
    dict(name="Upper crust",       z0=0.0,  z1=15.0,  k=3.0, A=1.5),
    dict(name="Lower crust",       z0=15.0, z1=40.0,  k=2.1, A=0.2),
    dict(name="Mantle lithosphere",z0=MOHO, z1=80.0, k=3.5, A=0.02),
    dict(name="Mantle lithosphere",z0=80, z1=100.0, k=3.5, A=0.02),
    dict(name="Mantle lithosphere",z0=100, z1=200.0, k=3.5, A=0.02),
]


def solve_layer(z0, k, A, q_in, T_in, z1):
    """
    Exact solution within one layer of constant k, A, from z0 to z1,
    given q and T already known at z0. Returns (q_out, T_out) at z1.

    q(z) = q_in - A*(z-z0)                          [linear]
    T(z) = T_in + (q_in*(z-z0) - 0.5*A*(z-z0)**2)/k  [quadratic, exact]
    """
    dz = z1 - z0
    q_out = q_in - A * dz
    T_out = T_in + (q_in * dz - 0.5 * A * dz**2) / k
    return q_out, T_out


def q_of_z(z, z0, A, q_in):
    return q_in - A * (z - z0)


def T_of_z(z, z0, k, A, q_in, T_in):
    dz = z - z0
    return T_in + (q_in * dz - 0.5 * A * dz**2) / k


def build_column(layers, q0=Q0, T0=T0):
    """
    March through a list of layers (each a dict with z0,z1,k,A), and
    return a list of the same dicts augmented with q_in,T_in,q_out,T_out
    at each layer's top and base.
    """
    out = []
    q_prev, T_prev = q0, T0
    for layer in layers:
        q_out, T_out = solve_layer(layer["z0"], layer["k"], layer["A"],
                                    q_prev, T_prev, layer["z1"])
        rec = dict(layer)
        rec.update(q_in=q_prev, T_in=T_prev, q_out=q_out, T_out=T_out)
        out.append(rec)
        q_prev, T_prev = q_out, T_out
    return out


def find_adiabat_crossing(column):
    """
    Search the built column layer by layer for where T(z) meets
    ADIABAT_T0 + ADIABAT_GRAD*z, using the exact quadratic form within
    each layer (linear if A=0). Returns (z_cross, T_cross) for the
    first crossing found, or None.
    """
    for rec in column:
        z0, k, A = rec["z0"], rec["k"], rec["A"]
        q_in, T_in = rec["q_in"], rec["T_in"]
        # T(z0+x) - adiabat(z0+x) = 0, solve for x = z - z0
        a = -0.5 * A
        b = q_in - ADIABAT_GRAD * k
        c = k * T_in - k * ADIABAT_T0 - ADIABAT_GRAD * k * z0
        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                continue
            roots = [-c / b]
        else:
            disc = b**2 - 4 * a * c
            if disc < 0:
                continue
            roots = [(-b + np.sqrt(disc)) / (2 * a),
                     (-b - np.sqrt(disc)) / (2 * a)]
        valid = [x for x in roots if 0 <= x <= (rec["z1"] - z0)]
        if valid:
            x = min(valid)
            z_cross = z0 + x
            T_cross = T_of_z(z_cross, z0, k, A, q_in, T_in)
            return z_cross, T_cross
    return None


def fine_table(layers, points, q0=Q0, T0=T0, label=""):
    """
    Reproduce the guided worksheet table: q and T at a specific list
    of depths (which must include every layer boundary you want a
    row for), stepping through `layers` and re-solving at each
    requested point.
    """
    print(f"\n--- {label} ---")
    print(f"{'z':>6} {'layer':<16} {'dz':>5} {'q(z)':>8} {'T(z)':>9}")
    q_prev, T_prev, z_prev = q0, T0, 0.0
    li = 0
    print(f"{0.0:6.1f} {'--':<16} {'--':>5} {q_prev:8.2f} {T_prev:9.2f}")
    for z in points:
        if z == 0.0:
            continue
        # find which layer this point falls in (by its upper boundary)
        while layers[li]["z1"] < z - 1e-9:
            li += 1
        layer = layers[li]
        dz = z - z_prev
        q_new, T_new = solve_layer(z_prev, layer["k"], layer["A"],
                                    q_prev, T_prev, z)
        print(f"{z:6.1f} {layer['name']:<16} {dz:5.1f} {q_new:8.2f} {T_new:9.2f}")
        q_prev, T_prev, z_prev = q_new, T_new, z
    return q_prev, T_prev


def full_geotherm_table(layers, moho, z_max=120.0, crust_step=5.0,
                         mantle_step=10.0, q0=Q0, T0=T0, label=""):
    """
    Full geotherm table for plotting/checking: every `crust_step` km
    from the surface to `moho`, then every `mantle_step` km from there
    to `z_max`. Reuses `fine_table`'s point-by-point solver, so results
    are exact within each constant-(k,A) layer.
    """
    crust_points = np.arange(crust_step, moho + 1e-6, crust_step)
    mantle_points = np.arange(moho + mantle_step, z_max + 1e-6, mantle_step)
    points = list(crust_points) + list(mantle_points)
    return fine_table(layers, points, q0=q0, T0=T0, label=label)


def full_column_summary(layers, label=""):
    col = build_column(layers)
    cross = find_adiabat_crossing(col)
    print(f"\n=== {label} ===")
    for rec in col:
        print(f"  {rec['name']:<20} z={rec['z0']:>5.1f}-{rec['z1']:<5.1f} "
              f"k={rec['k']:.2f} A={rec['A']:.3f}  "
              f"q: {rec['q_in']:.2f} -> {rec['q_out']:.2f} mW/m2   "
              f"T: {rec['T_in']:.1f} -> {rec['T_out']:.1f} C")
    if cross:
        print(f"  Adiabat crossing: z={cross[0]:.1f} km, T={cross[1]:.1f} C")
    else:
        print("  No adiabat crossing found within the modelled column.")
    return col, cross


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Base case: guided worksheet table (Questions 2-3)
    # ------------------------------------------------------------------
    fine_table(LAYERS_BASE, [5, 10, 15, MOHO], label="Base case (worksheet table)")
    full_column_summary(LAYERS_BASE, label="Base case (full column)")

    # ------------------------------------------------------------------
    # Full geotherm to 120 km (Question 4 plot check): 5 km steps in
    # the crust, 10 km steps in the mantle.
    # ------------------------------------------------------------------
    full_geotherm_table(LAYERS_BASE, MOHO, z_max=120.0,
                         label="Full geotherm to 120 km (5 km crust / 10 km mantle)")

    # ------------------------------------------------------------------
    # k perturbation (Question 5): k1 = 1.5 instead of 3.0, A unchanged.
    # q(z) should come out identical to the base case.
    # ------------------------------------------------------------------
    layers_k = [dict(l) for l in LAYERS_BASE]
    layers_k[0] = dict(layers_k[0], k=1.5)
    fine_table(layers_k, [5, 10, 15], label="k1=1.5 (worksheet table)")
    full_column_summary(layers_k, label="k1=1.5 (full column)")

    # ------------------------------------------------------------------
    # A perturbation (Question 6): A1 = 3.0 instead of 1.5, k unchanged.
    # ------------------------------------------------------------------
    layers_A = [dict(l) for l in LAYERS_BASE]
    layers_A[0] = dict(layers_A[0], A=3.0)
    fine_table(layers_A, [5, 10, 15], label="A1=3.0 (worksheet table)")
    full_column_summary(layers_A, label="A1=3.0 (full column)")
