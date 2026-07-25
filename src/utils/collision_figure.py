"""
Collisional activity -- Phase 1: immediate post-collision geotherms,
via thrust-stack thickening.

This represents the EARLY phase of orogenic thermal history: a 40 km
pre-collision crust (typical average continental thickness) has its
upper 20 km thrust over the ENTIRE original 40 km column, giving a
60 km total post-collision thickness. Immediately after thrust
stacking, the geotherm is a SAWTOOTH (the thrust sheet and the buried
footwall each retain their own original temperature profile,
unchanged, just juxtaposed) -- colder than the eventual equilibrium
at every depth, because the thickened, more-radiogenic column hasn't
yet built up the extra heat it will eventually trap. Given long
enough, it warms toward a NEW, HIGHER equilibrium surface heat flow.

This is deliberately distinct from Fischer et al.'s model (see
fischer_thermal_model.py), which represents a separate, LATER-stage
process: once a thickened root has equilibrated (or otherwise reached
a locally hot state via magmatism, shear heating, etc.), erosion and
continued relaxation cool it from a high q0 toward a lower qf over
much longer timescales. The two models are not competing descriptions
of the same phase -- they describe different parts of the same
orogen's history, and most of the real R-vs-age data (mostly older
than 200 Ma) falls in the regime the second (cooling) model addresses,
while this sawtooth/warming model addresses the earliest, most poorly
sampled part of the record.

Reuses solve_layer()/T_of_z() from geotherm_8d.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.geotherm_activity import T_of_z, build_column, Q0, T0

FIGDIR = Path(__file__).resolve().parent.parent.parent
FIGDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Pre-collision layer structure: 40 km crust (typical average thickness),
# same k/A per lithology as Activity 8.D's base case -- just a thicker
# lower crust to reach 40 km instead of 8.D's 35 km.
# ----------------------------------------------------------------------
PRECOLLISION_LAYERS = [
    dict(name="Upper crust", z0=0.0, z1=15.0, k=3.0, A=1.5),
    dict(name="Lower crust", z0=15.0, z1=40.0, k=2.1, A=0.2),
    dict(name="Mantle lithosphere", z0=40.0, z1=500.0, k=3.5, A=0.02),
]
FAULT_DEPTH = 20.0  # km, thickness of the thrust sheet duplicated


def pre_collision_column():
    return build_column(PRECOLLISION_LAYERS, q0=Q0, T0=T0)


def pre_collision_T(z):
    col = pre_collision_column()
    for rec in col:
        if rec["z0"] <= z <= rec["z1"]:
            return T_of_z(z, rec["z0"], rec["k"], rec["A"], rec["q_in"], rec["T_in"])
    rec = col[-1]
    return T_of_z(z, rec["z0"], rec["k"], rec["A"], rec["q_in"], rec["T_in"])


def slice_layers(layers, z_start, z_end):
    """Portion of `layers` within [z_start, z_end], clipping any layer
    that straddles a boundary. Depths remain in the ORIGINAL coordinate."""
    out = []
    for l in layers:
        lo, hi = max(l["z0"], z_start), min(l["z1"], z_end)
        if lo < hi:
            out.append(dict(name=l["name"], z0=lo, z1=hi, k=l["k"], A=l["A"]))
    return out


def shift_layers(layers, shift):
    return [dict(name=l["name"], z0=l["z0"] + shift, z1=l["z1"] + shift,
                  k=l["k"], A=l["A"]) for l in layers]


def thickened_crustal_layers():
    """
    Thrust sheet: the original 0-FAULT_DEPTH slice, unshifted (sits on
    top, unchanged). Footwall: the ENTIRE original crustal column,
    shifted down by FAULT_DEPTH (buried, not truncated). Because
    FAULT_DEPTH != the upper crust's own thickness, the thrust sheet
    spans both original layers -- slice_layers() handles that split
    automatically rather than assuming a single clean cut.
    """
    crust_only = [l for l in PRECOLLISION_LAYERS if l["name"] != "Mantle lithosphere"]
    old_crust_base = crust_only[-1]["z1"]  # 40 km

    thrust_sheet = slice_layers(crust_only, 0.0, FAULT_DEPTH)
    footwall = shift_layers(slice_layers(crust_only, 0.0, old_crust_base), FAULT_DEPTH)

    new_moho = FAULT_DEPTH + old_crust_base  # 20 + 40 = 60 km
    return thrust_sheet + footwall, new_moho


def initial_postthickening_T(z):
    """
    T(z) immediately after thrust stacking: the thrust sheet (0 to
    FAULT_DEPTH) retains its own original T(z), unchanged; the footwall
    (the entire original 40 km column, now buried beneath the thrust
    sheet) also retains its own original T(z), just offset downward by
    FAULT_DEPTH. This produces a sawtooth at z=FAULT_DEPTH: temperature
    drops from whatever the thrust sheet reached at its base straight
    back down to the footwall's own surface temperature T0, then rises
    again following the full original 40 km profile.
    """
    _, new_moho = thickened_crustal_layers()
    if z <= FAULT_DEPTH:
        return pre_collision_T(z)                  # thrust sheet: unchanged
    elif z <= new_moho:
        return pre_collision_T(z - FAULT_DEPTH)      # footwall: full original profile, offset
    else:
        d = z - new_moho
        return pre_collision_T(40.0 + d)              # undisturbed mantle lithosphere


def new_equilibrium_column():
    """
    Solve the thickened crust bottom-up from a fixed basal heat flow,
    through the new (four-sub-layer) crustal structure, giving a new,
    higher surface heat flow.
    """
    new_crust_layers, new_moho = thickened_crustal_layers()
    pre_col = pre_collision_column()
    q_base_original_moho = pre_col[-2]["q_out"]  # q at the ORIGINAL (40 km) Moho

    # Deliberate, documented tuning choice (not derived) -- adjust if
    # you have a firmer basis for how much the basal flow should drop
    # with the extra mantle lithosphere thickness in between. Tuned to
    # land the new-Moho equilibrium temperature in the 1000-1100 C
    # range judged physically reasonable.
    Q_BASE_FRACTION = 0.85
    q_base = q_base_original_moho * Q_BASE_FRACTION

    q, T_rel = q_base, 0.0
    records = []
    for layer in reversed(new_crust_layers):
        z0, z1, k, A = layer["z0"], layer["z1"], layer["k"], layer["A"]
        dz = z1 - z0
        q_new = q + A * dz
        dT = (q + q_new) / 2 / k * dz
        T_rel_new = T_rel - dT   # going upward (shallower), T decreases
        records.append(dict(name=layer["name"], z0=z0, z1=z1, k=k, A=A,
                             q_out=q, q_in=q_new, T_out=T_rel, T_in=T_rel_new))
        q, T_rel = q_new, T_rel_new
    records = records[::-1]

    shift = T0 - records[0]["T_in"]
    for rec in records:
        rec["T_in"] += shift
        rec["T_out"] += shift
    return records, records[0]["q_in"]


def T_new_equilibrium(z):
    records, _ = new_equilibrium_column()
    for rec in records:
        if rec["z0"] <= z <= rec["z1"]:
            return T_of_z(z, rec["z0"], rec["k"], rec["A"], rec["q_in"], rec["T_in"])
    rec = records[-1]
    return T_of_z(z, rec["z0"], rec["k"], rec["A"], rec["q_in"], rec["T_in"])


def make_figure():
    _, new_moho = thickened_crustal_layers()

    z = np.linspace(0, new_moho + 20, 400)
    eps = 1e-6
    z = np.sort(np.concatenate([z, [FAULT_DEPTH - eps, FAULT_DEPTH + eps]]))

    T_pre = [pre_collision_T(zz) for zz in z]
    T_init = [initial_postthickening_T(zz) for zz in z]
    T_eq = [T_new_equilibrium(zz) for zz in z]

    records, q0_new = new_equilibrium_column()
    print(f"New equilibrium surface heat flow: q0 = {q0_new:.1f} mW/m2 "
          f"(pre-collision q0 = {Q0:.1f} mW/m2)")
    print(f"New Moho depth: {new_moho:.0f} km")
    print(f"New equilibrium T at new Moho: {T_new_equilibrium(new_moho):.0f} C")

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.plot(T_pre, z, color='0.6', lw=1.5, ls=':', label='Pre-collision equilibrium (40 km crust)')
    ax.plot(T_init, z, color='tab:blue', lw=2, label='Initial (just after thickening)')
    ax.plot(T_eq, z, color='tab:red', lw=2, label='New equilibrium')
    ax.axhline(new_moho, color='0.7', lw=0.8, ls='--')
    ax.text(5, new_moho - 2, 'new Moho (60 km)', fontsize=8, color='0.5')
    ax.axhline(FAULT_DEPTH, color='0.85', lw=0.6, ls=':')
    ax.text(5, FAULT_DEPTH - 2, 'thrust fault (20 km)', fontsize=7, color='0.6')
    ax.set_xlabel('Temperature ($^\\circ$C)')
    ax.set_ylabel('Depth (km)')
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'collisional_geotherms.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'collisional_geotherms.png', dpi=200, bbox_inches='tight')
    print("Wrote collisional_geotherms.{pdf,png}")


if __name__ == '__main__':
    make_figure()