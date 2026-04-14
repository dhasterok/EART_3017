import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.physics.magnetics.mag2d_model import compute_bx_bz, compute_bt


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def polygon_circle(xc, zc, r, n=64):
    """
    Approximate a circle with an n-sided polygon.
    All units in km.
    """
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    return [(xc + r*np.cos(t), zc + r*np.sin(t)) for t in th]


def polygon_rectangle(xc, z_top, width, height):
    """
    Rectangle centered at xc, with top at z_top.
    Width > height recommended.
    All units in km.
    """
    w = width / 2.0
    z_bot = z_top + height
    return [
        (xc - w, z_top),
        (xc + w, z_top),
        (xc + w, z_bot),
        (xc - w, z_bot),
    ]


# ---------------------------------------------------------------------
# Physical parameters (clean validation case)
# ---------------------------------------------------------------------

F_nT = 50000.0        # Earth's field magnitude (nT)
IE_deg = 90.0         # vertical field
susceptibility = 0.01
remanence_Am = 0.0    # OFF
remanence_inc = 0.0

# Observation profile
x_obs = np.linspace(-2.0, 2.0, 401)   # km


# ---------------------------------------------------------------------
# Cylinder model
# ---------------------------------------------------------------------

cyl_vertices = polygon_circle(
    xc=0.0,
    zc=0.30,      # 300 m depth to centre
    r=0.10,       # 100 m radius
    n=64
)

Bx_cyl, Bz_cyl = compute_bx_bz(
    x_obs, cyl_vertices,
    susceptibility, F_nT, IE_deg,
    remanence_Am, remanence_inc
)

Bt_cyl = compute_bt(
    x_obs, cyl_vertices,
    susceptibility, F_nT, IE_deg,
    remanence_Am, remanence_inc
)


# ---------------------------------------------------------------------
# Rectangle model (wide slab)
# ---------------------------------------------------------------------

rect_vertices = polygon_rectangle(
    xc=0.0,
    z_top=0.20,   # 200 m depth to top
    width=2.0,    # 2 km wide
    height=0.20   # 200 m thick
)

Bx_rec, Bz_rec = compute_bx_bz(
    x_obs, rect_vertices,
    susceptibility, F_nT, IE_deg,
    remanence_Am, remanence_inc
)

Bt_rec = compute_bt(
    x_obs, rect_vertices,
    susceptibility, F_nT, IE_deg,
    remanence_Am, remanence_inc
)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# --- Cylinder ---
axs[0].plot(x_obs, Bx_cyl, label="Bx")
axs[0].plot(x_obs, Bz_cyl, label="Bz")
axs[0].plot(x_obs, Bt_cyl, label="ΔT", linewidth=2)
axs[0].set_title("Buried Infinite Cylinder (vertical field)")
axs[0].set_ylabel("Magnetic field (nT)")
axs[0].legend()
axs[0].grid(True)

# --- Rectangle ---
axs[1].plot(x_obs, Bx_rec, label="Bx")
axs[1].plot(x_obs, Bz_rec, label="Bz")
axs[1].plot(x_obs, Bt_rec, label="ΔT", linewidth=2)
axs[1].set_title("Buried Wide Rectangle (vertical field)")
axs[1].set_xlabel("x (km)")
axs[1].set_ylabel("Magnetic field (nT)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()