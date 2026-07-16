"""
antarctica_profile.py
----------------------
Topography (ETOPO 2022 ice surface) and crustal thickness (ECM1) along a
great-circle profile across Antarctica and the South Pole.

Usage
-----
    python week_5_isostasy/workshop/antarctica_profile.py
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

_HERE   = Path(__file__).resolve().parent
_COURSE = _HERE.parent.parent
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from src.utils.geo_profile import sample_profile, load_crustal_thickness

# ── Paths ────────────────────────────────────────────────────────────────
DATAPATH     = _COURSE / "data"
ETOPO_PATH   = DATAPATH / "topography/ETOPO_2022_v1_30s_N90W180_surface.nc"
ECM1_PATH    = DATAPATH / "crustal_thickness/ECM1.txt"
ECM1_CACHE   = DATAPATH / "crustal_thickness/ECM1_Hc_grid.nc"
OUTPUT_PATH  = _HERE / "antarctica_profile.eps"

# ── Profile endpoints ────────────────────────────────────────────────────
LON0, LAT0 = -130.0, -72.0
LON1, LAT1 =   50.0, -65.0
N_POINTS   = 500

# ── Sample topography ───────────────────────────────────────────────────
print("Sampling ETOPO topography …")
dist_km, lons, lats, elevation = sample_profile(
    ETOPO_PATH, "z", LON0, LAT0, LON1, LAT1, n=N_POINTS, method="linear"
)

# ── Sample crustal thickness ────────────────────────────────────────────
print("Building crustal thickness grid …")
if ECM1_CACHE.exists():
    crust_grid = ECM1_CACHE
    crust_var  = "Hc"
else:
    crust_grid = load_crustal_thickness(ECM1_PATH, variable="Hc", cache_path=ECM1_CACHE)
    crust_var  = "Hc"

print("Sampling crustal thickness …")
_, _, _, thickness = sample_profile(
    crust_grid, crust_var, LON0, LAT0, LON1, LAT1, n=N_POINTS, method="linear"
)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, (ax_topo, ax_crust) = plt.subplots(
    2, 1, figsize=(6.5, 5.5), sharex=True,
    gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.08},
)

ax_crust.plot(dist_km, -elevation/1000, color="k", lw=1)
ax_topo.axhline(0, color="0.6", lw=0.5, zorder=0)
ax_topo.set_ylabel("Elevation (m)")

ax_crust.plot(dist_km, thickness-elevation/1000, color="firebrick", lw=1.2)
ax_crust.invert_yaxis()
ax_crust.set_ylabel("Crustal thickness (km)")
ax_crust.set_xlabel("Distance along profile (km)")

i_pole = int(np.argmin(np.abs(lats + 90.0)))
for ax in (ax_topo, ax_crust):
    ax.axvline(dist_km[i_pole], color="0.5", lw=0.7, ls="--", zorder=0)
ax_topo.text(dist_km[i_pole], ax_topo.get_ylim()[0], "South Pole",
             ha="center", va="bottom", fontsize=8, color="0.4")

ax_topo.set_title(
    f"Profile: ({LAT0:.1f}°, {LON0:.1f}°) → "
    f"({LAT1:.1f}°, {LON1:.1f}°)", fontsize=10
)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, format="eps", bbox_inches="tight")
print(f"Saved → {OUTPUT_PATH}")
