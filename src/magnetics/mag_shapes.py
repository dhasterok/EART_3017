"""
mag_shapes.py  –  demo of mag_volume for six geometric bodies
=============================================================
Python mirror of mag_shapes.m.  Run from this directory:

    python mag_shapes.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mag_volume import mag_volume

# Observation grid
x = np.arange(-10, 10.2, 0.2)
y = np.arange(-10, 10.2, 0.2)

# Magnetisation parameters
I  = 30.0    # body inclination  (degrees)
D  =  0.0    # body declination  (degrees)

# Earth-field parameters
IE = 45.0    # Earth-field inclination  (degrees)
DE = 20.0    # Earth-field declination  (degrees)

zc = 6.0     # depth to centre of body

# ------------------------------------------------------------------
# Cone  (apex at zt=2, base diameter 4, height 5)
# ------------------------------------------------------------------
print("--- cone ---")
zt = 2.0
_, Bt = mag_volume(x, y, 'cone', zt, [4, 5], I, D, IE, DE)

# ------------------------------------------------------------------
# Rectangle  (30 × 30 × 2)
# ------------------------------------------------------------------
print("--- rect ---")
mag_volume(x, y, 'rect', zc, [30, 30, 2], I, D)

# ------------------------------------------------------------------
# Cylinder  (diameter 3, height 5)
# ------------------------------------------------------------------
print("--- cyl ---")
mag_volume(x, y, 'cyl', zc, [3, 5], I, D)

# ------------------------------------------------------------------
# Sphere  (diameter 4) — exact analytical formula by default
# ------------------------------------------------------------------
print("--- sphere ---")
Bc, Bt = mag_volume(x, y, 'sphere', zc, [4], I, D, IE, DE)

# ------------------------------------------------------------------
# Sheet  (60 × 20 × 1, dip 60°)
# ------------------------------------------------------------------
print("--- sheet ---")
mag_volume(x, y, 'sheet', zc, [60, 20, 1, 60], I, D)

# ------------------------------------------------------------------
# Plane  (20 × 20, height 10, dip 60°, thickness 1)
# ------------------------------------------------------------------
print("--- plane ---")
mag_volume(x, y, 'plane', zc, [20, 20, 10, 60, 1], I, D, IE, DE)

plt.show()
