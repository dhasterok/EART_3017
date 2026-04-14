import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

mu0 = 4*np.pi*1e-7
M   = 1.0           # A/m (arbitrary scaling)
a   = 0.1           # radius (km)
z0  = 0.3           # depth to centre (km)

x = np.linspace(-2.0, 2.0, 801)   # profile (km)

# Strike coordinate for finite cylinder
ny = 2001
y_max_list = [0.2, 1.0, 5.0, 20.0]   # half-lengths (km)


# ---------------------------------------------------------------------
# Vertical field of a vertical point dipole
# ---------------------------------------------------------------------

def Bz_dipole(x, y, z, m):
    r2 = x*x + y*y + z*z
    r5 = r2**2.5
    return mu0 * m * (2*z*z - x*x - y*y) / r5


# ---------------------------------------------------------------------
# Compute Bz by integrating dipoles along strike
# ---------------------------------------------------------------------

def Bz_finite_cylinder(x, z0, y_max):
    y = np.linspace(-y_max, y_max, ny)
    dy = y[1] - y[0]
    Bz = np.zeros_like(x)

    for i, xi in enumerate(x):
        rBz = Bz_dipole(xi, y, z0, M * np.pi * a*a)
        Bz[i] = np.sum(rBz) * dy

    return Bz

def Bz_infinite_cylinder(x, z0):
    r2 = x**2 + z0**2
    return (mu0 * M * a**2 * ( 2 * np.pi)) *(z0*z0 - x*x) / (r2*r2)


# ---------------------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------------------

plt.figure(figsize=(8,6))

for y_max in y_max_list:
    Bz = Bz_finite_cylinder(x, z0, y_max)
    plt.plot(x, Bz, label=f"L = {2*y_max:.1f} km")

plt.plot(x,Bz_infinite_cylinder(x, z0), 'k--', label="Infinite cylinder")

plt.axhline(0, color='k', linewidth=0.8)
plt.xlabel("x (km)")
plt.ylabel("Bz (arbitrary units)")
plt.title("Vertical magnetic field: finite vs infinite cylinder")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()