"""
mag_volume.py  –  Magnetic anomaly of simple geometric bodies
=============================================================

Computes the total-field magnetic anomaly for six elementary shapes using
vectorised NumPy broadcasting.  A chunked loop over dipoles keeps peak
memory bounded even for large bodies.

For the **sphere** an exact analytical formula is available (default):
a uniformly magnetised sphere produces an external field identical to that
of a single magnetic dipole located at its centre, so no approximation is
needed.

Shapes and their *dim* arguments
---------------------------------
    rect   – rectangular prism  : dim = [xwidth, ywidth, zheight]
    cyl    – vertical cylinder  : dim = [diameter, zheight]
    sphere – sphere             : dim = [diameter]
    cone   – vertical cone      : dim = [base_diameter, height]
                                  (zc = depth to apex / top)
    sheet  – one-sided layer    : dim = [2*xwidth, ywidth, zthickness, dip_deg]
    plane  – finite dipping slab: dim = [xwidth, ywidth, height, dip_deg, thickness]

Usage
-----
    import numpy as np
    from mag_volume import mag_volume

    x  = np.arange(-10, 10.2, 0.2)
    y  = np.arange(-10, 10.2, 0.2)
    Bc, Bt = mag_volume(x, y, 'sphere', zc=6, dim=[4], I=30, D=0, IE=45, DE=20)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Physical constant
MU0 = 4.0 * np.pi * 1e-7   # permeability of free space (T·m/A)

# Default sub-dipole grid spacing (matches MATLAB default of 0.2)
_DEFAULT_SPACING = 0.2

# Dipoles processed per chunk; keeps M×chunk arrays ~< 400 MB
_DEFAULT_CHUNK = 1000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dircos(I_deg: float, D_deg: float, A_deg: float = 0.0):
    """Direction cosines for inclination I, declination D, azimuth A."""
    I = np.radians(I_deg)
    D = np.radians(D_deg)
    A = np.radians(A_deg)
    return np.cos(I)*np.cos(D - A), np.cos(I)*np.sin(D - A), np.sin(I)


def _dipole_field(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
    I: float, D: float,
    chunk: int = _DEFAULT_CHUNK,
) -> np.ndarray:
    """
    Total magnetic field from N unit-moment dipoles at M observation points.

    Parameters
    ----------
    x, y, z : (M,) – observation coordinates
    xd, yd, zd : (N,) – dipole centre coordinates
    I, D : magnetisation inclination / declination (degrees)
    chunk : number of dipoles processed per iteration (memory control)

    Returns
    -------
    Bc : (M, 3) array  [Bx, By, Bz]  in relative units (m = 1 per dipole)
    """
    mx, my, mz = _dircos(I, D)

    x  = np.asarray(x ).ravel()[:, None]   # M×1
    y  = np.asarray(y ).ravel()[:, None]
    z  = np.asarray(z ).ravel()[:, None]
    xd = np.asarray(xd).ravel()
    yd = np.asarray(yd).ravel()
    zd = np.asarray(zd).ravel()

    M_obs = x.size
    N     = xd.size

    Bx = np.zeros(M_obs)
    By = np.zeros(M_obs)
    Bz = np.zeros(M_obs)

    for k in range(0, N, chunk):
        sl = slice(k, min(k + chunk, N))

        rx = x - xd[sl]           # M×chunk  (broadcast)
        ry = y - yd[sl]
        rz = z - zd[sl]

        r2  = rx**2 + ry**2 + rz**2
        r   = np.sqrt(r2)
        mdr = mx*rx + my*ry + mz*rz
        fac = MU0 / (4.0*np.pi * r**5)

        Bx += np.sum(fac * (3.0*mdr*rx - r2*mx), axis=1)
        By += np.sum(fac * (3.0*mdr*ry - r2*my), axis=1)
        Bz += np.sum(fac * (3.0*mdr*rz - r2*mz), axis=1)

    return np.column_stack([Bx, By, Bz])


def _build_dipoles(
    shape: str,
    zc: float,
    dim: list[float],
    spacing: float = _DEFAULT_SPACING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (xd, yd, zd) flat arrays of dipole positions for *shape*.

    The grid is first constructed on a bounding box then masked to the
    actual shape geometry, matching the approach used in the MATLAB
    mag_volume.m.
    """
    half0 = dim[0] / 2.0

    xd_1d = np.arange(-half0, half0 + spacing/2, spacing)

    if shape in ('rect', 'sheet', 'plane'):
        yd_1d = np.arange(-dim[1]/2,   dim[1]/2   + spacing/2, spacing)
        zd_1d = np.arange(zc - dim[2]/2, zc + dim[2]/2 + spacing/2, spacing)
    elif shape == 'cone':
        yd_1d = np.arange(-half0, half0 + spacing/2, spacing)
        zd_1d = np.arange(zc, zc + dim[1] + spacing/2, spacing)
    else:  # cyl, sphere
        yd_1d = np.arange(-half0, half0 + spacing/2, spacing)
        zd_1d = np.arange(zc - dim[-1]/2, zc + dim[-1]/2 + spacing/2, spacing)

    Xd, Yd, Zd = np.meshgrid(xd_1d, yd_1d, zd_1d, indexing='ij')
    Xd, Yd, Zd = Xd.ravel(), Yd.ravel(), Zd.ravel()

    if shape == 'rect':
        mask = (np.abs(Xd) <= half0) & (np.abs(Yd) <= dim[1]/2)

    elif shape == 'cyl':
        mask = np.sqrt(Xd**2 + Yd**2) <= half0

    elif shape == 'sphere':
        mask = np.sqrt(Xd**2 + Yd**2 + (Zd - zc)**2) <= half0

    elif shape == 'cone':
        # apex at zc (top), base at zc + dim[1]; radius tapers from base to apex
        frac = 1.0 - (Zd - zc) / dim[1]
        mask = (Zd < zc + dim[1]) & (np.sqrt(Xd**2 + Yd**2) <= half0 * frac)

    elif shape == 'sheet':
        dip_rad = np.radians(dim[3])
        mask = Xd >= (Zd - zc) / np.tan(dip_rad)

    elif shape == 'plane':
        dip_rad = np.radians(dim[3])
        half_w  = dim[4] / 2.0
        strike  = (Zd - zc) / np.tan(dip_rad)
        mask = (Xd + half_w >= strike) & (Xd - half_w <= strike)

    else:
        raise ValueError(f"Unknown shape '{shape}'. "
                         "Choose from: rect, cyl, sphere, cone, sheet, plane")

    # Keep only sub-surface dipoles
    mask &= Zd > 0.1
    if not np.any(mask):
        raise ValueError(
            "No dipoles remain after depth/shape filter. "
            "Check that zc and dim place the body below the surface (z > 0)."
        )

    return Xd[mask], Yd[mask], Zd[mask]


# ---------------------------------------------------------------------------
# Analytical sphere formula
# ---------------------------------------------------------------------------

def _sphere_analytic(
    X: np.ndarray, Y: np.ndarray,
    zc: float, R: float,
    I: float, D: float,
) -> np.ndarray:
    """
    Exact field of a uniformly magnetised sphere (radius R, centred at depth zc).

    A sphere of radius R and unit magnetisation is equivalent to a single
    magnetic dipole at its centre with moment m = (4/3) π R³.

    Returns Bc : (M, 3) array [Bx, By, Bz]
    """
    moment = (4.0/3.0) * np.pi * R**3
    Z_obs  = np.zeros(X.size)
    Bc = _dipole_field(
        X.ravel(), Y.ravel(), Z_obs,
        np.array([0.0]), np.array([0.0]), np.array([zc]),
        I, D,
    )
    return Bc * moment


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_result(
    x: np.ndarray, y: np.ndarray,
    xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
    Bz: np.ndarray,
    shape: str,
) -> None:
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f"Shape: {shape}", fontsize=13)

    # Left panel – dipole geometry
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(xd, yd, zd, s=2, alpha=0.35, c=zd, cmap='viridis_r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z (depth)')
    ax1.set_title('Dipole positions')
    ax1.invert_zaxis()

    # Right panel – Bz map
    ax2 = fig.add_subplot(122)
    X, Y = np.meshgrid(x, y)
    vmax = np.percentile(np.abs(Bz), 98)
    cf = ax2.contourf(X, Y, Bz, levels=40, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax)
    plt.colorbar(cf, ax=ax2, label='$B_z$ (rel. units)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Vertical magnetic field $B_z$')
    ax2.set_aspect('equal')

    plt.tight_layout()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mag_volume(
    x: np.ndarray,
    y: np.ndarray,
    shape: str,
    zc: float,
    dim: list[float],
    I: float,
    D: float,
    IE: float = 90.0,
    DE: float = 0.0,
    plot: bool = True,
    spacing: float = _DEFAULT_SPACING,
    chunk: int = _DEFAULT_CHUNK,
    analytical_sphere: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Magnetic anomaly of a geometric body filled with discrete dipoles.

    Parameters
    ----------
    x, y   : 1-D arrays defining the surface observation grid
    shape  : 'rect' | 'cyl' | 'sphere' | 'cone' | 'sheet' | 'plane'
    zc     : depth to body centre (top for cone), positive downward (km or m)
    dim    : shape dimensions — see module docstring for per-shape conventions
    I, D   : inclination / declination of body magnetisation (degrees)
    IE, DE : inclination / declination of Earth's field (degrees)
    plot   : produce a two-panel figure when True (default)
    spacing: sub-dipole grid spacing (default 0.2, matches MATLAB default)
    chunk  : dipoles per vectorisation chunk (controls peak memory)
    analytical_sphere : use exact equivalent-dipole formula for spheres (default True)

    Returns
    -------
    Bc : (ny, nx, 3) ndarray  – field components [Bx, By, Bz] on the grid
    Bt : (ny, nx) ndarray     – total-field anomaly projected onto Earth's field
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X, Y = np.meshgrid(x, y)
    nx, ny = len(x), len(y)
    Z_obs = np.zeros(X.size)

    # ------------------------------------------------------------------
    # Compute field at observation points
    # ------------------------------------------------------------------
    if shape == 'sphere' and analytical_sphere:
        R = dim[0] / 2.0
        Bc_flat = _sphere_analytic(X, Y, zc, R, I, D)
        # For the geometry plot, show one equivalent dipole
        xd = yd = np.array([0.0])
        zd = np.array([zc])
        print(f"sphere: exact analytical formula (equivalent single dipole at depth {zc}).")
    else:
        xd, yd, zd = _build_dipoles(shape, zc, dim, spacing)
        print(f"Number of dipoles: {xd.size}")
        Bc_flat = _dipole_field(X.ravel(), Y.ravel(), Z_obs, xd, yd, zd, I, D, chunk)

    # ------------------------------------------------------------------
    # Total-field projection onto Earth's field direction
    # ------------------------------------------------------------------
    ex, ey, ez = _dircos(IE, DE)
    Bt = (Bc_flat[:, 0]*ex + Bc_flat[:, 1]*ey + Bc_flat[:, 2]*ez).reshape(ny, nx)
    Bc = Bc_flat.reshape(ny, nx, 3)

    if plot:
        _plot_result(x, y, xd, yd, zd, Bc[:, :, 2], shape)

    return Bc, Bt
