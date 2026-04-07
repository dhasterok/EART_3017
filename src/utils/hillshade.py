import numpy as np

# ── Hillshade ─────────────────────────────────────────────────────────────
def hillshade(z, dx_m=2000.0, dy_m=2000.0, azimuth=315.0, altitude=45.0):
    """Lambertian hillshade from a DEM on a regular grid.

    Parameters
    ----------
    z        : 2-D elevation array (m)
    dx_m,dy_m: grid spacing in metres (E-W, N-S)
    azimuth  : solar azimuth (degrees clockwise from north), default 315 (NW)
    altitude : solar elevation above horizon (degrees), default 45
    """
    az  = np.radians(360 - azimuth + 90)
    alt = np.radians(altitude)
    dz_dx = np.gradient(z, dx_m, axis=1)
    dz_dy = np.gradient(z, dy_m, axis=0)
    slope  = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dy, dz_dx)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)