def sphangle_fast(lon, lat, lon0, lat0, h=0):
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    lon0 = np.deg2rad(lon0)
    lat0 = np.deg2rad(lat0)

    delta = np.arccos(
        np.sin(lat0)*np.sin(lat) +
        np.cos(lat0)*np.cos(lat)*np.cos(lon - lon0)
    )

    delta = np.degrees(delta)

    # Same hemisphere logic as before
    if h != 0:
        sgn = np.ones_like(delta)

        if h == 1:
            change = lon < lon0
        elif h == 2:
            change = lon > lon0
        elif h == 3:
            change = lat < lat0
        elif h == 4:
            change = lat > lat0
        else:
            return delta

        sgn[change] = -1
        delta *= sgn

    return delta