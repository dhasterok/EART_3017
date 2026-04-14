"""
ss_thermal.py

Steady-state thermal structure and thermal isostasy utilities.
Direct translation of MATLAB code to Python/NumPy.

All depths in km
Temperatures in deg C
Heat flow in mW/m^2
Heat production in µW/m^3
"""

import numpy as np


# ============================================================
# Geotherm calculation (classic layered conductive geotherm)
# ============================================================
def geotherms(z, zi, k, A, q0, T0):
    """
    Computes temperatures along a geotherm.

    Parameters
    ----------
    z : array_like
        Depth vector (km)
    zi : array_like
        Interface depths (km)
    k : array_like
        Thermal conductivity per layer (W/m/K)
    A : array_like
        Heat production per layer (µW/m^3)
    q0 : float
        Surface heat flow (mW/m^2)
    T0 : float
        Surface temperature (deg C)

    Returns
    -------
    T : ndarray
        Temperature profile (deg C)
    """

    z = np.asarray(z, dtype=float)
    zi = np.asarray(zi, dtype=float)
    k = np.asarray(k, dtype=float)
    A = np.asarray(A, dtype=float)

    T = np.zeros_like(z)

    # Heat flow at interfaces
    q = np.zeros(len(A))
    q[0] = q0
    for n in range(1, len(A)):
        q[n] = q[n - 1] - A[n - 1] * (zi[n] - zi[n - 1])

    Tp = 1300.0  # mantle potential temperature

    n = 0
    T[0] = T0
    Tn = T0

    for j in range(1, len(z)):
        if z[j] > zi[n + 1]:
            n += 1
            Tn = T[j - 1]

        dz = z[j] - zi[n]
        T[j] = (
            Tn
            + q[n] * dz / k[n]
            - 0.5 * A[n] * dz**2 / k[n]
        )

        if T[j] > Tp:
            T[j:] = Tp
            break

    return T


# ============================================================
# Thermal elevation (integral formulation)
# ============================================================
def elevation(z, zi, T0, T, k, A, q0, Tref, kref, Aref, qref, alpha):
    """
    Computes thermal elevation relative to a reference geotherm.
    """

    tau = Tintegral(z, T, zi, k, A, q0, T0)
    tau_ref = Tintegral(z, Tref, zi, kref, Aref, qref, T0)

    elev = alpha * np.sum(tau - tau_ref)
    return elev


# ============================================================
# Integral of temperature within layers
# ============================================================
def Tintegral(z, T, zi, k, A, q0, T0):
    """
    Computes integral of T(z) dz within each stratigraphic layer.
    """

    z = np.asarray(z)
    T = np.asarray(T)
    zi = np.asarray(zi)
    k = np.asarray(k)
    A = np.asarray(A)

    Tp = 1300.0

    # Index of temperature reaching adiabat
    ind = np.where(T >= Tp)[0]
    if len(ind) == 0:
        ind = len(z) - 1
    else:
        ind = ind[0]

    zi = np.concatenate((zi, [z[-1]]))
    zi[-2] = z[ind]

    # Heat flow at layer tops
    qi = np.zeros(len(A))
    qi[0] = q0
    for n in range(1, len(A)):
        qi[n] = qi[n - 1] - A[n - 1] * (zi[n] - zi[n - 1])

    Ti = np.zeros(len(zi))
    Ti[0] = T0

    tau = np.zeros(len(zi) - 1)

    for n in range(len(zi) - 2):
        dz = zi[n + 1] - zi[n]

        Ti[n + 1] = (
            Ti[n]
            + qi[n] * dz / k[n]
            - A[n] * dz**2 / (2 * k[n])
        )

        tau[n] = (
            Ti[n] * dz
            + qi[n] * dz**2 / (2 * k[n])
            - A[n] * dz**3 / (6 * k[n])
        )

    # Add adiabatic mantle contribution
    tau[-1] += Tp * (zi[-1] - zi[-2])

    return tau


# ============================================================
# Compute steady-state conductive geotherm
# ============================================================
def compute_temperature(z, T0, q0, ztop, k, A):
    """
    Steady-state layered conductive geotherm with adiabatic truncation.
    """

    z = np.asarray(z, dtype=float)
    ztop = np.asarray(ztop, dtype=float)
    k = np.asarray(k, dtype=float)
    A = np.asarray(A, dtype=float)

    ztop = np.concatenate((ztop, [z.max()]))

    Tp = 1300.0       # deg C
    Ga = 0.3          # deg/km

    T = np.zeros_like(z)
    T[0] = T0

    istart = 1

    for i in range(len(ztop) - 1):
        iend = np.where(z <= ztop[i + 1])[0][-1]

        dz = z[istart:iend + 1] - ztop[i]
        T[istart:iend + 1] = T0 + (q0 * dz - 0.5 * A[i] * dz**2) / k[i]

        if z[iend] == ztop[i + 1]:
            T0 = T[iend]
            q0 = q0 - A[i] * dz[-1]
        else:
            dz_layer = ztop[i + 1] - ztop[i]
            q0 = q0 - A[i] * dz_layer
            T0 = T0 + (q0 * dz_layer - 0.5 * A[i] * dz_layer**2) / k[i]

        istart = iend + 1

    # Enforce adiabat
    for i in range(len(z)):
        Ta = Tp + Ga * z[i]
        if T[i] > Ta:
            T[i] = Ta

    return T.reshape(-1)


# ============================================================
# Simple elevation (rectangular integration)
# ============================================================
def compute_elevation(dz, T, Tref, alpha):
    """
    Computes thermal elevation by rectangular integration.
    """

    T = np.asarray(T)
    Tref = np.asarray(Tref)

    elev = alpha * np.sum(T - Tref) * dz
    return elev