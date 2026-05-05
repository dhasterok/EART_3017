"""
borehole_climate.py

Physics functions for borehole palaeoclimate analysis.

Borehole temperature logs record the integrated thermal history of the ground.
Removing the steady-state geothermal gradient (the "reduced temperature") isolates
anomalies caused by past surface temperature changes.  These can then be inverted
for the magnitude and timing of those changes.

Units
-----
Depth              : metres (m)
Temperature        : degrees Celsius (°C)
Gradient           : °C km⁻¹
Thermal diffusivity: m² yr⁻¹  (rock ≈ 31.5 m² yr⁻¹ ≈ 1 × 10⁻⁶ m² s⁻¹)
Time               : years before logging

References
----------
Lachenbruch & Marshall (1986) Changing environment of the Arctic: Thermal
    response of the permafrost.
Beltrami & Mareschal (1991) Recent warming in eastern Canada inferred from
    geothermal measurements.
Carslaw & Jaeger (1959) Conduction of Heat in Solids, 2nd edn.
"""

import glob
from pathlib import Path

import numpy as np
from scipy.special import erfc, erf


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_borehole(filepath):
    """
    Load a single borehole temperature log.

    Accepts whitespace-delimited ASCII files with 2 or 3 columns:
        2 columns: depth (m), temperature (°C)
        3 columns: depth (m), resistance (ignored), temperature (°C)

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    dict with keys: 'name', 'z', 'T', 'filepath'
        or None if the file cannot be parsed.
    """
    filepath = Path(filepath)
    try:
        data = np.loadtxt(filepath)
    except Exception:
        return None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncol = data.shape[1]
    if ncol == 2:
        z, T = data[:, 0], data[:, 1]
    elif ncol == 3:
        z, T = data[:, 0], data[:, 2]
    else:
        return None

    if len(z) == 0:
        return None

    return {
        'name': filepath.stem,
        'z': z.astype(float),
        'T': T.astype(float),
        'filepath': str(filepath),
        # Processing results (populated by process_borehole)
        'ztop': None, 'grad': None, 'sd_grad': None,
        'tsurf': None, 'sd_tsurf': None,
        'curv': None, 'rms': None,
        'ind_curv': None, 'ind_grad': None,
        'Tred': None,
        # Inversion results (populated by inversion functions)
        'dT': None, 'tau': None, 'inv_rms': None,
    }


def load_boreholes(directory, pattern='*.dat'):
    """
    Load all matching borehole files from *directory*.

    Parameters
    ----------
    directory : str or Path
    pattern   : glob pattern (case-insensitive on macOS/Linux)

    Returns
    -------
    list of borehole dicts (skips unreadable files)
    """
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    # Also pick up capitalised extensions (.DAT)
    upper = pattern.replace('*', '*').upper()
    if upper != pattern:
        paths += sorted(directory.glob(upper))
        paths = sorted(set(paths))

    boreholes = []
    for p in paths:
        log = load_borehole(p)
        if log is not None:
            boreholes.append(log)
    return boreholes


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def clip_log(log, start_depth=20.0, min_depth=140.0):
    """
    Remove data shallower than *start_depth* and reject boreholes whose
    maximum depth is less than *min_depth*.

    Returns a new dict (copy with trimmed arrays), or None if rejected.
    """
    z = log['z']
    T = log['T']

    if np.max(z) < min_depth:
        return None

    mask = z >= start_depth
    if not np.any(mask):
        mask = np.ones(len(z), dtype=bool)

    new = dict(log)
    new['z'] = z[mask].copy()
    new['T'] = T[mask].copy()
    return new


# ---------------------------------------------------------------------------
# Rolling regression: gradient, curvature
# ---------------------------------------------------------------------------

def compute_curvature(log, min_points=5):
    """
    Rolling least-squares from each possible start depth.

    For start index *i*, fits both a line (gradient / intercept) and a
    parabola (curvature) to the data from z[i] to the bottom of the hole.

    Parameters
    ----------
    log        : borehole dict (must have 'z' and 'T')
    min_points : minimum number of points required for a fit

    Returns
    -------
    Tuple of 1-D arrays, all indexed by start depth:
        ztop, grad (°C km⁻¹), sd_grad, tsurf (°C), sd_tsurf, curv, rms
    """
    z = log['z']
    T = log['T']
    n_total = len(z)

    ztop_list, grad_list, sd_grad_list = [], [], []
    tsurf_list, sd_tsurf_list, curv_list, rms_list = [], [], [], []

    for i in range(n_total - min_points):
        zi = z[i:]
        Ti = T[i:]
        n = len(zi)

        # --- Linear fit (gradient + intercept) ---
        A = np.column_stack([np.ones(n), zi])
        m_lin, _, _, _ = np.linalg.lstsq(A, Ti, rcond=None)

        res = Ti - A @ m_lin
        var_i = np.dot(res, res) / max(n - 1, 1)

        AtA_inv = np.linalg.inv(A.T @ A)
        sd_m = np.sqrt(var_i * np.diag(AtA_inv))

        grad_list.append(m_lin[1] * 1e3)   # °C/m → °C/km
        tsurf_list.append(m_lin[0])
        sd_tsurf_list.append(sd_m[0])
        sd_grad_list.append(sd_m[1] * 1e3)
        rms_list.append(np.sqrt(np.mean(res ** 2)))
        ztop_list.append(zi[0])

        # --- Quadratic fit (curvature) ---
        B = np.column_stack([zi ** 2, zi, np.ones(n)])
        cond = np.linalg.cond(B.T @ B)
        if cond > 1e15:
            curv_list.append(np.nan)
        else:
            m_quad, _, _, _ = np.linalg.lstsq(B, Ti, rcond=None)
            curv_list.append(2.0 * m_quad[0])

    return (
        np.array(ztop_list),
        np.array(grad_list),
        np.array(sd_grad_list),
        np.array(tsurf_list),
        np.array(sd_tsurf_list),
        np.array(curv_list),
        np.array(rms_list),
    )


def depth2linear(log):
    """
    Find the start depth that best separates the climate-disturbed shallow
    zone from the undisturbed linear geotherm below.

    Two criteria:
      ind_curv : index of minimum |curvature|
      ind_grad : index of minimum |sd_grad|

    Returns
    -------
    (ind_curv, ind_grad) : int indices into log['ztop'] etc.
    """
    curv = log['curv']
    sd_grad = log['sd_grad']

    valid = ~np.isnan(curv)
    if np.any(valid):
        ind_curv = int(np.nanargmin(np.abs(curv)))
    else:
        ind_curv = 0

    ind_grad = int(np.argmin(np.abs(sd_grad)))
    return ind_curv, ind_grad


def process_borehole(log, start_depth=20.0, min_depth=140.0,
                     criterion='curvature'):
    """
    Clip, compute curvature, find fit depth, and compute reduced temperature.

    Modifies *log* in-place.  Returns the modified log, or None if the log
    is too shallow.

    Parameters
    ----------
    criterion : 'curvature' or 'sd_grad'
    """
    clipped = clip_log(log, start_depth, min_depth)
    if clipped is None:
        return None

    log['z'] = clipped['z']
    log['T'] = clipped['T']

    ztop, grad, sd_grad, tsurf, sd_tsurf, curv, rms = compute_curvature(log)
    log['ztop'] = ztop
    log['grad'] = grad
    log['sd_grad'] = sd_grad
    log['tsurf'] = tsurf
    log['sd_tsurf'] = sd_tsurf
    log['curv'] = curv
    log['rms'] = rms

    ind_curv, ind_grad = depth2linear(log)
    log['ind_curv'] = ind_curv
    log['ind_grad'] = ind_grad

    idx = ind_curv if criterion == 'curvature' else ind_grad
    log['Tred'] = reduce_temperature(log['T'], log['z'],
                                     grad[idx], tsurf[idx])
    return log


# ---------------------------------------------------------------------------
# Reduced temperature
# ---------------------------------------------------------------------------

def reduce_temperature(T, z, gradient, intercept):
    """
    Remove the linear geothermal background.

    T_red = T − (z · gradient / 1000 + intercept)

    Parameters
    ----------
    T         : array, observed temperature (°C)
    z         : array, depth (m)
    gradient  : float, background gradient (°C km⁻¹)
    intercept : float, surface temperature intercept (°C)
    """
    return np.asarray(T) - (np.asarray(z) * gradient / 1e3 + intercept)


# ---------------------------------------------------------------------------
# Forward model: step change in surface temperature
# ---------------------------------------------------------------------------

def _forward_kernel(x):
    """
    4 · i²erfc(x)  =  (1 + 2x²) · erfc(x) − 2x · exp(−x²) / √π

    This is the shape function for a step change in surface temperature.
    """
    return (1.0 + 2.0 * x ** 2) * erfc(x) - 2.0 * x * np.exp(-x ** 2) / np.sqrt(np.pi)


def forward_step(dT, tau, alpha, z):
    """
    Temperature anomaly at depth due to a step change in surface temperature.

    Models the reduced temperature profile expected if the ground surface
    temperature changed by *dT* degrees *tau* years before the borehole was
    logged.

    Parameters
    ----------
    dT    : float, surface temperature step (°C); positive = warming
    tau   : float, time since step change (yr)
    alpha : float, thermal diffusivity (m² yr⁻¹); ~31.5 for granite
    z     : array_like, depths (m)

    Returns
    -------
    T_anomaly : ndarray (°C)
    """
    z = np.asarray(z, dtype=float)
    x = z / np.sqrt(4.0 * alpha * max(tau, 1e-6))
    return dT * _forward_kernel(x)


def _forward_jacobian(dT, tau, alpha, z):
    """
    Jacobian of forward_step with respect to [dT, tau].

    Returns F : (len(z), 2) array
        F[:, 0] = ∂T / ∂dT
        F[:, 1] = ∂T / ∂tau
    """
    z = np.asarray(z, dtype=float)
    tau = max(tau, 1e-6)
    x = z / np.sqrt(4.0 * alpha * tau)
    x2 = x ** 2

    dF_ddT = _forward_kernel(x)

    # Analytical derivative w.r.t. tau (from Gauss-Newton formulation)
    dF_dtau = (2.0 * dT * x / tau) * (
        np.exp(-x2) * (1.0 - x2) / np.sqrt(np.pi)
        + x * erf(x)
        + x2 / np.sqrt(np.pi)
        - x
    )

    return np.column_stack([dF_ddT, dF_dtau])


# ---------------------------------------------------------------------------
# Inversion: Newton (Gauss-Newton) method
# ---------------------------------------------------------------------------

def invert_newton(z, Tred, alpha=31.5, dT0=1.0, tau0=150.0,
                  max_iter=500, tol=1e-8):
    """
    Gauss-Newton inversion for step-change surface temperature parameters.

    Delegates to :func:`src.inversion.gauss_newton.gauss_newton`.

    Parameters
    ----------
    z, Tred    : arrays, depth (m) and reduced temperature (°C)
    alpha      : float, thermal diffusivity (m² yr⁻¹)
    dT0, tau0  : float, initial guesses (°C, yr)
    max_iter   : int, maximum iterations
    tol        : float, step-norm convergence tolerance

    Returns
    -------
    dT        : float, best-fit temperature change (°C)
    tau       : float, best-fit time before logging (yr)
    rms       : float, root-mean-square misfit (°C)
    Cov       : (2, 2) ndarray, parameter covariance matrix
    history   : dict with keys 'iter', 'rms', 'step_norm', 'path'
    """
    from src.inversion.gauss_newton import gauss_newton

    z = np.asarray(z, dtype=float)
    Tred = np.asarray(Tred, dtype=float)

    def fwd(z_, p):
        return forward_step(p[0], max(p[1], 1.0), alpha, z_)

    def jac(z_, p):
        return _forward_jacobian(p[0], max(p[1], 1.0), alpha, z_)

    p0 = np.array([float(dT0), float(tau0)])
    p, Cov, history = gauss_newton(fwd, jac, z, Tred, p0,
                                   maxit=max_iter, tol=tol)

    T_pred = forward_step(p[0], max(p[1], 1.0), alpha, z)
    rms = float(np.sqrt(np.mean((T_pred - Tred) ** 2)))
    return float(p[0]), float(p[1]), rms, Cov, history


# ---------------------------------------------------------------------------
# Inversion: grid search
# ---------------------------------------------------------------------------

def invert_gridsearch(z, Tred, alpha=31.5,
                      dT_min=-5.0, dT_max=5.0, dT_step=0.1,
                      tau_min=10.0, tau_max=500.0, tau_step=5.0):
    """
    Brute-force grid search for best-fitting step-change parameters.

    A DC bias (mean offset between observed and predicted) is removed before
    computing RMS so the search focuses on shape rather than absolute level.

    Parameters
    ----------
    z, Tred       : arrays, depth (m) and reduced temperature (°C)
    alpha         : float, thermal diffusivity (m² yr⁻¹)
    dT_min/max/step : grid limits and step for dT (°C)
    tau_min/max/step: grid limits and step for tau (yr)

    Returns
    -------
    best_dT  : float (°C)
    best_tau : float (yr)
    best_rms : float (°C)
    rms_grid : 2-D ndarray, shape (n_dT, n_tau)
    dT_vals  : 1-D ndarray
    tau_vals : 1-D ndarray
    """
    z = np.asarray(z, dtype=float)
    Tred = np.asarray(Tred, dtype=float)

    dT_vals = np.arange(dT_min, dT_max + dT_step * 0.5, dT_step)
    tau_vals = np.arange(tau_min, tau_max + tau_step * 0.5, tau_step)

    rms_grid = np.full((len(dT_vals), len(tau_vals)), np.inf)
    best_rms = np.inf
    best_dT = dT_vals[len(dT_vals) // 2]
    best_tau = tau_vals[len(tau_vals) // 2]

    for i, dT in enumerate(dT_vals):
        for j, tau in enumerate(tau_vals):
            T_pred = forward_step(dT, tau, alpha, z)
            bias = float(np.mean(Tred - T_pred))
            residuals = Tred - (T_pred + bias)
            rms = float(np.std(residuals))
            rms_grid[i, j] = rms
            if rms < best_rms:
                best_rms = rms
                best_dT = dT
                best_tau = tau

    return best_dT, best_tau, best_rms, rms_grid, dT_vals, tau_vals


# ---------------------------------------------------------------------------
# Forward model from air temperature record
# ---------------------------------------------------------------------------

def air_temp_to_reduced(z, years, T_air, alpha=31.5):
    """
    Predict reduced temperature at depths *z* from an air temperature record.

    Uses numerical convolution of the surface temperature history with the
    diffusion Green's function.  The mean is subtracted from T_air first so
    the result represents anomalies relative to the long-term mean.

    Parameters
    ----------
    z      : array, depths (m)
    years  : array, calendar years (increasing), same length as T_air
    T_air  : array, annual mean air temperature (°C)
    alpha  : float, thermal diffusivity (m² yr⁻¹)

    Returns
    -------
    Tred_predicted : 1-D ndarray, shape (len(z),)
    """
    z = np.asarray(z, dtype=float)
    years = np.asarray(years, dtype=float)
    T_air = np.asarray(T_air, dtype=float)

    # Time before most recent year
    t_elapsed = years[-1] - years           # years before present
    dt = np.diff(years, prepend=years[0])   # annual step sizes

    T_anom = T_air - np.mean(T_air)

    Tred = np.zeros(len(z))
    for k, tk in enumerate(t_elapsed):
        if tk <= 0:
            continue
        x = z / np.sqrt(4.0 * alpha * tk)
        Tred += T_anom[k] * _forward_kernel(x) * dt[k]

    return Tred
