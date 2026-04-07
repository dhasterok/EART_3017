from __future__ import annotations
import numpy as np

class StopFlag:
    def __init__(self):
        self.stop = False


def gauss_newton(fwd, jac, x, d_obs, p0, lam=0.0, maxit=20, tol=1e-8, Wd=None, stop_flag: StopFlag|None=None):
    p = np.asarray(p0, dtype=float)
    npar = p.size

    if Wd is None:
        def W_apply(v):
            return v
        def WT_W():
            return np.eye(len(d_obs))
    else:
        W = np.diag(Wd) if np.ndim(Wd)==1 else Wd
        def W_apply(v):
            return W @ v
        def WT_W():
            return W.T @ W

    history = {"iter":[], "rms":[], "step_norm":[], "path": [p.copy()]}

    for it in range(1, maxit+1):
        if stop_flag is not None and stop_flag.stop:
            break
        g = fwd(x, p)
        r = d_obs - g
        J = jac(x, p)
        A = J.T @ WT_W() @ J + (lam**2) * np.eye(npar)
        b = J.T @ WT_W() @ r
        try:
            dp = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            dp, *_ = np.linalg.lstsq(A, b, rcond=None)
        p_new = p + dp
        step_norm = np.linalg.norm(dp)
        rms = np.sqrt(np.mean((W_apply(r))**2))
        history["iter"].append(it)
        history["rms"].append(float(rms))
        history["step_norm"].append(float(step_norm))
        history["path"].append(p_new.copy())
        if step_norm < tol * (np.linalg.norm(p) + tol):
            p = p_new
            break
        p = p_new

    # final cov
    g = fwd(x, p)
    r = d_obs - g
    dof = max(len(d_obs) - npar, 1)
    sigma2 = (r @ r) / dof
    J = jac(x, p)
    A = J.T @ J + (lam**2) * np.eye(npar)
    try:
        Cov = sigma2 * np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Cov = sigma2 * np.linalg.pinv(A)

    return p, Cov, history
