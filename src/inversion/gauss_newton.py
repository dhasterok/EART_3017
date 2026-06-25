from __future__ import annotations
import numpy as np

class StopFlag:
    """Thread-safe flag for interrupting a running Gauss-Newton iteration.

    Attributes
    ----------
    stop : bool
        Set to ``True`` from another thread to request early termination.
    """

    def __init__(self):
        self.stop = False


def gauss_newton(fwd, jac, x, d_obs, p0, lam=0.0, maxit=20, tol=1e-8, Wd=None, stop_flag: StopFlag|None=None):
    """Iterative Gauss-Newton inversion with optional Tikhonov regularisation.

    Minimises the weighted misfit

    .. math::

        \\phi(\\mathbf{p}) = \\|\\mathbf{W}(\\mathbf{d} - \\mathbf{g}(\\mathbf{p}))\\|^2
                           + \\lambda^2 \\|\\mathbf{p}\\|^2

    by iterating

    .. math::

        \\mathbf{p}_{k+1} = \\mathbf{p}_k +
        (\\mathbf{J}^T \\mathbf{W}^T \\mathbf{W} \\mathbf{J} + \\lambda^2 \\mathbf{I})^{-1}
        \\mathbf{J}^T \\mathbf{W}^T \\mathbf{W} \\mathbf{r}_k

    where :math:`\\mathbf{r}_k = \\mathbf{d} - \\mathbf{g}(\\mathbf{p}_k)`.

    Parameters
    ----------
    fwd : callable
        Forward model ``fwd(x, p) -> ndarray`` returning predicted data of
        shape ``(N,)`` given inputs ``x`` and parameter vector ``p``.
    jac : callable
        Jacobian ``jac(x, p) -> ndarray`` returning the ``(N, M)`` matrix of
        partial derivatives of ``fwd`` with respect to ``p``.
    x : array-like
        Independent variable(s) passed unchanged to ``fwd`` and ``jac``.
    d_obs : array-like, shape (N,)
        Observed data vector.
    p0 : array-like, shape (M,)
        Initial parameter estimate.
    lam : float, optional
        Tikhonov regularisation parameter :math:`\\lambda`. Zero (default)
        gives unregularised least-squares.
    maxit : int, optional
        Maximum number of iterations. Default is 20.
    tol : float, optional
        Convergence tolerance. Iteration stops when
        ``‖dp‖ < tol * (‖p‖ + tol)``. Default is 1e-8.
    Wd : array-like or None, optional
        Data-weighting matrix. A 1-D array is treated as the diagonal of a
        diagonal weight matrix; a 2-D array is used as-is. ``None`` (default)
        applies equal unit weights.
    stop_flag : StopFlag or None, optional
        When provided, each iteration checks ``stop_flag.stop`` and exits
        early if it is ``True``. Useful for interrupting from another thread.

    Returns
    -------
    p : ndarray, shape (M,)
        Best-fit parameter vector at convergence (or after ``maxit``
        iterations).
    Cov : ndarray, shape (M, M)
        Estimated parameter covariance matrix,
        :math:`\\hat{\\sigma}^2 (\\mathbf{J}^T \\mathbf{J} + \\lambda^2 \\mathbf{I})^{-1}`,
        where :math:`\\hat{\\sigma}^2 = \\|\\mathbf{r}\\|^2 / (N - M)`.
    history : dict
        Iteration log with keys:

        ``'iter'`` : list of int
            Iteration numbers.
        ``'rms'`` : list of float
            Weighted RMS misfit after each iteration.
        ``'step_norm'`` : list of float
            Euclidean norm of the parameter update ``dp`` at each iteration.
        ``'path'`` : list of ndarray
            Parameter vector after each iteration, starting from ``p0``.
    """
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
