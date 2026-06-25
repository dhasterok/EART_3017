from __future__ import annotations
import numpy as np


def forward(x, p):
    """Forward model for plagioclase thermal conductivity.

    Computes

    .. math::

        k(\\mathrm{An}, T) = (a + b\\,\\mathrm{An} + c\\,\\mathrm{An}^2)
                             \\left(\\frac{298}{T}\\right)^n

    Parameters
    ----------
    x : array-like, shape (N, 2)
        Input matrix whose columns are anorthite fraction An (dimensionless,
        0–1) and temperature T (K).
    p : array-like, shape (4,)
        Model parameters ``[a, b, c, n]`` where *a*, *b*, *c* are
        composition coefficients (W m⁻¹ K⁻¹) and *n* is the temperature
        exponent (dimensionless).

    Returns
    -------
    k : ndarray, shape (N,)
        Predicted thermal conductivity (W m⁻¹ K⁻¹).
    """
    x = np.asarray(x)
    An, T = x[:, 0], x[:, 1]
    a, b, c, n = p
    return (a + b * An + c * An**2) * (298.0 / T) ** n


def jacobian(x, p):
    """Jacobian of :func:`forward` with respect to the model parameters.

    Returns the matrix of partial derivatives
    :math:`J_{ij} = \\partial k_i / \\partial p_j`.

    Parameters
    ----------
    x : array-like, shape (N, 2)
        Input matrix whose columns are anorthite fraction An (dimensionless,
        0–1) and temperature T (K).
    p : array-like, shape (4,)
        Model parameters ``[a, b, c, n]``.

    Returns
    -------
    J : ndarray, shape (N, 4)
        Jacobian matrix with columns
        :math:`[\\partial k/\\partial a,\\; \\partial k/\\partial b,\\;
        \\partial k/\\partial c,\\; \\partial k/\\partial n]`.
    """
    x = np.asarray(x)
    An, T = x[:, 0], x[:, 1]
    a, b, c, n = p
    ratio = (298.0 / T) ** n
    base  = a + b * An + c * An**2
    dk_da = ratio
    dk_db = An * ratio
    dk_dc = An**2 * ratio
    dk_dn = base * ratio * np.log(298.0 / T)
    return np.column_stack([dk_da, dk_db, dk_dc, dk_dn])
