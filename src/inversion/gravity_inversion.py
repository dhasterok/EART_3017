"""
gravity_inversion.py
--------------------
Gauss-Newton / Levenberg-Marquardt inversion for the 2-D gravity modeller.

Tweaks vertex positions (keeping linked vertices together) and/or density
contrasts to minimise the weighted or unweighted misfit between the computed
Talwani gravity and the loaded observed data.

The iteration loop runs in a QThread (InversionWorker) and emits per-step
signals so the GUI can update in real time.

Parameter vector layout
-----------------------
  p = [ ρ_contrast_0, ..., ρ_contrast_N,   (if invert_densities)
        x_g0, z_g0, x_g1, z_g1, ... ]      (if invert_vertices)

where each (x_gi, z_gi) is the shared position of a unique vertex group
(multiple polygon vertices at the same location move together).
"""

from __future__ import annotations

import sys
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# ── path setup ────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).resolve().parent          # gravity2d/
_COURSE = _HERE.parent.parent                       # new_version/
for _p in [str(_HERE), str(_COURSE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.gravity.talwani_model import compute_gz

_LINK_EPS = 1e-9    # km -- vertices closer than this are treated as linked
_H_XZ     = 1e-3   # km -- finite-difference step for vertex coordinates
_H_RHO    = 1.0    # kg/m³ -- finite-difference step for density contrasts


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InversionConfig:
    """User-configurable inversion settings."""
    invert_vertices:  bool  = True
    invert_densities: bool  = True
    use_weights:      bool  = False
    damping:          float = 1e-2   # λ in (J^T W J + λ² I) Δp = J^T W r
    max_iter:         int   = 20
    tol:              float = 1e-4   # convergence on ||Δp||


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter vector helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_params(bodies, bg_density: float, cfg: InversionConfig):
    """
    Construct initial parameter vector and the index mappings.

    Returns
    -------
    p            : (M,) float array
    density_bidx : list of body indices for density parameters
    vert_groups  : list of lists of (body_idx, vert_idx); one list per unique
                   vertex location (linked vertices share a single (x, z) pair)
    """
    parts = []
    density_bidx = []
    vert_groups  = []

    if cfg.invert_densities:
        for bi, body in enumerate(bodies):
            parts.append(body.density - bg_density)
            density_bidx.append(bi)

    if cfg.invert_vertices:
        assigned = {}   # (body_idx, vert_idx) -> group index already created
        for bi, body in enumerate(bodies):
            for vi, (vx, vz) in enumerate(body.vertices):
                key = (bi, vi)
                if key in assigned:
                    continue
                # Collect all coincident vertices across all bodies
                group = [key]
                for bj, bdy in enumerate(bodies):
                    for vj, (ux, uz) in enumerate(bdy.vertices):
                        if (bj, vj) == key or (bj, vj) in assigned:
                            continue
                        if math.hypot(vx - ux, vz - uz) < _LINK_EPS:
                            group.append((bj, vj))
                for k in group:
                    assigned[k] = len(vert_groups)
                vert_groups.append(group)
                parts.extend([vx, vz])

    return np.array(parts, dtype=float), density_bidx, vert_groups


def _apply_params(p, bodies, bg_density, density_bidx, vert_groups, cfg):
    """Write parameter vector back into bodies in-place."""
    idx = 0
    if cfg.invert_densities:
        for bi in density_bidx:
            bodies[bi].density = float(p[idx]) + bg_density
            idx += 1
    if cfg.invert_vertices:
        for group in vert_groups:
            vx = float(p[idx])
            vz = max(0.0, float(p[idx + 1]))   # z ≥ 0 (depth is positive)
            for bi, vi in group:
                bodies[bi].vertices[vi] = [vx, vz]
            idx += 2


def _read_params(bodies, bg_density, density_bidx, vert_groups, cfg):
    """Read current body state back into a parameter vector (reflects clamping)."""
    parts = []
    if cfg.invert_densities:
        for bi in density_bidx:
            parts.append(bodies[bi].density - bg_density)
    if cfg.invert_vertices:
        for group in vert_groups:
            bi, vi = group[0]
            vx, vz = bodies[bi].vertices[vi]
            parts.extend([vx, vz])
    return np.array(parts, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
#  Forward model wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _forward_gz(bodies, bg_density, x_obs):
    """Compute gz at x_obs for the current body set."""
    gz = np.zeros(len(x_obs))
    for body in bodies:
        if body.visible and len(body.vertices) >= 3:
            contrast = body.density - bg_density
            gz += compute_gz(x_obs, body.vertices, contrast)
    return gz


def _jacobian_fd(p, bodies, bg_density, x_obs, gz_base,
                 density_bidx, vert_groups, cfg):
    """
    Finite-difference Jacobian  J[i, j] = ∂gz[i] / ∂p[j].

    Uses forward differences with steps _H_RHO (density) and _H_XZ (vertices).
    Restores bodies to the state corresponding to p on exit.
    """
    N = len(x_obs)
    M = len(p)
    J = np.zeros((N, M))
    p_pert = p.copy()

    col = 0
    if cfg.invert_densities:
        for _ in density_bidx:
            p_pert[:] = p
            p_pert[col] += _H_RHO
            _apply_params(p_pert, bodies, bg_density, density_bidx, vert_groups, cfg)
            gz_p = _forward_gz(bodies, bg_density, x_obs)
            J[:, col] = (gz_p - gz_base) / _H_RHO
            col += 1

    if cfg.invert_vertices:
        for _ in vert_groups:
            for coord_off, h in enumerate([_H_XZ, _H_XZ]):
                p_pert[:] = p
                p_pert[col + coord_off] += h
                _apply_params(p_pert, bodies, bg_density, density_bidx, vert_groups, cfg)
                gz_p = _forward_gz(bodies, bg_density, x_obs)
                J[:, col + coord_off] = (gz_p - gz_base) / h
            col += 2

    # Restore bodies to the unperturbed state
    _apply_params(p, bodies, bg_density, density_bidx, vert_groups, cfg)
    return J


# ─────────────────────────────────────────────────────────────────────────────
#  QThread worker
# ─────────────────────────────────────────────────────────────────────────────

class InversionWorker(QThread):
    """
    Runs Gauss-Newton iterations in a background thread.

    Signals
    -------
    iteration_done(iter_num, rms, body_dicts)
        Emitted after each accepted step; body_dicts is a serialisable list
        so the GUI can rebuild bodies in the main thread.
    converged(rms)
        Emitted when inversion finishes (converged or max_iter reached).
    stopped()
        Emitted when the user requests early termination.
    """

    iteration_done = pyqtSignal(int, float, list)
    converged      = pyqtSignal(float)
    stopped        = pyqtSignal()

    def __init__(self, bodies, bg_density: float, obs_data,
                 cfg: InversionConfig, parent=None):
        super().__init__(parent)
        # Deep-copy bodies so we don't modify the GUI model mid-run
        self._bodies = [b.clone() for b in bodies]
        self._bg     = bg_density
        self._obs    = obs_data
        self._cfg    = cfg
        self._abort  = False

    def request_stop(self):
        self._abort = True

    def _bodies_to_dicts(self):
        return [
            {
                "name":     b.name,
                "density":  b.density,
                "color":    b.color,
                "visible":  b.visible,
                "vertices": [v[:] for v in b.vertices],
            }
            for b in self._bodies
        ]

    def run(self):
        obs = self._obs
        cfg = self._cfg

        # Select unmasked observations
        mask     = obs.masked if obs.masked is not None \
                   else np.zeros(len(obs.x), dtype=bool)
        unmasked = ~mask
        if not unmasked.any():
            self.converged.emit(0.0)
            return

        x_obs  = obs.x[unmasked]
        gz_obs = obs.gz[unmasked]

        # Data weights (1/σ²) or None
        Wd = None
        if cfg.use_weights and obs.gz_unc is not None:
            unc = obs.gz_unc[unmasked]
            unc = np.where(unc > 0, unc, 1e-6)
            Wd  = 1.0 / (unc ** 2)

        p, density_bidx, vert_groups = _build_params(self._bodies, self._bg, cfg)

        if len(p) == 0:
            self.converged.emit(0.0)
            return

        # Weight matrix helpers (reuse gauss_newton.py convention: Wd is diagonal)
        if Wd is not None:
            W_mat = np.diag(Wd)
            def WtW():
                return W_mat.T @ W_mat
        else:
            def WtW():
                return np.eye(len(x_obs))

        lam = cfg.damping

        for it in range(1, cfg.max_iter + 1):
            if self._abort:
                self.stopped.emit()
                return

            gz_calc = _forward_gz(self._bodies, self._bg, x_obs)
            r       = gz_obs - gz_calc
            rms     = float(np.sqrt(np.mean(r ** 2)))

            J  = _jacobian_fd(p, self._bodies, self._bg, x_obs, gz_calc,
                              density_bidx, vert_groups, cfg)
            WW = WtW()
            A  = J.T @ WW @ J + (lam ** 2) * np.eye(len(p))
            b  = J.T @ WW @ r
            try:
                dp = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dp, *_ = np.linalg.lstsq(A, b, rcond=None)

            # Apply update and enforce z ≥ 0 via _apply_params clamping
            p_new = p + dp
            _apply_params(p_new, self._bodies, self._bg,
                          density_bidx, vert_groups, cfg)
            # Read back (so p reflects any z-clamping)
            p_new = _read_params(self._bodies, self._bg, density_bidx, vert_groups, cfg)

            gz_new  = _forward_gz(self._bodies, self._bg, x_obs)
            rms_new = float(np.sqrt(np.mean((gz_obs - gz_new) ** 2)))

            self.iteration_done.emit(it, rms_new, self._bodies_to_dicts())
            p = p_new

            step_norm = float(np.linalg.norm(dp))
            if step_norm < cfg.tol * (float(np.linalg.norm(p)) + cfg.tol):
                break

        # Final RMS
        gz_final  = _forward_gz(self._bodies, self._bg, x_obs)
        rms_final = float(np.sqrt(np.mean((gz_obs - gz_final) ** 2)))
        self.converged.emit(rms_final)
