from __future__ import annotations
from PyQt6 import QtCore
import numpy as np

class GridSearchWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(dict)
    aborted = QtCore.pyqtSignal()

    def __init__(self, fwd_r, r_stations, d_obs, p_fixed, z_vals, R_vals, dr_vals, stop_flag):
        super().__init__()
        self.fwd_r = fwd_r
        self.r_stations = r_stations
        self.d_obs = d_obs
        self.p_fixed = p_fixed.copy()
        self.z_vals = z_vals
        self.R_vals = R_vals
        self.dr_vals = dr_vals
        self.stop_flag = stop_flag

    def _rms(self, r):
        return np.sqrt(np.mean(r*r))

    def _grid_pair(self, ix, iy, xvals, yvals, pfix):
        ny, nx = len(yvals), len(xvals)
        G = np.empty((ny, nx), dtype=float)
        total = max(nx*ny, 1)
        k = 0
        for j, yv in enumerate(yvals):
            if self.stop_flag.stop:
                self.aborted.emit(); return None
            for i, xv in enumerate(xvals):
                p = pfix.copy()
                p[ix] = xv
                p[iy] = yv
                g = self.fwd_r(self.r_stations, p)
                G[j,i] = self._rms(self.d_obs - g)
                k += 1
                if k % 200 == 0:
                    self.progress.emit(int(100 * k / (3*total)))
        jmin, imin = np.unravel_index(np.argmin(G), G.shape)
        return G, (imin, jmin)

    def run(self):
        pfix = self.p_fixed.copy()
        G1, (i1, j1) = self._grid_pair(1, 2, self.R_vals, self.dr_vals, pfix)
        if G1 is None: return
        G2, (i2, j2) = self._grid_pair(0, 2, self.z_vals, self.dr_vals, pfix)
        if G2 is None: return
        G3, (i3, j3) = self._grid_pair(0, 1, self.z_vals, self.R_vals, pfix)
        if G3 is None: return
        self.progress.emit(100)
        z1 = pfix[0]; R1 = self.R_vals[i1]; dr1 = self.dr_vals[j1]; rms1 = G1[j1,i1]
        z2 = self.z_vals[i2]; R2 = pfix[1]; dr2 = self.dr_vals[j2]; rms2 = G2[j2,i2]
        z3 = self.z_vals[i3]; R3 = self.R_vals[j3]; dr3 = pfix[2]; rms3 = G3[j3,i3]
        self.finished.emit({
            'p_fixed': pfix,
            'G_dr_R': G1, 'R_vals': self.R_vals, 'dr_vals': self.dr_vals, 'min_dr_R': (R1, dr1, z1, rms1), 'idx_dr_R': (i1, j1),
            'G_dr_z': G2, 'z_vals': self.z_vals, 'dr_vals2': self.dr_vals, 'min_dr_z': (z2, dr2, R2, rms2), 'idx_dr_z': (i2, j2),
            'G_R_z' : G3, 'z_vals2': self.z_vals, 'R_vals2': self.R_vals, 'min_R_z' : (z3, R3, dr3, rms3), 'idx_R_z': (i3, j3),
        })
