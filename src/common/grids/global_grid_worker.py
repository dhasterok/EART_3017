from __future__ import annotations
from PyQt6 import QtCore
import numpy as np

class GlobalGridWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(dict)
    aborted = QtCore.pyqtSignal()

    def __init__(self, fwd_r, r_stations, d_obs, z_vals, R_vals, dr_vals, stop_flag):
        super().__init__()
        self.fwd_r = fwd_r
        self.r_stations = r_stations
        self.d_obs = d_obs
        self.z_vals = z_vals
        self.R_vals = R_vals
        self.dr_vals = dr_vals
        self.stop_flag = stop_flag

    def _rms(self, r):
        return np.sqrt(np.mean(r*r))

    def run(self):
        nz, nR, ndr = len(self.z_vals), len(self.R_vals), len(self.dr_vals)
        total = nz*nR*ndr
        k = 0
        best = {'rms': np.inf, 'z': None, 'R': None, 'drho': None, 'idx': (None,None,None)}
        for iz, z in enumerate(self.z_vals):
            if self.stop_flag.stop:
                self.aborted.emit(); return
            for iR, R in enumerate(self.R_vals):
                for idr, dr in enumerate(self.dr_vals):
                    p = np.array([z, R, dr], float)
                    g = self.fwd_r(self.r_stations, p)
                    rms = self._rms(self.d_obs - g)
                    if rms < best['rms']:
                        best = {'rms': rms, 'z': z, 'R': R, 'drho': dr, 'idx': (iz,iR,idr)}
                    k += 1
                    if k % 500 == 0:
                        self.progress.emit(int(100*k/total))
        self.progress.emit(100)
        self.finished.emit({'best': best, 'z_vals': self.z_vals, 'R_vals': self.R_vals, 'dr_vals': self.dr_vals})
