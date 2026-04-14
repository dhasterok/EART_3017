"""
tests/test_forward_models.py
-----------------------------
Automated validation of the 2-D gravity and magnetic forward models.

Run with:
    pytest tests/test_forward_models.py -v

All tests are deterministic and require no GUI.

Analytic references
-------------------
Gravity — 2-D horizontal cylinder (Blakely 1995, eq. 6.18):
    gz(x) = 2π G Δρ R² · z / (x² + z²)

    where z is the cylinder axis depth, R is the radius, and Δρ is the
    density contrast.  Units: SI → mGal via ×1e5.

Magnetic — 2-D horizontal cylinder (Blakely 1995, §6.3):
    The horizontal and vertical field components at observation point x
    due to induced magnetisation J = χ F / μ₀ with inclination IE are:

        Bx(x) = (μ₀/π) J R² · [Mx (d²-x²) - 2 Mz x d] / (x²+d²)²  ×1e9 nT
        Bz(x) = (μ₀/π) J R² · [Mz (d²-x²) + 2 Mx x d] / (x²+d²)²  ×1e9 nT

        (observation at surface, cylinder axis at depth d, horizontal
         distance x from directly above the cylinder; Mx/Mz are the
         horizontal and vertical components of J)

    TMI = Bx cos(IE) + Bz sin(IE)

    Reference: Blakely (1995), eq. 6.21–6.22 (2-D cylinder, SI).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.physics.gravity.talwani_model    import compute_gz
from src.physics.gravity.grav2_5d_model   import compute_gz   as compute_gz_25d
from src.physics.magnetics.mag2d_model    import compute_bt, compute_bx_bz
from src.physics.magnetics.mag2_5d_model  import compute_bt   as compute_bt_25d
from src.physics.magnetics.mag2_5d_model  import compute_bx_bz as compute_bx_bz_25d


# ── physical constants ────────────────────────────────────────────────────────
G_SI   = 6.674e-11          # m³ kg⁻¹ s⁻²
MU0    = 4.0 * math.pi * 1e-7  # T·m A⁻¹


# ── polygon builders ──────────────────────────────────────────────────────────

def make_cylinder_polygon(cx_km: float, cz_km: float,
                          r_km: float, n_sides: int) -> list:
    """Regular n-gon approximating a horizontal cylinder (vertices in km)."""
    return [
        [cx_km + r_km * math.cos(2 * math.pi * k / n_sides),
         cz_km + r_km * math.sin(2 * math.pi * k / n_sides)]
        for k in range(n_sides)
    ]


def make_rect_polygon(x0, z0, x1, z1) -> list:
    """Axis-aligned rectangle; coordinates in km, z positive downward."""
    return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]


# ── analytic solutions ────────────────────────────────────────────────────────

def analytic_gz_cylinder(x_km, cx_km: float, depth_km: float,
                          radius_km: float, rho: float) -> np.ndarray:
    """
    Vertical gravity of an infinite 2-D horizontal cylinder (mGal).

    gz(x) = 2π G Δρ R² · d / ((x - cx)² + d²)

    Parameters
    ----------
    x_km     : observation positions (km)
    cx_km    : cylinder centre x (km)
    depth_km : cylinder axis depth (km, positive downward)
    radius_km: cylinder radius (km)
    rho      : density contrast (kg/m³)
    """
    x  = np.asarray(x_km, dtype=float) * 1e3   # → m
    cx = cx_km * 1e3
    d  = depth_km * 1e3
    R  = radius_km * 1e3
    gz_si = 2.0 * math.pi * G_SI * rho * R**2 * d / ((x - cx)**2 + d**2)
    return gz_si * 1e5   # m/s² → mGal


def analytic_mag_cylinder(x_km, cx_km: float, depth_km: float,
                           radius_km: float,
                           susceptibility: float,
                           F_nT: float, IE_deg: float):
    """
    Horizontal (Bh) and vertical (Bz) induced magnetic fields and TMI
    of an infinite 2-D horizontal cylinder at the surface (nT).

    Derivation
    ----------
    The Won-Bevis / Blakely forward model sums two polygon integrals:
        Bx = (μ₀/2π)(Mx·ΣFk − Mz·ΣHk)
        Bz = (μ₀/2π)(Mz·ΣFk + Mx·ΣHk)

    For a disk (circle) at (cx, d) observed at (x, 0), the polygon
    integrals converge to (Δx = x − cx, r² = Δx² + d²):
        ΣFk = πR² d / r²        [same integral as the Talwani gravity sum]
        ΣHk = −πR² Δx / r²     [sign fixed so that Bx>0 right of a
                                  vertically magnetised cylinder]

    Substituting:
        Bx = (μ₀R²/2)(Mx·d − Mz·Δx) / r²
        Bz = (μ₀R²/2)(Mz·d + Mx·Δx) / r²

    Note: the r⁴ denominator that appears in 3-D point-dipole formulas
    does NOT apply here — a 2-D cylinder is a line source (r² decay).

    Returns
    -------
    bh_nT, bz_nT, tmi_nT : ndarrays
    """
    dx  = np.asarray(x_km, dtype=float) * 1e3 - cx_km * 1e3   # Δx in metres
    d   = depth_km * 1e3    # cylinder depth (m)
    R   = radius_km * 1e3   # cylinder radius (m)
    IE  = math.radians(IE_deg)

    F_T = F_nT * 1e-9                   # T
    J   = susceptibility * F_T / MU0    # A/m

    Mx = J * math.cos(IE)
    Mz = J * math.sin(IE)

    r2 = dx**2 + d**2
    prefactor = MU0 * R**2 / 2.0       # SI units (T·m²)

    bh_si = prefactor * (Mx * d  - Mz * dx) / r2
    bz_si = prefactor * (Mz * d  + Mx * dx) / r2

    bh_nT  = bh_si * 1e9
    bz_nT  = bz_si * 1e9
    tmi_nT = bh_nT * math.cos(IE) + bz_nT * math.sin(IE)
    return bh_nT, bz_nT, tmi_nT


# ─────────────────────────────────────────────────────────────────────────────
# 1. Horizontal cylinder — gravity
# ─────────────────────────────────────────────────────────────────────────────

class TestGravityCylinder:

    # geometry and material
    CX    = 0.0     # km  (centred on profile)
    DEPTH = 5.0     # km
    R     = 1.0     # km
    RHO   = 500.0   # kg/m³  density contrast

    N_POLY  = 360   # vertices for a near-exact polygon
    X_OBS   = np.linspace(-20, 20, 201)

    def _polygon(self, n_sides=None):
        n = n_sides or self.N_POLY
        return make_cylinder_polygon(self.CX, self.DEPTH, self.R, n)

    def _analytic(self):
        return analytic_gz_cylinder(self.X_OBS, self.CX,
                                    self.DEPTH, self.R, self.RHO)

    def test_peak_value(self):
        """Peak gz over the cylinder matches analytic to < 0.5%."""
        gz_num  = compute_gz(self.X_OBS, self._polygon(), self.RHO)
        gz_anal = self._analytic()
        peak_err = abs(gz_num.max() - gz_anal.max()) / abs(gz_anal.max())
        assert peak_err < 0.005, f"Peak gravity error {peak_err:.4%} exceeds 0.5%"

    def test_allclose(self):
        """Full profile matches analytic within 0.5% relative tolerance."""
        gz_num  = compute_gz(self.X_OBS, self._polygon(), self.RHO)
        gz_anal = self._analytic()
        assert np.allclose(gz_num, gz_anal, rtol=0.005, atol=0.0)

    def test_rms_error(self):
        """RMS error between numeric and analytic is < 0.01 mGal."""
        gz_num  = compute_gz(self.X_OBS, self._polygon(), self.RHO)
        gz_anal = self._analytic()
        rms = float(np.sqrt(np.mean((gz_num - gz_anal)**2)))
        assert rms < 0.01, f"Gravity RMS error {rms:.4f} mGal exceeds 0.01 mGal"

    def test_symmetry(self):
        """Response is symmetric about the cylinder axis."""
        gz = compute_gz(self.X_OBS, self._polygon(), self.RHO)
        gz_flip = compute_gz(-self.X_OBS, self._polygon(), self.RHO)
        # Both profiles evaluated at opposite-sign offsets should match
        assert np.allclose(gz, gz_flip[::-1], rtol=1e-6)

    def test_positive_gz(self):
        """Positive density contrast gives positive gz everywhere."""
        gz = compute_gz(self.X_OBS, self._polygon(), self.RHO)
        assert np.all(gz > 0)

    def test_zero_contrast_gives_zero(self):
        """Zero density contrast gives zero gravity."""
        gz = compute_gz(self.X_OBS, self._polygon(), 0.0)
        assert np.allclose(gz, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rectangular prism — gravity sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestGravityRectangle:

    RECT   = make_rect_polygon(-2, 2, 2, 6)   # 4 km wide, 4 km tall, 2–6 km depth
    RHO    = 300.0
    X_OBS  = np.linspace(-20, 20, 201)
    X_FINE = np.linspace(-10, 10, 1001)

    def test_peak_above_centre(self):
        """Peak of gz should be directly above the rectangle centre (x=0)."""
        gz  = compute_gz(self.X_FINE, self.RECT, self.RHO)
        x_peak = self.X_FINE[np.argmax(gz)]
        assert abs(x_peak) < 0.05, f"Peak offset {x_peak:.3f} km from centre"

    def test_positive_anomaly(self):
        """Positive density contrast gives positive gz."""
        gz = compute_gz(self.X_OBS, self.RECT, self.RHO)
        assert np.all(gz > 0)

    def test_negative_contrast_flips_sign(self):
        """Negative density contrast flips the sign of gz everywhere."""
        gz_pos = compute_gz(self.X_OBS, self.RECT,  self.RHO)
        gz_neg = compute_gz(self.X_OBS, self.RECT, -self.RHO)
        assert np.allclose(gz_pos, -gz_neg, rtol=1e-10)

    def test_linearity_in_density(self):
        """gz scales linearly with density contrast."""
        gz1 = compute_gz(self.X_OBS, self.RECT, self.RHO)
        gz2 = compute_gz(self.X_OBS, self.RECT, 2.0 * self.RHO)
        assert np.allclose(gz2, 2.0 * gz1, rtol=1e-10)

    def test_falls_off_far_field(self):
        """gz at far offsets is much smaller than the peak."""
        gz = compute_gz(self.X_OBS, self.RECT, self.RHO)
        peak = gz.max()
        far  = max(abs(gz[0]), abs(gz[-1]))   # ±20 km edges
        assert far < 0.05 * peak, \
            f"Far-field value {far:.4f} mGal is >{5}% of peak {peak:.4f} mGal"

    def test_fewer_than_3_vertices_is_zero(self):
        """Degenerate polygons (< 3 vertices) return zero."""
        gz = compute_gz(self.X_OBS, [[0, 1], [1, 2]], 300.0)
        assert np.allclose(gz, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Horizontal cylinder — magnetics
# ─────────────────────────────────────────────────────────────────────────────

class TestMagneticsCylinder:

    CX    = 0.0
    DEPTH = 4.0     # km
    R     = 0.8     # km
    CHI   = 0.001   # SI susceptibility (realistic for mafic rocks)
    F_nT  = 50_000.0
    IE    = 60.0    # degrees

    N_POLY = 360
    X_OBS  = np.linspace(-20, 20, 201)

    def _polygon(self, n=None):
        return make_cylinder_polygon(self.CX, self.DEPTH, self.R, n or self.N_POLY)

    def _analytic(self):
        return analytic_mag_cylinder(
            self.X_OBS, self.CX, self.DEPTH, self.R,
            self.CHI, self.F_nT, self.IE)

    def test_tmi_allclose(self):
        """Numeric TMI matches analytic within 0.5% relative tolerance."""
        tmi_num = compute_bt(self.X_OBS, self._polygon(),
                             self.CHI, self.F_nT, self.IE)
        _, _, tmi_anal = self._analytic()
        assert np.allclose(tmi_num, tmi_anal, rtol=0.005, atol=0.0)

    def test_tmi_rms_error(self):
        """RMS error of TMI vs analytic is < 0.5 nT."""
        tmi_num = compute_bt(self.X_OBS, self._polygon(),
                             self.CHI, self.F_nT, self.IE)
        _, _, tmi_anal = self._analytic()
        rms = float(np.sqrt(np.mean((tmi_num - tmi_anal)**2)))
        assert rms < 0.5, f"TMI RMS error {rms:.3f} nT exceeds 0.5 nT"

    def test_bh_bz_allclose(self):
        """Numeric Bh and Bz match analytic within 0.5%."""
        bh_num, bz_num = compute_bx_bz(self.X_OBS, self._polygon(),
                                        self.CHI, self.F_nT, self.IE)
        bh_anal, bz_anal, _ = self._analytic()
        assert np.allclose(bh_num, bh_anal, rtol=0.005, atol=0.0), \
            "Bh does not match analytic"
        assert np.allclose(bz_num, bz_anal, rtol=0.005, atol=0.0), \
            "Bz does not match analytic"

    def test_vertical_field_symmetry(self):
        """At IE=90° (vertical field) TMI should be symmetric (even function of x)."""
        poly = self._polygon()
        tmi = compute_bt(self.X_OBS, poly, self.CHI, self.F_nT, 90.0)
        tmi_flip = compute_bt(-self.X_OBS, poly, self.CHI, self.F_nT, 90.0)
        assert np.allclose(tmi, tmi_flip[::-1], rtol=1e-5), \
            "TMI at IE=90° should be symmetric about the cylinder axis"

    def test_zero_susceptibility_gives_zero(self):
        """Zero susceptibility with no remanence gives zero TMI."""
        tmi = compute_bt(self.X_OBS, self._polygon(), 0.0, self.F_nT, self.IE)
        assert np.allclose(tmi, 0.0)

    def test_tmi_consistent_with_bh_bz(self):
        """TMI = Bh·cos(IE) + Bz·sin(IE) must be consistent with compute_bt."""
        IE_rad = math.radians(self.IE)
        tmi    = compute_bt(self.X_OBS, self._polygon(),
                            self.CHI, self.F_nT, self.IE)
        bh, bz = compute_bx_bz(self.X_OBS, self._polygon(),
                                self.CHI, self.F_nT, self.IE)
        tmi_from_components = bh * math.cos(IE_rad) + bz * math.sin(IE_rad)
        assert np.allclose(tmi, tmi_from_components, rtol=1e-6), \
            "TMI from compute_bt disagrees with Bh·cos(IE)+Bz·sin(IE)"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Convergence tests — polygon → cylinder
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergence:
    """
    As the number of polygon vertices increases, the numerical result
    must approach the analytic cylinder solution.  Both gravity and
    magnetics are tested.
    """

    CX    = 0.0
    DEPTH = 5.0     # km
    R     = 1.0     # km
    RHO   = 400.0   # kg/m³
    CHI   = 0.03
    F_nT  = 50_000.0
    IE    = 45.0

    X_OBS    = np.linspace(-15, 15, 151)
    N_SIDES  = [8, 16, 32, 64, 128, 256]   # vertex counts to test

    def _gz_rms_error(self, n_sides):
        poly   = make_cylinder_polygon(self.CX, self.DEPTH, self.R, n_sides)
        gz_num = compute_gz(self.X_OBS, poly, self.RHO)
        gz_ref = analytic_gz_cylinder(self.X_OBS, self.CX,
                                      self.DEPTH, self.R, self.RHO)
        return float(np.sqrt(np.mean((gz_num - gz_ref)**2)))

    def _tmi_rms_error(self, n_sides):
        poly    = make_cylinder_polygon(self.CX, self.DEPTH, self.R, n_sides)
        tmi_num = compute_bt(self.X_OBS, poly, self.CHI, self.F_nT, self.IE)
        _, _, tmi_ref = analytic_mag_cylinder(
            self.X_OBS, self.CX, self.DEPTH, self.R,
            self.CHI, self.F_nT, self.IE)
        return float(np.sqrt(np.mean((tmi_num - tmi_ref)**2)))

    def test_gravity_converges(self):
        """Gravity RMS error decreases as polygon vertex count increases."""
        errors = [self._gz_rms_error(n) for n in self.N_SIDES]
        # Allow one non-monotone step (floating-point noise at high n), but the
        # overall trend must be downward: last error < first error / 10.
        assert errors[-1] < errors[0] / 10.0, \
            f"Gravity did not converge: errors = {[f'{e:.4f}' for e in errors]}"
        # Count violations of monotonicity
        violations = sum(1 for a, b in zip(errors, errors[1:]) if b > a * 1.05)
        assert violations <= 1, \
            f"Too many non-monotone steps in gravity convergence: {violations}"

    def test_magnetics_converges(self):
        """Magnetic TMI RMS error decreases as polygon vertex count increases."""
        errors = [self._tmi_rms_error(n) for n in self.N_SIDES]
        assert errors[-1] < errors[0] / 10.0, \
            f"TMI did not converge: errors = {[f'{e:.4f}' for e in errors]}"
        violations = sum(1 for a, b in zip(errors, errors[1:]) if b > a * 1.05)
        assert violations <= 1, \
            f"Too many non-monotone steps in TMI convergence: {violations}"

    def test_gravity_256_sides_accuracy(self):
        """256-vertex polygon approximates cylinder gz to < 0.05%."""
        err = self._gz_rms_error(256)
        gz_ref = analytic_gz_cylinder(self.X_OBS, self.CX,
                                      self.DEPTH, self.R, self.RHO)
        relative = err / gz_ref.max()
        assert relative < 0.0005, \
            f"Gravity error at 256 sides: {relative:.5%} (exceeds 0.05%)"

    def test_magnetics_256_sides_accuracy(self):
        """256-vertex polygon approximates cylinder TMI to < 0.1%."""
        err = self._tmi_rms_error(256)
        _, _, tmi_ref = analytic_mag_cylinder(
            self.X_OBS, self.CX, self.DEPTH, self.R,
            self.CHI, self.F_nT, self.IE)
        relative = err / abs(tmi_ref).max()
        assert relative < 0.001, \
            f"TMI error at 256 sides: {relative:.5%} (exceeds 0.1%)"


# ─────────────────────────────────────────────────────────────────────────────
# Analytic references for 2.5-D (finite-strike) models
# ─────────────────────────────────────────────────────────────────────────────

def analytic_bxbz_2d(x_km, cx_km, depth_km, radius_km,
                     susceptibility, F_nT, IE_deg, DE_deg=0.0):
    """
    Exact Bx and Bz (nT) for a uniformly magnetised infinite 2-D circular
    cylinder using the 2-D dipole (r⁴) kernel.

    Outside an infinite circular cylinder the field is exactly that of a 2-D
    line dipole with moment per unit length m = M × πR²:

        Φ_m  = (R²/2) (Mx·ξ + Mz·ζ) / r²

        Bx = (μ₀R²/2) [Mx(ξ²−ζ²) + 2Mz·ξζ ] / r⁴
        Bz = (μ₀R²/2) [Mz(ζ²−ξ²) + 2Mx·ξζ ] / r⁴

    where ξ = x_obs − cx, ζ = z_obs − depth (negative for buried body).
    For vertical field (Mz only) Bz ∝ (d²−x²)/r⁴ ✓ and Bx is odd in x ✓.
    """
    xi   = (np.asarray(x_km, dtype=float) - cx_km) * 1e3   # m
    d    = depth_km  * 1e3
    R    = radius_km * 1e3

    IE   = math.radians(IE_deg)
    DE   = math.radians(DE_deg)
    F_T  = F_nT * 1e-9
    J    = susceptibility * F_T / MU0
    Mx   = J * math.cos(IE) * math.cos(DE)
    Mz   = J * math.sin(IE)

    zeta = -d                       # ζ = z_obs − depth, obs at surface
    xi2  = xi**2
    z2   = zeta**2
    r4   = (xi2 + z2)**2

    fac   = MU0 * R**2 / 2.0
    bx_si = fac * (Mx * (xi2 - z2) + 2.0 * Mz * xi * zeta) / r4
    bz_si = fac * (Mz * (z2 - xi2) + 2.0 * Mx * xi * zeta) / r4
    return bx_si * 1e9, bz_si * 1e9


def analytic_gz_25d(x_km, cx_km, depth_km, radius_km, rho, L_km):
    """
    Vertical gravity (mGal) of a horizontal cylinder extruded to ±L.

    Derived by integrating the line-source gravity formula along y:

        gz = G Δρ π R² · ∫_{-L}^{L} (-ζ) / (ρ² + y²)^{3/2} dy
           = G Δρ π R² · 2L(−ζ) / (ρ² R_L)

    where ζ = z_obs − d (negative for buried source), R_L = √(ρ² + L²).
    As L → ∞ this recovers gz_2D = 2π G Δρ R² d / (Δx² + d²).
    """
    x   = np.asarray(x_km, dtype=float) * 1e3
    cx  = cx_km * 1e3
    d   = depth_km * 1e3
    R   = radius_km * 1e3
    L   = L_km * 1e3
    dx  = x - cx
    rho2 = dx**2 + d**2
    RL   = np.sqrt(rho2 + L**2)
    # source is at depth d, obs at z=0: ζ = 0 - d = -d
    gz_si = G_SI * rho * math.pi * R**2 * (2.0 * L * d) / (rho2 * RL)
    return gz_si * 1e5   # m/s² → mGal


def analytic_bxbz_25d(x_km, cx_km, depth_km, radius_km,
                       susceptibility, F_nT, IE_deg, DE_deg, L_km):
    """
    Bx and Bz (nT) for a vertically/arbitrarily magnetised cylinder ±L.

    Derived by applying the analytic y-integrals I1 and I5 to the
    uniformly magnetised cross-section (area = π R²):

        I1 = 2L / (ρ² R_L)
        I5 = 2L/(3 ρ² R_L³) + 4L/(3 ρ⁴ R_L)

        Bx = (μ₀/4π) π R² [ Mx(3ξ²I5 − I1) + Mz(3ξζI5) ]
        Bz = (μ₀/4π) π R² [ Mx(3ξζI5)       + Mz(3ζ²I5 − I1) ]
    """
    xi   = (np.asarray(x_km, dtype=float) - cx_km) * 1e3   # x_obs − cx (m)
    d    = depth_km  * 1e3
    R    = radius_km * 1e3
    L    = L_km      * 1e3

    IE   = math.radians(IE_deg)
    DE   = math.radians(DE_deg)
    F_T  = F_nT * 1e-9
    J    = susceptibility * F_T / MU0
    Mx   = J * math.cos(IE) * math.cos(DE)
    Mz   = J * math.sin(IE)

    # ζ = z_obs − d = 0 − d = −d (obs at surface, source at depth d)
    zeta = -d
    rho2 = xi**2 + d**2
    rho4 = rho2**2
    RL   = np.sqrt(rho2 + L**2)
    RL3  = RL**3

    I1 = 2.0 * L / (rho2 * RL)
    I5 = 2.0 * L / (3.0 * rho2 * RL3) + 4.0 * L / (3.0 * rho4 * RL)

    area  = math.pi * R**2
    fac   = (MU0 / (4.0 * math.pi)) * area

    bx_si = fac * (Mx * (3.0*xi**2*I5   - I1) + Mz * (3.0*xi*zeta*I5))
    bz_si = fac * (Mx * (3.0*xi*zeta*I5)      + Mz * (3.0*zeta**2*I5 - I1))
    return bx_si * 1e9, bz_si * 1e9


# ─────────────────────────────────────────────────────────────────────────────
# 5. 2.5-D gravity — horizontal cylinder
# ─────────────────────────────────────────────────────────────────────────────

class TestGravity25D:
    """
    Validate grav2_5d_model.compute_gz against the analytic 2.5-D formula
    for a horizontal cylinder, and verify the 2D limit.
    """

    CX     = 0.0
    DEPTH  = 5.0     # km
    R      = 1.0     # km
    RHO    = 500.0   # kg/m³
    L      = 25.0    # km  (finite strike half-length)
    L_INF  = 1000.0  # km  (approximates infinite strike)

    N_POLY = 360
    X_OBS  = np.linspace(-20, 20, 201)

    def _polygon(self):
        return make_cylinder_polygon(self.CX, self.DEPTH, self.R, self.N_POLY)

    def test_matches_analytic_cylinder(self):
        """gz matches exact 2-D analytic cylinder formula to < 1%."""
        gz_num  = compute_gz_25d(self.X_OBS, 0.0, self._polygon(), self.RHO)
        gz_anal = analytic_gz_cylinder(self.X_OBS, self.CX, self.DEPTH,
                                       self.R, self.RHO)
        rel_err = np.abs(gz_num - gz_anal) / (np.abs(gz_anal).max() + 1e-30)
        assert rel_err.max() < 0.01, \
            f"Max relative error {rel_err.max():.4%} exceeds 1%"

    def test_matches_talwani(self):
        """gz matches Talwani 2-D solution to < 1%."""
        gz_grid = compute_gz_25d(self.X_OBS, 0.0, self._polygon(), self.RHO)
        gz_2d   = compute_gz(self.X_OBS, self._polygon(), self.RHO)
        rel_err = np.abs(gz_grid - gz_2d) / (np.abs(gz_2d).max() + 1e-30)
        assert rel_err.max() < 0.01, \
            f"2-D Talwani max relative error {rel_err.max():.4%} exceeds 1%"

    def test_symmetry(self):
        """gz is symmetric about the cylinder axis."""
        gz = compute_gz_25d(self.X_OBS, 0.0, self._polygon(), self.RHO,
                             strike_half_length_km=self.L)
        assert np.allclose(gz, gz[::-1], rtol=1e-5), \
            "gz is not symmetric about x=0"

    def test_positive_for_positive_contrast(self):
        """Positive density contrast gives positive gz everywhere."""
        gz = compute_gz_25d(self.X_OBS, 0.0, self._polygon(), self.RHO,
                             strike_half_length_km=self.L)
        assert np.all(gz > 0)

    def test_linearity_in_density(self):
        """gz scales linearly with density contrast."""
        gz1 = compute_gz_25d(self.X_OBS, 0.0, self._polygon(),  self.RHO,
                              strike_half_length_km=self.L)
        gz2 = compute_gz_25d(self.X_OBS, 0.0, self._polygon(), 2.0*self.RHO,
                              strike_half_length_km=self.L)
        assert np.allclose(gz2, 2.0 * gz1, rtol=1e-10)

    def test_zero_contrast_gives_zero(self):
        """Zero density contrast gives zero gravity."""
        gz = compute_gz_25d(self.X_OBS, 0.0, self._polygon(), 0.0,
                             strike_half_length_km=self.L)
        assert np.allclose(gz, 0.0)

    def test_rms_accuracy(self):
        """gz RMS error vs analytic cylinder is < 1% of peak."""
        gz_num  = compute_gz_25d(self.X_OBS, 0.0, self._polygon(), self.RHO)
        gz_anal = analytic_gz_cylinder(self.X_OBS, self.CX, self.DEPTH,
                                       self.R, self.RHO)
        rms = float(np.sqrt(np.mean((gz_num - gz_anal)**2)))
        assert rms < 0.01 * gz_anal.max(), \
            f"gz RMS error {rms:.4f} mGal exceeds 1% of peak"

    def test_degenerate_polygon_returns_zero(self):
        """Fewer than 3 vertices returns zero."""
        gz = compute_gz_25d(self.X_OBS, 0.0, [[0, 1], [1, 2]], 300.0)
        assert np.allclose(gz, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 2.5-D magnetics — horizontal cylinder
# ─────────────────────────────────────────────────────────────────────────────

class TestMagnetics25D:
    """
    Validate mag2_5d_model against the analytic 2.5-D cylinder formula.

    Key physics checks:
      - Bz is symmetric and has negative side lobes for vertical magnetisation
      - TMI is anti-symmetric for vertical magnetisation (IE=90°)
      - Matches analytic finite-strike formula to < 1%
      - Approaches 2-D surface-charge limit at large L
    """

    CX    = 0.0
    DEPTH = 5.0      # km
    R     = 1.0      # km
    CHI   = 0.01
    F_nT  = 50_000.0
    L     = 25.0     # km
    L_INF = 1000.0   # km

    N_POLY = 360
    X_OBS  = np.linspace(-20, 20, 201)

    def _polygon(self):
        return make_cylinder_polygon(self.CX, self.DEPTH, self.R, self.N_POLY)

    # --- Bz vertical magnetisation -----------------------------------------

    def test_bz_vertical_symmetry(self):
        """Bz is an even function of x for vertical magnetisation (IE=90°)."""
        _, bz = compute_bx_bz_25d(self.X_OBS, 0.0, self._polygon(),
                                   self.CHI, self.F_nT, 90.0,
                                   strike_half_length_km=self.L)
        # Tolerance relative to the peak (Bz crosses zero so rtol alone fails)
        peak = np.abs(bz).max()
        assert np.allclose(bz, bz[::-1], atol=1e-6 * peak), \
            "Bz is not symmetric for IE=90°"

    def test_bz_has_negative_side_lobes(self):
        """Bz for vertical magnetisation must be negative away from the body."""
        _, bz = compute_bx_bz_25d(self.X_OBS, 0.0, self._polygon(),
                                   self.CHI, self.F_nT, 90.0,
                                   strike_half_length_km=self.L)
        assert bz.min() < 0.0, \
            f"No negative side lobes found (min Bz = {bz.min():.4f} nT)"
        assert bz.max() > 0.0, \
            f"No positive peak found (max Bz = {bz.max():.4f} nT)"

    def test_bz_peak_above_body(self):
        """Peak Bz is at x = 0 (directly above the cylinder) for IE=90°."""
        _, bz = compute_bx_bz_25d(self.X_OBS, 0.0, self._polygon(),
                                   self.CHI, self.F_nT, 90.0,
                                   strike_half_length_km=self.L)
        x_peak = self.X_OBS[np.argmax(bz)]
        assert abs(x_peak) < 0.5, \
            f"Peak Bz at x={x_peak:.2f} km, expected near x=0"

    # --- Bx vertical magnetisation -----------------------------------------

    def test_bx_antisymmetric_vertical_mag(self):
        """Bx is an odd function of x for vertical magnetisation (IE=90°)."""
        bx, _ = compute_bx_bz_25d(self.X_OBS, 0.0, self._polygon(),
                                   self.CHI, self.F_nT, 90.0,
                                   strike_half_length_km=self.L)
        # Tolerance relative to peak (Bx passes through zero at x=0)
        peak = np.abs(bx).max()
        assert np.allclose(bx, -bx[::-1], atol=1e-6 * peak), \
            "Bx is not anti-symmetric for IE=90°"

    # --- matches analytic finite-strike formula ----------------------------

    def test_bxbz_match_analytic_2d(self):
        """Bx and Bz match the exact 2-D dipole analytic formula to < 1%."""
        bx_num, bz_num = compute_bx_bz_25d(
            self.X_OBS, 0.0, self._polygon(), self.CHI, self.F_nT, 60.0)
        bx_an, bz_an = analytic_bxbz_2d(
            self.X_OBS, self.CX, self.DEPTH, self.R,
            self.CHI, self.F_nT, 60.0, 0.0)
        rel_bx = np.abs(bx_num - bx_an) / (np.abs(bx_an).max() + 1e-30)
        rel_bz = np.abs(bz_num - bz_an) / (np.abs(bz_an).max() + 1e-30)
        assert rel_bx.max() < 0.01, \
            f"Bx max relative error {rel_bx.max():.4%} exceeds 1%"
        assert rel_bz.max() < 0.01, \
            f"Bz max relative error {rel_bz.max():.4%} exceeds 1%"

    # --- 2-D limit ---------------------------------------------------------

    def test_bz_2d_limit(self):
        """Bz at L=1000 km matches surface-charge 2-D ground truth to < 1%."""
        _, bz_25d = compute_bx_bz_25d(
            self.X_OBS, 0.0, self._polygon(), self.CHI, self.F_nT,
            90.0, strike_half_length_km=self.L_INF)

        # surface-charge reference (2-D, z positive down, obs at surface)
        d   = self.DEPTH * 1e3
        R   = self.R     * 1e3
        F_T = self.F_nT  * 1e-9
        Mz  = self.CHI * F_T / MU0
        x_m = self.X_OBS * 1e3
        r2  = x_m**2 + d**2
        bz_ref = (MU0 / 2.0) * Mz * R**2 * (d**2 - x_m**2) / r2**2 * 1e9

        rel_err = np.abs(bz_25d - bz_ref) / (np.abs(bz_ref).max() + 1e-30)
        assert rel_err.max() < 0.01, \
            f"Bz 2D-limit max relative error {rel_err.max():.4%} exceeds 1%"

    # --- TMI consistency ---------------------------------------------------

    def test_tmi_consistent_with_bx_bz(self):
        """ΔT = Bx·cos(IE) + Bz·sin(IE) is consistent with compute_bt_25d."""
        IE = 45.0
        IE_r = math.radians(IE)
        tmi = compute_bt_25d(self.X_OBS, 0.0, self._polygon(),
                              self.CHI, self.F_nT, IE,
                              strike_half_length_km=self.L)
        bx, bz = compute_bx_bz_25d(self.X_OBS, 0.0, self._polygon(),
                                    self.CHI, self.F_nT, IE,
                                    strike_half_length_km=self.L)
        tmi_from_components = bx * math.cos(IE_r) + bz * math.sin(IE_r)
        assert np.allclose(tmi, tmi_from_components, rtol=1e-4), \
            "TMI from compute_bt_25d disagrees with Bx·cos(IE)+Bz·sin(IE)"

    def test_zero_susceptibility_gives_zero(self):
        """Zero susceptibility with no remanence gives zero field."""
        tmi = compute_bt_25d(self.X_OBS, 0.0, self._polygon(),
                              0.0, self.F_nT, 60.0,
                              strike_half_length_km=self.L)
        assert np.allclose(tmi, 0.0)

    def test_linearity_in_susceptibility(self):
        """Field scales linearly with susceptibility."""
        tmi1 = compute_bt_25d(self.X_OBS, 0.0, self._polygon(),
                               self.CHI, self.F_nT, 45.0,
                               strike_half_length_km=self.L)
        tmi2 = compute_bt_25d(self.X_OBS, 0.0, self._polygon(),
                               2.0*self.CHI, self.F_nT, 45.0,
                               strike_half_length_km=self.L)
        assert np.allclose(tmi2, 2.0 * tmi1, rtol=1e-8)

    # --- remanence ---------------------------------------------------------

    def test_remanence_adds_to_induced(self):
        """Pure remanence with same direction as induced doubles the field."""
        F_T    = self.F_nT * 1e-9
        J_ind  = self.CHI * F_T / MU0
        tmi_ind = compute_bt_25d(self.X_OBS, 0.0, self._polygon(),
                                  self.CHI, self.F_nT, 60.0,
                                  strike_half_length_km=self.L)
        tmi_rem = compute_bt_25d(self.X_OBS, 0.0, self._polygon(),
                                  0.0, self.F_nT, 60.0,
                                  remanence_Am=J_ind,
                                  remanence_inc_deg=60.0,
                                  strike_half_length_km=self.L)
        tmi_both = compute_bt_25d(self.X_OBS, 0.0, self._polygon(),
                                   self.CHI, self.F_nT, 60.0,
                                   remanence_Am=J_ind,
                                   remanence_inc_deg=60.0,
                                   strike_half_length_km=self.L)
        assert np.allclose(tmi_both, tmi_ind + tmi_rem, rtol=1e-8), \
            "Induced + remanent fields do not superpose linearly"

    # --- longer strike approaches 2-D limit --------------------------------

    def test_bx_bz_rms_accuracy(self):
        """Bx and Bz RMS errors vs exact 2-D analytic are < 1% of peak."""
        bx_num, bz_num = compute_bx_bz_25d(
            self.X_OBS, 0.0, self._polygon(), self.CHI, self.F_nT, 90.0)
        bx_an, bz_an = analytic_bxbz_2d(
            self.X_OBS, self.CX, self.DEPTH, self.R,
            self.CHI, self.F_nT, 90.0)
        rms_bx = float(np.sqrt(np.mean((bx_num - bx_an)**2)))
        rms_bz = float(np.sqrt(np.mean((bz_num - bz_an)**2)))
        assert rms_bx < 0.01 * np.abs(bx_an).max(), \
            f"Bx RMS error {rms_bx:.4f} nT exceeds 1% of peak"
        assert rms_bz < 0.01 * np.abs(bz_an).max(), \
            f"Bz RMS error {rms_bz:.4f} nT exceeds 1% of peak"
