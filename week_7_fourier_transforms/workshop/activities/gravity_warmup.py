"""
Week 7 warm-up: "Two Bumps, One Line"
--------------------------------------
Builds two buried-sphere gravity anomalies at different depths, sums them
into a single "mystery" profile, and reveals the components + their
amplitude spectra.

Physics: point-mass / buried-sphere gravity anomaly along a surface profile,

    dg(x) = (4/3) * pi * G * drho * R^3 * z / (x^2 + z^2)^(3/2)

With R, z in km and drho in g/cm^3, this reduces to the working form

    dg(0) [mGal] = 27.95 * drho * R^3 / z^2

and FWHM = 2*z*sqrt(2^(2/3) - 1) ~= 1.53 * z

Both facts are used below only to choose R for a target peak amplitude;
everything else is computed directly from the exact analytic profile.

Edit SHALLOW / DEEP below to retune the reveal. Two knobs matter most:
  - z (depth) controls wavelength content (this is the whole point)
  - R, drho control amplitude (kept comparable on purpose so the summed
    curve doesn't visually telegraph "two sources")
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Source parameters — edit freely
# ----------------------------------------------------------------------
G = 6.674e-11  # m^3 kg^-1 s^-2 (not used directly; kept for reference)

SHALLOW = dict(z=1.0, R=0.71, drho=0.25, shift=2.0,label="Shallow source")   # km, km, g/cc
DEEP    = dict(z=8.0, R=3.37, drho=0.4, shift=-3.0, label="Deep source")      # km, km, g/cc

X_HALF_WIDTH = 30.0   # km, profile extends from -X_HALF_WIDTH to +X_HALF_WIDTH
N_POINTS = 2048        # samples along the profile (power of 2 keeps FFT tidy)


def buried_sphere_anomaly(x_km, z_km, R_km, drho_gcc, x_shift=0.0):
    """
    Exact buried-sphere (point-mass equivalent) gravity anomaly, in mGal.

    x_km : array of station positions (km), source at x = 0
    z_km : depth to sphere centre (km)
    R_km : sphere radius (km)
    drho_gcc : density contrast (g/cm^3)
    """
    # 27.95 mGal * (drho[g/cc] * R[km]^3) / z[km]^2 at x = 0, then apply
    # the standard (1 + (x/z)^2)^(-3/2) shape factor.
    peak = 27.95 * drho_gcc * R_km**3 / z_km**2
    return peak / (1.0 + ((x_km - x_shift) / z_km) ** 2) ** 1.5


def amplitude_spectrum(profile, dx_km):
    """
    One-sided amplitude spectrum vs wavelength (km).
    Returns wavelength array (longest first) and normalized amplitude.
    """
    n = len(profile)
    spec = np.abs(np.fft.rfft(profile))
    freq = np.fft.rfftfreq(n, d=dx_km)  # cycles / km
    # Skip the zero-frequency (DC) term; convert freq -> wavelength
    wavelength = np.full_like(freq, np.inf)
    wavelength[1:] = 1.0 / freq[1:]
    spec_norm = spec / spec.max()
    return wavelength, spec_norm


def main():
    x = np.linspace(-X_HALF_WIDTH, X_HALF_WIDTH, N_POINTS)
    dx = x[1] - x[0]

    g_shallow = buried_sphere_anomaly(x, SHALLOW["z"], SHALLOW["R"], SHALLOW["drho"], x_shift=SHALLOW["shift"])
    g_deep    = buried_sphere_anomaly(x, DEEP["z"], DEEP["R"], DEEP["drho"], x_shift=DEEP["shift"])
    g_sum     = g_shallow + g_deep

    print(f"Shallow peak: {g_shallow.max():.2f} mGal, "
          f"FWHM ~= {1.53*SHALLOW['z']:.2f} km")
    print(f"Deep peak:    {g_deep.max():.2f} mGal, "
          f"FWHM ~= {1.53*DEEP['z']:.2f} km")

    # ------------------------------------------------------------------
    # Figure 1: the "mystery" curve only — this is what goes on the
    # handout BEFORE the reveal. No labels beyond axes.
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6.5, 3.2))
    ax1.plot(x, g_sum, color="black", linewidth=1.8)
    ax1.set_xlabel("Distance (km)")
    ax1.set_ylabel("Gravity anomaly (mGal)")
    ax1.axhline(0, color="grey", linewidth=0.6)
    ax1.set_title("Mystery profile")
    fig1.tight_layout()
    fig1.savefig("fig_mystery.pdf")
    fig1.savefig("fig_mystery.png", dpi=200)

    # ------------------------------------------------------------------
    # Figure 2: the reveal — components, sum, and spectra side by side.
    # This is what goes on the handout AFTER discussion.
    # ------------------------------------------------------------------
    wl_shallow, amp_shallow = amplitude_spectrum(g_shallow, dx)
    wl_deep, amp_deep = amplitude_spectrum(g_deep, dx)
    wl_sum, amp_sum = amplitude_spectrum(g_sum, dx)

    fig2, axes = plt.subplots(2, 1, figsize=(6.5, 6.4))

    ax = axes[0]
    ax.plot(x, g_sum, color="black", linewidth=1.8, label="Sum (what you saw)")
    ax.plot(x, g_shallow, color="tab:red", linewidth=1.3, linestyle="--",
            label=f"{SHALLOW['label']} (z = {SHALLOW['z']:.0f} km)")
    ax.plot(x, g_deep, color="tab:blue", linewidth=1.3, linestyle="--",
            label=f"{DEEP['label']} (z = {DEEP['z']:.0f} km)")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Gravity anomaly (mGal)")
    ax.set_title("Components")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    # Only plot out to a sensible wavelength range; drop the inf (DC) point
    mask_s = np.isfinite(wl_shallow) & (wl_shallow <= 2 * X_HALF_WIDTH)
    mask_d = np.isfinite(wl_deep) & (wl_deep <= 2 * X_HALF_WIDTH)
    mask_t = np.isfinite(wl_sum) & (wl_sum <= 2 * X_HALF_WIDTH)
    ax.plot(wl_shallow[mask_s], amp_shallow[mask_s], color="tab:red",
            linewidth=1.5, label=SHALLOW["label"])
    ax.plot(wl_deep[mask_d], amp_deep[mask_d], color="tab:blue",
            linewidth=1.5, label=DEEP["label"])
    ax.plot(wl_sum[mask_t], amp_sum[mask_t], color="black",
            linewidth=1.5, label="Sum")
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Amplitude spectra")
    ax.legend(fontsize=8)

    fig2.tight_layout()
    fig2.savefig("fig_reveal.pdf")
    fig2.savefig("fig_reveal.png", dpi=200)

    print("Wrote fig_mystery.{pdf,png} and fig_reveal.{pdf,png}")


if __name__ == "__main__":
    main()
