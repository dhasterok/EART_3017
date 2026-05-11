import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
nu_true = 1.0          # true frequency: 1 cycle per unit xi
T = 40.0                # duration (units of xi)
n_dense = 5000         # dense sampling for "true" signal
phase_true = 0.0

# Dense grid for true signal
xi_dense = np.linspace(0, T, n_dense)
signal_true = np.sin(2*np.pi*nu_true*xi_dense)

# plot only first 2 units
mask = xi_dense <= 2.0

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def sample_signal(delta_xi, shift):
    """Sample the true signal at interval delta_xi with phase shift."""
    xi = np.arange(0, T, delta_xi) + shift
    y = np.sin(2 * np.pi * nu_true * xi + phase_true)
    return xi, y

def dft_amplitude_phase(xi, y, freq):
    """Recover amplitude and phase at a target frequency using projection."""
    cos_part = np.cos(2 * np.pi * freq * xi)
    sin_part = np.sin(2 * np.pi * freq * xi)

    a = 2 * np.dot(y, cos_part) / len(y)
    b = 2 * np.dot(y, sin_part) / len(y)

    amplitude = np.sqrt(a**2 + b**2)
    phase = np.arctan2(b, a)
    return amplitude, np.degrees(phase)

def dominant_frequency(xi, y):
    """Find dominant frequency via FFT."""
    y = y - np.mean(y)
    n = len(y)
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=(xi[1] - xi[0]))
    return freqs[np.argmax(np.abs(fft))]

# ------------------------------------------------------------
# Panel (b): Sampling interval vs recovered frequency
# ------------------------------------------------------------
delta_xis = np.linspace(0.05, 0.75, 29)
recovered_freqs = []

for dx in delta_xis:
    xi, y = sample_signal(dx, shift=0.0)
    recovered_freqs.append(dominant_frequency(xi, y))

# Nyquist and practical limits
dx_nyquist = 1 / (2 * nu_true)
dx_practical = 1 / (4 * nu_true)

# ------------------------------------------------------------
# Panel (d): Phase envelope vs sampling interval
# ------------------------------------------------------------
_shifts = np.linspace(0, 1.0 / nu_true, 200)   # one full cycle of shifts
amps_min, amps_max = [], []
phase_min, phase_max = [], []

for dx in delta_xis:
    amps = []
    phases = []
    for s in _shifts:
        xi, y = sample_signal(dx, s)
        amp, phase = dft_amplitude_phase(xi, y, nu_true)
        amps.append(amp)
        phases.append(phase)
    amps_min.append(np.min(amps))
    amps_max.append(np.max(amps))
    phase_min.append(np.min(phases))
    phase_max.append(np.max(phases))

phase_min = np.array(phase_min)
phase_max = np.array(phase_max)

# ------------------------------------------------------------
# Panel (c): Phase & amplitude vs sampling shift
# ------------------------------------------------------------
def phase_amp_vs_shift(delta_xi):
    period = 1.0 / nu_true           # one complete cycle of the wave
    shifts = np.linspace(0, period, 200)
    amps, phases = [], []

    for s in shifts:
        xi, y = sample_signal(delta_xi, s)
        amp, phase = dft_amplitude_phase(xi, y, nu_true)
        amps.append(amp)
        phases.append(phase)

    return shifts / period, np.array(amps), np.array(phases)

x_ny, amp_ny, ph_ny = phase_amp_vs_shift(dx_nyquist)
x_2ny, amp_2ny, ph_2ny = phase_amp_vs_shift(dx_nyquist / 2)

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(8, 14))
plt.subplots_adjust(hspace=0.55, top=0.97, bottom=0.07)

def _caption(ax, text, offset=-0.15):
    """Place a panel caption below the axes, like a figure sub-caption."""
    ax.text(0.5, offset, text, transform=ax.transAxes,
            ha='center', va='top', fontsize=10, clip_on=False)

# Panel (a): True signal with oversampled and undersampled examples
axes[0].plot(xi_dense[mask], signal_true[mask], 'k', lw=2, label='True signal')

for dx, color, label in [
    (0.4, 'C0', r'$\Delta\xi = 0.4$ (oversampled)'),
    (0.6, 'C3', r'$\Delta\xi = 0.6$ (undersampled)'),
]:
    xi_s, y_s = sample_signal(dx, shift=0.0)
    in_view = xi_s <= 2.0
    axes[0].plot(xi_s[in_view], y_s[in_view], 'o--', color=color,
                 markersize=5, lw=1.0, label=label)

axes[0].set_ylabel("Amplitude")
axes[0].legend(frameon=False, fontsize=8, loc='upper right')
_caption(axes[0], r"(a) True signal: 1 cycle per unit $\tilde{\xi}$")

# Panel (b): Sampling interval vs recovered frequency
axes[1].plot(delta_xis, recovered_freqs, 'o-', label="Recovered frequency")
axes[1].axhline(nu_true, color='k', ls='--', label="True frequency")
axes[1].axvline(dx_nyquist, color='r', ls=':', label=r"Nyquist, $\nu_{Ny}$")
axes[1].axvline(dx_practical, color='b', ls=':', label="Practical limit")
axes[1].set_xlabel("Sampling interval $\\Delta\\xi$")
axes[1].set_ylabel("Recovered frequency")
_caption(axes[1], "(b) Frequency recovery vs sampling interval", offset=-0.22)
axes[1].legend(frameon=False, loc='lower left', ncol=1)

def _freq_to_wav(f):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(f) > 1e-10, 1.0 / f, np.inf)

ax_wav = axes[1].secondary_yaxis('right', functions=(_freq_to_wav, _freq_to_wav))
ax_wav.set_ylabel("Wavelength ($\\xi$)")

# Panel (c): Amplitude and phase vs sampling shift
ax_amp = axes[2]
ax_phase = ax_amp.twinx()

ax_amp.plot(x_ny, amp_ny, 'r-', label=r"Amplitude ($\nu_{Ny}$)")
ax_amp.plot(x_2ny, amp_2ny, 'r--', label=r"Amplitude ($2\nu_{Ny}$)")
ax_phase.plot(x_ny, ph_ny, 'b-', label=r"Phase ($\nu_{Ny}$)")
ax_phase.plot(x_2ny, ph_2ny, 'b--', label=r"Phase ($2\nu_{Ny}$)")

ax_amp.set_xlabel("Sampling shift $\\delta / \\lambda$")
ax_amp.set_ylabel("Recovered amplitude")
ax_phase.set_ylabel("Recovered phase (degrees)")
_caption(axes[2], "(c) Amplitude and phase sensitivity to sampling shift", offset=-0.22)

# Manual combined legend
lines = ax_amp.get_lines() + ax_phase.get_lines()
labels = [l.get_label() for l in lines]
axes[2].legend(lines, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))

# Panel (d): Amplitude (left) and phase (right) envelopes vs sampling interval
ax_d_amp   = axes[3]
ax_d_phase = ax_d_amp.twinx()

ax_d_amp.fill_between(delta_xis, amps_min, amps_max,
                      color='r', alpha=0.35, label='Amplitude envelope')
ax_d_phase.fill_between(delta_xis, phase_min, phase_max,
                        color='b', alpha=0.35, label='Phase envelope')

ax_d_amp.axvline(dx_nyquist,   color='r', ls=':', label=r"Nyquist, $\nu_{Ny}$")
ax_d_amp.axvline(dx_practical, color='b', ls=':', label="Practical limit")

ax_d_amp.set_xlabel(r"Sampling interval $\Delta\xi$")
ax_d_amp.set_ylabel("Recovered amplitude", color='r')
ax_d_amp.tick_params(axis='y', labelcolor='r')
ax_d_phase.set_ylabel("Recovered phase (degrees)", color='b')
ax_d_phase.tick_params(axis='y', labelcolor='b')

_caption(axes[3], "(d) Amplitude and phase envelopes vs sampling interval", offset=-0.22)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax_d_amp.legend(handles=[
    Patch(facecolor='r', alpha=0.35, label='Amplitude envelope'),
    Patch(facecolor='b', alpha=0.35, label='Phase envelope'),
    Line2D([0], [0], color='r', ls=':', label=r"Nyquist, $\nu_{Ny}$"),
    Line2D([0], [0], color='b', ls=':', label="Practical limit"),
], frameon=False, loc='upper right', ncol=2)

plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('nyquist.svg')
plt.show()
