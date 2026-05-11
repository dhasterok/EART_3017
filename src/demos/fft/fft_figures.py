# Figure 2: Amplitude vs Phase
# Demonstrates that phase controls geometry, not amplitude

import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Domain
N = 2048
x = np.linspace(0, 1, N, endpoint=False)

# Construct a multi-frequency signal
frequencies = [5, 11, 23]
signal = np.zeros_like(x)
for f in frequencies:
    signal += np.cos(2 * np.pi * f * x)

# Fourier transform
F = np.fft.fft(signal)

# Extract amplitude spectrum
amplitude = np.abs(F)

# Construct new spectrum with identical amplitude but random phase
random_phase = np.exp(1j * np.random.uniform(0, 2*np.pi, N))
F_random_phase = amplitude * random_phase

# Inverse transforms
signal_original = np.real(np.fft.ifft(F))
signal_phase_randomized = np.real(np.fft.ifft(F_random_phase))

# Plot
plt.figure(figsize=(10, 4))
plt.plot(x, signal_original,
         label='Original signal', linewidth=2)
plt.plot(x, signal_phase_randomized,
         '--', label='Same amplitude, different phase', linewidth=2)

plt.xlabel('x')
plt.ylabel('Amplitude')
plt.title('Amplitude vs Phase: Identical Amplitude Spectrum, Different Signals')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 4: Convolution as Kernel Averaging
# Illustrates convolution as a weighted local averaging operation


# Spatial domain
x = np.linspace(-5, 5, 800)

# Define a signal: smooth background + sharp localized feature
signal = np.exp(-x**2) + 0.6 * np.exp(-(x - 2.0)**2 / 0.1)

# Define a Gaussian smoothing kernel
sigma = 0.4
kernel = np.exp(-(x**2) / (2 * sigma**2))
kernel /= kernel.sum()  # Normalize for averaging

# Perform convolution
convolved = np.convolve(signal, kernel, mode='same')

# Plot
plt.figure(figsize=(10, 4))
plt.plot(x, signal, label='Original signal', linewidth=2)
plt.plot(x, kernel, linestyle=':', label='Kernel (Gaussian)')
plt.plot(x, convolved, label='Convolved signal', linewidth=2)

plt.xlabel('x')
plt.ylabel('Amplitude')
plt.title('Convolution as Kernel Averaging')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 5: Filter Shape vs Convolution Kernel Shape cutoffs produce ringing kernels,
# while smooth filters produce localized kernels.

# Parameters
N = 2048                     # number of samples
dx = 1.0                     # sample spacing
k = np.fft.fftfreq(N, d=dx)  # frequency / wavenumber axis

# ----- Spectral filters -----

# Ideal (brick-wall) low-pass filter
kc = 0.05
H_ideal = np.abs(k) <= kc

# Gaussian low-pass filter
H_gaussian = np.exp(-(k / kc)**2)

# ----- Corresponding kernels (inverse FFT) -----

kernel_ideal = np.real(np.fft.ifft(H_ideal))
kernel_gaussian = np.real(np.fft.ifft(H_gaussian))

# Shift kernels for plotting
kernel_ideal = np.fft.fftshift(kernel_ideal)
kernel_gaussian = np.fft.fftshift(kernel_gaussian)

# Spatial axis for kernels
x = np.linspace(-N/2, N/2, N) * dx

# ----- Plot -----

plt.figure(figsize=(12, 4))

# (a) Spectral filters
plt.subplot(1, 2, 1)
plt.plot(k, H_ideal, label='Ideal low-pass', linewidth=2)
plt.plot(k, H_gaussian, label='Gaussian low-pass', linewidth=2)
plt.xlabel('Frequency / wavenumber')
plt.ylabel('Amplitude')
plt.title('Spectral Filters')
plt.legend()
plt.grid(True, alpha=0.3)

# (b) Spatial kernels
plt.subplot(1, 2, 2)
plt.plot(x, kernel_ideal, label='Ideal-filter kernel', linewidth=2)
plt.plot(x, kernel_gaussian, label='Gaussian-filter kernel', linewidth=2)
plt.xlim(-100, 100)  # zoom to show structure clearly
plt.xlabel('Space / time')
plt.ylabel('Amplitude')
plt.title('Convolution Kernels')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

