"""Signal generation for the Fourier Transform teaching tool.

FS = N = 1024 gives df = 1 Hz exactly, so named frequencies (5, 8, 20, 50 Hz)
fall exactly on FFT bins — no spectral leakage for the clean components.
"""
import numpy as np

FS   = 1024.0   # sample rate (Hz)
N    = 1024     # number of samples  →  duration = 1 s, df = 1 Hz

PRESET_NAMES = [
    'Default (mixed)',
    'Single sine',
    'Multi-tone',
    'Square wave',
    'Impulse',
    'Noise only',
    'Chirp',
]

# Fixed noise template so amplitude slider never changes the realization
_rng   = np.random.default_rng(42)
_NOISE = _rng.standard_normal(N)


def get_time():
    return np.arange(N) / FS


def get_freqs():
    return np.fft.rfftfreq(N, d=1.0 / FS)


def _square(t, freq):
    return np.sign(np.sin(2 * np.pi * freq * t))


def _chirp(t, f0=5.0, f1=200.0):
    k = (f1 - f0) / t[-1]
    return np.sin(2 * np.pi * (f0 + 0.5 * k * t) * t)


def make_signal(preset='Default (mixed)', noise_level=0.2):
    """Return (t, signal, components) where components = [(label, array), ...]."""
    t = get_time()
    noise = noise_level * _NOISE

    if preset == 'Default (mixed)':
        comps = [
            ('5 Hz sine',    np.sin(2*np.pi*5*t)),
            ('20 Hz sine',   0.5 * np.sin(2*np.pi*20*t)),
            ('50 Hz sine',   0.3 * np.sin(2*np.pi*50*t)),
            ('8 Hz square',  0.4 * _square(t, 8)),
            ('Noise',        noise),
        ]
    elif preset == 'Single sine':
        comps = [
            ('10 Hz sine', np.sin(2*np.pi*10*t)),
            ('Noise',      noise),
        ]
    elif preset == 'Multi-tone':
        comps = [
            ('10 Hz',  np.sin(2*np.pi*10*t)),
            ('30 Hz',  0.7 * np.sin(2*np.pi*30*t)),
            ('70 Hz',  0.5 * np.sin(2*np.pi*70*t)),
            ('150 Hz', 0.3 * np.sin(2*np.pi*150*t)),
            ('Noise',  noise),
        ]
    elif preset == 'Square wave':
        comps = [
            ('8 Hz square', _square(t, 8)),
            ('Noise',       noise),
        ]
    elif preset == 'Impulse':
        sig = np.zeros(N)
        sig[N // 2] = 1.0
        comps = [('Impulse', sig)]
    elif preset == 'Noise only':
        comps = [('Noise', noise_level * _NOISE)]
    elif preset == 'Chirp':
        comps = [
            ('Chirp', _chirp(t)),
            ('Noise', noise),
        ]
    else:
        comps = [('10 Hz sine', np.sin(2*np.pi*10*t))]

    signal = sum(arr for _, arr in comps)
    return t, signal, comps
