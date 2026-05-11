"""Filter construction and application for the Fourier Transform teaching tool."""
import numpy as np

FILTER_TYPES = ['None', 'Low-pass', 'High-pass', 'Band-pass', 'Band-stop', 'Custom']
TAPER_TYPES  = ['None', 'Cosine', 'Hanning', 'Hamming', 'Gaussian']
PHASE_MODES  = ['Normal', 'Randomize phase', 'Flatten amplitude', 'Zero phase']

_phase_rng = np.random.default_rng(99)   # deterministic randomized-phase realisation


# ── Taper shapes ──────────────────────────────────────────────────────────────

def _taper_shape(x, taper):
    """Smooth 0→1 transition for x ∈ [0, 1]."""
    if taper in ('Cosine', 'Hanning'):
        return 0.5 * (1.0 - np.cos(np.pi * x))
    if taper == 'Hamming':
        w = 0.54 - 0.46 * np.cos(np.pi * x)
        # normalise so it spans [0, 1]
        return (w - 0.08) / 0.92
    if taper == 'Gaussian':
        # Logistic sigmoid — visually indistinguishable from Gaussian CDF
        return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))
    return x   # linear fallback


# ── Single edge ───────────────────────────────────────────────────────────────

def _edge(freqs, fc, taper, taper_width, rising):
    """
    Build one filter edge at fc.
    rising=True  → passband is f ≥ fc (high-pass sense).
    rising=False → passband is f <  fc (low-pass  sense).
    """
    nf = len(freqs)
    df = freqs[1] - freqs[0] if nf > 1 else 1.0
    H  = np.zeros(nf)

    if taper == 'None':
        H[freqs >= fc] = 1.0 if rising else 0.0
        H[freqs <  fc] = 0.0 if rising else 1.0
        return H

    tw   = max(taper_width, df)
    f_lo = fc - tw / 2.0
    f_hi = fc + tw / 2.0

    # Flat flanks
    if rising:
        H[freqs >= f_hi] = 1.0
    else:
        H[freqs <  f_lo] = 1.0

    # Transition band
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if mask.any():
        x = (freqs[mask] - f_lo) / (f_hi - f_lo)
        t = _taper_shape(x, taper)
        H[mask] = t if rising else (1.0 - t)

    return H


# ── Public API ────────────────────────────────────────────────────────────────

def make_filter(freqs, ftype, fc1, fc2=None, taper='None', taper_width=5.0):
    """
    Build a real-valued filter H(f) ∈ [0, 1].

    Parameters
    ----------
    freqs       : 1-D array of positive frequencies from rfftfreq
    ftype       : one of FILTER_TYPES
    fc1         : primary cutoff (Hz)
    fc2         : secondary cutoff for band filters (Hz)
    taper       : one of TAPER_TYPES
    taper_width : full width of the transition band (Hz)
    """
    nf = len(freqs)

    if ftype == 'None' or ftype == 'Custom':
        return np.ones(nf)

    if ftype == 'Low-pass':
        return _edge(freqs, fc1, taper, taper_width, rising=False)

    if ftype == 'High-pass':
        return _edge(freqs, fc1, taper, taper_width, rising=True)

    if fc2 is None:
        fc2 = min(fc1 * 2.0, freqs[-1])
    fc1, fc2 = min(fc1, fc2), max(fc1, fc2)

    if ftype == 'Band-pass':
        return (_edge(freqs, fc1, taper, taper_width, rising=True) *
                _edge(freqs, fc2, taper, taper_width, rising=False))

    if ftype == 'Band-stop':
        return np.clip(
            _edge(freqs, fc1, taper, taper_width, rising=False) +
            _edge(freqs, fc2, taper, taper_width, rising=True),
            0.0, 1.0)

    return np.ones(nf)


def apply_filter(F, H, phase_mode='Normal', phase_shift_rad=0.0):
    """
    Apply filter H and optional phase manipulation to rfft spectrum F.

    Returns the modified complex spectrum.
    """
    amp   = np.abs(F)
    phase = np.angle(F)
    phs   = phase_shift_rad

    if phase_mode == 'Randomize phase':
        rnd = _phase_rng.uniform(-np.pi, np.pi, size=F.shape)
        return H * amp * np.exp(1j * (rnd + phs))

    if phase_mode == 'Flatten amplitude':
        # Equal amplitudes, preserve phase structure
        flat = np.where(amp > 1e-12, 1.0, 0.0)
        return H * flat * np.exp(1j * (phase + phs))

    if phase_mode == 'Zero phase':
        return (H * amp + 0j)   # real → zero phase

    # Normal
    return H * F * np.exp(1j * phs)
