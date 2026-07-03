import numpy as np
from scipy.special import erf

def lognormal_mu_sigma_from_mean_std(m: float, s: float):
    m = max(1e-9, float(m))
    s = max(1e-12, float(s))
    sigma2 = float(np.log(1.0 + (s*s)/(m*m)))
    sigma = float(np.sqrt(sigma2))
    mu = float(np.log(m)) - 0.5*sigma2
    return mu, sigma

def moving_average_2d(arr: np.ndarray, win: int, iters: int = 1) -> np.ndarray:
    if win <= 1:
        return arr
    a = arr.copy()
    H, W = a.shape
    for _ in range(iters):
        py = px = win//2
        p = np.pad(a, ((py,py),(px,px)), mode='reflect')
        ii = p.cumsum(0).cumsum(1)
        y2 = np.arange(win, H+win)
        x2 = np.arange(win, W+win)
        y1 = y2 - win
        x1 = x2 - win
        s = ii[y2[:,None], x2] - ii[y1[:,None], x2] - ii[y2[:,None], x1] + ii[y1[:,None], x1]
        a = s / float(win*win)
    return a

# -------------------- Random granular --------------------
def build_random_field(H: int, W: int, fracs, mus, sigs, corr_len: int, sigma_ln_noise: float):
    K = len(fracs)
    comps = np.random.choice(K, size=(H,W), p=np.array(fracs))
    if corr_len and corr_len > 1:
        score = []
        for i in range(K):
            m = (comps==i).astype(float)
            sc = moving_average_2d(m, corr_len, 1)
            score.append(sc)
        score = np.stack(score, axis=-1)
        comps = np.argmax(score, axis=-1)
    k = np.zeros((H,W), float)
    for i in range(K):
        mu, sig = mus[i], sigs[i]
        mask = (comps==i)
        if mask.any():
            lnk = np.random.normal(mu, sig, size=mask.sum())
            if sigma_ln_noise>0:
                lnk += np.random.normal(0, sigma_ln_noise, size=mask.sum())
            k[mask] = np.exp(lnk)
    return k, comps.astype(np.uint8)

# -------------------- Layer builders --------------------

def _draw_thicknesses(rng, N, target_sum, CV, dist):
    if N <= 0:
        return np.array([])
    if CV <= 1e-12:
        t = np.ones(N)
    else:
        if dist == "lognormal":
            s = np.sqrt(np.log(1.0 + CV**2))
            m = -0.5*s**2
            t = np.exp(rng.normal(m, s, size=N))
        else:
            a = np.sqrt(3.0) * CV
            t = np.clip(1.0 + rng.uniform(-a, +a, size=N), 1e-3, None)
    ssum = t.sum()
    if ssum>0:
        t *= (target_sum/ssum)
    return t

def build_two_component_layers(
    H, W,
    f_A=0.5,
    N_A=1, N_B=2,
    CV_thick=0.0, dist_thick="lognormal",
    w_int=0.0,
    p_cont=1.0, L_gap=10,
    A_meander=0.0, L_meander=999999,
    muA=0.0, sigA=0.3,
    muB=-0.4, sigB=0.3,
    sigma_ln_noise=0.0,
    rng=None
):
    """Series-oriented layering (layers ⟂ flow). Parallel achieved by rotation."""
    if rng is None:
        rng = np.random.default_rng()
    f_B = 1.0 - f_A
    N_A = max(1, int(N_A)); N_B = max(2, int(N_B))
    tA = _draw_thicknesses(rng, N_A, f_A*H, CV_thick, dist_thick)
    tB = _draw_thicknesses(rng, N_B, f_B*H, CV_thick, dist_thick)

    # Interleave starting with larger fraction
    seq, iA, iB, turnA = [], 0, 0, (f_A >= f_B)
    while iA < N_A or iB < N_B:
        if turnA and iA < N_A: seq.append(("A", tA[iA])); iA += 1
        elif (not turnA) and iB < N_B: seq.append(("B", tB[iB])); iB += 1
        turnA = not turnA
    for j in range(iA, N_A): seq.append(("A", tA[j]))
    for j in range(iB, N_B): seq.append(("B", tB[j]))

    # Straight cumulative edges
    y_edges = [0.0]
    for _, th in seq: y_edges.append(y_edges[-1] + th)
    y_edges[-1] = H

    # Sinusoidal interfaces — all share one phase so peaks align across layers
    x = np.arange(W)
    n_if = len(y_edges) - 2          # number of interior interfaces
    phase = np.random.uniform(0, 2*np.pi) if A_meander > 0 and L_meander > 0 else 0.0
    wave  = A_meander * np.sin(2*np.pi*x / max(1, L_meander) + phase) if A_meander > 0 and L_meander > 0 else np.zeros(W)
    y_if  = [np.full(W, y_edges[i + 1]) + wave for i in range(n_if)]

    # Sinusoidal outer boundaries: fold the top (y=0) and bottom (y=H) edges
    # so each component bleeds onto the far edge of the other.
    y_top = wave.copy()              # y=0 + same wave
    y_bot = wave + H                 # y=H + same wave

    # All interfaces used for erf softening (outer + interior)
    y_if_all = [y_top] + y_if + [y_bot]

    # Assign labels by locating each pixel between its bounding interfaces.
    # Work backward: start with every pixel in the last layer, then override
    # pixels above each interface with the layer above it.
    yy = np.arange(H)[:, None]      # (H, 1) — broadcast against (1, W)
    layer_idx = np.full((H, W), len(seq) - 1, dtype=int)
    for i in range(len(y_if) - 1, -1, -1):
        layer_idx[yy < y_if[i][None, :]] = i

    # Wrap outer boundaries: pixels above the folded top edge belong to the
    # last layer; pixels below the folded bottom edge belong to the first.
    layer_idx[yy < y_top[None, :]] = len(seq) - 1
    layer_idx[yy >= y_bot[None, :]] = 0

    label = np.zeros((H, W), np.uint8)
    for i, (lab, _) in enumerate(seq):
        label[layer_idx == i] = 1 if lab == 'A' else 0

    # Discontinuity leaks (use straight row ranges as an approximation)
    if p_cont < 1.0:
        for i,(lab,th) in enumerate(seq):
            if np.random.random() > p_cont:
                ngaps = np.random.randint(1,4)
                r0,r1 = int(round(y_edges[i])), int(round(y_edges[i+1]))
                r0, r1 = max(0, r0), min(H, r1)
                for _ in range(ngaps):
                    glen = max(1, int(np.random.normal(L_gap, 0.3*L_gap)))
                    cx = np.random.randint(0, max(1, W-glen))
                    label[r0:r1, cx:cx+glen] = 1 - (1 if lab=='A' else 0)

    # Soft alpha via erf distance to nearest interface (including outer boundaries)
    if w_int <= 1e-12 or not y_if_all:
        alphaA = label.astype(float)
    else:
        yy = np.arange(H)[:, None] * np.ones((1,W))
        dist = np.full((H,W), np.inf)
        for yi in y_if_all:
            dist = np.minimum(dist, np.abs(yy - yi[None,:]))
        sign = np.where(label>0, 1.0, -1.0)
        d_signed = sign * dist
        alphaA = 0.5*(1.0 + erf(d_signed/(np.sqrt(2.0)*max(1e-9, w_int))))
    alphaB = 1.0 - alphaA

    # ln k fields and geometric blend
    lnkA = np.random.normal(muA, sigA, size=(H, W))
    lnkB = np.random.normal(muB, sigB, size=(H, W))
    if sigma_ln_noise>0:
        noise = np.random.normal(0, sigma_ln_noise, size=(H, W))
        lnkA += noise; lnkB += noise
    lnk = alphaA*lnkA + alphaB*lnkB
    k = np.exp(lnk)
    # comp_img: 0=A, 1=B (hard label; label array has 1=A so we flip)
    comp_img = (1 - label).astype(np.uint8)
    return k, comp_img

def build_layers_ABC_affinity(
    H, W,
    fA=0.45, fB=0.45, fC=0.10,
    N_A=1, N_B=2,
    CV_thick=0.0, dist_thick="lognormal",
    w_int=0.0, p_cont=1.0, L_gap=10,
    A_meander=0.0, L_meander=999999,
    phi_CA=0.8,
    muA=0.0,  sigA=0.3,
    muB=-0.4, sigB=0.3,
    muC=-0.2, sigC=0.35,
    sigma_ln_noise=0.0,
    rng=None,
):
    # 1) Build A/B scaffold to get alphaA (ignore absolute k and comp_img)
    alphaA, _ = build_two_component_layers(
        H, W, f_A=fA, N_A=N_A, N_B=N_B, CV_thick=CV_thick, dist_thick=dist_thick,
        w_int=w_int, p_cont=p_cont, L_gap=L_gap, A_meander=A_meander, L_meander=L_meander,
        muA=0.0, sigA=1e-6, muB=-10.0, sigB=1e-6, sigma_ln_noise=0.0, rng=rng,
    )
    alphaA = np.clip(alphaA, 0.0, 1.0)
    alphaB = 1.0 - alphaA

    # 2) Distribute all of C between A and B regions by affinity.
    # phi_CA is the fraction of C that goes into A-type pixels; (1-phi_CA) goes into B.
    eps = 1e-12
    fC_toA = phi_CA * fC
    fC_toB = (1.0 - phi_CA) * fC
    WA = alphaA.copy(); WB = alphaB.copy()
    SA, SB = WA.sum(), WB.sum()
    if SA > eps: WA *= (fC_toA * H * W) / SA
    else: WA[:] = 0
    if SB > eps: WB *= (fC_toB * H * W) / SB
    else: WB[:] = 0
    alphaC = WA + WB
    overflow = np.maximum(0.0, (alphaA + alphaB + alphaC) - 1.0)
    if np.any(overflow > 0):
        denomAB = alphaA + alphaB + eps
        alphaA -= overflow * (alphaA / denomAB)
        alphaB -= overflow * (alphaB / denomAB)

    # 3) ln k and geometric blend
    lnkA = np.random.normal(muA, sigA, size=(H, W))
    lnkB = np.random.normal(muB, sigB, size=(H, W))
    lnkC = np.random.normal(muC, sigC, size=(H, W))
    if sigma_ln_noise>0:
        noise = np.random.normal(0, sigma_ln_noise, size=(H, W))
        lnkA += noise; lnkB += noise; lnkC += noise
    lnk = alphaA*lnkA + alphaB*lnkB + alphaC*lnkC
    k = np.exp(lnk)
    # comp_img: dominant component per pixel (0=A, 1=B, 2=C)
    comp_img = np.argmax(np.stack([alphaA, alphaB, alphaC], axis=0), axis=0).astype(np.uint8)
    return k, comp_img