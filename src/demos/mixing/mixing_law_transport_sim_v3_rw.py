
# -*- coding: utf-8 -*-
"""
Mixing-Law Transport Demonstration (v3, with Biased Random Walk)
----------------------------------------------------------------
Adds a mid-fidelity **biased random walk** (BRW) transport surrogate alongside
Dijkstra shortest-path. This better approximates diffusive transport while
remaining cheap to simulate.

Features inherited from v2:
  • Cell property is conductivity k(x,y); traversal *cost* for Dijkstra = 1/k.
  • Normalize area-mean(k) = 1 for fair comparisons.
  • Generators: random (lognormal, optional correlation), parallel corridors,
    series barriers.
  • Example snapshot/animation + histograms.

New in v3:
  • Biased Random Walk (BRW): walkers start on the left edge (uniform row),
    step to 4-neighbors with probability proportional to neighbor conductivity
    and a rightward drift (beta in [0,1)). Each hop accrues time based on the
    current-cell conductivity (dt = 1/k_current by default). Walk ends upon
    reaching the right edge (absorbing), with optional max_steps to guard
    against pathologically long traces at tiny beta.
  • Batch interface mirrors the v2 helpers for easy comparison.

Usage examples:
  from mixing_law_transport_sim_v3_rw import batch_times_rw, batch_times_dij
  times_rw, ex_rw = batch_times_rw('series', trials=400, beta=0.25)
  times_dj, ex_dj = batch_times_dij('series', trials=400)
  save_example_and_hist(times_rw, ex_rw, 'SERIES_BRW')

Author: updated for BRW by M365 Copilot
"""

import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from matplotlib import animation

# -------------------- Utility --------------------

def normalize_mean_k(k: np.ndarray) -> np.ndarray:
    m = float(k.mean())
    return k if m <= 0 else (k / m)


def moving_average_2d(arr: np.ndarray, win_y=3, win_x=3, iters=1) -> np.ndarray:
    a = arr.copy()
    H, W = a.shape
    for _ in range(iters):
        pad_y, pad_x = win_y//2, win_x//2
        p = np.pad(a, ((pad_y,pad_y),(pad_x,pad_x)), mode='reflect')
        ii = p.cumsum(0).cumsum(1)
        y2 = np.arange(win_y, H+win_y)
        x2 = np.arange(win_x, W+win_x)
        y1 = y2 - win_y
        x1 = x2 - win_x
        s = ii[y2[:,None], x2] - ii[y1[:,None], x2] - ii[y2[:,None], x1] + ii[y1[:,None], x1]
        a = s / float(win_y*win_x)
    return a

# -------------------- Media Generators (k fields) --------------------

def make_random_k(H=60, W=90, logsigma=0.6, corr_len=0):
    """Random granular medium with lognormal k; normalized to mean(k)=1."""
    mu = -0.5*(logsigma**2)
    ln_k = np.random.normal(mu, logsigma, size=(H, W))
    if corr_len and corr_len > 1:
        ln_k = moving_average_2d(ln_k, win_y=corr_len, win_x=corr_len, iters=1)
        ln_k = (ln_k - ln_k.mean())
        sd = ln_k.std() + 1e-12
        ln_k = mu + (logsigma/sd)*ln_k
    k = np.exp(ln_k)
    k *= (0.98 + 0.04*np.random.rand(H, W))
    return normalize_mean_k(k)


def make_parallel_k(H=60, W=90, corridor_fraction=0.25, k_high=3.0, k_low=0.4, jitter=0.1):
    """High-k corridors aligned with flow (left->right). Mean(k) normalized."""
    k = np.full((H, W), k_low, dtype=float)
    num = max(1, int(W * corridor_fraction))
    cols = np.linspace(0, W-1, num, dtype=int)
    for c in cols:
        k[:, c] = k_high
        if c-1 >= 0 and np.random.rand() < 0.5:
            k[:, c-1] = (k_high + k_low) / 2
        if c+1 < W and np.random.rand() < 0.5:
            k[:, c+1] = (k_high + k_low) / 2
    k *= (1.0 - jitter/2.0 + jitter*np.random.rand(H, W))
    return normalize_mean_k(k)


def make_series_k(H=60, W=90, barrier_fraction=0.18, k_high=1.2, k_low=0.2, thickness=1, jitter=0.1):
    """Low-k barriers perpendicular to flow (must be crossed). Mean(k) normalized."""
    k = np.full((H, W), k_high, dtype=float)
    num = max(1, int(H * barrier_fraction))
    rows = np.linspace(3, H-4, num, dtype=int)
    for r in rows:
        k[r:r+thickness, :] = k_low
    k *= (1.0 - jitter/2.0 + jitter*np.random.rand(H, W))
    return normalize_mean_k(k)

# -------------------- Dijkstra Shortest-Path --------------------

def dijkstra_cost(cost: np.ndarray, start, goal):
    H, W = cost.shape
    INF = 1e18
    dist = np.full((H, W), INF, dtype=float)
    prev = np.full((H, W, 2), -1, dtype=int)
    sx, sy = start
    gx, gy = goal
    dist[sx, sy] = cost[sx, sy]
    pq = [(dist[sx, sy], (sx, sy))]
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    while pq:
        d, (x, y) = heappop(pq)
        if (x, y) == (gx, gy):
            break
        if d != dist[x, y]:
            continue
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                nd = d + cost[nx, ny]
                if nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    prev[nx, ny] = [x, y]
                    heappush(pq, (nd, (nx, ny)))
    path = []
    cx, cy = gx, gy
    if dist[gx, gy] >= INF/2:
        return float('inf'), path
    while not (cx == sx and cy == sy):
        path.append((cx, cy))
        px, py = prev[cx, cy]
        cx, cy = px, py
    path.append((sx, sy))
    path.reverse()
    return dist[gx, gy], path

# -------------------- Biased Random Walk (BRW) --------------------

def biased_random_walk_first_passage(
    k: np.ndarray,
    beta: float = 0.2,
    step_time: str = 'inv_k_current',  # 'inv_k_current' or 'const'
    max_steps: int | None = None,
    rng: np.random.Generator | None = None,
    record_path: bool = True,
):
    """Simulate one biased random walk from a uniform-left start to a free right exit.

    Transition weights to 4-neighbors j are:
        w_j = max(0, k[j] * (1 + beta * dx)),  dx = {+1 right, -1 left, 0 up/down}
    Then P(i->j) = w_j / sum(w).

    Time increment per hop:
        'inv_k_current' : dt = 1 / k[current]
        'const'         : dt = 1

    The walk begins at (row ~ U{0..H-1}, col=0) and stops when col==W-1
    (absorbing boundary). Out-of-bounds neighbors are ignored. If all
    weights are numerically zero, falls back to uniform over valid neighbors.

    Returns
    -------
    total_time : float
    path       : list[(r,c)]  (possibly empty if record_path=False)
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = k.shape
    r = int(rng.integers(0, H)); c = 0
    t = 0.0
    path = [(r, c)] if record_path else []

    if max_steps is None:
        max_steps = H * W * 20  # generous cap for small beta / tortuous media

    # neighbor direction vectors and their dx for bias
    neigh = [(1,0,0), (-1,0,0), (0,1,1), (0,-1,-1)]  # (dr, dc, dx)

    for _ in range(max_steps):
        if c == W-1:
            break
        # collect valid neighbors and weights
        nbrs = []
        weights = []
        for dr, dc, dx in neigh:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                w = k[nr, nc] * (1.0 + beta * dx)
                if w < 0:
                    w = 0.0
                nbrs.append((nr, nc))
                weights.append(w)
        if not nbrs:
            # Stuck (shouldn't happen in normal grids)
            break
        ws = np.array(weights, dtype=float)
        s = ws.sum()
        if s <= 1e-15:
            # fallback: uniform over neighbors
            probs = np.full(len(nbrs), 1.0/len(nbrs))
        else:
            probs = ws / s
        # sample next
        idx = rng.choice(len(nbrs), p=probs)
        nr, nc = nbrs[idx]
        # time increment
        if step_time == 'inv_k_current':
            dt = 1.0 / (k[r, c] + 1e-12)
        else:
            dt = 1.0
        t += dt
        r, c = nr, nc
        if record_path:
            path.append((r, c))
        if c == W-1:
            break
    else:
        # exceeded max_steps: treat as failure; return inf time but keep path
        t = float('inf')
    return t, path if record_path else []

# -------------------- Simulation helpers --------------------

def simulate_once_dij(geom='random', H=60, W=90, logsigma=0.6, corr_len=0,
                      corr_frac=0.25, barrier_frac=0.18):
    if geom == 'random':
        k = make_random_k(H, W, logsigma=logsigma, corr_len=corr_len)
    elif geom == 'parallel':
        k = make_parallel_k(H, W, corridor_fraction=corr_frac)
    elif geom == 'series':
        k = make_series_k(H, W, barrier_fraction=barrier_frac)
    else:
        raise ValueError('unknown geometry')
    cost = 1.0 / (k + 1e-9)
    start = (H//2, 0)
    goal  = (H//2, W-1)
    time, path = dijkstra_cost(cost, start, goal)
    return time, k, path


def simulate_once_rw(geom='random', H=60, W=90, logsigma=0.6, corr_len=0,
                     corr_frac=0.25, barrier_frac=0.18,
                     beta=0.2, step_time='inv_k_current', rng=None):
    if geom == 'random':
        k = make_random_k(H, W, logsigma=logsigma, corr_len=corr_len)
    elif geom == 'parallel':
        k = make_parallel_k(H, W, corridor_fraction=corr_frac)
    elif geom == 'series':
        k = make_series_k(H, W, barrier_fraction=barrier_frac)
    else:
        raise ValueError('unknown geometry')
    t, path = biased_random_walk_first_passage(k, beta=beta, step_time=step_time, rng=rng)
    return t, k, path


def batch_times_dij(geom, trials=200, **kwargs):
    times = []
    example = None
    for t in range(trials):
        tt, k, path = simulate_once_dij(geom=geom, **kwargs)
        times.append(tt)
        if t == 0:
            example = (k, path, tt)
    return np.array(times), example


def batch_times_rw(geom, trials=200, walkers_per_field=1, rng_seed=None,
                   beta=0.2, step_time='inv_k_current', **kwargs):
    """Run BRW trials; by default 1 walker per field (like Dijkstra),
    but you can set walkers_per_field>1 to average many walkers on each field.
    Returns times array (one per field) and one example (first field/path).
    """
    rng = np.random.default_rng(rng_seed)
    times = []
    example = None
    for t in range(trials):
        # build one field per trial
        if geom == 'random':
            k = make_random_k(kwargs.get('H',60), kwargs.get('W',90),
                              logsigma=kwargs.get('logsigma',0.6), corr_len=kwargs.get('corr_len',0))
        elif geom == 'parallel':
            k = make_parallel_k(kwargs.get('H',60), kwargs.get('W',90), corridor_fraction=kwargs.get('corr_frac',0.25))
        elif geom == 'series':
            k = make_series_k(kwargs.get('H',60), kwargs.get('W',90), barrier_fraction=kwargs.get('barrier_frac',0.18))
        else:
            raise ValueError('unknown geometry')
        # simulate walkers
        acc = 0.0
        last_path = []
        for _ in range(max(1, walkers_per_field)):
            tt, path = biased_random_walk_first_passage(k, beta=beta, step_time=step_time, rng=rng)
            acc += tt
            last_path = path
        mean_t = acc / max(1, walkers_per_field)
        times.append(mean_t)
        if t == 0:
            example = (k, last_path, mean_t)
    return np.array(times), example

# -------------------- Plotting & Animation --------------------

def save_example_and_hist(times, example, title_prefix):
    k, path, t = example
    H, W = k.shape
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    im = ax[0].imshow(k, cmap='magma', origin='lower')
    ax[0].set_title(f'{title_prefix}: k field (mean=1), time={t:.2f}')
    if path:
        yp = [p[0] for p in path]
        xp = [p[1] for p in path]
        ax[0].plot(xp, yp, color='cyan', lw=1.5)
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label='Conductivity k')
    ax[1].hist(times, bins=24, color='#4477AA', edgecolor='white')
    ax[1].set_title(f'{title_prefix}: time distribution (n={len(times)})')
    ax[1].set_xlabel('Traversal time'); ax[1].set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(f'mixing_v3_{title_prefix}_example.png', dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5.5,4))
    ax2.hist(times, bins=24, color='#4477AA', edgecolor='white')
    ax2.set_title(f'{title_prefix}: traversal-time histogram')
    ax2.set_xlabel('Traversal time'); ax2.set_ylabel('Count')
    fig2.tight_layout()
    fig2.savefig(f'mixing_v3_{title_prefix}_hist.png', dpi=180)
    plt.close(fig2)


def animate_trace(k, path, title_prefix, fps=20, step=1):
    H, W = k.shape
    fig, ax = plt.subplots(figsize=(6,4))
    im = ax.imshow(k, cmap='magma', origin='lower')
    ax.set_title(f'{title_prefix}: traversal animation')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Conductivity k')
    (ln,) = ax.plot([], [], color='cyan', lw=1.5)
    (pt,) = ax.plot([], [], 'wo', ms=6)
    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    def init():
        ln.set_data([], [])
        pt.set_data([], [])
        return (ln, pt)
    def update(i):
        if len(xs)==0:
            return (ln, pt)
        j = min(i*step, len(xs)-1)
        ln.set_data(xs[:j+1], ys[:j+1])
        pt.set_data([xs[j]], [ys[j]])
        return (ln, pt)
    frames = max(1, len(xs)//max(1,step))
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=1000/fps, blit=True)
    try:
        anim.save(f'mixing_v3_{title_prefix}_anim.gif', writer=animation.PillowWriter(fps=fps))
    except Exception as e:
        print('Animation save failed:', e)
    plt.close(fig)
