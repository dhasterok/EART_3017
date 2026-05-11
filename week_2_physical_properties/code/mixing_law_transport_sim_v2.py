# -*- coding: utf-8 -*-
"""
Mixing-Law Transport Demonstration (v2)
--------------------------------------
Goal:
  Ensure traversal-time ordering consistent with mixing intuition:
    PARALLEL (arithmetic mean of conductivity)  -> fastest
    RANDOM   (geometric mean of conductivity)   -> middle
    SERIES   (harmonic mean of conductivity)    -> slowest

Key changes vs v1:
  • Cell property is conductivity k(x,y); traversal cost = 1/k.
  • Normalize area-mean(k) = 1 for fair comparisons.
  • Random: lognormal k with optional correlation length.
  • Parallel: high-k corridors aligned with transport.
  • Series: low-k barriers perpendicular to transport.
  • Optional GIF animation of the least-cost path.
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
    # Choose mu so E[k]=1 for lognormal: E[k]=exp(mu+0.5*sigma^2)=1 -> mu = -0.5*sigma^2
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
    k = np.full((H, W), k_high, dtype=float)
    num = max(1, int(H * barrier_fraction))
    rows = np.linspace(3, H-4, num, dtype=int)
    for r in rows:
        k[r:r+thickness, :] = k_low
    k *= (1.0 - jitter/2.0 + jitter*np.random.rand(H, W))
    return normalize_mean_k(k)

# -------------------- Path Solver --------------------

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

# -------------------- Simulation helpers --------------------

def simulate_once(geom='random', H=60, W=90, logsigma=0.6, corr_len=0, corr_frac=0.25, barrier_frac=0.18):
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


def batch_times(geom, trials=200, **kwargs):
    times = []
    example = None
    for t in range(trials):
        tt, k, path = simulate_once(geom=geom, **kwargs)
        times.append(tt)
        if t == 0:
            example = (k, path, tt)
    return np.array(times), example


def save_example_and_hist(times, example, title_prefix):
    k, path, t = example
    H, W = k.shape
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    im = ax[0].imshow(k, cmap='magma', origin='lower')
    ax[0].set_title(f'{title_prefix}: k field (mean=1), path time={t:.2f}')
    if path:
        yp = [p[0] for p in path]
        xp = [p[1] for p in path]
        ax[0].plot(xp, yp, color='cyan', lw=1.5)
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label='Conductivity k')
    ax[1].hist(times, bins=18, color='#4477AA', edgecolor='white')
    ax[1].set_title(f'{title_prefix}: traversal-time distribution (n={len(times)})')
    ax[1].set_xlabel('Traversal time (sum of 1/k along least-cost path)')
    ax[1].set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(f'mixing_v2_{title_prefix}_example.png', dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5.5,4))
    ax2.hist(times, bins=24, color='#4477AA', edgecolor='white')
    ax2.set_title(f'{title_prefix}: traversal-time histogram')
    ax2.set_xlabel('Traversal time')
    ax2.set_ylabel('Count')
    fig2.tight_layout()
    fig2.savefig(f'mixing_v2_{title_prefix}_hist.png', dpi=180)
    plt.close(fig2)


def animate_path(k, path, title_prefix, fps=20, step=1):
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
        j = min(i*step, len(xs)-1)
        ln.set_data(xs[:j+1], ys[:j+1])
        pt.set_data(xs[j], ys[j])
        return (ln, pt)

    frames = max(1, len(xs)//step)
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=1000/fps, blit=True)
    try:
        anim.save(f'mixing_v2_{title_prefix}_anim.gif', writer=animation.PillowWriter(fps=fps))
    except Exception as e:
        print('Animation save failed:', e)
    plt.close(fig)

