
"""
Mixing-Law Transport Demonstration
----------------------------------
Simulate traversal time across 2D media representing different mixing geometries:
  1) RANDOM granular medium  -> geometric-like behavior for transport
  2) PARALLEL corridors      -> arithmetic-like (parallel pathways)
  3) SERIES barriers         -> harmonic-like (series resistances)

We model a walker (heat/particle) moving from left to right across a grid.
Each cell has a traversal cost. We compute the least-cost path (Dijkstra) and
record total time. Repeat to build distributions and plot histograms.

Usage (basic):
  python mixing_law_transport_sim.py

This will generate PNGs with path examples and histograms in the working directory.
"""
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random

np.random.seed(42)
random.seed(42)

# ---------------- Dijkstra on 2D grid (4-neighbour) ----------------

def dijkstra_cost(grid, start, goal):
    H, W = grid.shape
    INF = 1e18
    dist = np.full((H, W), INF, dtype=float)
    prev = np.full((H, W, 2), -1, dtype=int)
    sx, sy = start
    gx, gy = goal
    dist[sx, sy] = grid[sx, sy]
    pq = []
    heappush(pq, (dist[sx, sy], (sx, sy)))
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
                nd = d + grid[nx, ny]
                if nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    prev[nx, ny] = [x, y]
                    heappush(pq, (nd, (nx, ny)))
    # reconstruct path
    path = []
    cx, cy = gx, gy
    if dist[gx, gy] >= INF/2:
        return INF, path
    while not (cx == sx and cy == sy):
        path.append((cx, cy))
        px, py = prev[cx, cy]
        cx, cy = px, py
    path.append((sx, sy))
    path.reverse()
    return dist[gx, gy], path

# ---------------- Media Generators ----------------

def make_random_medium(H=60, W=90, logmean=0.0, logsigma=0.5):
    """Random granular medium: log-normal costs produce right-skewed times."""
    grid = np.random.lognormal(mean=logmean, sigma=logsigma, size=(H, W))
    return grid


def make_parallel_corridors(H=60, W=90, low_cost=0.2, high_cost=1.0, corridor_fraction=0.2):
    """Parallel corridors aligned with transport (left->right): arithmetic-like behavior."""
    grid = np.full((H, W), high_cost, dtype=float)
    # create vertical corridors (parallel to motion): low cost columns
    num_corridors = max(1, int(W * corridor_fraction))
    cols = np.linspace(0, W-1, num_corridors, dtype=int)
    for c in cols:
        grid[:, c] = low_cost
    # add small noise
    grid *= (0.95 + 0.1*np.random.rand(H, W))
    return grid


def make_series_barriers(H=60, W=90, low_cost=0.3, high_cost=2.0, barrier_fraction=0.15):
    """Series barriers perpendicular to transport: harmonic-like behavior dominated by bottlenecks."""
    grid = np.full((H, W), low_cost, dtype=float)
    # add horizontal high-cost barrier rows that must be crossed in series
    num_barriers = max(1, int(H * barrier_fraction))
    rows = np.linspace(5, H-6, num_barriers, dtype=int)
    for r in rows:
        grid[r:r+1, :] = high_cost
    # add noise
    grid *= (0.95 + 0.1*np.random.rand(H, W))
    return grid

# ---------------- Simulation ----------------

def simulate(geom='random', trials=50, H=60, W=90):
    times = []
    example = None
    for t in range(trials):
        if geom == 'random':
            grid = make_random_medium(H, W, logmean=0.0, logsigma=0.6)
        elif geom == 'parallel':
            grid = make_parallel_corridors(H, W, corridor_fraction=0.25)
        elif geom == 'series':
            grid = make_series_barriers(H, W, barrier_fraction=0.18)
        else:
            raise ValueError('unknown geometry')
        start = (H//2, 0)
        goal  = (H//2, W-1)
        time, path = dijkstra_cost(grid, start, goal)
        times.append(time)
        if t == 0:
            example = (grid, path, time)
    return np.array(times), example


def plot_example_and_hist(times, example, title, out_prefix):
    grid, path, t = example
    H, W = grid.shape
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    im = ax[0].imshow(grid, cmap='viridis', origin='lower')
    ax[0].set_title(f'{title}: example grid and least-cost path (time={t:.1f})')
    if path:
        yp = [p[0] for p in path]
        xp = [p[1] for p in path]
        ax[0].plot(xp, yp, color='r', lw=1.5)
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label='Cell cost')
    ax[1].hist(times, bins=12, color='#4477AA', edgecolor='white')
    ax[1].set_title(f'{title}: distribution of times (n={len(times)})')
    ax[1].set_xlabel('Traversal time (least cost)')
    ax[1].set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_{title.replace(" ","_")}.png', dpi=180)
    plt.close(fig)


def main():
    configs = [
        ('RANDOM (granular)', 'random'),
        ('PARALLEL (corridors)', 'parallel'),
        ('SERIES (barriers)', 'series'),
    ]
    for title, geom in configs:
        times, example = simulate(geom=geom, trials=1000)
        plot_example_and_hist(times, example, title, out_prefix='mixing_sim')
    print('Saved figures: mixing_sim_RANDOM_(granular).png, mixing_sim_PARALLEL_(corridors).png, mixing_sim_SERIES_(barriers).png')

if __name__ == '__main__':
    main()
