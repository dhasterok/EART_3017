import numpy as np

# -------------------- Mixing-law predictions --------------------

def mixing_law_predictions(fracs, mus, sigs):
    """Return (k_harm, k_geom, k_arith) for the given lognormal component parameters."""
    fracs = np.asarray(fracs, float)
    mus   = np.asarray(mus,   float)
    sigs  = np.asarray(sigs,  float)
    means = np.exp(mus + 0.5*(sigs**2))          # E[k] per lognormal component
    k_harm  = 1.0 / float(np.sum(fracs / means))
    k_geom  = float(np.exp(np.sum(fracs * mus))) # exp(E[ln k]) — geometric mean
    k_arith = float(np.sum(fracs * means))
    return k_harm, k_geom, k_arith


# -------------------- Dijkstra (random start + free exit) --------------------

def dijkstra_random_start_free_exit(cost, rng=None, start_row=None,
                                    return_explored=False):
    """
    Minimum-cost path from a left-boundary cell to any right-boundary cell.

    Parameters
    ----------
    cost        : 2-D array, shape (H, W); cost[r,c] = 1/k[r,c]
    rng         : numpy Generator (optional)
    start_row   : int or None; if None, chosen uniformly from [0, H)
    return_explored : if True, also return the list of cells in settlement order

    Returns
    -------
    (t, exit_row, path)                    when return_explored=False
    (t, exit_row, path, explored)          when return_explored=True

    path and explored are lists of (row, col) tuples.
    """
    H, W = cost.shape
    if rng is None:
        rng = np.random.default_rng()
    if start_row is None:
        start_row = int(rng.integers(0, H))
    sx, sy = start_row, 0

    INF = 1e18
    dist = np.full((H, W), INF, float)
    prev = np.full((H, W, 2), -1, int)
    dist[sx, sy] = cost[sx, sy]

    from heapq import heappush, heappop
    pq = [(dist[sx, sy], (sx, sy))]
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    explored = [] if return_explored else None
    goal = None

    while pq:
        d, (x, y) = heappop(pq)
        if d != dist[x, y]:
            continue
        if return_explored:
            explored.append((x, y))
        if y == W - 1:
            goal = (x, y)
            break
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                nd = d + cost[nx, ny]
                if nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    prev[nx, ny] = [x, y]
                    heappush(pq, (nd, (nx, ny)))

    if goal is None:
        if return_explored:
            return float('inf'), -1, [], explored or []
        return float('inf'), -1, []

    # Trace path back via prev pointers
    path = []
    cx, cy = goal
    while not (cx == sx and cy == sy):
        path.append((cx, cy))
        cx, cy = prev[cx, cy]
    path.append((sx, sy))
    path.reverse()

    t = float(dist[goal])
    exit_row = goal[0]

    if return_explored:
        return t, exit_row, path, explored
    return t, exit_row, path


# -------------------- Biased random walk (directed diffusion) --------------------

def first_passage_time_random_walk(k, *, beta=0.2, max_steps=10000,
                                   step_time_model='1_over_k',
                                   rng=None, record_path=False,
                                   start_row=None):
    """
    Single-walker first passage to the right edge on a 4-neighbour grid.

    Start: start_row (or uniformly random on the left edge if None).
    Absorbing boundary: right edge.  Reflecting: top/bottom.
    Transition probability to neighbour j:
        P ∝ k_target × (1 + β × dot(e_x, step))   (clipped at 0)
    Time increment per step: 1/k_target ('1_over_k') or 1 ('constant').

    Returns (t, exit_row, path, k_geom_path).
    k_geom_path is the geometric mean of k values visited (independent of path
    length, invariant to β, converges to approximately field k_geom).
    path is populated only when record_path=True.
    """
    if rng is None:
        rng = np.random.default_rng()
    H, W = k.shape
    if start_row is None:
        r = int(rng.integers(0, H))
    else:
        r = int(start_row)
    c = 0
    t = 0.0
    sum_ln_k = 0.0
    n_steps = 0
    path = [(r, c)] if record_path else []
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # S N E W

    while True:
        if c == W - 1:
            k_geom_path = float(np.exp(sum_ln_k / n_steps)) if n_steps > 0 else float('nan')
            return t, r, path, k_geom_path
        if n_steps > max_steps:
            k_geom_path = float(np.exp(sum_ln_k / n_steps)) if n_steps > 0 else float('nan')
            return float('inf'), r, path, k_geom_path
        weights = []
        dests = []
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                bias = 1.0 + beta * (1 if dc > 0 else (-1 if dc < 0 else 0))
                w = max(0.0, k[nr, nc] * bias)
                if w > 0:
                    weights.append(w)
                    dests.append((nr, nc))
        if not weights:
            k_geom_path = float(np.exp(sum_ln_k / n_steps)) if n_steps > 0 else float('nan')
            return float('inf'), r, path, k_geom_path
        weights = np.array(weights, float)
        p = weights / weights.sum()
        idx = rng.choice(len(dests), p=p)
        nr, nc = dests[idx]
        dt = 1.0 if step_time_model == 'constant' else 1.0 / max(1e-12, k[nr, nc])
        t += dt
        sum_ln_k += np.log(max(1e-12, k[nr, nc]))
        n_steps += 1
        r, c = nr, nc
        if record_path:
            path.append((r, c))


def mean_fpt_random_walk(k, *, beta, max_steps, step_time_model,
                         walkers, rng=None):
    """Run `walkers` independent walkers and return their mean finite FPT."""
    if rng is None:
        rng = np.random.default_rng()
    acc = 0.0
    count = 0
    for _ in range(walkers):
        t, _, _, _ = first_passage_time_random_walk(
            k, beta=beta, max_steps=max_steps,
            step_time_model=step_time_model, rng=rng, record_path=False)
        if np.isfinite(t):
            acc += t
            count += 1
    return acc / count if count > 0 else float('inf')


# -------------------- Unified trial interface --------------------

def one_trial(k, solver, W, start_row=None, *, beta=0.2, max_steps=10000,
              step_time_model='1_over_k', rng=None, record_path=False):
    """
    Run one trial and return (t, k_eff, exit_row, path).

    k_eff = W / t  (effective bulk conductivity for the path).
    exit_row = row index on the right boundary where the path terminates.
    path is populated only when record_path=True.
    """
    if rng is None:
        rng = np.random.default_rng()
    if solver == 'dijkstra':
        cost = 1.0 / (k + 1e-9)
        t, exit_row, path = dijkstra_random_start_free_exit(
            cost, rng, start_row, return_explored=False)
        k_eff = W / t if np.isfinite(t) and t > 0 else float('nan')
    else:
        t, exit_row, path, k_geom_path = first_passage_time_random_walk(
            k, beta=beta, max_steps=max_steps,
            step_time_model=step_time_model, rng=rng,
            record_path=record_path, start_row=start_row)
        k_eff = k_geom_path  # geometric mean of k along path — independent of path length and β
    return t, k_eff, exit_row, path


# -------------------- Kirchhoff resistor-network solver --------------------

def solve_kirchhoff(k, periodic_tb=False):
    """
    Solve Kirchhoff's current law on the pixel resistor network.

    Edge conductance between adjacent pixels i, j:
        g_ij = 2 k_i k_j / (k_i + k_j)  (two half-cells in series)

    Boundary conditions
    -------------------
    T = 1 at the left column  (column 0)
    T = 0 at the right column (column W-1)
    Top/bottom: insulated (no edge added) or periodic (wrap-around edge).

    Returns
    -------
    k_eff : float
        Effective conductivity = total flux × W / H.
    T : ndarray, shape (H, W)
        Temperature field (solution).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    H, W = k.shape
    N = H * W

    ri, ci, vi = [], [], []

    def _add(i_arr, j_arr, g_arr):
        ri.extend([i_arr, j_arr, i_arr, j_arr])
        ci.extend([i_arr, j_arr, j_arr, i_arr])
        vi.extend([ g_arr,  g_arr, -g_arr, -g_arr])

    # Horizontal edges: (r, c) ↔ (r, c+1)
    r, c = np.mgrid[0:H, 0:W-1]
    i = (r * W + c).ravel();  j = i + 1
    g = 2*k[r, c].ravel()*k[r, c+1].ravel() / (k[r, c].ravel() + k[r, c+1].ravel() + 1e-30)
    _add(i, j, g)

    # Vertical edges: (r, c) ↔ (r+1, c)
    r, c = np.mgrid[0:H-1, 0:W]
    i = (r * W + c).ravel();  j = i + W
    g = 2*k[r, c].ravel()*k[r+1, c].ravel() / (k[r, c].ravel() + k[r+1, c].ravel() + 1e-30)
    _add(i, j, g)

    # Periodic top-bottom wrap: (0, c) ↔ (H-1, c)
    if periodic_tb:
        c_arr = np.arange(W)
        i_arr = c_arr.copy()
        j_arr = (H - 1) * W + c_arr
        g = 2*k[0]*k[H-1] / (k[0] + k[H-1] + 1e-30)
        _add(i_arr, j_arr, g)

    L = coo_matrix(
        (np.concatenate(vi), (np.concatenate(ri), np.concatenate(ci))),
        shape=(N, N)
    ).tocsr()

    # Boundary masks
    left  = np.arange(H) * W            # column 0
    right = np.arange(H) * W + (W - 1)  # column W-1
    bc = np.zeros(N, bool)
    bc[left] = True;  bc[right] = True
    interior = np.where(~bc)[0]
    bc_idx   = np.where(bc)[0]

    T_full = np.zeros(N)
    T_full[left] = 1.0  # T_full[right] = 0 already

    T_full[interior] = spsolve(
        L[interior][:, interior],
        -L[interior][:, bc_idx].dot(T_full[bc_idx])
    )
    T = T_full.reshape(H, W)

    # Total flux crossing the left→right edge at column 0.
    # Distance between boundary nodes = W-1 spacings, so the correct
    # conductivity formula uses (W-1) not W (matches node-to-node distance).
    g_edge = 2*k[:, 0]*k[:, 1] / (k[:, 0] + k[:, 1] + 1e-30)
    Q = float(np.sum(g_edge * (T[:, 0] - T[:, 1])))
    k_eff = Q * (W - 1) / H
    return k_eff, T


# -------------------- Legacy prediction proxy --------------------

def predicted_keff(geom, fracs, mus, sigs):
    fracs = np.array(fracs)
    means = np.exp(mus + 0.5*(sigs**2))
    inv_means = np.exp(-mus + 0.5*(sigs**2))
    mean_lnk = mus
    if geom == 'parallel':
        keff = float(np.sum(fracs * means))
    elif geom == 'series':
        keff = 1.0 / float(np.sum(fracs * inv_means))
    else:
        keff = float(np.exp(np.sum(fracs * mean_lnk)))
    return keff
