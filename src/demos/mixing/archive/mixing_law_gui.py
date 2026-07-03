# -*- coding: utf-8 -*-
"""
Mixing-Law Transport GUI (v3)
-----------------------------
Interactive demonstration for effective transport in media with two or three
mineral components. Students can:
  • Choose geometry: RANDOM (granular), PARALLEL (lanes), SERIES (barriers)
  • Set 2--3 components with mean k and std k (assumed lognormal), and area fractions
  • Adjust grain correlation (corr_len), fuzziness of boundaries, corridor/barrier fraction
  • Choose number of realizations (log slider), run trials, and watch histogram grow
  • Animate a single traversal path in real time with speed control
  • See predicted effective conductivity (k_eff) from mixing laws (arith/geom/harm) and
    a predicted traversal time overlaid as a vertical line on the histogram
Notes
-----
We construct a conductivity field k(x,y). Traversal cost per cell is 1/k. The fastest
path from left to right is computed with Dijkstra on a 4-neighbor grid.
For geometric predictions, we:
  - PARALLEL: k_eff = sum_i f_i * E[k_i]            (arithmetic)
  - SERIES:   1/k_eff = sum_i f_i * E[1/k_i]       (harmonic)
  - RANDOM:   ln k_eff = sum_i f_i * E[ln k_i]     (geometric)
The domain is then globally normalized by mean(k) so comparisons are fair; we scale
predicted k_eff by the same factor before predicting traversal time ~ W / k_eff.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, TextBox
from heapq import heappush, heappop
from matplotlib import animation

# -------------------- Core math --------------------

def lognormal_mu_sigma_from_mean_std(m, s):
    m = max(1e-9, float(m))
    s = max(1e-12, float(s))
    sigma2 = np.log(1.0 + (s*s)/(m*m))
    sigma = np.sqrt(sigma2)
    mu = np.log(m) - 0.5*sigma2
    return mu, sigma


def normalize_mean_k(k):
    mk = float(k.mean())
    return k if mk <= 0 else (k / mk), (1.0/mk if mk>0 else 1.0)


def moving_average_2d(arr, win_y=3, win_x=3, iters=1):
    a = arr.copy()
    H, W = a.shape
    for _ in range(iters):
        py, px = win_y//2, win_x//2
        p = np.pad(a, ((py,py),(px,px)), mode='reflect')
        ii = p.cumsum(0).cumsum(1)
        y2 = np.arange(win_y, H+win_y)
        x2 = np.arange(win_x, W+win_x)
        y1 = y2 - win_y
        x1 = x2 - win_x
        s = ii[y2[:,None], x2] - ii[y1[:,None], x2] - ii[y2[:,None], x1] + ii[y1[:,None], x1]
        a = s/float(win_y*win_x)
    return a

# -------------------- Media factories --------------------

def sample_component_k(component_id, comp_map, mus, sigs):
    mu = mus[component_id]
    sig = sigs[component_id]
    return np.exp(np.random.normal(mu, sig, size=comp_map.shape))


def build_random_field(H, W, fracs, mus, sigs, corr_len=0, fuzz=0.15):
    # Assign components by multinomial
    comps = np.random.choice(len(fracs), size=(H,W), p=np.array(fracs))
    # induce clustering by majority smoothing on one-hot maps
    if corr_len and corr_len>1:
        K = len(fracs)
        score = []
        for i in range(K):
            m = (comps==i).astype(float)
            sc = moving_average_2d(m, corr_len, corr_len, 1)
            score.append(sc)
        score = np.stack(score, axis=-1)
        comps = np.argmax(score, axis=-1)
    # Small random pixel flips for fuzziness
    if fuzz>0:
        mask = np.random.rand(H,W) < min(0.49, fuzz)
        comps[mask] = np.random.randint(0, len(fracs), size=mask.sum())
    # sample k per component
    k = np.zeros((H,W), float)
    for i in range(len(fracs)):
        k[comps==i] = sample_component_k(i, comps==i, mus, sigs)[comps==i]
    return k


def build_parallel_field(H, W, fracs, mus, sigs, fuzz=0.10, corridor_frac=0.25):
    # vertical lanes approximate fractions
    lanes = np.floor(np.cumsum(np.array(fracs))*W).astype(int)
    borders = [0] + list(lanes)
    comps = np.zeros((H,W), int)
    start=0
    for i,stop in enumerate(lanes):
        comps[:, start:stop] = i
        start = stop
    # jitter boundaries row-wise
    rng = np.random.default_rng()
    for i in range(1, len(lanes)):
        b = lanes[i-1]
        # shift boundary per row by ±2 columns with 50% chance
        shifts = rng.integers(-2,3,size=H)
        for r in range(H):
            c = np.clip(b+shifts[r], 1, W-2)
            comps[r, c-1:c+1] = rng.integers(0, len(fracs))
    # fuzziness random swaps
    if fuzz>0:
        mask = np.random.rand(H,W) < min(0.49, fuzz)
        comps[mask] = np.random.randint(0, len(fracs), size=mask.sum())
    # sample k
    k = np.zeros((H,W), float)
    for i in range(len(fracs)):
        k[comps==i] = sample_component_k(i, comps==i, mus, sigs)[comps==i]
    # Add explicit high-k corridors with small width if corridor_frac>0
    if corridor_frac>0:
        num = max(1, int(W*corridor_frac))
        cols = np.linspace(0, W-1, num, dtype=int)
        for c in cols:
            k[:, c] *= 1.5
    return k


def build_series_field(H, W, fracs, mus, sigs, fuzz=0.10, barrier_frac=0.15, thickness=1, strict_barriers=False):
    # horizontal layers approximate fractions
    layers = np.floor(np.cumsum(np.array(fracs))*H).astype(int)
    comps = np.zeros((H,W), int)
    start=0
    for i,stop in enumerate(layers):
        comps[start:stop, :] = i
        start = stop
    # fuzz boundaries column-wise
    rng = np.random.default_rng()
    for i in range(1, len(layers)):
        b = layers[i-1]
        shifts = rng.integers(-2,3,size=W)
        for c in range(W):
            r = np.clip(b+shifts[c], 1, H-2)
            comps[r-1:r+1, c] = rng.integers(0, len(fracs))
    # optional sparse gaps to break strict barriers unless strict requested
    if not strict_barriers:
        gap_cols = rng.choice(W, size=max(1,W//20), replace=False)
        comps[::max(1,H//20), gap_cols] = rng.integers(0, len(fracs), size=(len(range(0,H,max(1,H//20))), len(gap_cols)))
    # fuzziness random swaps
    if fuzz>0:
        mask = np.random.rand(H,W) < min(0.49, fuzz)
        comps[mask] = np.random.randint(0, len(fracs), size=mask.sum())
    # sample k
    k = np.zeros((H,W), float)
    for i in range(len(fracs)):
        k[comps==i] = sample_component_k(i, comps==i, mus, sigs)[comps==i]
    # enforce extra low-k barrier rows if barrier_frac>0
    if barrier_frac>0:
        num = max(1, int(H*barrier_frac))
        rows = np.linspace(0, H-1, num, dtype=int)
        for r in rows:
            k[r:r+thickness, :] *= 0.5
    return k

# -------------------- Path solver --------------------

def dijkstra(cost, start, goal):
    H,W = cost.shape
    INF = 1e18
    dist = np.full((H,W), INF, float)
    prev = np.full((H,W,2), -1, int)
    sx,sy = start; gx,gy = goal
    dist[sx,sy] = cost[sx,sy]
    pq = [(dist[sx,sy], (sx,sy))]
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    while pq:
        d,(x,y) = heappop(pq)
        if (x,y)==(gx,gy): break
        if d!=dist[x,y]: continue
        for dx,dy in moves:
            nx,ny = x+dx, y+dy
            if 0<=nx<H and 0<=ny<W:
                nd = d + cost[nx,ny]
                if nd < dist[nx,ny]:
                    dist[nx,ny] = nd
                    prev[nx,ny] = [x,y]
                    heappush(pq, (nd, (nx,ny)))
    path=[]; cx,cy=gx,gy
    if dist[gx,gy] >= INF/2: return float('inf'), path
    while not (cx==sx and cy==sy):
        path.append((cx,cy))
        px,py = prev[cx,cy]
        cx,cy = px,py
    path.append((sx,sy)); path.reverse()
    return dist[gx,gy], path

# -------------------- Predictions --------------------

def predicted_keff(geom, fracs, mus, sigs):
    fracs = np.array(fracs)
    means = np.exp(mus + 0.5*(sigs**2))            # E[k]
    inv_means = np.exp(-mus + 0.5*(sigs**2))       # E[1/k] for lognormal
    mean_lnk = mus                                 # E[ln k] = mu
    if geom=='parallel':
        keff = np.sum(fracs * means)               # arithmetic
    elif geom=='series':
        keff = 1.0 / np.sum(fracs * inv_means)     # harmonic
    else:
        keff = np.exp(np.sum(fracs * mean_lnk))    # geometric
    return keff

# -------------------- GUI --------------------
class MixingGUI:
    def __init__(self):
        self.H, self.W = 60, 90
        self.geom = 'random'
        self.ncomp = 2
        self.fracs = [0.5, 0.5, 0.0]
        self.means = [1.2, 0.6, 0.9]
        self.stds  = [0.3, 0.2, 0.2]
        self.corr_len = 0
        self.fuzz = 0.10
        self.corridor_frac = 0.25
        self.barrier_frac  = 0.18
        self.strict_barriers = False
        self.logN = 2.0   # 10^2 trials
        self.animate_flag = False
        self.speed = 30
        self.times = []
        self.example = None
        self._setup_fig()

    def _setup_fig(self):
        self.fig = plt.figure(figsize=(12,6))
        gs = self.fig.add_gridspec(3, 4, width_ratios=[1.1,1.1,1.1,0.8], height_ratios=[1.2,1.0,0.7], wspace=0.35, hspace=0.35)
        self.ax_img = self.fig.add_subplot(gs[:2, :2])
        self.ax_hist = self.fig.add_subplot(gs[:2, 2])
        self.ax_text = self.fig.add_subplot(gs[2, :3])
        self.ax_text.axis('off')
        # Controls panel
        self.ax_geom = self.fig.add_subplot(gs[0, 3])
        self.ax_comp = self.fig.add_subplot(gs[1, 3])
        self.ax_run  = self.fig.add_subplot(gs[2, 3])

        # Geometry radio
        self.rb_geom = RadioButtons(self.ax_geom, ('random','parallel','series'), active=0)
        self.rb_geom.on_clicked(self._on_geom)
        self.ax_geom.set_title('Geometry')

        # Component boxes (means, stds, fracs)
        # Use TextBox to allow numeric entry
        self._make_component_boxes()

        # Sliders under histogram area
        axN = self.fig.add_axes([0.56, 0.08, 0.33, 0.03])
        self.sl_N = Slider(axN, 'log10 N', 1.0, 4.0, valinit=self.logN, valstep=0.05)
        self.sl_N.on_changed(self._on_params)

        axC = self.fig.add_axes([0.56, 0.04, 0.33, 0.03])
        self.sl_corr = Slider(axC, 'corr_len', 0, 9, valinit=self.corr_len, valstep=1)
        self.sl_corr.on_changed(self._on_params)

        axF = self.fig.add_axes([0.56, 0.12, 0.33, 0.03])
        self.sl_fuzz = Slider(axF, 'fuzz', 0.0, 0.45, valinit=self.fuzz, valstep=0.01)
        self.sl_fuzz.on_changed(self._on_params)

        # Geometry-specific sliders
        axCorr = self.fig.add_axes([0.15, 0.02, 0.28, 0.03])
        self.sl_corridor = Slider(axCorr, 'corridors(frac)', 0.0, 0.5, valinit=self.corridor_frac, valstep=0.01)
        self.sl_corridor.on_changed(self._on_params)
        axBarr = self.fig.add_axes([0.15, 0.06, 0.28, 0.03])
        self.sl_barrier  = Slider(axBarr, 'barriers(frac)', 0.0, 0.5, valinit=self.barrier_frac,  valstep=0.01)
        self.sl_barrier.on_changed(self._on_params)

        # Checkboxes: animate, strict barriers
        self.ck = CheckButtons(self.ax_run, ['Animate path','Strict barriers'], [self.animate_flag, self.strict_barriers])
        self.ck.on_clicked(self._on_checks)

        # Buttons
        bax1 = self.fig.add_axes([0.35, 0.905, 0.16, 0.05])
        bax2 = self.fig.add_axes([0.53, 0.905, 0.16, 0.05])
        self.bt_run = Button(bax1, 'Run N trials')
        self.bt_anim= Button(bax2, 'Animate 1 path')
        self.bt_run.on_clicked(self._run_trials)
        self.bt_anim.on_clicked(self._animate_one)

        # Speed slider
        axS = self.fig.add_axes([0.72, 0.905, 0.2, 0.03])
        self.sl_speed = Slider(axS, 'speed (fps)', 5, 60, valinit=self.speed, valstep=1)
        self.sl_speed.on_changed(self._on_params)

        self._refresh_prediction_text()
        self._reset_plot()

    def _make_component_boxes(self):
        self.ax_comp.clear()
        self.ax_comp.set_title('Components (2 or 3)')
        # radio for ncomp
        axr = self.ax_comp.inset_axes([0.02,0.68,0.4,0.3])
        self.rb_n = RadioButtons(axr, ('2','3'), active=(self.ncomp-2))
        self.rb_n.on_clicked(self._on_ncomp)
        # TextBoxes for means, stds, fracs
        # Comp1
        axm1 = self.ax_comp.inset_axes([0.02,0.45,0.28,0.12]); self.tb_m1 = TextBox(axm1, 'k1 mean', initial=f"{self.means[0]:.3f}")
        axs1 = self.ax_comp.inset_axes([0.32,0.45,0.28,0.12]); self.tb_s1 = TextBox(axs1, 'k1 std',  initial=f"{self.stds[0]:.3f}")
        axf1 = self.ax_comp.inset_axes([0.62,0.45,0.28,0.12]); self.tb_f1 = TextBox(axf1, 'f1',      initial=f"{self.fracs[0]:.3f}")
        # Comp2
        axm2 = self.ax_comp.inset_axes([0.02,0.28,0.28,0.12]); self.tb_m2 = TextBox(axm2, 'k2 mean', initial=f"{self.means[1]:.3f}")
        axs2 = self.ax_comp.inset_axes([0.32,0.28,0.28,0.12]); self.tb_s2 = TextBox(axs2, 'k2 std',  initial=f"{self.stds[1]:.3f}")
        axf2 = self.ax_comp.inset_axes([0.62,0.28,0.28,0.12]); self.tb_f2 = TextBox(axf2, 'f2',      initial=f"{self.fracs[1]:.3f}")
        # Comp3
        axm3 = self.ax_comp.inset_axes([0.02,0.11,0.28,0.12]); self.tb_m3 = TextBox(axm3, 'k3 mean', initial=f"{self.means[2]:.3f}")
        axs3 = self.ax_comp.inset_axes([0.32,0.11,0.28,0.12]); self.tb_s3 = TextBox(axs3, 'k3 std',  initial=f"{self.stds[2]:.3f}")
        axf3 = self.ax_comp.inset_axes([0.62,0.11,0.28,0.12]); self.tb_f3 = TextBox(axf3, 'f3',      initial=f"{self.fracs[2]:.3f}")
        # hook
        for tb in [self.tb_m1,self.tb_s1,self.tb_f1,self.tb_m2,self.tb_s2,self.tb_f2,self.tb_m3,self.tb_s3,self.tb_f3]:
            tb.on_submit(self._on_params_text)

    # ---------- Events ----------
    def _on_geom(self, label):
        self.geom = label
        self._reset_plot()

    def _on_ncomp(self, label):
        self.ncomp = int(label)
        # renormalize fractions
        f1 = float(self.tb_f1.text); f2=float(self.tb_f2.text); f3=float(self.tb_f3.text)
        if self.ncomp==2:
            s = max(1e-9, f1+f2)
            self.fracs=[f1/s, f2/s, 0.0]
        else:
            s = max(1e-9, f1+f2+f3)
            self.fracs=[f1/s, f2/s, f3/s]
        self._make_component_boxes()
        self._reset_plot()

    def _on_checks(self, labels):
        states = self.ck.get_status()
        self.animate_flag = states[0]
        self.strict_barriers = states[1]
        self._reset_plot()

    def _on_params_text(self, text):
        self._read_components()
        self._reset_plot()

    def _on_params(self, val):
        self.logN = self.sl_N.val
        self.corr_len = int(self.sl_corr.val)
        self.fuzz = float(self.sl_fuzz.val)
        self.corridor_frac = float(self.sl_corridor.val)
        self.barrier_frac = float(self.sl_barrier.val)
        self.speed = int(self.sl_speed.val)
        self._reset_plot()

    def _read_components(self):
        def clamp01(x):
            try:
                return max(0.0, min(1.0, float(x)))
            except:
                return 0.0
        self.means = [float(self.tb_m1.text or 1.0), float(self.tb_m2.text or 1.0), float(self.tb_m3.text or 1.0)]
        self.stds  = [float(self.tb_s1.text or 0.2), float(self.tb_s2.text or 0.2), float(self.tb_s3.text or 0.2)]
        f1, f2, f3 = clamp01(self.tb_f1.text), clamp01(self.tb_f2.text), clamp01(self.tb_f3.text)
        if self.ncomp==2:
            total = max(1e-9, f1+f2)
            self.fracs = [f1/total, f2/total, 0.0]
            self.tb_f3.set_val('0.000')
        else:
            total = max(1e-9, f1+f2+f3)
            self.fracs = [f1/total, f2/total, f3/total]

    # ---------- Build field and predict ----------
    def _build_field(self):
        self._read_components()
        fr = self.fracs[:self.ncomp]
        means = self.means[:self.ncomp]
        stds  = self.stds[:self.ncomp]
        mus, sigs = [], []
        for m,s in zip(means,stds):
            mu, sg = lognormal_mu_sigma_from_mean_std(m, s)
            mus.append(mu); sigs.append(sg)
        mus = np.array(mus); sigs=np.array(sigs)
        if self.geom=='random':
            k_raw = build_random_field(self.H, self.W, fr, mus, sigs, corr_len=self.corr_len, fuzz=self.fuzz)
        elif self.geom=='parallel':
            k_raw = build_parallel_field(self.H, self.W, fr, mus, sigs, fuzz=self.fuzz, corridor_frac=self.corridor_frac)
        else:
            k_raw = build_series_field(self.H, self.W, fr, mus, sigs, fuzz=self.fuzz, barrier_frac=self.barrier_frac, strict_barriers=self.strict_barriers)
        # Normalize globally
        k, alpha = normalize_mean_k(k_raw)  # k = alpha * k_raw; alpha=1/mean(k_raw)
        # Predicted keff scaled by same alpha
        keff_raw = predicted_keff(self.geom, fr, mus, sigs)
        keff = alpha * keff_raw
        # Predict time ~ W / keff (straight path assumption)
        t_pred = self.W / max(1e-9, keff)
        return k, t_pred, keff

    def _reset_plot(self):
        self.times = []
        self.ax_hist.cla()
        self.ax_hist.set_title('Traversal-time histogram')
        self.ax_hist.set_xlabel('time (sum of 1/k along path)')
        self.ax_hist.set_ylabel('count')
        self._refresh_prediction_text()
        # draw initial field
        k, t_pred, _ = self._build_field()
        self._last_field = k
        self._last_pred  = t_pred
        self.ax_img.cla()
        im = self.ax_img.imshow(k, cmap='magma', origin='lower')
        self.ax_img.set_title(f"{self.geom.upper()} k-field (mean≈1)")
        self.fig.colorbar(im, ax=self.ax_img, fraction=0.046, pad=0.04, label='k')
        # vertical prediction line placeholder
        self._pred_line = self.ax_hist.axvline(t_pred, color='orange', lw=2, ls='--', label='predicted')
        self.ax_hist.legend(loc='upper right')
        self.fig.canvas.draw_idle()

    def _refresh_prediction_text(self):
        txt = (
            f"Geometry: {self.geom} | Components: {self.ncomp}\n"
            f"fractions: {np.array(self.fracs[:self.ncomp]).round(3)}\n"
            f"means k:   {np.array(self.means[:self.ncomp]).round(3)}\n"
            f"stds  k:   {np.array(self.stds[:self.ncomp]).round(3)}\n"
            f"corr_len={self.corr_len}, fuzz={self.fuzz:.2f}, corr(frac)={self.corridor_frac:.2f}, barr(frac)={self.barrier_frac:.2f}\n"
            f"log10 N={self.logN:.2f}, animate={self.animate_flag}, speed={self.speed} fps"
        )
        self.ax_text.cla(); self.ax_text.axis('off'); self.ax_text.text(0.01,0.9, txt, fontsize=9, va='top')

    # ---------- Actions ----------
    def _run_trials(self, evt):
        N = int(10**self.logN)
        # incremental update so histogram appears to grow
        chunk = max(1, N//50)
        remaining = N
        t_pred = self._last_pred
        while remaining>0:
            m = min(chunk, remaining)
            # build a new field each realization and measure time
            for _ in range(m):
                k, _, _ = self._build_field()
                cost = 1.0/(k+1e-9)
                t, path = dijkstra(cost, (self.H//2,0), (self.H//2,self.W-1))
                self.times.append(t)
            remaining -= m
            # update histogram
            self.ax_hist.cla()
            self.ax_hist.hist(self.times, bins=24, color='#4477AA', edgecolor='white')
            self.ax_hist.axvline(t_pred, color='orange', lw=2, ls='--', label='predicted')
            self.ax_hist.set_title(f"Histogram (n={len(self.times)})")
            self.ax_hist.set_xlabel('time'); self.ax_hist.set_ylabel('count')
            self.ax_hist.legend(loc='upper right')
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def _animate_one(self, evt):
        # build field and animate a single path
        k, t_pred, _ = self._build_field()
        cost = 1.0/(k+1e-9)
        t, path = dijkstra(cost, (self.H//2,0), (self.H//2,self.W-1))
        self._last_field = k
        self._last_pred = t_pred
        self.ax_img.cla()
        im = self.ax_img.imshow(k, cmap='magma', origin='lower')
        self.ax_img.set_title(f"{self.geom.upper()} (animate path), t={t:.2f}")
        self.fig.colorbar(im, ax=self.ax_img, fraction=0.046, pad=0.04, label='k')
        xs = [p[1] for p in path]; ys = [p[0] for p in path]
        (ln,) = self.ax_img.plot([], [], color='cyan', lw=1.8)
        (pt,) = self.ax_img.plot([], [], 'wo', ms=6)
        self.fig.canvas.draw_idle()
        fps = max(1, int(self.speed))
        step = 1
        frames = max(1, len(xs)//step)
        def init():
            ln.set_data([],[]); pt.set_data([],[]); return (ln,pt)
        def update(i):
            j = min(i*step, len(xs)-1)
            ln.set_data(xs[:j+1], ys[:j+1])
            pt.set_data(xs[j], ys[j])
            return (ln,pt)
        anim = animation.FuncAnimation(self.fig, update, init_func=init, frames=frames, interval=1000/fps, blit=True)
        plt.pause(frames*(1.0/fps)+0.1)


def main():
    gui = MixingGUI()
    plt.show()

if __name__ == '__main__':
    main()
