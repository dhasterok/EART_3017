"""
Activity 9.A -- Half-space cooling figure
-------------------------------------------------------------------
Plots T(z,t) = T_s + (T_b - T_s) * erf( z / (2 sqrt(kappa t)) ) at
several times, on generic (symbolic, not numeric) axes: T increasing
to the right, z increasing downward -- matching the convention used
throughout this course's geothermics activities (see 8.D).

A vertical dashed line marks T = T_s + 0.84*(T_b - T_s), since
erf(1) ~= 0.8427 -- the temperature level reached at z = ell_T for
any given curve. Because ell_T = 2*sqrt(kappa*t) differs per curve,
this single vertical line crosses each curve at that curve's own
thermal length, which is the point of showing several times at once.

Axes are deliberately unlabelled with numbers -- only T_s, T_b, and
per-curve ell_T markers -- since the point is the shape and scaling
relationship, not particular values.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv
from pathlib import Path

FIGDIR = Path(__file__).resolve().parent
FIGDIR.mkdir(parents=True, exist_ok=True)

# Normalised temperatures: T_s = 0, T_b = 1 (i.e. plotting
# (T - T_s)/(T_b - T_s) but labelling the axis T_s / T_b directly).
T_s, T_b = 0.0, 1.0

# Representative kappa*t products for three times t1 < t2 < t3,
# chosen only for clean, well-separated curve spacing -- not tied to
# any specific kappa or real time value (axes are symbolic).
KAPPA_T_VALUES = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]   # -> ell_T = 2*sqrt(kappa t) = 2, 4, 6, 8, 10
CURVE_LABELS = [r"$t_0$", r"$t_1$", r"$t_2$", r"$t_3$", r"$t_4$", r"$t_5$"]
CURVE_COLORS = ["#aa0000", "#103954", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0"]

ERF_LEVEL = erf(1.0)   # ~0.8427, the level reached at z = ell_T
LABEL_LEVEL = 0.75     # fraction of (T_b - T_s) at which curve labels sit

Z_MAX = 9.0  # generic depth axis extent
LABEL_DZ = 0.3  # downward offset (in z) placing each label just below its curve


def half_space_T(z, kappa_t):
    eta = z / (2.0 * np.sqrt(kappa_t))
    return T_s + (T_b - T_s) * erf(eta)


def main():
    z = np.linspace(0, Z_MAX, 400)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    T_label = T_s + LABEL_LEVEL * (T_b - T_s)

    for kt, label, color in zip(KAPPA_T_VALUES, CURVE_LABELS, CURVE_COLORS):
        if kt == 0.0:
            # t=0: no time has passed for diffusion to act, so the
            # profile is a step -- uniform T_b in the interior, with an
            # instantaneous drop to the fixed surface temperature T_s
            # right at z=0. Draw both pieces explicitly instead of
            # evaluating erf(z/0), which is undefined.
            ax.plot([T_b, T_b], [0, Z_MAX], color=color, linewidth=2.2)
            ax.plot([T_s, T_b], [0, 0], color=color, linewidth=2.2)
            z_label = 0.0
        else:
            T = half_space_T(z, kt)
            ax.plot(T, z, color=color, linewidth=2.2)
            z_label = 2 * np.sqrt(kt) * erfinv(LABEL_LEVEL)

        # curve label, placed just below the curve at T_s + 0.75*DeltaT
        ax.text(T_label, z_label + LABEL_DZ, label, color=color,
                ha="center", va="top", fontsize=10)

        # mark this curve's thermal length: where T crosses the 0.84 level
        ell_T = 2 * np.sqrt(kt)
        ax.plot([T_s + ERF_LEVEL * (T_b - T_s)], [ell_T],
                marker="o", color=color, markersize=5, zorder=5)

    # dashed vertical line at T = T_s + 0.84*(T_b-T_s)
    T_level = T_s + ERF_LEVEL * (T_b - T_s)
    ax.axvline(T_s, color="gray", linestyle="--", linewidth=1.2)
    ax.axvline(T_level, color="gray", linestyle="--", linewidth=1.2)
    ax.text(T_level, -0.35, r"$T_s + 0.84\,\Delta T$",
            ha="center", va="bottom", fontsize=10, color="gray")

    # axis conventions: z increases downward, T increases rightward
    ax.set_xlim(T_s - 0.05, T_b + 0.12)
    ax.set_ylim(Z_MAX, 0)  # inverted: 0 at top

    ax.set_xlabel(r"$T$", fontsize=13)
    ax.set_ylabel(r"$z$", fontsize=13, rotation=0, labelpad=15)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # symbolic tick labels only -- no numeric values
    ax.set_xticks([T_s, T_level, T_b])
    ax.set_xticklabels([r"$T_s$", "", r"$T_b$"])
    ax.set_yticks([])

    # annotate one curve's ell_T explicitly, as a worked example for
    # reading the figure
    ell_T_mid = 2 * np.sqrt(KAPPA_T_VALUES[1])
    ax.annotate(r"$\ell_T=2\sqrt{\kappa t_2}$",
                xy=(T_level, ell_T_mid),
                xytext=(T_level + 0.18, ell_T_mid - 0.6),
                fontsize=10, color=CURVE_COLORS[1],
                arrowprops=dict(arrowstyle="->", color=CURVE_COLORS[1], lw=1.2))

    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGDIR / "halfspace_cooling.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "halfspace_cooling.png", dpi=200, bbox_inches="tight")
    print("Wrote halfspace_cooling.{pdf,png}")


if __name__ == "__main__":
    main()