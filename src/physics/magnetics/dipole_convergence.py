"""
dipole_convergence.py

Visualizes how the field of two opposite point "monopoles" (+q and -q)
converges toward the familiar dipole pattern as the separation between
them shrinks.

Physically, no isolated magnetic monopole exists -- this is a teaching
device (and the same math as two opposite point electric charges).
Each point pole is modeled with an inverse-square radial field,
B(r) = q * r_hat / r^2, matching a magnetic Coulomb-pole convention.
The combined field of the +q/-q pair is exactly a physical dipole in
the limit of small separation, which is the point of the figure.

Produces two products for teaching use:
  1. Streamline panels at several separations (large -> touching),
     showing the qualitative convergence to a dipole pattern.
  2. Scalar potential contour maps (V = q/r for each pole, summed for
     the pair) at the same separations, suitable for a "students add
     the contour values by hand" exercise.

Usage:
    python dipole_convergence.py
Outputs:
    dipole_convergence_streamlines.png
    dipole_convergence_potential.png
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Field and potential from a single point pole
# ----------------------------------------------------------------------

def pole_field(X, Y, x0, y0, q, eps=1e-3):
    """
    Vector field of a single point pole of strength q at (x0, y0).
    B = q * r_hat / r^2, with a softening length `eps` to avoid a
    singularity exactly at the source (for plotting only).
    """
    dx = X - x0
    dy = Y - y0
    r2 = dx**2 + dy**2 + eps**2
    r = np.sqrt(r2)
    Bx = q * dx / r2**1.5 * r  # = q*dx/r^3 with softening baked into r2
    By = q * dy / r2**1.5 * r
    return Bx, By


def pole_field_simple(X, Y, x0, y0, q, eps=1e-3):
    """Cleaner equivalent: B = q * (dx, dy) / r^3, softened."""
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2 + eps**2)
    Bx = q * dx / r**3
    By = q * dy / r**3
    return Bx, By


def pole_potential(X, Y, x0, y0, q, eps=1e-3):
    """Scalar potential V = q / r for a single point pole (softened)."""
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2 + eps**2)
    return q / r


# ----------------------------------------------------------------------
# Combined field / potential for a +q / -q pair separated by `sep`
# ----------------------------------------------------------------------

def pair_field(X, Y, sep, q=1.0, eps=1e-3):
    """+q pole at (+sep/2, 0), -q pole at (-sep/2, 0)."""
    Bx1, By1 = pole_field_simple(X, Y, sep / 2, 0, q, eps)
    Bx2, By2 = pole_field_simple(X, Y, -sep / 2, 0, -q, eps)
    return Bx1 + Bx2, By1 + By2


def pair_potential(X, Y, sep, q=1.0, eps=1e-3):
    V1 = pole_potential(X, Y, sep / 2, 0, q, eps)
    V2 = pole_potential(X, Y, -sep / 2, 0, -q, eps)
    return V1 + V2


# ----------------------------------------------------------------------
# Figure 1: streamlines at several separations
# ----------------------------------------------------------------------

def make_streamline_figure(separations, extent=4.0, n=400,
                            filename="dipole_convergence_streamlines.png"):
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, len(separations), figsize=(5 * len(separations), 5))
    if len(separations) == 1:
        axes = [axes]

    for ax, sep in zip(axes, separations):
        Bx, By = pair_field(X, Y, sep)
        speed = np.sqrt(Bx**2 + By**2)
        # log-scale color for visibility across large dynamic range
        lw = 1.0
        ax.streamplot(X, Y, Bx, By, color=np.log10(speed + 1e-12),
                      cmap="viridis", density=1.4, linewidth=lw,
                      broken_streamlines=False)
        ax.plot(sep / 2, 0, "o", color="crimson", markersize=10, label="+")
        ax.plot(-sep / 2, 0, "o", color="royalblue", markersize=10, label="-")
        ax.set_title(f"separation = {sep:g}")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Two opposite point poles: convergence to a dipole field", fontsize=14)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


# ----------------------------------------------------------------------
# Figure 2: potential contour maps (for a "students add the numbers" exercise)
# ----------------------------------------------------------------------

def make_potential_figure(separations, extent=4.0, n=300,
                           filename="dipole_convergence_potential.png"):
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, len(separations), figsize=(5 * len(separations), 5))
    if len(separations) == 1:
        axes = [axes]

    levels = np.linspace(-3, 3, 25)  # symmetric levels; clip extreme values near poles
    for ax, sep in zip(axes, separations):
        V = pair_potential(X, Y, sep)
        Vc = np.clip(V, -3, 3)
        cf = ax.contourf(X, Y, Vc, levels=levels, cmap="RdBu_r")
        ax.contour(X, Y, Vc, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
        ax.plot(sep / 2, 0, "o", color="k", markersize=6)
        ax.plot(-sep / 2, 0, "o", color="k", markersize=6)
        ax.set_title(f"separation = {sep:g}")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Scalar potential of a +q/-q pair (red = +, blue = -)", fontsize=14)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


# ----------------------------------------------------------------------
# Figure 3: each pole's own contours, plotted separately but on shared
# axes -- e.g. for a "trace one, then overlay the other" hand exercise
# ----------------------------------------------------------------------

def make_individual_pole_figure(separations, extent=4.0, n=300,
                                 filename="dipole_convergence_individual_poles.png"):
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(2, len(separations),
                              figsize=(5 * len(separations), 10), sharex=True, sharey=True)
    if len(separations) == 1:
        axes = axes.reshape(2, 1)

    # Contour levels: same magnitude scale used for both poles so the
    # two rows are directly comparable / overlay-able.
    levels = np.linspace(0.2, 3, 12)

    for col, sep in enumerate(separations):
        # Top row: positive pole alone, at (+sep/2, 0)
        Vpos = pole_potential(X, Y, sep / 2, 0, q=1.0)
        axp = axes[0, col]
        axp.contour(X, Y, np.clip(Vpos, 0, 3), levels=levels,
                    colors="royalblue", linewidths=1.0)
        axp.plot(sep / 2, 0, "o", color="royalblue", markersize=6)
        axp.plot(-sep / 2, 0, ".", color="lightgray", markersize=4)  # ghost of pair location
        axp.set_title(f"+ pole only, separation = {sep:g}")

        # Bottom row: negative pole alone, at (-sep/2, 0)
        Vneg = pole_potential(X, Y, -sep / 2, 0, q=-1.0)
        axn = axes[1, col]
        axn.contour(X, Y, np.clip(Vneg, -3, 0), levels=-levels[::-1],
                    colors="firebrick", linewidths=1.0)
        axn.plot(-sep / 2, 0, "o", color="firebrick", markersize=6)
        axn.plot(sep / 2, 0, ".", color="lightgray", markersize=4)  # ghost of pair location
        axn.set_title(f"- pole only, separation = {sep:g}")

        for ax in (axp, axn):
            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Each pole's potential contours, plotted separately on shared axes\n"
                 "(blue = positive pole, red = negative pole)", fontsize=14)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


if __name__ == "__main__":
    # Separations chosen to match the progression discussed for the
    # workshop: far apart -> half distance -> touching.
    separations = [6.0, 3.0, 0.2]
    make_streamline_figure(separations)
    make_potential_figure(separations)
    make_individual_pole_figure(separations)