"""
Real R (surface relief / root thickness) vs. age-since-collision,
from digitized values (mountain_buoyancy.csv), replacing the earlier
cropped-published-figure version.

Uncertainty: the data file now includes a per-point, one-sided
estimated uncertainty (column UNC). Per direct instruction, this is
applied symmetrically (same magnitude for + and -) rather than
treated as a genuinely asymmetric interval, since only one side was
actually estimated.

Ages: corrected in this version of the data file for an axis-break
the original digitization missed (the source figure switches to a
coarser Ma-per-pixel scale partway across the x-axis) -- no clipping
or other cleanup needed here now, unlike the previous version.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).resolve().parent


def load_R_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"ELEV_ROOT": "R"})
    df = df.dropna(subset=["Name", "Age", "R"])  # drop trailing empty rows
    df = df[["Name", "ID", "Age", "R", "UNC"]]
    return df


def make_figure(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(df["Age"], df["R"], yerr=df["UNC"], fmt='o', ms=4,
                color='0.25', ecolor='0.7', elinewidth=1, capsize=0, zorder=2)
    for _, row in df.iterrows():
        ax.annotate(row["ID"], (row["Age"], row["R"]), fontsize=7,
                    xytext=(4, 3), textcoords='offset points', color='0.4')
    ax.set_xlabel('Time since collision ceased (Ma)')
    ax.set_ylabel('$R$ (relief / root thickness)')
    ax.set_title('Root buoyancy ratio vs. orogen age (digitized from Fischer et al.)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R_vs_age.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'R_vs_age.png', dpi=200, bbox_inches='tight')
    print("Wrote R_vs_age.{pdf,png}")

    young = df[df["Age"] < 300]
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.errorbar(young["Age"], young["R"], yerr=young["UNC"], fmt='o', ms=4,
                 color='0.25', ecolor='0.7', elinewidth=1, capsize=0, zorder=2)
    for _, row in young.iterrows():
        ax2.annotate(row["ID"], (row["Age"], row["R"]), fontsize=7,
                     xytext=(4, 3), textcoords='offset points', color='0.4')
    ax2.set_xlabel('Time since collision ceased (Ma)')
    ax2.set_ylabel('$R$ (relief / root thickness)')
    ax2.set_title('Young orogens only (age < 300 Ma)')
    fig2.tight_layout()
    fig2.savefig(FIGDIR / 'R_vs_age_young.pdf', bbox_inches='tight')
    fig2.savefig(FIGDIR / 'R_vs_age_young.png', dpi=200, bbox_inches='tight')
    print("Wrote R_vs_age_young.{pdf,png}")


if __name__ == '__main__':
    df = load_R_data(Path(__file__).resolve().parent.parent.parent / "data" / "geothermics" / "mountain_buoyancy.csv")
    print(df.to_string(index=False))
    make_figure(df)
