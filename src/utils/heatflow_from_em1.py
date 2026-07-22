"""
Activity 8.C -- Radiogenic heat flow from crustal velocity structure
-----------------------------------------------------------------------
Applies the terrane-specific log10(A) = m0*(Vp - 6.0) + m1 models to the
ECM1 layered crustal velocity model, clipped to specific terrane
polygons, to estimate each terrane's crustal radiogenic heat flow
contribution -- properly integrated over REAL layer thicknesses and
velocities, rather than the earlier linear-Vp-to-40-km approximation.

ECM1 file format (data/crustal_thickness/README_ECM1.txt): one row per
1x1 degree (Lon, Lat) grid cell, in "wide" format with three explicit
crustal layers as separate columns -- DLy1-3 are cumulative BASE depths
(km) of layers 1-3 (DLy3 == Hc, the Moho depth), VP1-3 are each layer's
P-wave velocity (km/s). `load_em1_model()` reshapes this into the tidy
long format (lat, lon, layer_name, top_depth_km, base_depth_km, vp_kms)
that everything downstream operates on.

Requires: pandas, numpy, geopandas, shapely, matplotlib
"""

import sys
from pathlib import Path
from io import StringIO

_ROOT = Path(__file__).resolve().parents[2]   # src/utils/ -> project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from src.utils.figure_utils import figutils as fu

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

EM1_PATH = _ROOT / "data" / "crustal_thickness" / "EM1" / "ECM1.txt"

# AuSREM crustal Vp model (5 km depth blocks) + AusMoho2012 Moho-depth grid,
# a much finer alternative to the 1x1 degree, 3-layer EM1 model above.
AUSREM_VP_PATH   = _ROOT / "data" / "crustal_thickness" / "AuSREM" / "AuSREM-C-int5km.txt"
AUSREM_MOHO_PATH = _ROOT / "data" / "crustal_thickness" / "AuSREM" / "AusMoho2012.xyz"

# Global heat flow observation database
HEAT_FLOW_MASTER_PATH = _ROOT / "data" / "geothermics" / "heat_flow" / "master.csv"

# Global geologic provinces shapefile (external repo, not part of EART_3017)
GLOBAL_PROVINCES_SHP = Path(
    "/Users/dhasterok/Documents/GitHub/global_tectonics/plates&provinces/shp/global_gprv.shp"
)

# `prov_name` values in GLOBAL_PROVINCES_SHP to select and union for each terrane
PROVINCE_NAMES = {
    "Macquarie Arc": ["Macquarie Arc"],
    "Yilgarn Craton": ["Eastern Yilgarn Craton", "Western Yilgarn Craton"],
}

FIGDIR = Path(__file__).resolve().parent / "tmp"
fu = fu(FIGDIR)


# ----------------------------------------------------------------------
# Terrane-specific Vp-A models (from vp_hp_models.xlsx), and the
# empirical log-normal mean/median correction measured directly from
# each terrane's raw scatter data (macquarie_arc.csv, yilgarn_craton.csv).
# log10(A) = m0*(Vp - VSHIFT) + m1
# ----------------------------------------------------------------------
VSHIFT = 6.0
HP_VP_DIR = _ROOT / "data" / "geothermics" / "province_hp_vp"

TERRANE_MODELS = {
    "Macquarie Arc":  dict(m0=-0.727209, m1=0.551203, scatter_csv=HP_VP_DIR / "macquarie_arc.csv"),
    "Yilgarn Craton": dict(m0=-1.192703, m1=0.695138, scatter_csv=HP_VP_DIR / "yilgarn_craton.csv"),
}


def measure_lognormal_correction(csv_path, m0, m1, vshift=VSHIFT):
    """
    Compute the empirical log10(A) residual scatter (sigma10) about the
    terrane's own fitted line, directly from its raw (Vp, A) data, and
    return the mean/median correction factor
        K = 10**(sigma10**2 * ln(10) / 2)
    Applying K to a median-based A(z) estimate gives the mean-based
    estimate needed for a physically correct heat-production integral.
    """
    df = pd.read_csv(csv_path)
    logA = np.log10(df["heat_production"])
    pred = m0 * (df["p_velocity"] - vshift) + m1
    sigma10 = (logA - pred).std()
    K = 10 ** (sigma10 ** 2 * np.log(10) / 2)
    return sigma10, K


# ----------------------------------------------------------------------
# ECM1 loading
# ----------------------------------------------------------------------
def load_em1_model(path):
    """
    Load the ECM1 crustal model (wide format, 3 layers/columns per 1x1
    degree grid cell) and reshape it into a tidy long DataFrame:
        lat, lon, layer_name, top_depth_km, base_depth_km, vp_kms
    ECM1's DLy1-3 are cumulative BASE depths of layers 1-3 (DLy3 == Hc,
    the Moho depth), so each layer's top is the previous layer's base
    (0 for layer 1).
    """
    df = pd.read_csv(path, sep="\t")
    required = {"Lon", "Lat", "DLy1", "DLy2", "DLy3", "VP1", "VP2", "VP3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ECM1 file is missing expected columns {missing}.")

    tops  = {1: 0.0,      2: df["DLy1"], 3: df["DLy2"]}
    bases = {1: df["DLy1"], 2: df["DLy2"], 3: df["DLy3"]}

    layers = [
        pd.DataFrame({
            "lat": df["Lat"],
            "lon": df["Lon"],
            "layer_name": f"Ly{n}",
            "top_depth_km": tops[n],
            "base_depth_km": bases[n],
            "vp_kms": df[f"VP{n}"],
        })
        for n in (1, 2, 3)
    ]

    long = pd.concat(layers, ignore_index=True)
    return long.sort_values(["lat", "lon", "top_depth_km"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# AuSREM loading (much finer alternative to EM1)
# ----------------------------------------------------------------------
def load_ausrem_model(vp_path=AUSREM_VP_PATH, moho_path=AUSREM_MOHO_PATH,
                       layer_thickness_km=5.0):
    """
    Load the AuSREM crustal Vp model and the AusMoho2012 Moho-depth grid,
    and combine them into the same tidy long format used by
    `load_em1_model()`:
        lat, lon, layer_name, top_depth_km, base_depth_km, vp_kms

    AuSREM-C-int5km.txt is a fixed-depth-block grid (71 lat x 101 lon x
    9 depths, 0.5 deg / 5 km spacing) with a repeated column-label line
    re-inserted before every depth level's block -- those get stripped
    out here. Each depth sample is treated as the representative velocity
    of a `layer_thickness_km`-thick block whose BASE is at that depth
    (e.g. the depth=5 sample represents 0-5 km, depth=10 represents
    5-10 km, ...).

    AusMoho2012.xyz gives the actual Moho depth on the same 0.5 deg grid
    (a subset of AuSREM's extent). Layers entirely below a cell's Moho
    are dropped, and the deepest crustal layer's base is truncated to the
    real Moho depth -- i.e. Moho sets the depth to integrate heat
    production to, rather than a fixed crustal-model thickness.
    """
    with open(vp_path) as f:
        lines = f.readlines()[14:]                 # drop the 14-line file header
    data_lines = [ln for ln in lines if "|" not in ln and ln.strip()]

    vp = pd.read_csv(
        StringIO("".join(data_lines)), sep=r"\s+",
        names=["depth_km", "lat", "lon", "vp_kms", "vsv_kms", "rho", "mask"],
    )
    vp = vp[vp["mask"] == 1.0].drop(columns=["vsv_kms", "rho", "mask"])

    moho = pd.read_csv(moho_path, sep=r"\s+", names=["lat", "lon", "moho_km"])

    df = vp.merge(moho, on=["lat", "lon"], how="inner")
    if df.empty:
        raise RuntimeError(
            "No overlap between AuSREM velocity grid and AusMoho2012 grid -- "
            "check both files use the same lat/lon convention."
        )

    df["base_depth_km"] = df["depth_km"]
    df["top_depth_km"] = df["depth_km"] - layer_thickness_km

    df = df[df["top_depth_km"] < df["moho_km"]].copy()   # drop sub-Moho layers
    df["base_depth_km"] = np.minimum(df["base_depth_km"], df["moho_km"])  # truncate at Moho
    df["layer_name"] = df["depth_km"].map(lambda d: f"L{d:.0f}")

    long = df[["lat", "lon", "layer_name", "top_depth_km", "base_depth_km", "vp_kms"]]
    return long.sort_values(["lat", "lon", "top_depth_km"]).reset_index(drop=True)


def load_terrane_geometry(provinces_gdf, prov_names):
    """Select province polygon(s) by `prov_name` from the global geologic
    provinces GeoDataFrame and return their union as a single geometry
    (used for the two-part Yilgarn Craton: east + west)."""
    selected = provinces_gdf[provinces_gdf["prov_name"].isin(prov_names)]
    if selected.empty:
        raise RuntimeError(
            f"No provinces matched {prov_names!r} in the global provinces shapefile."
        )
    return selected.union_all()


def clip_em1_to_terrane(em1_long, geometry):
    """Keep only (lat, lon) grid nodes falling inside `geometry`."""
    unique_pts = em1_long[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    points = gpd.GeoDataFrame(
        unique_pts,
        geometry=gpd.points_from_xy(unique_pts["lon"], unique_pts["lat"]),
        crs="EPSG:4326",
    )
    inside = points[points.within(geometry)]
    key = inside[["lat", "lon"]]
    return em1_long.merge(key, on=["lat", "lon"], how="inner")


# ----------------------------------------------------------------------
# Observed heat flow (data/geothermics/heat_flow/master.csv)
# ----------------------------------------------------------------------
def load_heat_flow_data(path=HEAT_FLOW_MASTER_PATH):
    """
    Load the global heat flow observation database and build a single
    best-available heat flow value per site: `heat_flow_corrected` where
    present, falling back to `heat_flow_uncorrected` otherwise. Returns
    columns: lat, lon, heat_flow_mwm2, heat_flow_source ('corrected' or
    'uncorrected'). Sites with neither value are dropped.
    """
    df = pd.read_csv(
        path, encoding="utf-8-sig", low_memory=False,
        usecols=["latitude", "longitude", "heat_flow_uncorrected", "heat_flow_corrected"],
    )
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    has_corrected = df["heat_flow_corrected"].notna()
    df["heat_flow_mwm2"] = df["heat_flow_corrected"].where(
        has_corrected, df["heat_flow_uncorrected"]
    )
    df["heat_flow_source"] = np.where(has_corrected, "corrected", "uncorrected")

    df = df.dropna(subset=["heat_flow_mwm2", "lat", "lon"])
    return df[["lat", "lon", "heat_flow_mwm2", "heat_flow_source"]].reset_index(drop=True)


def clip_points_to_terrane(points_df, geometry):
    """Keep only rows of `points_df` (with lat/lon columns) falling inside
    `geometry`."""
    points = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df["lon"], points_df["lat"]),
        crs="EPSG:4326",
    )
    return points[points.within(geometry)].drop(columns="geometry").reset_index(drop=True)


def compute_terrane_heat_flow_stats(terrane_name, heat_flow_df, provinces_gdf, prov_names):
    """
    Clip the observed heat flow database to a terrane's polygon and
    compute summary statistics (count, mean, median, std, min, max) of
    the best-available heat flow (corrected preferred, uncorrected as
    fallback -- see `load_heat_flow_data`).
    """
    geometry = load_terrane_geometry(provinces_gdf, prov_names)
    clipped = clip_points_to_terrane(heat_flow_df, geometry)

    n_corrected = (clipped["heat_flow_source"] == "corrected").sum()
    n_uncorrected = (clipped["heat_flow_source"] == "uncorrected").sum()

    stats = {
        "n": len(clipped),
        "n_corrected": int(n_corrected),
        "n_uncorrected": int(n_uncorrected),
        "mean": clipped["heat_flow_mwm2"].mean(),
        "median": clipped["heat_flow_mwm2"].median(),
        "std": clipped["heat_flow_mwm2"].std(),
        "min": clipped["heat_flow_mwm2"].min(),
        "max": clipped["heat_flow_mwm2"].max(),
    }

    print(f"\n--- {terrane_name}: observed heat flow (master.csv) ---")
    if stats["n"] == 0:
        print("  No heat flow observations found inside this terrane's polygon.")
    else:
        print(f"  Sites: {stats['n']}  "
              f"({stats['n_corrected']} corrected, {stats['n_uncorrected']} uncorrected)")
        print(f"  mean={stats['mean']:.1f}  median={stats['median']:.1f}  "
              f"std={stats['std']:.1f}  min={stats['min']:.1f}  max={stats['max']:.1f}  mW/m2")

    return clipped, stats


# ----------------------------------------------------------------------
# Core physics: layer Vp -> layer A -> integrated crustal heat flow
# ----------------------------------------------------------------------
def _log10_A(vp_kms, m0, m1, vshift=VSHIFT):
    """
    log10(A) = m0*(Vp - vshift) + m1, with Vp clipped to a floor of
    `vshift` (6 km/s) before evaluating the model.

    The Vp-A regressions are calibrated against crystalline crustal rock
    samples (Vp >~ 6 km/s); below that, m0's negative slope extrapolates
    to implausibly high heat production (e.g. for AuSREM's near-surface
    depth blocks, which include unconsolidated sediment/weathered
    material far outside the calibration range). Clipping Vp at `vshift`
    caps A at its value there (m0*(vshift-vshift) = 0, so log10(A) = m1)
    instead of extrapolating further.
    """
    vp_clipped = np.maximum(vp_kms, vshift)
    return m0 * (vp_clipped - vshift) + m1


def heat_flow_contribution_per_cell(cell_layers, m0, m1, K, vshift=VSHIFT):
    """
    For one (lat, lon) grid cell's layers (a DataFrame with
    top_depth_km, base_depth_km, vp_kms), compute:
        q_median  -- crustal radiogenic heat flow using median-based A
        q_mean    -- using the log-normal-corrected mean-based A
    via q = sum_i( A_i * thickness_i ), A_i piecewise-constant per layer.
    Returns (q_median_mW_m2, q_mean_mW_m2).
    """
    cl = cell_layers.dropna(subset=["base_depth_km"])
    thickness_km = cl["base_depth_km"] - cl["top_depth_km"]
    logA = _log10_A(cl["vp_kms"], m0, m1, vshift)
    A_median = 10 ** logA          # uW/m3
    A_mean = A_median * K          # uW/m3

    # integral of A [uW/m3] * thickness [km] -> uW/m2 -> mW/m2
    # (1 uW/m3 * 1 km = 1 uW/m3 * 1000 m = 1000 uW/m2 = 1 mW/m2)
    q_median = (A_median * thickness_km).sum() * 1.0   # mW/m2, see note below
    q_mean = (A_mean * thickness_km).sum() * 1.0
    return q_median, q_mean


def _step_xy(tops, bases, values):
    """Interleave per-layer (top, base, value) triples into (value, depth)
    arrays that trace a piecewise-constant staircase when plotted with
    `ax.plot(value, depth)`."""
    tops = np.asarray(tops, dtype=float)
    bases = np.asarray(bases, dtype=float)
    values = np.asarray(values, dtype=float)
    depth = np.empty(2 * len(values))
    value = np.empty(2 * len(values))
    depth[0::2] = tops
    depth[1::2] = bases
    value[0::2] = values
    value[1::2] = values
    return value, depth


def compute_depth_profiles(clipped, m0, m1, K, vshift=VSHIFT):
    """
    Build mean-based heat-production-vs-depth staircase profiles for every
    EM1 grid cell in `clipped`, plus a single profile from the terrane's
    typical velocity-depth structure (mean top/base depth and median Vp
    per layer, then converted to A). Returns (cell_profiles, mean_profile),
    each profile a (heat_production, depth_km) pair of arrays.
    """
    cell_profiles = []
    for (lat, lon), cell in clipped.sort_values("top_depth_km").groupby(["lat", "lon"]):
        logA = _log10_A(cell["vp_kms"], m0, m1, vshift)
        A_mean = (10 ** logA) * K
        cell_profiles.append(
            _step_xy(cell["top_depth_km"], cell["base_depth_km"], A_mean)
        )

    layer_avg = (
        clipped.groupby("layer_name")
        .agg(top_depth_km=("top_depth_km", "mean"),
             base_depth_km=("base_depth_km", "mean"),
             vp_kms=("vp_kms", "median"))
        .sort_values("top_depth_km")
    )
    logA_avg = _log10_A(layer_avg["vp_kms"], m0, m1, vshift)
    A_avg = (10 ** logA_avg) * K
    mean_profile = _step_xy(layer_avg["top_depth_km"], layer_avg["base_depth_km"], A_avg)

    return cell_profiles, mean_profile


def compute_velocity_profiles(clipped):
    """
    Build Vp-vs-depth staircase profiles for every EM1 grid cell in
    `clipped`, plus the terrane's typical velocity-depth structure (mean
    top/base depth and median Vp per layer). Returns (cell_profiles,
    mean_profile), each profile a (vp_kms, depth_km) pair of arrays.
    """
    cell_profiles = []
    for (lat, lon), cell in clipped.sort_values("top_depth_km").groupby(["lat", "lon"]):
        cell_profiles.append(
            _step_xy(cell["top_depth_km"], cell["base_depth_km"], cell["vp_kms"])
        )

    layer_avg = (
        clipped.groupby("layer_name")
        .agg(top_depth_km=("top_depth_km", "mean"),
             base_depth_km=("base_depth_km", "mean"),
             vp_kms=("vp_kms", "median"))
        .sort_values("top_depth_km")
    )
    mean_profile = _step_xy(layer_avg["top_depth_km"], layer_avg["base_depth_km"],
                             layer_avg["vp_kms"])

    return cell_profiles, mean_profile


def compute_velocity_depth_stats(clipped):
    """
    Per-depth-interval summary statistics of Vp for a terrane's clipped
    long-format DataFrame. Returns a DataFrame with columns:
        layer_name, top_depth_km, base_depth_km, n, vp_median, vp_mean, vp_std
    sorted by top_depth_km.
    """
    stats = (
        clipped.groupby("layer_name")
        .agg(top_depth_km=("top_depth_km", "mean"),
             base_depth_km=("base_depth_km", "mean"),
             n=("vp_kms", "size"),
             vp_median=("vp_kms", "median"),
             vp_mean=("vp_kms", "mean"),
             vp_std=("vp_kms", "std"))
        .sort_values("top_depth_km")
        .reset_index(drop=True)
    )
    return stats


def compute_terrane_heat_flow(terrane_name, em1_long, provinces_gdf, prov_names, model):
    geometry = load_terrane_geometry(provinces_gdf, prov_names)
    clipped = clip_em1_to_terrane(em1_long, geometry)
    if clipped.empty:
        raise RuntimeError(
            f"No EM1 grid cells found inside {terrane_name}'s polygon -- "
            "check the shapefile CRS matches EM1's lat/lon (EPSG:4326), "
            "and that the shapefile actually overlaps the EM1 grid extent."
        )

    sigma10, K = measure_lognormal_correction(
        model["scatter_csv"], model["m0"], model["m1"]
    )

    results = []
    for (lat, lon), cell in clipped.groupby(["lat", "lon"]):
        q_med, q_mean = heat_flow_contribution_per_cell(
            cell, model["m0"], model["m1"], K
        )
        results.append(dict(lat=lat, lon=lon, q_median=q_med, q_mean=q_mean))

    results = pd.DataFrame(results)
    print(f"\n--- {terrane_name} ---")
    print(f"  EM1 grid cells inside polygon: {len(results)}")
    print(f"  sigma10 = {sigma10:.3f}, log-normal correction K = {K:.3f}")
    print(f"  q_median: mean={results['q_median'].mean():.1f}  "
          f"median={results['q_median'].median():.1f}  mW/m2")
    print(f"  q_mean:   mean={results['q_mean'].mean():.1f}  "
          f"median={results['q_mean'].median():.1f}  mW/m2")
    return results


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_heat_flow_distribution(results_by_terrane, savename="11_terrane_heatflow_distribution"):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"Macquarie Arc": "tab:orange", "Yilgarn Craton": "tab:blue"}
    for name, df in results_by_terrane.items():
        ax.hist(df["q_mean"], bins=25, alpha=0.5, label=f"{name} (mean-based)",
                color=colors.get(name))
    ax.set_xlabel("Crustal radiogenic heat flow contribution (mW/m$^2$)")
    ax.set_ylabel("EM1 grid cells")
    ax.set_title("Predicted crustal radiogenic heat flow, by terrane")
    ax.legend(fontsize=8)
    ax.tick_params(direction="out", top=True, right=True)
    fig.tight_layout()
    fu.savefig(fig, savename)
    plt.show()


def plot_heat_production_depth(profiles_by_terrane, savename="12_heat_production_depth"):
    """Side-by-side A(z) staircase plots per terrane: every EM1 grid
    cell's profile in grey (transparent), with the terrane's average
    velocity-depth-derived profile overplotted."""
    colors = {"Macquarie Arc": "tab:orange", "Yilgarn Craton": "tab:blue"}
    n = len(profiles_by_terrane)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, (cell_profiles, mean_profile)) in zip(axes, profiles_by_terrane.items()):
        for A_cell, z_cell in cell_profiles:
            ax.plot(A_cell, z_cell, color="0.4", alpha=0.25, lw=0.8)
        ax.plot(*mean_profile, color=colors.get(name, "k"), lw=2.2,
                label="Median velocity-depth profile")
        ax.set_xlabel("Heat production (µW/m$^3$)")
        ax.set_title(name)
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(direction="out", top=True, right=True)

    axes[0].set_ylabel("Depth (km)")
    axes[0].invert_yaxis()
    fig.tight_layout()
    fu.savefig(fig, savename)
    plt.show()


def plot_velocity_depth(profiles_by_terrane, savename="12b_velocity_depth"):
    """Side-by-side Vp(z) staircase plots per terrane: every EM1 grid
    cell's velocity profile in grey (transparent), with the terrane's
    median velocity-depth profile overplotted."""
    colors = {"Macquarie Arc": "tab:orange", "Yilgarn Craton": "tab:blue"}
    n = len(profiles_by_terrane)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, (cell_profiles, mean_profile)) in zip(axes, profiles_by_terrane.items()):
        for vp_cell, z_cell in cell_profiles:
            ax.plot(vp_cell, z_cell, color="0.4", alpha=0.25, lw=0.8)
        ax.plot(*mean_profile, color=colors.get(name, "k"), lw=2.2,
                label="Median velocity-depth profile")
        ax.set_xlabel("P-wave velocity (km/s)")
        ax.set_title(name)
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(direction="out", top=True, right=True)

    axes[0].set_ylabel("Depth (km)")
    axes[0].invert_yaxis()
    fig.tight_layout()
    fu.savefig(fig, savename)
    plt.show()


def export_velocity_depth_latex(stats_by_terrane, savename="15_velocity_depth_stats"):
    """
    Export per-depth-interval P-wave velocity statistics (median, mean,
    std) for each terrane as a single LaTeX table, written to
    FIGDIR/`savename`.tex.
    """
    rows = []
    for name, stats in stats_by_terrane.items():
        for _, r in stats.iterrows():
            rows.append({
                "Terrane": name,
                "Depth (km)": f"{r.top_depth_km:.0f}--{r.base_depth_km:.1f}",
                "n": int(r.n),
                "Median $V_p$ (km/s)": f"{r.vp_median:.2f}",
                "Mean $V_p$ (km/s)": f"{r.vp_mean:.2f}",
                "Std $V_p$ (km/s)": f"{r.vp_std:.2f}" if pd.notna(r.vp_std) else "--",
            })
    table_df = pd.DataFrame(rows)

    latex = table_df.to_latex(
        index=False, escape=False,
        caption="P-wave velocity statistics by depth interval for each terrane.",
        label="tab:velocity_depth_stats",
        column_format="llrccc",
    )

    out_path = fu.FIGDIR / f"{savename}.tex"
    out_path.write_text(latex)
    print(f"Wrote LaTeX table to {out_path}")
    return table_df


def plot_vp_vs_heat_production(model_dict, log=False,
                                savename="13_vp_vs_heat_production"):
    """Side-by-side Vp vs. heat production scatter plots per terrane,
    from each terrane's raw (Vp, A) scatter data."""
    colors = {"Macquarie Arc": "tab:orange", "Yilgarn Craton": "tab:blue"}
    n = len(model_dict)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, model_dict.items()):
        df = pd.read_csv(model["scatter_csv"])
        yvals = np.log10(df["heat_production"]) if log else df["heat_production"]
        ax.scatter(df["p_velocity"], yvals, s=8, alpha=0.35,
                   color=colors.get(name, "k"), edgecolor="none")
        ax.set_xlabel("P-wave velocity (km/s)")
        ax.set_ylabel(r"$\log_{10}$ heat production (µW/m$^3$)" if log
                      else "Heat production (µW/m$^3$)")
        ax.set_title(name)
        ax.tick_params(direction="out", top=True, right=True)

    fig.tight_layout()
    fu.savefig(fig, savename)
    plt.show()


def main():
    print("Loading AuSREM crustal velocity model + AusMoho2012 Moho grid ...")
    em1_long = load_ausrem_model(AUSREM_VP_PATH, AUSREM_MOHO_PATH)

    print("Loading global geologic provinces shapefile ...")
    provinces_gdf = gpd.read_file(GLOBAL_PROVINCES_SHP)

    print("Loading observed heat flow database (master.csv) ...")
    heat_flow_df = load_heat_flow_data(HEAT_FLOW_MASTER_PATH)

    results_by_terrane = {}
    profiles_by_terrane = {}
    velocity_profiles_by_terrane = {}
    velocity_stats_by_terrane = {}
    observed_heat_flow_by_terrane = {}
    for name, model in TERRANE_MODELS.items():
        results_by_terrane[name] = compute_terrane_heat_flow(
            name, em1_long, provinces_gdf, PROVINCE_NAMES[name], model
        )

        geometry = load_terrane_geometry(provinces_gdf, PROVINCE_NAMES[name])
        clipped = clip_em1_to_terrane(em1_long, geometry)
        _, K = measure_lognormal_correction(
            model["scatter_csv"], model["m0"], model["m1"]
        )
        profiles_by_terrane[name] = compute_depth_profiles(
            clipped, model["m0"], model["m1"], K
        )
        velocity_profiles_by_terrane[name] = compute_velocity_profiles(clipped)
        velocity_stats_by_terrane[name] = compute_velocity_depth_stats(clipped)

        observed_heat_flow_by_terrane[name] = compute_terrane_heat_flow_stats(
            name, heat_flow_df, provinces_gdf, PROVINCE_NAMES[name]
        )

    plot_heat_flow_distribution(results_by_terrane)
    plot_heat_production_depth(profiles_by_terrane)
    plot_velocity_depth(velocity_profiles_by_terrane)
    export_velocity_depth_latex(velocity_stats_by_terrane)
    plot_vp_vs_heat_production(TERRANE_MODELS, log=False)
    plot_vp_vs_heat_production(TERRANE_MODELS, log=True,
                                savename="14_vp_vs_log_heat_production")

    print("\nWrote figures to", FIGDIR)


if __name__ == "__main__":
    main()
