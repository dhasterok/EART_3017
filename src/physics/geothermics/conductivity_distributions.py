from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── xlsx parsing ──────────────────────────────────────────────────────────────
# The file uses the strict OOXML namespace, which openpyxl does not support.
_NS = 'http://purl.oclc.org/ooxml/spreadsheetml/main'

def _shared_strings(z):
    root = ET.fromstring(z.read('xl/sharedStrings.xml'))
    result = []
    for si in root.findall(f'{{{_NS}}}si'):
        result.append(''.join(
            t.text for t in si.iter(f'{{{_NS}}}t') if t.text
        ))
    return result

def _col_index(ref):
    col_str = ''.join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in col_str:
        idx = idx * 26 + ord(ch) - ord('A') + 1
    return idx

def _cell_value(cell, strings):
    v = cell.find(f'{{{_NS}}}v')
    if v is None or v.text is None:
        return None
    if cell.get('t') == 's':
        return strings[int(v.text)]
    try:
        return float(v.text)
    except ValueError:
        return v.text

def _load_database(path):
    """Return lists of (rock_type, conductivity, sio2, rock_origin) from 'A. Database'."""
    rock_types, conductivities, sio2_vals, origins = [], [], [], []
    with zipfile.ZipFile(path, 'r') as z:
        strings = _shared_strings(z)
        root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        for row in root.iter(f'{{{_NS}}}row'):
            if int(row.get('r')) == 1:        # skip header
                continue
            cells = {_col_index(c.get('r')): _cell_value(c, strings)
                     for c in row.findall(f'{{{_NS}}}c')}
            rt  = cells.get(5)                # column E: Rock Type
            k   = cells.get(50)               # column AX: Thermal Conductivity
            si  = cells.get(6)                # column F: SiO2
            ori = cells.get(3)                # column C: Rock Origin
            if rt is not None and k is not None:
                rock_types.append(rt)
                conductivities.append(float(k))
                sio2_vals.append(float(si) if si is not None else float('nan'))
                origins.append(ori or '')
    return rock_types, conductivities, sio2_vals, origins


# ── classification ────────────────────────────────────────────────────────────
_MERGE = {
    'quartzolite': 'quartzite',
}
_MIN_N = 5

_IGNEOUS_ORIGINS = {'plutonic', 'volcanic', 'metaplutonic', 'metavolcanic'}
_SEDIM_ORIGINS   = {'clastic', 'metasedimentary'}


def _classify(origin_counts):
    """Return 'igneous' or 'sedimentary' based on dominant Rock Origin."""
    ign = sum(origin_counts.get(o, 0) for o in _IGNEOUS_ORIGINS)
    sed = sum(origin_counts.get(o, 0) for o in _SEDIM_ORIGINS)
    return 'sedimentary' if sed > ign else 'igneous'


# ── statistics & ordering ─────────────────────────────────────────────────────

def _group_stats(rock_types, conductivities, sio2_vals, origins):
    k_groups      = defaultdict(list)
    sio2_groups   = defaultdict(list)
    origin_counts = defaultdict(lambda: defaultdict(int))

    for rt, k, si, ori in zip(rock_types, conductivities, sio2_vals, origins):
        rt = _MERGE.get(rt, rt)
        k_groups[rt].append(k)
        if not np.isnan(si):
            sio2_groups[rt].append(si)
        if ori:
            origin_counts[rt][ori] += 1

    stats = {}
    for rt, vals in k_groups.items():
        if len(vals) < _MIN_N:
            continue
        a = np.array(vals)
        stats[rt] = {
            'n':        len(a),
            'p5':       np.percentile(a,  5),
            'p25':      np.percentile(a, 25),
            'p50':      np.percentile(a, 50),
            'p75':      np.percentile(a, 75),
            'p95':      np.percentile(a, 95),
            'sio2':     np.mean(sio2_groups[rt]) if sio2_groups[rt] else float('nan'),
            'category': _classify(origin_counts[rt]),
        }

    ordered = sorted(stats.keys(), key=lambda rt: stats[rt]['sio2'])
    return stats, ordered


# ── plotting ──────────────────────────────────────────────────────────────────
_COLORS = {
    'igneous':     '#4878CF',   # blue
    'sedimentary': '#D65F5F',   # red
}

def _blend_with_white(hex_color, alpha=0.4):
    """Return a solid RGB tuple equivalent to hex_color at given alpha over white."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (alpha*r + (1 - alpha), alpha*g + (1 - alpha), alpha*b + (1 - alpha))

def plot_conductivity_distributions(ax=None):
    data_file = Path(__file__).resolve().parents[3] / 'data' / 'thermal_conductivity_data.xlsx'
    rock_types, conductivities, sio2_vals, origins = _load_database(data_file)
    stats, ordered = _group_stats(rock_types, conductivities, sio2_vals, origins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(6, len(ordered) * 0.4)))
    else:
        fig = ax.figure

    rect_half = 0.125   # half-height of p25–p75 rectangle (total = 0.25)

    for y, rt in enumerate(ordered):
        s = stats[rt]
        color      = _COLORS[s['category']]
        fill_color = _blend_with_white(color, alpha=0.4)

        # p5–p95 horizontal line
        ax.plot([s['p5'], s['p95']], [y, y], color=color, linewidth=1.2, zorder=2)

        # p25–p75 filled rectangle (solid color, no alpha — keeps PDF vector)
        rect = mpatches.FancyBboxPatch(
            (s['p25'], y - rect_half),
            s['p75'] - s['p25'],
            2 * rect_half,
            boxstyle='square,pad=0',
            linewidth=0.8,
            edgecolor=color,
            facecolor=fill_color,
            zorder=3,
        )
        ax.add_patch(rect)

        # p50 vertical tick
        ax.plot([s['p50'], s['p50']], [y - rect_half, y + rect_half],
                color=color, linewidth=1.5, zorder=4)

        # count text above the rectangle
        ax.text(s['p50'], y + rect_half + 0.04, f"N={s['n']}",
                ha='center', va='bottom', fontsize=7, color='#333333', zorder=5)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered)
    ax.set_ylim(-0.6, len(ordered) - 0.4)
    ax.set_xlabel(r'Thermal Conductivity (W m$^{-1}$ K$^{-1}$)')
    ax.set_title('Thermal Conductivity by Rock Type')
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(facecolor=_blend_with_white(_COLORS['igneous']),
                       edgecolor=_COLORS['igneous'], linewidth=0.8,
                       label='Igneous / Metaigneous'),
        mpatches.Patch(facecolor=_blend_with_white(_COLORS['sedimentary']),
                       edgecolor=_COLORS['sedimentary'], linewidth=0.8,
                       label='Sedimentary / Metasedimentary'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

    fig.tight_layout()
    return fig, ax


if __name__ == '__main__':
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plot_conductivity_distributions()
    fig.savefig('conductivity_distributions.svg')
    plt.show()
