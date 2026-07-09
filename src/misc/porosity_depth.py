from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_DATA = Path(__file__).resolve().parents[2] / 'data' / 'porosity' / 'MallonSwarbrick2002_fig8.csv'
_OUT  = Path(__file__).resolve().parents[2] / 'data' / 'porosity' / 'MallonSwarbrick2002_fig8.eps'

df = pd.read_csv(_DATA, comment='#')
depth_km  = df['depth'] / 1000.0
porosity  = df['porosity']

phi = (1 + np.sqrt(5)) / 2          # golden ratio ≈ 1.618
fig_w = 3.5                          # inches — long dimension in y
fig, ax = plt.subplots(figsize=(fig_w, fig_w * phi))

ax.scatter(porosity, depth_km, s=12, color='k', marker='o', zorder=3)

ax.set_xlabel('Porosity')
ax.set_ylabel('Depth (km)')

# 0,0 in upper-left: x starts at 0, y inverted with 0 at top
ax.set_xlim(left=0)
ax.set_ylim(bottom=depth_km.max() * 1.05, top=0)

ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

fig.tight_layout()
fig.savefig(_OUT, format='eps', bbox_inches='tight')
print(f'Saved {_OUT}')
