"""ARI performance across scenarios, drawn from the result tables in data/study in the style of house_style.py:
(a) mean ARI of Static, No-Context and AGRI-BRAIN by scenario with 95% BCa intervals; (b) paired ARI gain of AGRI-BRAIN
over No-Context by scenario, labeled with the paired gain. Every plotted value is read from the tables.

usage: python ari_performance_by_scenario.py STUDY_DIR OUTPUT_STEM
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_style import POLICY, WIDTH, axes_style, bar_kw, patch, plt, save, title_and_legend  # noqa: E402

A = Path(sys.argv[1])
OUT = Path(sys.argv[2])
MODES = [('static', 'Static'), ('no_context', 'No-Context'), ('agribrain', 'AGRI-BRAIN')]
SCEN = [('heatwave', 'Heatwave'), ('overproduction', 'Overproduction'), ('cyber_outage', 'Cyber outage'),
        ('adaptive_pricing', 'Adaptive pricing'), ('baseline', 'Baseline')]

m = pd.read_csv(A / 'modes.csv')
m = m[(m.setting == 'main_fail_closed') & (m.metric == 'ari')].set_index(['scenario', 'mode'])
c = pd.read_csv(A / 'comparisons_by_scenario.csv')
c = c[(c.comparison == 'agribrain_minus_no_context') & (c.condition == 'fail-closed')].set_index('scenario')

fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 5.2), gridspec_kw={'width_ratios': [1.35, 1], 'wspace': 0.32})
x = np.arange(len(SCEN))
width = 0.26
for k, (mode, _) in enumerate(MODES):
    r = m.loc[[(s, mode) for s, _ in SCEN]]
    err = [r['mean'] - r['low'], r['high'] - r['mean']]
    ax.bar(x + (k - 1) * width, r['mean'], width, yerr=err, error_kw={'elinewidth': 1.1, 'capsize': 2.5},
           **bar_kw(*POLICY[mode]))
ax.set_xticks(x, [s[1] for s in SCEN], rotation=30, ha='right', rotation_mode='anchor')
ax.set_ylabel('Mean ARI')
ax.set_ylim(0, 0.7)
ax.set_yticks([0, 0.2, 0.4, 0.6], ['0.0', '0.2', '0.4', '0.6'])
axes_style(ax, grid='y')
ax.set_title('(a) Resilience performance under disruption')

y = np.arange(len(SCEN))
rows = c.loc[[s for s, _ in SCEN]]
bx.barh(y, rows['mean'], 0.55, xerr=[rows['mean'] - rows['low'], rows['high'] - rows['mean']],
        error_kw={'elinewidth': 1.1, 'capsize': 2.5}, **bar_kw(*POLICY['agribrain']))
for yi, (s, _) in zip(y, SCEN):
    bx.text(rows.loc[s, 'high'] + 0.0008, yi, f'+{rows.loc[s, "mean"]:.4f}', va='center')
bx.set_yticks(y, [s[1] for s in SCEN])
bx.invert_yaxis()
bx.set_xlim(0, 0.043)
bx.set_xticks([0, 0.01, 0.02, 0.03], ['0', '0.01', '0.02', '0.03'])
bx.set_xlabel('Paired ARI gain')
axes_style(bx, grid=None)
bx.set_title('(b) AGRI-BRAIN’s ARI gain over No-Context')

fig.subplots_adjust(top=0.74)
title_and_legend(fig, 'ARI performance across scenarios', [patch(*POLICY[k], n) for k, n in MODES], ncol=3,
                 legend_y=0.875, title_y=0.985)
save(fig, str(OUT))
for s, name in SCEN:
    print(f'{name:17s}', ' '.join(f'{m.loc[(s, k), "mean"]:.4f}' for k, _ in MODES),
          f'gain {c.loc[s, "mean"]:.4f} [{c.loc[s, "low"]:.4f}, {c.loc[s, "high"]:.4f}]',
          f'+{100 * c.loc[s, "mean"] / m.loc[(s, "no_context"), "mean"]:.2f}%')
