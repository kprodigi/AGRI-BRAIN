"""Sensitivity to the weight settings, drawn from the result tables in data/study in the style of house_style.py.
Every plotted value is read from the tables.

usage: python weight_sensitivity.py STUDY_DIR OUTPUT_STEM
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_style import (AGB, NOCTX, RETRIEVAL, ROUTE, STATIC, TOOL, WIDTH, axes_style, bar_kw, patch, plt,  # noqa: E402
                         save, title_and_legend)

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.parent.mkdir(parents=True, exist_ok=True)
SCEN = [('heatwave', 'Heatwave'), ('overproduction', 'Overproduction'), ('cyber_outage', 'Cyber outage'),
        ('adaptive_pricing', 'Adaptive pricing'), ('baseline', 'Baseline')]
LABEL_BOX = dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.9)

# ---------------------------------------------------------------- Fig. S1: weight settings
d = pd.read_csv(A / 'sensitivity_settings.csv')
d = d[d.family == 'weights'].reset_index(drop=True)
NAMES = {'base_policy': 'Base-policy prior', 'tool_prior': 'Tool-context prior', 'retrieval_prior': 'Retrieval-context prior',
         'w_c': 'Carbon weight', 'w_l': 'Labor weight', 'w_r': 'Community weight', 'w_p': 'Price weight',
         'eta': 'Waste penalty', 'eta_rho': 'Risk penalty'}


def lab(s):
    if s.startswith('joint'):
        return f'Joint {s[5:7]}% {s[-1]}'
    base, level = s.rsplit('_', 1)
    return f'{NAMES[base]} {"−20%" if level == "minus20" else "+20%"}'


fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 10.5), sharey=True, gridspec_kw={'wspace': 0.1})
y = list(range(len(d)))
g, glo, ghi = d.agribrain_minus_no_context, d.agribrain_minus_no_context_low, d.agribrain_minus_no_context_high
ax.errorbar(g, y, xerr=[g - glo, ghi - g], fmt='none', ecolor='black', elinewidth=1.2, capsize=3)
ax.plot(g, y, '^', color=AGB, markersize=10, markeredgecolor='black', markeredgewidth=0.8, linestyle='', zorder=3)
main = pd.read_csv(A / 'principal_comparisons.csv')
main = main[(main.comparison == 'agribrain_minus_no_context') & (main.condition == 'fail-closed')].iloc[0]
ax.axvline(main['mean'], color='black', linewidth=1.2, linestyle='--')
ax.set_xlim(0, 0.03)
ax.set_xticks([0, 0.01, 0.02, 0.03], ['0', '0.01', '0.02', '0.03'])
ax.set_xlabel('Paired ARI gain')
c, clo, chi = (d.agribrain_minus_no_context_change_from_reference, d.agribrain_minus_no_context_change_low,
               d.agribrain_minus_no_context_change_high)
bx.errorbar(c, y, xerr=[c - clo, chi - c], fmt='none', ecolor='black', elinewidth=1.2, capsize=3)
bx.plot(c, y, 's', color=STATIC, markersize=9, markeredgecolor='black', markeredgewidth=0.8, linestyle='', zorder=3)
bx.axvline(0, color='black', linewidth=1.2, linestyle='--')
bx.set_xlabel('Change in paired ARI gain')
for a_, t in ((ax, '(a) Gain over No-Context'), (bx, '(b) Change from nominal')):
    axes_style(a_, grid='x')
    a_.set_title(t, pad=24)
bx.tick_params(axis='y', left=False)
ax.set_yticks(y, [lab(s) for s in d.setting])
ax.invert_yaxis()
ax.text(main['mean'] + 0.0004, -0.9, 'nominal')
fig.subplots_adjust(top=0.885)
fig.suptitle('Sensitivity to the weight settings', y=0.985)
save(fig, str(OUT))
