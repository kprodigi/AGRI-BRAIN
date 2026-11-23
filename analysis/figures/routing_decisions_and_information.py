"""Routing decisions of AGRI-BRAIN over time and the information that shaped them, for four scenarios (columns).

Rows, top to bottom:
  disturbance   the scenario's disturbance signal as a grey strip, scaled to its range in that scenario
  routes        probability of each route after the guard (stacked; cold chain, redistribution, recovery)
  retrieval     share of decisions on which the guard removes a route because of the criteria from knowledge retrieval
  capacity      share of decisions on which the guard removes a route because of the capacity reports (MCP tool):
                no room, or no report
  MCP tool      the tool (MCP) contribution to each route's score
Means over the 20 seeds in 1.5-h bins, from the per-decision records.

usage: python routing_decisions_and_information.py AGRIBRAIN_DIR OUTPUT_STEM
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_style import RETRIEVAL, ROUTE, TOOL, WIDTH, patch, plt, save  # noqa: E402

AGB_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = range(5001, 5021)
BIN = 6
SCEN = [
    ('heatwave', '(a) Heatwave', lambda r: r['temp_outcome_environmental']),
    ('overproduction', '(b) Overproduction', lambda r: r['inventory_outcome_environmental']),
    ('cyber_outage', '(c) Cyber outage', lambda r: float(r['mcp_tool_call_count_step'] == 0)),   # tools unavailable
    ('adaptive_pricing', '(d) Adaptive pricing', lambda r: r['phi'][9]),
]
ROUTES = [('Cold chain', 0), ('Local redistribution', 1), ('Recovery', 2)]
CAPACITY = {'capacity', 'no_capacity', 'no_capacity_report'}
STRIP = dict(aspect='auto', interpolation='nearest', extent=(0, 72, 0, 1), vmin=0, vmax=1)
cmap = lambda c: LinearSegmentedColormap.from_list('c', ['white', c])  # noqa: E731


def rows(path):
    return [r for r in json.load(gzip.open(path))['rows'] if not r.get('_header')]


def binmean(series):
    return np.asarray(series, dtype=float).reshape(len(SEEDS), -1, BIN).mean(axis=2).mean(axis=0)


def removed(r, reasons):
    return float(any(x in reasons for x in ((r.get('guard') or {}).get('reasons') or []) if x))


def strip(ax, values, color, label=None):
    ax.imshow(values[None, :], cmap=cmap(color), **STRIP)
    ax.set_yticks([])
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)


t = (np.arange(288 // BIN) * BIN + BIN / 2) * 0.25
edges = np.arange(0, 72.01, BIN * 0.25)
fig, axes = plt.subplots(5, 4, figsize=(WIDTH, 9.6), sharex=True,
                         gridspec_kw={'height_ratios': [0.28, 1.6, 0.28, 0.28, 1.0], 'hspace': 0.18, 'wspace': 0.12})
for j, (s, title, dval) in enumerate(SCEN):
    run = [rows(AGB_DIR / f'direct_{seed}_{s}.json.gz') for seed in SEEDS]
    dist, prob, ret, cap, adj = axes[:, j]
    d = binmean([[dval(r) for r in x] for x in run])
    strip(dist, (d - d.min()) / (d.max() - d.min()), '0.25')
    dist.set_title(title)
    bottom = np.zeros_like(t)
    for name, k in ROUTES:
        share = binmean([[r['probs'][k] for r in x] for x in run])
        color, hatch = ROUTE[name]
        prob.stairs(bottom + share, edges, baseline=bottom, fill=True, facecolor=color, hatch=hatch,
                    edgecolor='black', linewidth=0.5)
        bottom = bottom + share
    prob.set_ylim(0, 1)
    prob.set_yticks([0, 0.5, 1], ['0', '0.5', '1'] if j == 0 else [])
    strip(ret, binmean([[removed(r, {'acceptance'}) for r in x] for x in run]), RETRIEVAL)
    strip(cap, binmean([[removed(r, CAPACITY) for r in x] for x in run]), TOOL)
    for name, k in ROUTES:
        color, _ = ROUTE[name]
        adj.plot(t, binmean([[(r['modifier_mcp'] or [0, 0, 0])[k] for r in x] for x in run]), color=color,
                 linewidth=2)
    adj.axhline(0, color='0.6', linewidth=0.8)
    adj.set_ylim(-0.6, 0.6)
    adj.set_yticks([-0.5, 0, 0.5], ['−0.5', '0', '0.5'] if j == 0 else [])
    adj.grid(axis='y', color='0.9')
    adj.set_xlabel('Time (hr.)')
    for ax in (prob, adj):
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, j]:
        ax.set_xlim(0, 72)
        ax.set_xticks([0, 24, 48, 72])
for i, lab in enumerate(['Disturbance', 'Route\nprobability', 'Knowledge\nretrieval:\nroute removed',
                         'MCP capacity\nreport:\nroute removed', 'MCP tool push\non route score\n(+ favours,\n− disfavours)']):
    axes[i, 0].set_ylabel(lab, rotation=0, ha='right', va='center', labelpad=8)
fig.subplots_adjust(top=0.84, bottom=0.11, left=0.16, right=0.985)
fig.suptitle('Routing decisions and the information that shaped them', y=0.985)
fig.legend(handles=[patch(*ROUTE[n], n) for n, _ in ROUTES], loc='center', bbox_to_anchor=(0.57, 0.925), ncol=3,
           frameon=False, handlelength=2.2, handleheight=1.1, columnspacing=2.2)
fig.text(0.57, 0.02, 'Strips: darker shade = stronger disturbance, or a larger share of decisions with the route '
         'removed (white 0, full colour 1)', ha='center', va='center', fontsize=12)
save(fig, str(OUT))
