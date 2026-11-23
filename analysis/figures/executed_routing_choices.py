"""Executed routing choices across scenarios, drawn from the per-decision records of the frozen evaluation
(20 seeds x 5 scenarios) in the style of house_style.py.

Each panel stacks the share of decisions on each executed route in 1.5-h bins over the 72-h episode, pooled over the
20 seeds. Decision difference: the share of decisions on which No-Context and AGRI-BRAIN execute different routes at the
same step, per seed, with a 95% BCa interval over seeds.

usage: python executed_routing_choices.py AGRIBRAIN_DIR NOCONTEXT_DIR STUDY_DIR OUTPUT_STEM
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_style import AGB, NOCTX, ROUTE, WIDTH, patch, plt, save, title_and_legend  # noqa: E402

AGB_DIR, NC_DIR, A, OUT = (Path(x) for x in sys.argv[1:5])
SCEN = [('heatwave', 'Heatwave'), ('overproduction', 'Overproduction'), ('cyber_outage', 'Cyber outage'),
        ('adaptive_pricing', 'Adaptive pricing'), ('baseline', 'Baseline')]
SEEDS = range(5001, 5021)
ROUTES = ['cold_chain', 'local_redistribute', 'recovery']
NAMES = dict(zip(ROUTES, ROUTE))
BIN = 6                                   # steps of 15 min -> 1.5 h


def actions(path):
    d = json.load(gzip.open(path))
    rows = [r for r in d['rows'] if not r.get('_header')]
    assert len(rows) == 288, (path, len(rows))
    return d['ari'], [r['action'] for r in rows]


ends = pd.read_csv(A / 'seed_endpoints.csv')
ends = ends[ends.setting == 'main_fail_closed'].set_index(['seed', 'scenario', 'mode'])
data = {}
for s, _ in SCEN:
    for seed in SEEDS:
        a_ari, a = actions(AGB_DIR / f'direct_{seed}_{s}.json.gz')
        n_ari, n = actions(NC_DIR / f'no_context_{seed}_{s}.json.gz')
        assert abs(a_ari - ends.loc[(seed, s, 'agribrain'), 'ari']) < 1e-9, (seed, s, 'agribrain')
        assert abs(n_ari - ends.loc[(seed, s, 'no_context'), 'ari']) < 1e-9, (seed, s, 'no_context')
        for mode, acts in (('agribrain', a), ('no_context', n)):
            shares = {r: np.mean([x == r for x in acts]) for r in ROUTES}
            for r, c in (('cold_chain', 'share_cold_chain'), ('local_redistribute', 'share_redistribution'),
                         ('recovery', 'share_recovery')):
                assert abs(shares[r] - ends.loc[(seed, s, mode), c]) < 1e-9, (seed, s, mode, r)
        data[(s, seed)] = (n, a)

fig, axes = plt.subplots(len(SCEN), 2, figsize=(WIDTH, 12.4), sharex=True, sharey=True,
                         gridspec_kw={'hspace': 0.42, 'wspace': 0.08})
t = np.arange(0, 288 + BIN, BIN) * 0.25
for i, (s, name) in enumerate(SCEN):
    for j, idx in enumerate((0, 1)):                       # 0 No-Context, 1 AGRI-BRAIN
        ax = axes[i, j]
        acts = np.array([data[(s, seed)][idx] for seed in SEEDS])          # 20 x 288
        bottom = np.zeros(288 // BIN)
        for r in ROUTES:
            share = (acts == r).reshape(len(SEEDS), -1, BIN).mean(axis=(0, 2)) * 100
            color, hatch = ROUTE[NAMES[r]]
            ax.stairs(bottom + share, t, baseline=bottom, fill=True, facecolor=color, hatch=hatch, edgecolor='black',
                      linewidth=0.6)
            bottom = bottom + share
        assert np.allclose(bottom, 100)
        ax.set_xlim(0, 72)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 24, 48, 72])
        ax.set_yticks([0, 50, 100])
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if j == 0:
            ax.set_title(f'({"abcde"[i]}) {name}')
        else:
            diff = np.array([np.mean(np.array(data[(s, seed)][0]) != np.array(data[(s, seed)][1])) for seed in SEEDS])
            ci = bootstrap((diff,), np.mean, method='BCa', confidence_level=0.95, n_resamples=9999,
                           random_state=np.random.default_rng(20261122)).confidence_interval
            ax.set_title(f'Decision difference: {100 * diff.mean():.1f}% (95% CI: {100 * ci.low:.1f}–'
                         f'{100 * ci.high:.1f}%)', fontsize=12, fontweight='normal')
            print(f'{name:17s} decision difference {100 * diff.mean():.1f}% [{100 * ci.low:.1f}, {100 * ci.high:.1f}]')
for j in range(2):
    axes[-1, j].set_xlabel('Time (hr.)')
fig.supylabel('Executed route share (%)', fontweight='bold', fontsize=13, x=0.065)
fig.text(0.29, 0.885, 'No-Context', ha='center', color=NOCTX, fontsize=15, fontweight='bold')
fig.text(0.71, 0.885, 'AGRI-BRAIN', ha='center', color=AGB, fontsize=15, fontweight='bold')
fig.subplots_adjust(top=0.85)
title_and_legend(fig, 'Executed routing choices across scenarios', [patch(*ROUTE[r], r) for r in ROUTE], ncol=3,
                 legend_y=0.925, title_y=0.975)
save(fig, str(OUT))
