"""How one routing decision is rebuilt from its record, in the style of house_style.py.

The decision is overproduction, hour 48.25, seed 5001: three context components are active, the guard removes
redistribution because the outlet has no room for the dispatch, and the recorded draw selects the cold chain without
the context and recovery with it.

(a) the recorded contribution of each context component to the score of each route; the diamond is the recorded total
    adjustment, which the contributions sum to
(b) the score of each route without and with the context
(c) the routing probabilities as cumulative bars, without the context, with it, and after the guard; the dashed line
    is the recorded random draw, and the route it falls on is the route taken

Every plotted value is read from the decision record. The script checks that the contributions sum to the recorded
adjustment, that the recorded scores give the recorded probabilities, and that the recorded draw gives the recorded
routes.

usage: python routing_decision_explanation.py AGRIBRAIN_DIR OUTPUT_STEM [SCENARIO HOUR]
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_style import AGB, NOCTX, ROUTE, WIDTH, axes_style, bar_kw, patch, plt, save  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

AGB_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
NAMES = {'heatwave': 'Heatwave', 'overproduction': 'Overproduction', 'cyber_outage': 'Cyber outage',
         'adaptive_pricing': 'Adaptive pricing', 'baseline': 'Baseline'}
SEED = 5001
SCENARIO, HOUR = (sys.argv[3], float(sys.argv[4])) if len(sys.argv) > 4 else ('overproduction', 48.25)
NAME = NAMES[SCENARIO]
ROUTES = ['Cold chain', 'Redistribution', 'Recovery']
ROUTE_STYLE = [ROUTE['Cold chain'], ROUTE['Local redistribution'], ROUTE['Recovery']]
COMPONENTS = [('Operating-envelope severity', '#E69F00', '//'), ('Forecast urgency', '#F0E442', '\\\\'),
              ('Redistribution shortfall', '#56B4E9', '..'), ('Redistribution ineligibility', '#882255', 'xx'),
              ('Recovery shortfall', '#CC79A7', '--')]
BOX = dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.9)
LEGEND = dict(loc='upper left', bbox_to_anchor=(0.0, 1.47), frameon=False, handlelength=2.0, handleheight=1.0,
              borderaxespad=0.0, fontsize=12)


def reason(i, why):
    if why == 'capacity':
        return 'no room'
    if why in ('no_capacity_report', 'missing_report'):
        return 'no capacity report'
    if why == 'acceptance':
        return 'not yet at risk' if i == 2 else 'too spoiled to sell'
    return why


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def drawn(p, u):
    return min(int(np.searchsorted(np.cumsum(p), u, side='right')), 2)


rows = [r for r in json.load(gzip.open(AGB_DIR / f'direct_{SEED}_{SCENARIO}.json.gz'))['rows'] if not r.get('_header')]
r = next(x for x in rows if abs(x['hour'] - HOUR) < 1e-9)
g = r['guard']
mod = np.array(r['context_modifier'], dtype=float)
con = np.array(r['context_feature_contributions'], dtype=float)          # route x component
post = np.array(r['post_context_logits_pre_override'], dtype=float)
without = post - mod
p_without = np.array(r['context_counterfactual_probs'], dtype=float)
p_with = np.array(g['probs_pre_guard'], dtype=float)
p_final = np.array(r['probs'], dtype=float)
u = float(r['policy_categorical_uniform'])
# the reconstruction the figure shows
assert np.abs(con.sum(axis=1) + np.array(r['context_nonfeature_residual'], dtype=float) - mod).max() < 1e-12
assert np.abs(softmax(post / float(r['policy_temperature'])) - p_with).max() < 1e-9
assert drawn(p_without, u) == r['context_counterfactual_action_idx'] != r['action_idx'] == drawn(p_final, u)
assert drawn(p_with, u) == g['sampled_action_pre_guard'] and abs(r['accepted_share'] - 1) < 1e-9
closed = [f'{ROUTES[i].lower()} closed by the guard ({reason(i, why)})' for i, why in enumerate(g['reasons']) if why]
print(f'{NAME}, hour {HOUR}: {ROUTES[r["context_counterfactual_action_idx"]]} without context -> '
      f'{ROUTES[r["action_idx"]]} executed; draw {u:.3f}; {closed}; psi {np.round(r["psi"], 2)}')

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(WIDTH, 5.6),
                                 gridspec_kw={'width_ratios': [1, 1, 1.25], 'wspace': 0.42})
x = np.arange(3)

# (a) contributions
up, down, used = np.zeros(3), np.zeros(3), []
for j, (label, color, hatch) in enumerate(COMPONENTS):
    v = con[:, j]
    if not np.any(np.abs(v) > 1e-12):
        continue
    ax.bar(x, v, 0.55, bottom=np.where(v >= 0, up, down), **bar_kw(color, hatch))
    up, down = up + np.clip(v, 0, None), down + np.clip(v, None, 0)
    used.append(patch(color, hatch, label))
ax.plot(x, mod, 'D', color='black', markersize=9, zorder=4)
ax.axhline(0, color='0.3', linewidth=1)
lim = 1.2 * max(up.max(), -down.min())
ax.set_ylim(-lim, lim)
ax.set_xlim(-0.6, 2.6)
ax.set_xticks(x, ROUTES, rotation=20, ha='right', rotation_mode='anchor')
ax.set_ylabel('Context adjustment')
axes_style(ax, grid='y')
used.append(Line2D([], [], color='black', marker='D', linestyle='', markersize=9, label='Recorded total adjustment'))
ax.legend(handles=used, ncol=1, **LEGEND)
ax.set_title('(a) Context contributions', y=1.49)

# (b) scores
w = 0.36
bx.bar(x - w / 2, without, w, **bar_kw(NOCTX, '..'))
bx.bar(x + w / 2, post, w, **bar_kw(AGB, 'xx'))
bx.axhline(0, color='0.3', linewidth=1)
bx.set_xlim(-0.6, 2.6)
bx.set_xticks(x, ROUTES, rotation=20, ha='right', rotation_mode='anchor')
bx.set_ylabel('Route score')
axes_style(bx, grid='y')
bx.legend(handles=[patch(NOCTX, '..', 'Without context'), patch(AGB, 'xx', 'With context')], ncol=1, **LEGEND)
bx.set_title('(b) Route scores', y=1.49)

# (c) probabilities and the draw
for row, p in zip((2, 1, 0), (p_without, p_with, p_final)):
    left, chosen = 0.0, drawn(p, u)
    for i in range(3):
        if p[i] <= 0:
            continue
        cx.barh(row, p[i], 0.6, left=left, **bar_kw(*ROUTE_STYLE[i]))
        if p[i] >= 0.09:
            cx.text(left + p[i] / 2, row, f'{100 * p[i]:.0f}%', ha='center', va='center', fontsize=12, bbox=BOX)
        if i == chosen:
            cx.add_patch(Rectangle((left, row - 0.3), p[i], 0.6, fill=False, edgecolor='black', linewidth=2.8, zorder=5))
        left += p[i]
    cx.text(1.03, row, f'→ {ROUTES[chosen]}', ha='left', va='center')
cx.axvline(u, color='black', linewidth=1.8, linestyle=(0, (4, 2)), zorder=6)
cx.set_xlim(0, 1)
cx.set_ylim(-0.5, 2.5)
cx.set_yticks([2, 1, 0], ['Without\ncontext', 'With\ncontext', 'After the\nguard'])
cx.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.5', '0.75', '1'])
cx.set_xlabel('Cumulative routing probability')
axes_style(cx, grid=None)
handles = [patch(*ROUTE_STYLE[i], ROUTES[i]) for i in range(3)]
handles.append(Line2D([], [], color='black', linewidth=1.8, linestyle=(0, (4, 2)), label=f'Recorded random draw ({u:.2f})'))
frame = patch('white', '', 'Route the draw falls on')
frame.set_linewidth(2.8)
handles.append(frame)
cx.legend(handles=handles, ncol=2, columnspacing=1.2, **LEGEND)
cx.set_title('(c) Routing probabilities and the draw', y=1.49)

fig.subplots_adjust(top=0.56, bottom=0.17, left=0.07, right=0.90)
fig.suptitle('Explaining an individual routing decision', y=0.985)
fig.text(0.5, 0.905, f'{NAME}, {HOUR:.2f} hr.: ' + ('; '.join(closed) if closed else 'all three routes open'),
         ha='center', va='center', fontsize=12)
save(fig, str(OUT))
