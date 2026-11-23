"""Analyses computed from the retained records of the study.

No adaptive mode is run here. Each function takes the retained records of
primary cells and returns rows for the tables written by ``analyze.py``:

* the exact split of a paired gain into route merit and refusal avoided, and
  the same gain by route switch;
* the paired gain before, during and after the disruption;
* the executed decisions re-scored under alternative index parameters;
* the contribution of each context component to the applied adjustment, how
  far the physical state predicts each component, and how often a capacity
  component is active on a route the guard has already removed;
* the size of the adjustment against the gaps between base scores, and how
  often the clip, the sign constraint and the change bound are active;
* rule policies replayed on the recorded environments, to show how the value
  of a diversion depends on the share of the dispatch that is accepted;
* loss, retained fraction and recovery time relative to the baseline scenario;
* activity of each role.

Re-scoring and the gain decomposition are accounting of decisions that were
made; they do not say how a policy would behave under other parameters. The
rule replays do follow the consequences of each rule through the episode: the
capacity of the destinations is updated by the rule's own decisions.
"""
from __future__ import annotations

import inference
import numpy as np
import validate

import study

ROUTES = validate.ACTIONS
STEPS = 288
ENVIRONMENT = {
    'rho_outcome_environmental': 'rho_outcome_environmental_trace',
    'inventory_outcome_environmental': 'inventory_outcome_environmental_trace',
    'temp_outcome_environmental': 'temp_outcome_environmental_trace',
    'rh_outcome_environmental': 'rh_outcome_environmental_trace',
    'transport_multiplier_outcome_environmental':
        'transport_multiplier_outcome_environmental_trace',
}
OBSERVED = {'rho_policy_observed': 'rho_policy_observed_trace',
            'inventory_policy_observed': 'inventory_policy_observed_trace',
            'temp_policy_observed': 'temp_policy_observed_trace',
            'rh_policy_observed': 'rh_policy_observed_trace'}
# (mode, reference): the gain decomposed is mode minus reference.
DECOMPOSITION_PAIRS = [
    ('agribrain', 'no_context'), ('agribrain_eligibility', 'no_context'),
    ('agribrain', 'agribrain_eligibility'),
    ('reservation_only', 'no_context'), ('eligibility_only', 'no_context'),
    ('agribrain_eligibility_no_peer', 'eligibility_only'),
    ('agribrain', 'rule_capacity_informed'),
]
RESCORED_MODES = ['static', 'no_context', 'eligibility_only', 'agribrain_eligibility_no_peer',
                  'reservation_only', 'agribrain_no_peer', 'agribrain',
                  'agribrain_eligibility', *study.RULES]
RESCORED_CONTRASTS = [('agribrain', 'no_context'), ('agribrain', 'agribrain_eligibility'),
                      ('agribrain_eligibility_no_peer', 'eligibility_only'),
                      ('agribrain', 'rule_capacity_informed')]
# Modes whose decision records are read for the component and score analyses.
LEDGER_MODES = ['adjustment_only', 'agribrain_eligibility_no_peer', 'agribrain_no_peer',
                'agribrain_eligibility', 'agribrain']
SOCIAL_WEIGHTS = {'w_c': 'emissions', 'w_l': 'labour', 'w_r': 'community',
                  'w_p': 'price_information'}
# First and last hour of the disturbance, counted from its onset offset
# (``scenario_engine``); ``None`` means it lasts to the end of the episode.
DISTURBANCE = {'heatwave': (24.0, 48.0), 'overproduction': (12.0, 60.0),
               'cyber_outage': (24.0, None)}
# The tool channel of the cyber outage goes down at this hour, without offset.
TOOL_OUTAGE_HOUR = 24.0
RECOVERY_LEVEL = 0.95
RECOVERY_WINDOW = 8  # steps: two hours
NEIGHBOURS = 10
# Share of the dispatch a priority rule waits for before it diverts.
RESERVATION_SHARES = [('any room', 1e-9), ('at least 20%', 0.2), ('at least 40%', 0.4),
                      ('at least 60%', 0.6), ('at least 80%', 0.8), ('100%', 1.0)]
QUANTILES = [5, 25, 50, 75, 95]


class Cell:
    """Retained records of one cell: environment, observations, traces."""

    def __init__(self, record):
        results = record['results']
        first = results.get('static') or next(iter(results.values()))
        self.task = record['task']
        self.parameters = record['actual_parameters']
        self.policy = self.parameters['policy']
        self.rows = [dict(zip(ENVIRONMENT, values, strict=True)) for values in zip(
            *(first[trace] for trace in ENVIRONMENT.values()), strict=True)]
        observer = results.get('rule_capacity_informed') or first
        self.observed = {key: observer[trace] for key, trace in OBSERVED.items()}
        self.hours = np.asarray(first['hours'], dtype=float)
        self.demand = first['demand_outcome_environmental_trace']
        self.onset_offset = float(first['scenario_onset_offset_hours'])
        self.spoilage = (first['effective_k_ref'], first['effective_Ea_R'])
        model = first.get('destination_capacity_model') or {}
        self.penalty_scale = float(model.get('refusal_penalty_scale', 1.0))
        self.refused_share_lost = bool(model.get('refused_share_lost', False))
        self.traces, self.learners = {}, {}
        for mode, result in results.items():
            if any(result[trace] != first[trace] for trace in ENVIRONMENT.values()):
                raise ValueError('Modes of one cell do not share the environment')
            shares = result.get('accepted_share_trace') or [1.0] * STEPS
            self.traces[mode] = {'ari': np.asarray(result['ari_trace'], dtype=float),
                                 'action': np.asarray(result['action_trace'], dtype=int),
                                 'share': np.asarray(shares, dtype=float)}
            summary = result.get('learner_summary') or {}
            if summary.get('initial_theta') is not None:
                self.learners[mode] = {key: summary.get(key) for key in (
                    'initial_theta', 'final_theta', 'sign_constrained', 'sign_reversal_count',
                    'magnitude_cap_mode', 'magnitude_cap_value', 'magnitude_cap_abs_floor',
                    'n_updates')}
        self._model = validate.OutcomeModel(
            self.policy, self.parameters, *self.spoilage, 'agribrain', self.penalty_scale,
            self.refused_share_lost)
        self._full = None

    def outcome(self, step, action, share, **alternative):
        return self._model.step(self.rows[step], ROUTES[action], share, **alternative)

    def full_acceptance_ari(self):
        """ARI of each route at each step if the destination took the whole dispatch."""
        if self._full is None:
            self._full = np.array([[self.outcome(t, a, 1.0)['ari'] for a in range(3)]
                                   for t in range(STEPS)])
        return self._full

    def check_reconstruction(self, mode):
        trace = self.traces[mode]
        rebuilt = np.array([self.outcome(t, trace['action'][t], trace['share'][t])['ari']
                            for t in range(STEPS)])
        error = float(np.max(np.abs(rebuilt - trace['ari'])))
        if error > 1e-9:
            raise ValueError(f'{self.task}: {mode} does not reconstruct from its traces ({error})')
        return error


def scenario_keys(seeds, scenario):
    scenarios = study.SCENARIOS if scenario == 'overall' else [scenario]
    return [(seed, sc) for seed in seeds for sc in scenarios]


# ---------------------------------------------------------------------------
# Gain decomposition
# ---------------------------------------------------------------------------
def decompose(cell, mode, reference):
    """Split the paired step differences of one cell; every part is a step mean.

    ``merit`` is the difference the two executed routes would give if each
    dispatch were accepted in full; ``refusal`` is the difference in what
    refusals cost. Their sum is the paired gain. ``switch[i][j]`` is the part
    of the gain on steps where the reference took route i and the mode route j.
    """
    full = cell.full_acceptance_ari()
    a, b = cell.traces[mode], cell.traces[reference]
    steps = np.arange(STEPS)
    full_a, full_b = full[steps, a['action']], full[steps, b['action']]
    merit = full_a - full_b
    refusal = (a['ari'] - full_a) - (b['ari'] - full_b)
    total = a['ari'] - b['ari']
    if float(np.max(np.abs(merit + refusal - total))) > 1e-12:
        raise ValueError('Decomposition is not exact')
    switch = np.zeros((3, 3, 3))  # reference route, mode route, (share, merit, refusal)
    for i in range(3):
        for j in range(3):
            mask = (b['action'] == i) & (a['action'] == j)
            switch[i, j] = (mask.mean(), merit[mask].sum() / STEPS, refusal[mask].sum() / STEPS)
    return {'gain': float(total.mean()), 'merit': float(merit.mean()),
            'refusal': float(refusal.mean()), 'switch': switch}


def decomposition_tables(cells, seeds, label):
    """Rows of the gain decomposition and of the gain by route switch."""
    parts, switches = [], []
    for mode, reference in DECOMPOSITION_PAIRS:
        split = {key: decompose(cell, mode, reference) for key, cell in cells.items()}
        for scenario in [*study.SCENARIOS, 'overall']:
            scenarios = study.SCENARIOS if scenario == 'overall' else [scenario]
            for part in ('gain', 'merit', 'refusal'):
                values = [np.mean([split[(seed, sc)][part] for sc in scenarios]) for seed in seeds]
                parts.append({'setting': label, 'mode': mode, 'reference': reference,
                              'scenario': scenario, 'part': part, **inference.estimate(
                                  values, ('decomposition', label, mode, reference, scenario, part))})
            table = np.mean([split[key]['switch'] for key in scenario_keys(seeds, scenario)], axis=0)
            for i in range(3):
                for j in range(3):
                    switches.append({
                        'setting': label, 'mode': mode, 'reference': reference,
                        'scenario': scenario,
                        'reference_route': ROUTES[i], 'mode_route': ROUTES[j],
                        'step_share': float(table[i, j, 0]),
                        'gain': float(table[i, j, 1] + table[i, j, 2]),
                        'merit': float(table[i, j, 1]), 'refusal': float(table[i, j, 2])})
    return parts, switches


# ---------------------------------------------------------------------------
# Gain before, during and after the disruption
# ---------------------------------------------------------------------------
def disruption_windows(cell):
    """Step masks of the windows of a disturbed cell; ``None`` without a disturbance.

    In the cyber outage the window starts with the earlier of the environmental
    disturbance and the loss of the tool channel.
    """
    scenario = cell.task['scenario']
    bounds = DISTURBANCE.get(scenario)
    if bounds is None:
        return None
    start = bounds[0] + cell.onset_offset
    if scenario == 'cyber_outage':
        start = min(start, TOOL_OUTAGE_HOUR)
    if bounds[1] is None:
        return {'before': cell.hours < start, 'during': cell.hours >= start}
    end = bounds[1] + cell.onset_offset
    return {'before': cell.hours < start,
            'during': (cell.hours >= start) & (cell.hours <= end),
            'after': cell.hours > end}


def window_table(cells, seeds, pairs, label):
    """Paired gain per step within each window of the disturbed scenarios."""
    rows = []
    for mode, reference in pairs:
        for scenario in DISTURBANCE:
            by_window = {}
            for seed in seeds:
                cell = cells[(seed, scenario)]
                difference = cell.traces[mode]['ari'] - cell.traces[reference]['ari']
                for name, mask in disruption_windows(cell).items():
                    if mask.any():
                        by_window.setdefault(name, []).append(float(difference[mask].mean()))
            for name, values in by_window.items():
                if len(values) != len(seeds):
                    continue  # the window is empty in some seed
                rows.append({'setting': label, 'mode': mode, 'reference': reference,
                             'scenario': scenario, 'window': name, **inference.estimate(
                                 values, ('window', label, mode, reference, scenario, name))})
    return rows


# ---------------------------------------------------------------------------
# Re-scoring under alternative index parameters
# ---------------------------------------------------------------------------
def alternatives(policy):
    """Alternative index parameters, declared before the run."""
    weights = {key: policy[key] for key in SOCIAL_WEIGHTS}
    variants = {'declared': {},
                'equal_social_weights': {'weights': dict.fromkeys(SOCIAL_WEIGHTS, 0.25)}}
    for key, name in SOCIAL_WEIGHTS.items():
        doubled = {k: v * (2.0 if k == key else 1.0) for k, v in weights.items()}
        total = sum(doubled.values())
        variants[f'{name}_weight_doubled'] = {
            'weights': {k: v / total for k, v in doubled.items()}}
    variants['emissions_normalizer_halved'] = {'carbon_cap': 0.5 * policy['carbon_cap']}
    variants['emissions_normalizer_doubled'] = {'carbon_cap': 2.0 * policy['carbon_cap']}
    return variants


def rescore_cell(cell, modes):
    """Episode means of the index of each mode under each variant, for one cell."""
    scores = {}
    for mode in modes:
        trace = cell.traces[mode]
        for variant, alternative in alternatives(cell.policy).items():
            steps = [cell.outcome(t, trace['action'][t], trace['share'][t], **alternative)
                     for t in range(STEPS)]
            scores[(variant, mode)] = float(np.mean([s['ari'] for s in steps]))
            if variant == 'declared':
                kept = 1.0 - np.array([s['waste'] for s in steps])
                social = np.array([s['slca'] for s in steps])
                sound = 1.0 - np.array([s['rho'] for s in steps])
                scores[('additive_aggregation', mode)] = float(np.mean((kept + social + sound) / 3.0))
                scores[('geometric_aggregation', mode)] = float(np.mean(np.cbrt(kept * social * sound)))
                scores[('factor_one_minus_waste', mode)] = float(kept.mean())
                scores[('factor_social_performance', mode)] = float(social.mean())
                scores[('factor_one_minus_risk', mode)] = float(sound.mean())
    return scores


def rescoring_tables(cells, seeds, label):
    modes = [m for m in RESCORED_MODES if all(m in cell.traces for cell in cells.values())]
    scores = {key: rescore_cell(cell, modes) for key, cell in cells.items()}
    variants = sorted({variant for variant, _ in next(iter(scores.values()))})

    def vector(variant, mode):
        return np.array([np.mean([scores[(seed, sc)][(variant, mode)] for sc in study.SCENARIOS])
                         for seed in seeds])

    by_mode, contrasts = [], []
    for variant in variants:
        means = {mode: float(vector(variant, mode).mean()) for mode in modes}
        order = sorted(modes, key=lambda m: -means[m])
        for mode in modes:
            by_mode.append({'setting': label, 'variant': variant, 'mode': mode,
                            'mean': means[mode], 'rank': order.index(mode) + 1})
        for mode, reference in RESCORED_CONTRASTS:
            if mode in modes and reference in modes:
                difference = vector(variant, mode) - vector(variant, reference)
                contrasts.append({'setting': label, 'variant': variant, 'mode': mode,
                                  'reference': reference, **inference.estimate(
                                      difference, ('rescoring', label, variant, mode, reference))})
    return by_mode, contrasts


# ---------------------------------------------------------------------------
# Decision records of the modes with a score adjustment
# ---------------------------------------------------------------------------
def ledger_arrays(rows):
    """The fields of an evaluation ledger used by the analyses, as arrays."""
    removed = np.zeros((len(rows), 3), dtype=bool)
    clipped = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        for route in (row.get('guard') or {}).get('removed_routes') or []:
            removed[index, route] = True
        # The clip is active where its recorded derivative is zero.
        for stage in ('primary', 'cooperative', 'composition'):
            derivative = ((row.get('context_integration') or {}).get(stage) or {}).get(
                'clip_derivative')
            clipped[index] |= derivative is not None and min(derivative) == 0.0
    return {
        'action': np.array([row['action_idx'] for row in rows], dtype=int),
        'probs': np.array([row['probs'] for row in rows], dtype=float),
        'clipped': clipped,
        'role': [row['role'] for row in rows],
        'psi': np.array([row['psi'] for row in rows], dtype=float),
        'phi': np.array([row['phi'] for row in rows], dtype=float),
        'contribution': np.array([row['context_feature_contributions'] for row in rows],
                                 dtype=float),
        'residual': np.array([row['context_nonfeature_residual'] for row in rows], dtype=float),
        'modifier': np.array([row['context_modifier'] for row in rows], dtype=float),
        'base': np.array([row['base_logits'] for row in rows], dtype=float),
        'temperature': np.array([row['policy_temperature'] for row in rows], dtype=float),
        'removed': removed,
        'peer_bias': np.array([row['peer_message_bias'] for row in rows], dtype=float),
        'tool_calls': np.array([len(row['primary_mcp_tools_invoked_step']) for row in rows]),
        'peer_messages_read': np.array(
            [row['step_channel_evidence']['peer']['consumed_count'] for row in rows]),
        'cooperative_overlay': np.array(
            [bool(row['step_channel_evidence']['cooperative']['active']) for row in rows]),
    }


def linear_r2(features, target):
    """In-sample R² of a linear fit; ``None`` when the target does not vary."""
    centred = target - target.mean()
    total = float(centred @ centred)
    if total <= 0.0:
        return None
    design = np.column_stack([np.ones(len(target)), features])
    residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
    return 1.0 - float(residual @ residual) / total


def neighbour_r2(features, target, groups, k=NEIGHBOURS):
    """Out-of-sample R² of a k-nearest-neighbour prediction, leaving one seed out.

    Unlike the linear fit this follows thresholds and other nonlinear
    functions of the state. ``None`` when the target does not vary.
    """
    centred = target - target.mean()
    total = float(centred @ centred)
    if total <= 0.0:
        return None
    spread = features.std(axis=0)
    scaled = features[:, spread > 0] / spread[spread > 0]
    predicted = np.empty(len(target))
    for group in np.unique(groups):
        test = groups == group
        train_x, train_y = scaled[~test], target[~test]
        train_norm = (train_x ** 2).sum(axis=1)
        index = np.flatnonzero(test)
        for start in range(0, len(index), 512):
            block = scaled[index[start:start + 512]]
            # Squared distances up to a term that is constant for each test point.
            distance = train_norm[None, :] - 2.0 * block @ train_x.T
            nearest = np.argpartition(distance, k, axis=1)[:, :k]
            predicted[index[start:start + 512]] = train_y[nearest].mean(axis=1)
    return 1.0 - float(((target - predicted) ** 2).sum()) / total


def component_tables(ledgers, seeds, mode, label):
    """Contribution of each context component, its overlap with the state and
    with the guard.

    ``ledgers`` maps (seed, scenario) to the arrays of one mode.
    """
    for key, arrays in ledgers.items():
        allocation = arrays['contribution'].sum(axis=2) + arrays['residual']
        if float(np.max(np.abs(allocation - arrays['modifier']))) > 1e-9:
            raise ValueError(f'{key}: component contributions do not sum to the adjustment')
    # Route whose guard removal coincides with each capacity or eligibility component.
    guarded_route = {2: 1, 3: 1, 4: 2}
    contributions, overlap = [], []
    for scenario in [*study.SCENARIOS, 'overall']:
        keys = scenario_keys(seeds, scenario)
        psi = np.concatenate([ledgers[k]['psi'] for k in keys])
        phi = np.concatenate([ledgers[k]['phi'] for k in keys])
        signed = np.concatenate([ledgers[k]['contribution'] for k in keys])
        removed = np.concatenate([ledgers[k]['removed'] for k in keys])
        magnitude = np.abs(signed)
        groups = np.concatenate([np.full(STEPS, seed) for seed, _ in keys])
        route_total = magnitude.sum(axis=(0, 2))
        for component in range(5):
            active = psi[:, component] != 0
            row = {'setting': label, 'mode': mode, 'scenario': scenario, 'component': component,
                   'mean_value': float(psi[:, component].mean()),
                   'active_step_share': float(active.mean())}
            for route in range(3):
                name = ROUTES[route]
                row[f'mean_contribution_{name}'] = float(signed[:, route, component].mean())
                row[f'share_of_adjustment_{name}'] = (
                    float(magnitude[:, route, component].sum() / route_total[route])
                    if route_total[route] > 0 else None)
            row['share_of_adjustment_all_routes'] = (
                float(magnitude[:, :, component].sum() / magnitude.sum())
                if magnitude.sum() > 0 else None)
            if component in guarded_route:
                # Of the steps where the component is active, how often the guard
                # has already removed the route the component speaks against.
                row['active_while_route_removed_share'] = (
                    float(removed[active, guarded_route[component]].mean())
                    if active.any() else None)
            contributions.append(row)
            overlap.append({
                'setting': label, 'mode': mode, 'scenario': scenario, 'component': component,
                'variance': float(psi[:, component].var()),
                'linear_r2': linear_r2(phi, psi[:, component]),
                'neighbour_r2_leave_one_seed_out': neighbour_r2(phi, psi[:, component], groups),
                'observations': int(len(psi))})
    return contributions, overlap


def softmax(values):
    values = np.exp(values - values.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def score_influence_table(ledgers_by_mode, seeds, label):
    """Size of the adjustment against the gaps between base scores, per mode."""
    rows = []
    for mode, ledgers in ledgers_by_mode.items():
        for scenario in [*study.SCENARIOS, 'overall']:
            keys = scenario_keys(seeds, scenario)
            base = np.concatenate([ledgers[k]['base'] for k in keys])
            modifier = np.concatenate([ledgers[k]['modifier'] for k in keys])
            removed = np.concatenate([ledgers[k]['removed'] for k in keys])
            temperature = np.concatenate([ledgers[k]['temperature'] for k in keys])[:, None]
            clipped = np.concatenate([ledgers[k]['clipped'] for k in keys])
            order = np.argsort(-base, axis=1)
            steps = np.arange(len(base))
            gap = base[steps, order[:, 0]] - base[steps, order[:, 1]]
            shift = modifier[steps, order[:, 0]] - modifier[steps, order[:, 1]]
            variation = 0.5 * np.abs(softmax(base / temperature)
                                     - softmax((base + modifier) / temperature)).sum(axis=1)
            available_base = np.where(removed, -np.inf, base)
            available_adjusted = np.where(removed, -np.inf, base + modifier)
            row = {'setting': label, 'mode': mode, 'scenario': scenario,
                   'decisions': int(len(base)),
                   'leading_route_changed_share': float(
                       (base.argmax(axis=1) != (base + modifier).argmax(axis=1)).mean()),
                   'leading_available_route_changed_share': float(
                       (available_base.argmax(axis=1) != available_adjusted.argmax(axis=1)).mean()),
                   'clip_active_decisions': int(clipped.sum())}
            for name, values in (('base_score_gap', gap), ('adjustment_leader_minus_runner_up', shift),
                                 ('largest_adjustment', np.abs(modifier).max(axis=1)),
                                 ('probability_change', variation)):
                for q in QUANTILES:
                    row[f'{name}_p{q:02d}'] = float(np.percentile(values, q))
            rows.append(row)
    return rows


def change_bound(learner):
    """Largest permitted distance of each context weight from its initial value."""
    initial = np.asarray(learner['initial_theta'], dtype=float)
    if learner['magnitude_cap_mode'] == 'relative_delta':
        return np.maximum(learner['magnitude_cap_value'] * np.abs(initial),
                          learner['magnitude_cap_abs_floor'])
    if learner['magnitude_cap_mode'] == 'absolute':
        # The bound is on the weight itself.
        return float(learner['magnitude_cap_value']) - np.abs(initial)
    raise ValueError(f"Unknown bound {learner['magnitude_cap_mode']}")


def constraint_activity_table(cells, seeds, label):
    """State of the constraints on the context weights after adaptation.

    A sign-constrained weight that an update would have carried through zero
    is held at zero, so a weight at zero with a nonzero initial value marks the
    sign constraint as binding. A reversal is a weight whose final sign is
    opposite to its initial sign; it can occur only without the constraint.
    """
    rows = []
    modes = sorted({mode for cell in cells.values() for mode in cell.learners})
    for mode in modes:
        reversals = at_zero = at_bound = cells_at_bound = cells_changed = updates = weights = 0
        largest_ratio, largest_change, constrained = 0.0, 0.0, None
        for key in scenario_keys(seeds, 'overall'):
            learner = cells[key].learners[mode]
            initial = np.asarray(learner['initial_theta'], dtype=float)
            final = np.asarray(learner['final_theta'], dtype=float)
            change, bound = np.abs(final - initial), change_bound(learner)
            constrained = bool(learner['sign_constrained'])
            weights += initial.size
            reversals += int(learner['sign_reversal_count'] or 0)
            at_zero += int(((initial != 0.0) & (final == 0.0)).sum()) if constrained else 0
            updates += int(learner['n_updates'] or 0)
            reached = change >= bound - 1e-9
            at_bound += int(reached.sum())
            cells_at_bound += int(reached.any())
            cells_changed += int(change.max() > 0.0)
            largest_change = max(largest_change, float(change.max()))
            largest_ratio = max(largest_ratio, float((change / bound).max()))
        rows.append({'setting': label, 'mode': mode, 'cells': len(seeds) * len(study.SCENARIOS),
                     'weights': weights, 'weight_updates': updates,
                     'cells_with_changed_weights': cells_changed,
                     'sign_constrained': constrained,
                     'weights_held_at_zero_by_the_sign_constraint': at_zero,
                     'sign_reversals_in_final_weights': reversals,
                     'weights_at_change_bound': at_bound,
                     'cells_with_a_weight_at_bound': cells_at_bound,
                     'largest_weight_change': largest_change,
                     'largest_change_as_share_of_bound': largest_ratio})
    return rows


def peer_influence_table(ledgers_by_mode, seeds, label):
    """Routing with and without peer messages, at each guard level.

    The two modes of a pair share the environment and the random draws.
    ``expected_changed_decisions`` is the sum over decisions of the total
    variation distance between their routing probabilities.
    """
    rows = []
    for mode, reference in (('agribrain', 'agribrain_no_peer'),
                            ('agribrain_eligibility', 'agribrain_eligibility_no_peer')):
        for scenario in [*study.SCENARIOS, 'overall']:
            keys = scenario_keys(seeds, scenario)
            a = {name: np.concatenate([ledgers_by_mode[mode][k][name] for k in keys])
                 for name in ('probs', 'action', 'peer_bias', 'peer_messages_read')}
            b = {name: np.concatenate([ledgers_by_mode[reference][k][name] for k in keys])
                 for name in ('probs', 'action')}
            distance = 0.5 * np.abs(a['probs'] - b['probs']).sum(axis=1)
            rows.append({
                'setting': label, 'mode': mode, 'reference': reference, 'scenario': scenario,
                'decisions': int(len(distance)),
                'decisions_with_a_peer_message_read': int((a['peer_messages_read'] > 0).sum()),
                'decisions_with_a_peer_bias': int(
                    (np.abs(a['peer_bias']).max(axis=1) > 1e-12).sum()),
                'largest_peer_bias': float(np.abs(a['peer_bias']).max()),
                'decisions_with_different_probabilities': int((distance > 1e-12).sum()),
                'largest_probability_change': float(distance.max()),
                'expected_changed_decisions': float(distance.sum()),
                'decisions_with_a_different_route': int((a['action'] != b['action']).sum())})
    return rows


# ---------------------------------------------------------------------------
# Rule policies replayed on the recorded environments
# ---------------------------------------------------------------------------
def replay_rule(cell, rule, reservation_share=None, closed=None):
    """Follow a rule reference through the recorded environment of a cell.

    Capacity is updated by the rule's own decisions. Returns the episode ARI,
    the routes taken, and the diverted and refused volume.
    """
    policy = cell.policy
    config = {key: policy[field] for key, field in validate.CAPACITY_PARAMETERS.items()}
    closed = bool(policy['missing_confirmation_closed']) if closed is None else closed
    share = float(policy['reservation_share']) if reservation_share is None else reservation_share
    criteria = (validate.published_criteria(policy) if policy['acceptance_document_present']
                else 'unavailable')
    predictor = validate.OutcomeModel(
        policy, cell.parameters, policy['k_ref'], policy['Ea_R'], 'static',
        policy['refusal_penalty_scale'], policy['refused_share_lost'])
    room = {1: config['bmax_lr'], 2: config['bmax_rec']}
    age = int(policy['capacity_report_age_steps'])
    history, ari, actions, diverted, refused = [], [], [], 0.0, 0.0
    for t in range(STEPS):
        row = {**cell.rows[t], **{key: values[t] for key, values in cell.observed.items()}}
        size = row['inventory_outcome_environmental'] / config['inventory_ref']
        observed_size = max(row['inventory_policy_observed'], 0.0) / config['inventory_ref']
        history.append(dict(room))
        offline = (cell.task['scenario'] == 'cyber_outage'
                   and cell.hours[t] >= TOOL_OUTAGE_HOUR)
        action = validate.rule_action(
            rule, row, criteria, 'unavailable' if offline else history[max(0, t - age)],
            observed_size, share, closed, predictor)
        accepted = 1.0
        if action:
            if not validate.accepted_by_destination(
                    ROUTES[action], row['rho_outcome_environmental'], config):
                accepted = 0.0
            elif size > 0.0:
                accepted = min(1.0, max(0.0, room[action]) / size)
            room[action] = max(0.0, room[action] - accepted * size)
            diverted += size
            refused += (1.0 - accepted) * size
        ari.append(cell.outcome(t, action, accepted)['ari'])
        actions.append(action)
        room[1] = min(config['bmax_lr'],
                      room[1] + config['kappa_lr'] * cell.demand[t] / config['demand_ref'])
        room[2] = min(config['bmax_rec'], room[2] + config['kappa_rec'])
    return float(np.mean(ari)), actions, diverted, refused


def check_rule_replay(cells):
    """The replay must give the routes the simulator recorded for the rules."""
    for key, cell in cells.items():
        for mode, rule in (('rule_capacity_informed', 'capacity_informed'),
                           ('rule_greedy_informed', 'greedy_informed'),
                           ('rule_risk_threshold', 'risk_threshold')):
            if replay_rule(cell, rule)[1] != cell.traces[mode]['action'].tolist():
                raise ValueError(f'{key}: replay of {mode} differs from the simulator')


def reservation_tradeoff_table(cells, seeds, label):
    """A priority rule that diverts at a given accepted share, the greedy rule
    and always-redistribute; paired with full reservation."""
    rules = [(f'priority rule, diverts with {name}', 'capacity_informed', share)
             for name, share in RESERVATION_SHARES]
    rules += [('greedy rule, partial acceptance allowed', 'greedy_informed', None),
              ('always redistribute', 'always_redistribute', None)]
    outcomes = {name: {key: replay_rule(cell, rule, share) for key, cell in cells.items()}
                for name, rule, share in rules}
    reference = 'priority rule, diverts with 100%'

    def vector(name, scenario):
        return np.array([np.mean([outcomes[name][k][0] for k in scenario_keys([seed], scenario)])
                         for seed in seeds])

    rows = []
    for name, _, _ in rules:
        cells_of = outcomes[name]
        diverted = sum(v[2] for v in cells_of.values())
        row = {'setting': label, 'rule': name,
               'ari': float(vector(name, 'overall').mean()),
               'refused_share_of_diverted_volume': (
                   sum(v[3] for v in cells_of.values()) / diverted if diverted else 0.0)}
        for route in range(3):
            row[f'share_{ROUTES[route]}'] = float(np.mean(
                [np.mean(np.asarray(v[1]) == route) for v in cells_of.values()]))
        for scenario in study.SCENARIOS:
            row[f'ari_{scenario}'] = float(vector(name, scenario).mean())
        if name != reference:
            difference = inference.estimate(
                vector(reference, 'overall') - vector(name, 'overall'),
                ('tradeoff', label, name))
            row.update({'full_reservation_minus_rule': difference['mean'],
                        'difference_low': difference['low'], 'difference_high': difference['high'],
                        'seeds_full_reservation_higher': difference['positive_seeds']})
        rows.append(row)
    return rows


def break_even_share(cell, step, action):
    """Smallest accepted share at which a diversion is worth at least the cold
    chain at one step: 0 if it always is, ``None`` if it never is.

    Waste and social performance are linear in the accepted share, so the
    step value is a quadratic in it.
    """
    full, none = cell.outcome(step, action, 1.0), cell.outcome(step, action, 0.0)
    cold = cell.outcome(step, 0, 1.0)
    target = (1.0 - cold['waste']) * cold['slca']

    def value(share):
        waste = none['waste'] + (full['waste'] - none['waste']) * share
        social = none['slca'] + (full['slca'] - none['slca']) * share
        return (1.0 - waste) * social - target

    if value(0.0) >= 0.0:
        return 0.0
    if value(1.0) < 0.0:
        return None
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if value(middle) < 0.0 else (low, middle)
    return high


def break_even_table(cells, seeds, label):
    """How often each destination accepts the product, and the distribution of
    the break-even accepted share over those steps."""
    rows = []
    policy = next(iter(cells.values())).policy
    config = {key: policy[field] for key, field in validate.CAPACITY_PARAMETERS.items()}
    for scenario in [*study.SCENARIOS, 'overall']:
        for action in (1, 2):
            shares, never, eligible = [], 0, 0
            for key in scenario_keys(seeds, scenario):
                cell = cells[key]
                for step in range(STEPS):
                    if not validate.accepted_by_destination(
                            ROUTES[action], cell.rows[step]['rho_outcome_environmental'], config):
                        continue
                    eligible += 1
                    share = break_even_share(cell, step, action)
                    if share is None:
                        never += 1
                    else:
                        shares.append(share)
            steps = STEPS * len(scenario_keys(seeds, scenario))
            row = {'setting': label, 'scenario': scenario, 'route': ROUTES[action],
                   'steps': steps, 'eligible_steps': eligible,
                   'destination_accepts_share': eligible / steps,
                   'never_worth_the_diversion_share': never / eligible if eligible else None}
            for q in QUANTILES:
                row[f'break_even_share_p{q:02d}'] = (
                    float(np.percentile(shares, q)) if shares else None)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Loss, retained fraction and recovery relative to the baseline scenario
# ---------------------------------------------------------------------------
def rolling_ratio(scenario, baseline):
    """Performance over the next two hours relative to the baseline scenario."""
    window = np.ones(RECOVERY_WINDOW)
    return (np.convolve(scenario, window, mode='valid')
            / np.convolve(baseline, window, mode='valid'))


def recovery_hours(scenario, baseline, end_hour):
    """Hours from the end of the disturbance until performance stays recovered.

    Recovered means at least 95% of the baseline-scenario performance of the
    same mode and seed over two hours, for two consecutive hours. ``None`` when
    this is not reached within the episode.
    """
    ratio = rolling_ratio(scenario, baseline)
    start = max(0, int(np.ceil(end_hour * 4)))
    for index in range(start, len(ratio) - RECOVERY_WINDOW + 1):
        if np.all(ratio[index:index + RECOVERY_WINDOW] >= RECOVERY_LEVEL):
            return index * 0.25 - end_hour
    return None


def resilience_tables(cells, seeds, modes, label):
    """Loss and recovery descriptors of each mode, and paired contrasts."""
    rows, per_seed = [], {}
    disturbed = [s for s in study.SCENARIOS if s != 'baseline']
    for scenario in disturbed:
        ends = DISTURBANCE.get(scenario, (None, None))[1]
        for mode in modes:
            loss, retained, drop, recovery = [], [], [], []
            for seed in seeds:
                q = cells[(seed, scenario)].traces[mode]['ari']
                q0 = cells[(seed, 'baseline')].traces[mode]['ari']
                loss.append(float((q0 - q).mean()))
                retained.append(float(q.mean() / q0.mean()))
                drop.append(float(1.0 - rolling_ratio(q, q0).min()))
                if ends is not None:
                    recovery.append(recovery_hours(
                        q, q0, ends + cells[(seed, scenario)].onset_offset))
            per_seed[(scenario, mode)] = {'loss': np.array(loss), 'retained': np.array(retained)}
            row = {'setting': label, 'scenario': scenario, 'mode': mode,
                   'loss': float(np.mean(loss)), 'retained_fraction': float(np.mean(retained)),
                   'largest_two_hour_shortfall': float(np.mean(drop))}
            if ends is not None:
                reached = [r for r in recovery if r is not None]
                row.update({
                    'seeds_recovered': len(reached),
                    'median_recovery_hours': float(np.median(reached)) if reached else None,
                    'recovery_hours_by_seed': recovery})
            rows.append(row)
    contrasts = []
    for scenario in disturbed:
        for mode, reference in (('agribrain', 'no_context'), ('agribrain', 'rule_capacity_informed')):
            for measure in ('loss', 'retained'):
                difference = per_seed[(scenario, mode)][measure] - per_seed[(scenario, reference)][measure]
                contrasts.append({'setting': label, 'scenario': scenario, 'mode': mode,
                                  'reference': reference, 'measure': measure,
                                  **inference.estimate(difference, (
                                      'resilience', label, scenario, mode, reference, measure))})
    return rows, contrasts


# ---------------------------------------------------------------------------
# Role activity
# ---------------------------------------------------------------------------
def role_activity(ledgers):
    activity = {}
    for arrays in ledgers.values():
        roles = np.asarray(arrays['role'])
        for role in sorted(set(arrays['role'])):
            mask = roles == role
            entry = activity.setdefault(role, {
                'decisions': 0, 'tool_calls': 0, 'decisions_with_peer_message_read': 0,
                'decisions_with_peer_bias': 0, 'decisions_with_cooperative_overlay': 0,
                'decisions_with_route_removed': 0})
            entry['decisions'] += int(mask.sum())
            entry['tool_calls'] += int(arrays['tool_calls'][mask].sum())
            entry['decisions_with_peer_message_read'] += int(
                (arrays['peer_messages_read'][mask] > 0).sum())
            entry['decisions_with_peer_bias'] += int(
                (np.abs(arrays['peer_bias'][mask]).max(axis=1) > 1e-12).sum())
            entry['decisions_with_cooperative_overlay'] += int(
                arrays['cooperative_overlay'][mask].sum())
            entry['decisions_with_route_removed'] += int(
                arrays['removed'][mask].any(axis=1).sum())
    return activity
