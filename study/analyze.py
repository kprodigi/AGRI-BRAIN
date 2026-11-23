"""Analysis of the completed study.

Verifies every task and every frozen-policy control, then writes the tables of
the study to ``<output>/analysis`` and the completion record
``<output>/COMPLETE.json``. Verification is deep: the retained evaluation
ledgers of every task are validated again and every endpoint they determine is
rebuilt from them and compared with the stored value. Nothing is selected on
the results: every mode, effect and setting of the design is written, and the
completion record is refused while a task or a control is missing.

Ten tests are made: the five principal comparisons under each of the two
missing-confirmation rules, in overall ARI, with a Holm adjustment across the
ten. Every other number is an estimate with its interval.

    python study/analyze.py --output DIR [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controls  # noqa: E402
import inference  # noqa: E402
import records  # noqa: E402
import validate  # noqa: E402

import study  # noqa: E402

PRACTICAL_THRESHOLD = 0.005
SCREENING_MARGIN = 0.01
# Main setting: name of its missing-confirmation rule.
CONDITIONS = {'main_fail_closed': 'fail-closed', 'main_fail_open': 'fail-open'}
DIAGNOSTIC = 'diagnostic_fail_closed'
RECORD_SETTINGS = [*CONDITIONS, DIAGNOSTIC]
# (name, mode, reference, what the difference estimates)
PRINCIPAL = [
    ('agribrain_minus_no_context', 'agribrain', 'no_context',
     'full system against the adaptive policy without protocol information'),
    ('reservation_minus_eligibility_guard', 'agribrain', 'agribrain_eligibility',
     'reservation guard against eligibility guard, both with the score adjustment'),
    ('adjustment_at_eligibility_guard', 'agribrain_eligibility_no_peer', 'eligibility_only',
     'score adjustment added to the eligibility guard'),
    ('condition_dependent_minus_constant_adjustment', 'agribrain_eligibility_no_peer',
     'constant_adjustment',
     'condition-dependent adjustment against a learned constant, eligibility guard'),
    ('agribrain_minus_capacity_informed_rule', 'agribrain', 'rule_capacity_informed',
     'full system against the capacity-informed priority rule'),
]
# Reported with intervals beside the principal comparisons; not tested.
SECONDARY = [
    ('reservation_minus_eligibility_guard_without_peers', 'agribrain_no_peer',
     'agribrain_eligibility_no_peer',
     'reservation guard against eligibility guard, with adjustment, without peer messages'),
]
# guard level: (mode without the score adjustment, mode with it); no peer messages
GUARD_LEVELS = {'none': ('no_context', 'adjustment_only'),
                'eligibility': ('eligibility_only', 'agribrain_eligibility_no_peer'),
                'reservation': ('reservation_only', 'agribrain_no_peer')}
FIXED_ROUTES = ['static', 'rule_always_redistribute', 'rule_always_recovery']
UNINFORMED = [*FIXED_ROUTES, 'rule_risk_threshold']
INFORMED = 'rule_capacity_informed'
GREEDY = 'rule_greedy_informed'
INTERVAL_METRICS = [*study.METRICS, 'refused_share_of_diverted_volume']
MEAN_METRICS = ['share_cold_chain', 'share_redistribution', 'share_recovery',
                'guard_removed_step_fraction', 'guard_changed_action_fraction',
                'mean_decision_latency_ms']
EFFECT_METRICS = ['ari', 'waste', 'slca', 'carbon', 'rle', 'refused_share_of_diverted_volume']
# name: ({mode: coefficient}, what the contrast estimates). Descriptive.
EFFECTS = {
    **{name: ({mode: 1, reference: -1}, meaning)
       for name, mode, reference, meaning in [*PRINCIPAL, *SECONDARY]},
    'agribrain_eligibility_minus_no_context': (
        {'agribrain_eligibility': 1, 'no_context': -1},
        'full system with the eligibility guard against the adaptive policy without '
        'protocol information'),
    'adjustment_without_guard': ({'adjustment_only': 1, 'no_context': -1},
                                 'score adjustment without a guard'),
    'adjustment_at_reservation_guard': ({'agribrain_no_peer': 1, 'reservation_only': -1},
                                        'score adjustment added to the reservation guard'),
    'adjustment_mean_over_guard_levels': (
        {'adjustment_only': 1 / 3, 'no_context': -1 / 3,
         'agribrain_eligibility_no_peer': 1 / 3, 'eligibility_only': -1 / 3,
         'agribrain_no_peer': 1 / 3, 'reservation_only': -1 / 3},
        'score adjustment, mean over the three guard levels'),
    'eligibility_guard_without_adjustment': ({'eligibility_only': 1, 'no_context': -1},
                                             'eligibility guard alone'),
    'eligibility_guard_with_adjustment': (
        {'agribrain_eligibility_no_peer': 1, 'adjustment_only': -1},
        'eligibility guard added to the adjustment'),
    'reservation_over_eligibility_without_adjustment': (
        {'reservation_only': 1, 'eligibility_only': -1},
        'reservation requirement added to the eligibility guard, no adjustment'),
    'reservation_guard_without_adjustment': ({'reservation_only': 1, 'no_context': -1},
                                             'reservation guard alone'),
    'reservation_guard_with_adjustment': ({'agribrain_no_peer': 1, 'adjustment_only': -1},
                                          'reservation guard added to the adjustment'),
    'adjustment_by_eligibility_interaction': (
        {'agribrain_eligibility_no_peer': 1, 'eligibility_only': -1, 'adjustment_only': -1,
         'no_context': 1}, 'adjustment effect at the eligibility guard minus without a guard'),
    'adjustment_by_reservation_interaction': (
        {'agribrain_no_peer': 1, 'reservation_only': -1, 'agribrain_eligibility_no_peer': -1,
         'eligibility_only': 1},
        'adjustment effect at the reservation guard minus at the eligibility guard'),
    'peer_messages_at_reservation_guard': ({'agribrain': 1, 'agribrain_no_peer': -1},
                                           'peer messages, reservation guard'),
    'peer_messages_at_eligibility_guard': (
        {'agribrain_eligibility': 1, 'agribrain_eligibility_no_peer': -1},
        'peer messages, eligibility guard'),
    'without_guard': ({'agribrain': 1, 'agribrain_no_guard': -1},
                      'full system minus the reservation guard'),
    'without_tools': (
        {'agribrain': 1, 'pirag_only': -1},
        'full system minus tool access: capacity information and the reservation check '
        'that depends on it (joint information-and-enforcement comparison)'),
    'without_retrieval': (
        {'agribrain': 1, 'mcp_only': -1},
        'full system minus retrieval: acceptance criteria and the eligibility check that '
        'depends on them (joint information-and-enforcement comparison)'),
    'context_weight_adaptation': ({'agribrain': 1, 'agribrain_fixed_context': -1},
                                  'full system minus adaptation of the context weights'),
    'sign_constraint': ({'agribrain': 1, 'agribrain_sign_unconstrained': -1},
                        'effect of the sign constraint'),
    'bounded_pathway_minus_feature_inputs': (
        {'agribrain_eligibility_no_peer': 1, 'feature_inputs_prior': -1},
        'bounded sign-constrained pathway against feature inputs (implementation check)'),
    'prescribed_weights': ({'feature_inputs_prior': 1, 'feature_inputs_zero': -1},
                           'value of the prescribed context weights'),
    'estimator': ({'agribrain': 1, 'no_pinn': -1},
                  'full system minus the neural residual of the spoilage estimator'),
    'agribrain_minus_static': ({'agribrain': 1, 'static': -1},
                               'full system against fixed cold chain'),
    'agribrain_minus_always_redistribute': ({'agribrain': 1, 'rule_always_redistribute': -1},
                                            'full system against always redistribute'),
    'agribrain_minus_always_recovery': ({'agribrain': 1, 'rule_always_recovery': -1},
                                        'full system against always recovery'),
    'agribrain_minus_risk_threshold_rule': ({'agribrain': 1, 'rule_risk_threshold': -1},
                                            'full system against the risk-threshold rule'),
    'agribrain_minus_greedy_rule': ({'agribrain': 1, GREEDY: -1},
                                    'full system against the model-informed greedy rule'),
    'agribrain_eligibility_minus_capacity_informed_rule': (
        {'agribrain_eligibility': 1, INFORMED: -1},
        'full system with the eligibility guard against the capacity-informed priority rule'),
    'agribrain_eligibility_minus_greedy_rule': (
        {'agribrain_eligibility': 1, GREEDY: -1},
        'full system with the eligibility guard against the model-informed greedy rule'),
    'capacity_informed_rule_minus_greedy_rule': (
        {INFORMED: 1, GREEDY: -1}, 'priority rule with reservation against the greedy rule'),
    'capacity_informed_rule_minus_always_redistribute': (
        {INFORMED: 1, 'rule_always_redistribute': -1},
        'priority rule with reservation against always redistribute'),
}
# Contrasts followed through the sensitivity settings and the graded series.
SENSITIVITY_CONTRASTS = {
    'agribrain_minus_no_context': ('agribrain', 'no_context'),
    'reservation_minus_eligibility_guard': ('agribrain', 'agribrain_eligibility'),
    'agribrain_minus_capacity_informed_rule': ('agribrain', INFORMED),
    'agribrain_eligibility_minus_no_context': ('agribrain_eligibility', 'no_context'),
}
GRADED_CONTRASTS = {
    **SENSITIVITY_CONTRASTS,
    'agribrain_minus_always_redistribute': ('agribrain', 'rule_always_redistribute'),
    'capacity_informed_rule_minus_greedy_rule': (INFORMED, GREEDY),
    'capacity_informed_rule_minus_always_redistribute': (INFORMED, 'rule_always_redistribute'),
}


def reading(estimate, threshold=PRACTICAL_THRESHOLD):
    """Where the interval of a difference lies relative to the practical threshold.

    ``unresolved`` means the interval contains a threshold: the data do not
    say whether the difference is larger or smaller than the threshold.
    """
    low, high = estimate['low'], estimate['high']
    if low > threshold:
        return 'larger than the threshold, positive'
    if high < -threshold:
        return 'larger than the threshold, negative'
    if low > -threshold and high < threshold:
        return 'smaller than the threshold'
    return 'unresolved'


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def endpoint(result):
    actions = np.asarray(result['action_trace'])
    row = {key: float(result[key]) for key in study.METRICS}
    for index, name in enumerate(('share_cold_chain', 'share_redistribution', 'share_recovery')):
        row[name] = float((actions == index).mean())
    for key in ('refused_share_of_diverted_volume', 'diverted_volume', 'refused_volume',
                'guard_removed_step_fraction', 'guard_changed_action_fraction',
                'mean_decision_latency_ms', 'p95_decision_latency_ms'):
        value = result.get(key)
        row[key] = float('nan') if value is None else float(value)
    row['routing_sha256'] = hashlib.sha256(bytes(result['action_trace'])).hexdigest()
    return row


def verify_task(job):
    """Deep verification of one task; returns what the tables need from it."""
    output, index = job
    record = study.verify_result(Path(output), index, deep=True)
    task = record['task']
    reconciled = {key: value for key, value in record['endpoint_reconciliation'].items()
                  if key != 'by_mode'}
    return ((task['setting'], task['seed'], task['scenario']),
            {mode: endpoint(result) for mode, result in record['results'].items()},
            record['revised_validation']['by_mode'], record['seconds'],
            record['attempt_directory'], reconciled)


class Data:
    """Endpoints and validation counts of every task, by (setting, seed, scenario)."""

    def __init__(self, cells, counts, seed_sets, reconciliation=None):
        self.cells, self.counts = cells, counts
        self.reconciliation = reconciliation
        self.settings = {s['id']: s for s in study.design()}
        self.seeds = {sid: list(seed_sets[s['seeds']]) for sid, s in self.settings.items()}

    def keys(self, setting, scenario):
        scenarios = study.SCENARIOS if scenario == 'overall' else [scenario]
        return [(setting, seed, sc) for seed in self.seeds[setting] for sc in scenarios]

    def vector(self, setting, mode, key, scenario):
        """Seed-level values; ``overall`` is the equal-scenario mean of each seed."""
        scenarios = study.SCENARIOS if scenario == 'overall' else [scenario]
        return np.array([np.mean([self.cells[(setting, seed, sc)][mode][key] for sc in scenarios])
                         for seed in self.seeds[setting]])

    def refused_share(self, setting, mode, scenario):
        """Refused share of the diverted volume of each seed, pooled over its cells."""
        scenarios = study.SCENARIOS if scenario == 'overall' else [scenario]
        values = []
        for seed in self.seeds[setting]:
            cells = [self.cells[(setting, seed, sc)][mode] for sc in scenarios]
            diverted = sum(c['diverted_volume'] for c in cells)
            values.append(sum(c['refused_volume'] for c in cells) / diverted if diverted > 0 else 0.0)
        return np.array(values)

    def metric(self, setting, mode, key, scenario):
        if key == 'refused_share_of_diverted_volume':
            return self.refused_share(setting, mode, scenario)
        return self.vector(setting, mode, key, scenario)

    def difference(self, setting, mode, reference, scenario='overall', key='ari'):
        return (self.metric(setting, mode, key, scenario)
                - self.metric(setting, reference, key, scenario))

    def same_routing(self, setting, other, mode):
        """Number of cells in which ``mode`` took the same routes in two settings."""
        return sum(self.cells[key][mode]['routing_sha256']
                   == self.cells[(other, *key[1:])][mode]['routing_sha256']
                   for key in self.keys(setting, 'overall'))

    def largest_cell_difference(self, setting, other, mode, key='ari'):
        return max(abs(self.cells[k][mode][key] - self.cells[(other, *k[1:])][mode][key])
                   for k in self.keys(setting, 'overall'))


def load_study(output, plan, workers, development_runtime, indices=None):
    """Verify every task; keep endpoints and validation counts of all of them
    and, for the main and diagnostic settings, the records used by the
    record-based analyses. ``indices`` limits the tasks, for checks on a part
    of a study; the analysis itself always loads every task."""
    jobs = [(str(output), task['index']) for task in study.tasks(plan['seed_sets'])
            if indices is None or task['index'] in indices]
    if workers > 1:
        with multiprocessing.get_context('spawn').Pool(
                workers, initializer=study.configure, initargs=(development_runtime,)) as pool:
            verified = pool.map(verify_task, jobs, chunksize=4)
    else:
        verified = [verify_task(job) for job in jobs]
    cells, counts, seconds = {}, {}, 0.0
    reconciliation = {'tasks': len(jobs)}
    record_cells = {setting: {} for setting in RECORD_SETTINGS}
    ledgers = {setting: {mode: {} for mode in records.LEDGER_MODES} for setting in CONDITIONS}
    for (_, index), (key, endpoints, by_mode, task_seconds, attempt, reconciled) in zip(
            jobs, verified, strict=True):
        cells[key], counts[key] = endpoints, by_mode
        seconds += task_seconds
        for name, value in reconciled.items():
            reconciliation[name] = (max(reconciliation.get(name, 0.0), value)
                                    if name == 'largest_metric_difference'
                                    else reconciliation.get(name, 0) + value)
        setting, cell_key = key[0], key[1:]
        if setting not in RECORD_SETTINGS:
            continue
        directory = output / 'tasks' / f'{index:04d}'
        record = study.load(directory / 'result.json')
        record_cells[setting][cell_key] = records.Cell(record)
        if setting in CONDITIONS:
            for mode in records.LEDGER_MODES:
                data = study.evaluation_ledger(
                    directory / attempt, record['retained_evidence'], mode, key[2])
                rows = [json.loads(line) for line in data.decode('utf-8').splitlines()[1:] if line]
                rows.sort(key=lambda row: row['step_index'])
                ledgers[setting][mode][cell_key] = records.ledger_arrays(rows)
    return Data(cells, counts, plan['seed_sets'], reconciliation), record_cells, ledgers, seconds


def write_csv(path, rows):
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                             for k, v in row.items()})


# ---------------------------------------------------------------------------
# Principal comparisons
# ---------------------------------------------------------------------------
def principal_comparisons(data):
    """The ten tests: five comparisons under two missing-confirmation rules."""
    rows, raw = [], {}
    for setting, condition in CONDITIONS.items():
        for name, mode, reference, meaning in PRINCIPAL:
            a = data.vector(setting, mode, 'ari', 'overall')
            b = data.vector(setting, reference, 'ari', 'overall')
            row = {'condition': condition, 'setting': setting, 'comparison': name,
                   'estimates': meaning, 'mode': mode, 'reference': reference,
                   'mode_ari': float(a.mean()), 'reference_ari': float(b.mean()),
                   **inference.estimate(a - b, ('principal', setting, name))}
            row['half_width'] = 0.5 * (row['high'] - row['low'])
            row['p_two_sided'], row['test'] = inference.signed_rank(a - b, 'two-sided')
            row['smallest_attainable_p'] = 2.0 * 0.5 ** row['n']
            raw[(setting, name)] = row['p_two_sided']
            rows.append(row)
    adjusted = inference.holm_adjust(raw)
    for row in rows:
        row['p_holm'] = adjusted[(row['setting'], row['comparison'])]
        row['tests_in_family'] = len(raw)
        row['significant'] = bool(row['p_holm'] < 0.05)
        row['threshold'] = PRACTICAL_THRESHOLD
        row['reading'] = reading(row)
    return rows


def comparisons_by_scenario(data):
    """The principal comparisons in each scenario, and the secondary comparison;
    intervals only."""
    rows = []
    for setting, condition in CONDITIONS.items():
        for family, comparisons in (('principal', PRINCIPAL), ('secondary', SECONDARY)):
            for name, mode, reference, meaning in comparisons:
                scenarios = [*study.SCENARIOS, 'overall'] if family == 'secondary' else study.SCENARIOS
                for scenario in scenarios:
                    estimate = inference.estimate(
                        data.difference(setting, mode, reference, scenario),
                        ('by_scenario', setting, name, scenario))
                    rows.append({'condition': condition, 'setting': setting, 'family': family,
                                 'comparison': name, 'estimates': meaning, 'scenario': scenario,
                                 **estimate, 'reading': reading(estimate)})
    return rows


def guard_by_adjustment(data, settings):
    """ARI by guard level, without and with the score adjustment (no peer messages)."""
    rows = []
    for setting in settings:
        for scenario in [*study.SCENARIOS, 'overall']:
            for guard, (without, adjusted) in GUARD_LEVELS.items():
                a = data.vector(setting, adjusted, 'ari', scenario)
                b = data.vector(setting, without, 'ari', scenario)
                estimate = inference.estimate(a - b, ('guard_by_adjustment', setting, scenario, guard))
                rows.append({
                    'setting': setting, 'scenario': scenario, 'guard': guard,
                    'mode_without_adjustment': without, 'mode_with_adjustment': adjusted,
                    'ari_without_adjustment': float(b.mean()),
                    'ari_with_adjustment': float(a.mean()),
                    'adjustment_effect': estimate['mean'], 'low': estimate['low'],
                    'high': estimate['high'], 'ci_method': estimate['ci_method'],
                    'sd': estimate['sd'], 'positive_seeds': estimate['positive_seeds'],
                    'n': estimate['n'], 'reading': reading(estimate)})
    return rows


def mode_summary(data, setting):
    rows = []
    for mode in data.settings[setting]['modes']:
        for scenario in [*study.SCENARIOS, 'overall']:
            for key in INTERVAL_METRICS:
                values = data.metric(setting, mode, key, scenario)
                if np.isfinite(values).all():
                    rows.append({'setting': setting, 'mode': mode, 'scenario': scenario,
                                 'metric': key, **inference.estimate(
                                     values, ('mode', setting, mode, scenario, key))})
            for key in MEAN_METRICS:
                values = data.metric(setting, mode, key, scenario)
                if np.isfinite(values).all():
                    rows.append({'setting': setting, 'mode': mode, 'scenario': scenario,
                                 'metric': key, 'mean': float(values.mean()), 'n': len(values)})
    return rows


def effect_rows(data, setting):
    """Descriptive contrasts: ARI in every scenario, the other metrics overall."""
    rows = []
    for name, (coefficients, meaning) in EFFECTS.items():
        for scenario in [*study.SCENARIOS, 'overall']:
            for key in (EFFECT_METRICS if scenario == 'overall' else ['ari']):
                values = sum(c * data.metric(setting, mode, key, scenario)
                             for mode, c in coefficients.items())
                estimate = inference.estimate(values, ('effect', setting, name, scenario, key))
                row = {'setting': setting, 'effect': name, 'estimates': meaning,
                       'scenario': scenario, 'metric': key, **estimate}
                if key == 'ari':
                    row['reading'] = reading(estimate)
                rows.append(row)
    return rows


def implementation_check(data, record_cells):
    """Bounded pathway against feature inputs, which must agree while the clip,
    the sign constraint and the change bound are inactive."""
    mode, reference = 'agribrain_eligibility_no_peer', 'feature_inputs_prior'
    rows = []
    for setting in CONDITIONS:
        cells = record_cells[setting]
        estimate = inference.estimate(data.difference(setting, mode, reference),
                                      ('implementation_check', setting))
        rows.append({
            'setting': setting, 'mode': mode, 'reference': reference, 'cells': len(cells),
            'cells_with_identical_routing': sum(
                np.array_equal(c.traces[mode]['action'], c.traces[reference]['action'])
                for c in cells.values()),
            'largest_step_ari_difference': max(
                float(np.max(np.abs(c.traces[mode]['ari'] - c.traces[reference]['ari'])))
                for c in cells.values()),
            **estimate, 'reading': reading(estimate)})
    return rows


# ---------------------------------------------------------------------------
# Submitted configuration and diagnostic family (seeds of the submitted study)
# ---------------------------------------------------------------------------
def submitted_model_record(data):
    """The submitted outcome model with its fixed-route references."""
    rows = []
    for mode in study.SUBMITTED_MODES:
        for scenario in [*study.SCENARIOS, 'overall']:
            rows.append({'mode': mode, 'scenario': scenario, **inference.estimate(
                data.vector('submitted', mode, 'ari', scenario),
                ('submitted', mode, scenario, 'ari'))})
    for reference in ('rule_always_redistribute', 'no_context', 'static'):
        rows.append({'mode': f'agribrain_minus_{reference}', 'scenario': 'overall',
                     **inference.estimate(data.difference('submitted', 'agribrain', reference),
                                          ('submitted', 'contrast', reference))})
    return rows


def scenario_identifier(data):
    """Submitted configuration with and without scenario terms in the query."""
    rows = []
    for scenario in [*study.SCENARIOS, 'overall']:
        with_terms = data.vector('submitted', 'agribrain', 'ari', scenario)
        without = data.vector('submitted_without_scenario_terms', 'agribrain', 'ari', scenario)
        rows.append({'scenario': scenario, 'with_scenario_terms': float(with_terms.mean()),
                     'without_scenario_terms': float(without.mean()), **inference.estimate(
                         with_terms - without, ('scenario_identifier', scenario))})
    return rows


def diagnostic_checks(data):
    """Identities that tie the revised outcome model to the submitted one."""
    unlimited, cells = 'diagnostic_unlimited_capacity', len(data.keys('submitted', 'overall'))
    without_constraints = {
        mode: {'cells_with_identical_routing': data.same_routing(unlimited, 'submitted', mode),
               'largest_ari_difference': data.largest_cell_difference(unlimited, 'submitted', mode)}
        for mode in ('static', 'no_context', 'rule_always_redistribute', 'rule_always_recovery')}
    return {
        'cells': cells,
        'static_between_outcome_models': {
            'cells_with_identical_routing': data.same_routing(DIAGNOSTIC, 'submitted', 'static'),
            'largest_ari_difference': data.largest_cell_difference(DIAGNOSTIC, 'submitted', 'static')},
        'without_destination_constraints': without_constraints,
        'refused_volume_without_destination_constraints': max(
            data.cells[key][mode]['refused_volume']
            for key in data.keys(unlimited, 'overall') for mode in data.cells[key]),
        'revised_context_design_under_the_submitted_outcome_model': inference.estimate(
            data.vector(unlimited, 'agribrain', 'ari', 'overall')
            - data.vector('submitted', 'agribrain', 'ari', 'overall'),
            ('diagnostic', 'revised_minus_submitted_context_design'))}


def screening(data, setting):
    """Screening criteria of the outcome model, from the rule references.

    They describe whether the benchmark separates informed from uninformed
    routing. They are diagnostics, not a validation of the model.
    """
    def by_scenario(mode, key='ari'):
        return {sc: float(data.metric(setting, mode, key, sc).mean()) for sc in study.SCENARIOS}

    informed = by_scenario(INFORMED)
    fixed = {mode: by_scenario(mode) for mode in FIXED_ROUTES}
    v1 = min(informed[sc] - max(fixed[mode][sc] for mode in FIXED_ROUTES)
             for sc in study.SCENARIOS)
    shares = {name: float(data.vector(setting, INFORMED, name, 'overall').mean())
              for name in ('share_cold_chain', 'share_redistribution', 'share_recovery')}
    refused = by_scenario('rule_always_redistribute', 'refused_share_of_diverted_volume')
    best = max(UNINFORMED, key=lambda m: data.vector(setting, m, 'ari', 'overall').mean())
    margin = data.difference(setting, INFORMED, best)
    needed = int(np.ceil(0.9 * len(margin)))
    met = {
        'v1': bool(v1 >= SCREENING_MARGIN), 'v2': bool(min(shares.values()) >= 0.05),
        'v3': bool(all(0.20 <= v <= 0.80 for v in refused.values())),
        'v4': bool(margin.mean() >= SCREENING_MARGIN and (margin > 0).sum() >= needed)}
    return {
        'setting': setting,
        'v1_informed_minus_best_fixed_route_worst_scenario': float(v1), 'v1_met': met['v1'],
        'v2_smallest_route_share_informed_rule': float(min(shares.values())), 'v2_met': met['v2'],
        'v3_refused_share_always_redistribute_min': float(min(refused.values())),
        'v3_refused_share_always_redistribute_max': float(max(refused.values())),
        'v3_met': met['v3'],
        'v4_best_uninformed_rule': best,
        'v4_informed_minus_best_uninformed': float(margin.mean()),
        'v4_seeds_positive': int((margin > 0).sum()), 'v4_met': met['v4'],
        'all_met': all(met.values())}


# ---------------------------------------------------------------------------
# Guard and acceptance document
# ---------------------------------------------------------------------------
def guard_activity(data, setting):
    """Guard removals by route and reason, and availability of the guard's inputs."""
    rows = []
    for mode in data.settings[setting]['modes']:
        for scenario in [*study.SCENARIOS, 'overall']:
            selected = [data.counts[key][mode] for key in data.keys(setting, scenario)]
            decisions = sum(c['guard_checked'] for c in selected)
            if not decisions:
                continue
            row = {'setting': setting, 'mode': mode, 'scenario': scenario, 'decisions': decisions,
                   'route_removed_share': sum(c['guard_removed_steps'] for c in selected) / decisions,
                   'pre_guard_sample_differed_share':
                       sum(c['guard_changed_steps'] for c in selected) / decisions,
                   'capacity_report_available_share':
                       sum(c['capacity_report_available_steps'] for c in selected) / decisions,
                   'acceptance_policy_available_share':
                       sum(c['acceptance_policy_available_steps'] for c in selected) / decisions}
            for route in validate.ACTIONS[1:]:
                for reason in validate.GUARD_REASONS:
                    row[f'{route}_removed_{reason}_share'] = sum(
                        c['guard_removed_by_reason'][route][reason] for c in selected) / decisions
            roles = [available for c in selected
                     for available in c['acceptance_policy_by_role'].values()]
            row['acceptance_policy_retrieved_share_of_role_stages'] = (
                sum(roles) / len(roles) if roles else None)
            rows.append(row)
    return rows


def document_cutoff(data):
    """Guard removals when the document states a cutoff other than the one applied."""
    setting, rows = 'acceptance_document_cutoff_040', []
    for mode in ('agribrain', 'agribrain_eligibility'):
        for scenario in [*study.SCENARIOS, 'overall']:
            def removed(name, mode=mode, scenario=scenario):
                return sum(
                    data.counts[key][mode]['guard_removed_by_reason']['local_redistribute']['acceptance']
                    for key in data.keys(name, scenario))
            change = data.vector(setting, mode, 'ari', scenario) - data.vector(
                'main_fail_closed', mode, 'ari', scenario)
            rows.append({
                'mode': mode, 'scenario': scenario,
                'decisions': 288 * len(data.keys(setting, scenario)),
                'redistribution_removed_for_acceptance_document_states_040': removed(setting),
                'redistribution_removed_for_acceptance_document_states_050':
                    removed('main_fail_closed'),
                'ari_change_from_main_setting': float(change.mean())})
    return rows


def document_identifier(data):
    """The same acceptance document under another identifier."""
    setting, cells = 'acceptance_document_renamed', len(data.keys('main_fail_closed', 'overall'))
    return [{'mode': mode, 'cells': cells,
             'cells_with_identical_routing': data.same_routing(setting, 'main_fail_closed', mode),
             'largest_ari_difference': data.largest_cell_difference(
                 setting, 'main_fail_closed', mode)}
            for mode in study.SENSITIVITY_MODES]


def document_absent(data):
    """No acceptance document in the corpus, under each missing-confirmation rule."""
    rows = []
    for setting, reference in (('acceptance_document_absent', 'main_fail_closed'),
                               ('acceptance_document_absent_fail_open', 'main_fail_open')):
        for mode in study.SENSITIVITY_MODES:
            change = inference.estimate(
                data.vector(setting, mode, 'ari', 'overall')
                - data.vector(reference, mode, 'ari', 'overall'),
                ('document_absent', setting, mode))
            rows.append({
                'setting': setting, 'reference_setting': reference, 'mode': mode,
                'ari': float(data.vector(setting, mode, 'ari', 'overall').mean()),
                'share_cold_chain': float(
                    data.vector(setting, mode, 'share_cold_chain', 'overall').mean()),
                'refused_share_of_diverted_volume': float(
                    data.refused_share(setting, mode, 'overall').mean()),
                'ari_change_from_reference': change['mean'], 'change_low': change['low'],
                'change_high': change['high'], 'change_ci_method': change['ci_method']})
    return rows


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------
def reference_setting(setting):
    """Main setting with the missing-confirmation rule of a sensitivity setting."""
    closed = setting['policy'].get('missing_confirmation_closed', True)
    return 'main_fail_closed' if closed else 'main_fail_open'


def sensitivity(data):
    by_scenario, by_setting = [], []
    for setting in study.design():
        if setting['family'] not in study.SENSITIVITY_FAMILIES:
            continue
        sid, reference = setting['id'], reference_setting(setting)
        if data.seeds[sid] != data.seeds[reference]:
            raise ValueError('Sensitivity settings are paired with the main settings by seed')
        summary = {'setting': sid, 'family': setting['family'],
                   'assumption': setting.get('assumption', 'weight magnitudes'),
                   'reference_setting': reference}
        for name, (mode, other) in SENSITIVITY_CONTRASTS.items():
            for scenario in [*study.SCENARIOS, 'overall']:
                gain = data.difference(sid, mode, other, scenario)
                estimate = inference.estimate(gain, ('sensitivity', sid, name, scenario))
                row = {'setting': sid, 'family': setting['family'], 'contrast': name,
                       'scenario': scenario, **estimate, 'reading': reading(estimate)}
                if scenario == 'overall':
                    change = inference.estimate(
                        gain - data.difference(reference, mode, other),
                        ('sensitivity', sid, name, 'change'))
                    row.update({'change_from_reference': change['mean'],
                                'change_low': change['low'], 'change_high': change['high'],
                                'change_ci_method': change['ci_method']})
                    summary.update({
                        name: estimate['mean'], f'{name}_low': estimate['low'],
                        f'{name}_high': estimate['high'], f'{name}_reading': row['reading'],
                        f'{name}_positive_seeds': estimate['positive_seeds'],
                        f'{name}_change_from_reference': change['mean'],
                        f'{name}_change_low': change['low'],
                        f'{name}_change_high': change['high']})
                by_scenario.append(row)
        for mode in study.SENSITIVITY_MODES:
            summary[f'ari_{mode}'] = float(data.vector(sid, mode, 'ari', 'overall').mean())
        for mode in ('no_context', 'agribrain', 'agribrain_eligibility', 'rule_always_redistribute'):
            summary[f'refused_share_{mode}'] = float(data.refused_share(sid, mode, 'overall').mean())
        summary.update({k: v for k, v in screening(data, sid).items() if k != 'setting'})
        by_setting.append(summary)
    return by_scenario, by_setting


def graded_series(data):
    """Contrasts along the graded series of the refusal penalty, the
    redistribution capacity and the share of the dispatch that is reserved."""
    rows = []
    for series, levels in study.GRADED_SERIES.items():
        for setting, level in levels:
            row = {'series': series, 'level': level, 'setting': setting}
            for mode in study.SENSITIVITY_MODES:
                row[f'ari_{mode}'] = float(data.vector(setting, mode, 'ari', 'overall').mean())
            for name, (mode, reference) in GRADED_CONTRASTS.items():
                estimate = inference.estimate(data.difference(setting, mode, reference),
                                              ('series', series, setting, name))
                row.update({name: estimate['mean'], f'{name}_low': estimate['low'],
                            f'{name}_high': estimate['high'],
                            f'{name}_positive_seeds': estimate['positive_seeds']})
            for mode in ('no_context', 'agribrain', 'agribrain_eligibility',
                         'rule_always_redistribute'):
                row[f'refused_share_{mode}'] = float(
                    data.refused_share(setting, mode, 'overall').mean())
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Frozen-policy controls
# ---------------------------------------------------------------------------
def control_tables(output, plan, data):
    results = {}
    for index in controls.primary_indices(plan['seed_sets']):
        record = controls.verify_cell(output, index)
        task = record['task']
        results[(task['setting'], task['seed'], task['scenario'])] = record['controls']
    means, contrasts = [], []
    for setting in CONDITIONS:
        for mode in controls.MODES:
            for scenario in [*study.SCENARIOS, 'overall']:
                scenarios = study.SCENARIOS if scenario == 'overall' else [scenario]

                def values(control, setting=setting, mode=mode, scenarios=scenarios):
                    return np.array([np.mean([results[(setting, seed, sc)][mode][control]['ari']
                                              for sc in scenarios])
                                     for seed in data.seeds[setting]])
                for control in controls.CONTROLS:
                    means.append({'setting': setting, 'mode': mode, 'scenario': scenario,
                                  'control': control, **inference.estimate(
                                      values(control), ('control', setting, mode, scenario, control))})
                for control in controls.CONTROLS[1:]:
                    estimate = inference.estimate(
                        values('live') - values(control),
                        ('control', setting, mode, scenario, 'live', control))
                    contrasts.append({'setting': setting, 'mode': mode, 'scenario': scenario,
                                      'contrast': f'live_minus_{control}', **estimate,
                                      'reading': reading(estimate)})
    integrity = {
        'cells': len(results),
        'executions': len(results) * len(controls.MODES) * len(controls.CONTROLS),
        'live_reproduces_the_stored_evaluation': True,
        'largest_step_ari_difference_live': max(
            r[mode]['live']['integrity']['stored_evaluation_step_ari']
            for r in results.values() for mode in controls.MODES)}
    return means, contrasts, integrity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def record_tables(out, data, record_cells, ledgers):
    """Tables computed from the retained records of the main and diagnostic settings."""
    tables = {name: [] for name in (
        'gain_decomposition', 'gain_by_route_switch', 'gain_by_disruption_window',
        'rescoring_modes', 'rescoring_contrasts', 'context_components',
        'context_overlap_with_state', 'score_influence', 'constraint_activity',
        'peer_influence', 'reservation_tradeoff', 'break_even_share', 'loss_and_recovery',
        'loss_and_recovery_contrasts')}
    roles = {}
    for setting in RECORD_SETTINGS:
        cells, seeds = record_cells[setting], data.seeds[setting]
        for cell in cells.values():
            for mode in cell.traces:
                cell.check_reconstruction(mode)
        records.check_rule_replay(cells)
        tables['reservation_tradeoff'] += records.reservation_tradeoff_table(cells, seeds, setting)
        tables['break_even_share'] += records.break_even_table(cells, seeds, setting)
        if setting not in CONDITIONS:
            continue
        parts, switches = records.decomposition_tables(cells, seeds, setting)
        tables['gain_decomposition'] += parts
        tables['gain_by_route_switch'] += switches
        tables['gain_by_disruption_window'] += records.window_table(
            cells, seeds, records.DECOMPOSITION_PAIRS, setting)
        by_mode, contrasts = records.rescoring_tables(cells, seeds, setting)
        tables['rescoring_modes'] += by_mode
        tables['rescoring_contrasts'] += contrasts
        for mode in controls.MODES:
            contributions, overlap = records.component_tables(
                ledgers[setting][mode], seeds, mode, setting)
            tables['context_components'] += contributions
            tables['context_overlap_with_state'] += overlap
        tables['score_influence'] += records.score_influence_table(ledgers[setting], seeds, setting)
        tables['constraint_activity'] += records.constraint_activity_table(cells, seeds, setting)
        tables['peer_influence'] += records.peer_influence_table(ledgers[setting], seeds, setting)
        resilience, resilience_contrasts = records.resilience_tables(
            cells, seeds, study.PRIMARY_MODES, setting)
        tables['loss_and_recovery'] += resilience
        tables['loss_and_recovery_contrasts'] += resilience_contrasts
        roles[setting] = {mode: records.role_activity(ledgers[setting][mode])
                          for mode in controls.MODES}
    for name, rows in tables.items():
        write_csv(out / f'{name}.csv', rows)
    study.save(out / 'role_activity.json', roles)
    return tables


def sensitivity_digest(by_scenario, principal):
    """How each followed contrast reads across the sensitivity settings."""
    main = {(row['setting'], row['comparison']): row for row in principal}
    digest = {}
    for name in SENSITIVITY_CONTRASTS:
        rows = [r for r in by_scenario if r['contrast'] == name and r['scenario'] == 'overall']
        readings = {}
        for row in rows:
            readings[row['reading']] = readings.get(row['reading'], 0) + 1
        entry = {'settings': len(rows), 'readings': readings,
                 'settings_with_negative_mean': [r['setting'] for r in rows if r['mean'] < 0],
                 'smallest': min(rows, key=lambda r: r['mean']),
                 'largest': max(rows, key=lambda r: r['mean'])}
        reference = main.get(('main_fail_closed', name))
        if reference is not None:
            entry['main_fail_closed'] = {k: reference[k] for k in ('mean', 'low', 'high', 'reading')}
            entry['settings_reading_differently_from_main_fail_closed'] = [
                r['setting'] for r in rows if r['reading'] != reference['reading']]
        digest[name] = entry
    return digest


def run(output, development, workers, development_runtime=None):
    output = Path(output).resolve()
    plan = study.read_plan(output, development)
    data, record_cells, ledgers, seconds = load_study(output, plan, workers, development_runtime)
    out = output / 'analysis'
    out.mkdir(exist_ok=True)

    write_csv(out / 'seed_endpoints.csv', [
        {'setting': setting, 'seed': seed, 'scenario': scenario, 'mode': mode, **row}
        for (setting, seed, scenario), modes in data.cells.items() for mode, row in modes.items()])
    write_csv(out / 'settings.csv', [
        {'setting': s['id'], 'family': s['family'], 'seeds': s['seeds'],
         'modes': ' '.join(s['modes']), 'policy': s['policy'], 'patches': s['patches'],
         **s.get('factors', {})} for s in study.design()])

    principal = principal_comparisons(data)
    write_csv(out / 'principal_comparisons.csv', principal)
    write_csv(out / 'comparisons_by_scenario.csv', comparisons_by_scenario(data))
    write_csv(out / 'guard_by_adjustment.csv', guard_by_adjustment(data, RECORD_SETTINGS))
    write_csv(out / 'modes.csv', [row for s in RECORD_SETTINGS for row in mode_summary(data, s)])
    effects = [row for s in RECORD_SETTINGS for row in effect_rows(data, s)]
    write_csv(out / 'effects.csv', effects)
    check = implementation_check(data, record_cells)
    write_csv(out / 'implementation_check.csv', check)
    write_csv(out / 'guard_activity.csv',
              [row for s in RECORD_SETTINGS for row in guard_activity(data, s)])
    write_csv(out / 'document_cutoff.csv', document_cutoff(data))
    identifier = document_identifier(data)
    write_csv(out / 'document_identifier.csv', identifier)
    write_csv(out / 'document_absent.csv', document_absent(data))

    write_csv(out / 'submitted_model.csv', submitted_model_record(data))
    write_csv(out / 'scenario_identifier.csv', scenario_identifier(data))
    diagnostics = diagnostic_checks(data)
    study.save(out / 'diagnostic_checks.json', diagnostics)
    screened = [screening(data, s) for s in RECORD_SETTINGS]
    write_csv(out / 'screening_criteria.csv', screened)

    tables = record_tables(out, data, record_cells, ledgers)

    by_scenario, by_setting = sensitivity(data)
    write_csv(out / 'sensitivity_by_scenario.csv', by_scenario)
    write_csv(out / 'sensitivity_settings.csv', by_setting)
    write_csv(out / 'graded_series.csv', graded_series(data))

    control_means, control_contrasts, control_integrity = control_tables(output, plan, data)
    write_csv(out / 'controls.csv', control_means)
    write_csv(out / 'control_contrasts.csv', control_contrasts)

    summary = {
        'counts': plan['counts'], 'seed_sets': plan['seed_sets'], 'development_only': development,
        'intervals': '95% BCa over seeds; pointwise, not simultaneous',
        'practical_threshold': PRACTICAL_THRESHOLD,
        'tests': 'two-sided exact Wilcoxon signed-rank on the ten principal comparisons, '
                 'Holm-adjusted across the ten; no other test is made',
        'endpoint_reconciliation': data.reconciliation,
        'principal_comparisons': principal,
        'implementation_check': check,
        'constraint_activity': tables['constraint_activity'],
        'screening_criteria': screened,
        'diagnostic_checks': diagnostics,
        'document_identifier': identifier,
        'controls': control_integrity,
        'sensitivity': sensitivity_digest(by_scenario, principal),
        'settings_not_meeting_a_screening_criterion': [
            r['setting'] for r in by_setting if not r['all_met']],
        'cpu_hours': seconds / 3600.0}
    study.save(out / 'summary.json', summary)
    study.save(output / 'COMPLETE.json', {
        'plan_sha256': study.sha(output / 'plan.json'), 'tasks_verified': len(data.cells),
        'controls_verified': control_integrity['cells'], 'development_only': development,
        'endpoint_reconciliation': data.reconciliation,
        'analysis_files': {p.name: study.sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    print(f"COMPLETE: {len(data.cells)} tasks and {control_integrity['cells']} control cells "
          'verified. Principal comparisons:')
    for row in principal:
        print(f"  {row['condition']:11s} {row['comparison']:46s} {row['mean']:+.4f} "
              f"({row['low']:+.4f} to {row['high']:+.4f}); Holm p {row['p_holm']:.4g}; "
              f"{row['reading']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--development-runtime', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=1,
                        help='processes used to verify the tasks')
    args = parser.parse_args()
    run(args.output, study.configure(args.development_runtime), max(1, args.workers),
        args.development_runtime)


if __name__ == '__main__':
    main()
