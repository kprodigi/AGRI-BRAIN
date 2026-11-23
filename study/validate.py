"""Validation of the evaluation ledgers produced by the study.

For every evaluation ledger of a task this module re-derives, from the recorded
quantities alone,

* that the capacity parameters in the ledger header are those of the setting;
* the step outcomes (waste, emissions indicator, social performance, ARI),
  including the refused share that continues by cold chain;
* the capacity of both destinations step by step and the share of each
  dispatch that the destination took;
* that a capacity report reaching the decision side states the reconstructed
  capacity, and that it is present exactly when the mode requests it and the
  tool channel is up;
* the routes the guard removed and the reason for each, and that the executed
  route was never one of them;
* that the acceptance criteria used by the guard are the values stated in the
  retrieved acceptance document;
* the three context components that the revised design reads from the capacity
  report and the retrieved acceptance policy;
* the route chosen by each rule reference, from what the rule may read;
* that no scenario identifier entered a retrieval query where the setting
  removes it, and that the probability-gap override of the submitted design
  never fired under the revised design;
* the endpoints stored with the result: every episode metric and step trace
  that the analysis reads and that the decision records determine is rebuilt
  from the records and must equal the stored value (``reconcile_endpoints``).

The arithmetic for capacity, acceptance, refusal, the guards and the rules is
written here independently of the simulator. The unchanged base equations
(waste, emissions, social score, spoilage rate, ARI) are imported from the
model package, so a defect in one of them would not be detected here; the unit
tests check them against values worked by hand. ``OutcomeModel`` is also used
by the analysis to evaluate decisions under alternative index parameters.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

TOLERANCE = 1e-9
ACTIONS = ['cold_chain', 'local_redistribute', 'recovery']
ROUTE_KM = {'cold_chain': 'km_coldchain', 'local_redistribute': 'km_local',
            'recovery': 'km_recovery'}
GUARD_REASONS = ['acceptance', 'no_acceptance_criteria', 'capacity', 'no_capacity',
                 'no_capacity_report']
SCENARIO_IDENTIFIERS = ['heatwave', 'overproduction', 'cyber', 'adaptive pricing',
                        'adaptive_pricing']
# Header field of the capacity model: policy field of the setting.
CAPACITY_PARAMETERS = {
    'enabled': 'destination_capacity_enabled',
    'kappa_lr': 'capacity_kappa_lr', 'kappa_rec': 'capacity_kappa_rec',
    'bmax_lr': 'capacity_bmax_lr', 'bmax_rec': 'capacity_bmax_rec',
    'demand_ref': 'capacity_demand_ref', 'inventory_ref': 'capacity_inventory_ref',
    'rho_at': 'acceptance_rho_at', 'rho_cut': 'acceptance_rho_cut',
    'strict_redistribution': 'acceptance_strict_redistribution',
    'refusal_penalty_scale': 'refusal_penalty_scale',
    'missing_confirmation_closed': 'missing_confirmation_closed',
    'acceptance_enabled': 'destination_acceptance_enabled',
    'refused_share_lost': 'refused_share_lost',
    'reservation_share': 'reservation_share',
}
OBSERVED = {'rho_outcome_environmental': 'rho_policy_observed',
            'inventory_outcome_environmental': 'inventory_policy_observed',
            'temp_outcome_environmental': 'temp_policy_observed',
            'rh_outcome_environmental': 'rh_policy_observed'}
RECONCILIATION_TOLERANCE = 1e-12
# Step trace stored with a result: field of the decision records it must equal.
RECONCILED_TRACES = {
    'ari_trace': 'ari', 'waste_trace': 'waste', 'slca_trace': 'slca', 'carbon_trace': 'carbon_kg',
    'action_trace': 'action_idx', 'prob_trace': 'probs', 'rho_trace': 'rho', 'hours': 'hour',
    'accepted_share_trace': 'accepted_share', 'dispatch_size_trace': 'dispatch_size',
    'redistribution_capacity_trace': 'redistribution_capacity_before',
    'recovery_capacity_trace': 'recovery_capacity_before',
    'rho_outcome_environmental_trace': 'rho_outcome_environmental',
    'inventory_outcome_environmental_trace': 'inventory_outcome_environmental',
    'temp_outcome_environmental_trace': 'temp_outcome_environmental',
    'rh_outcome_environmental_trace': 'rh_outcome_environmental',
    'demand_outcome_environmental_trace': 'demand_outcome_environmental',
    'transport_multiplier_outcome_environmental_trace':
        'transport_multiplier_outcome_environmental',
    'rho_policy_observed_trace': 'rho_policy_observed',
    'inventory_policy_observed_trace': 'inventory_policy_observed',
    'temp_policy_observed_trace': 'temp_policy_observed',
    'rh_policy_observed_trace': 'rh_policy_observed',
}
# Fields of a result that must equal the header of its ledger.
RECONCILED_HEADER = ['latent_environment_sha256', 'effective_k_ref', 'effective_Ea_R',
                     'scenario_onset_offset_hours', 'destination_capacity_model']


def read_ledger(path):
    """Return ``(metadata, records)`` of one decision ledger (plain or gzip)."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or '_header' not in rows[0]:
        raise ValueError(f'{path}: ledger header missing')
    records = [row for row in rows[1:] if 'step_index' in row]
    records.sort(key=lambda row: row['step_index'])
    if [row['step_index'] for row in records] != list(range(len(records))):
        raise ValueError(f'{path}: step indices are not contiguous')
    return rows[0]['metadata'], records


def close(a, b, tolerance=TOLERANCE):
    return abs(float(a) - float(b)) <= tolerance


def accepted_by_destination(action, rho, config):
    """Acceptance criterion of the destination, re-stated independently."""
    if action == 'cold_chain' or not config['acceptance_enabled']:
        return True
    if action == 'local_redistribute':
        if rho > config['rho_cut']:
            return False
        return (not config['strict_redistribution']) or rho > config['rho_at']
    return rho > config['rho_at']


def published_criteria(policy):
    """Criteria stated in the acceptance document of a setting."""
    at, cut = policy['published_rho_at'], policy['published_rho_cut']
    return {'rho_at': policy['acceptance_rho_at'] if at is None else at,
            'rho_cut': policy['acceptance_rho_cut'] if cut is None else cut,
            'strict_redistribution': policy['acceptance_strict_redistribution']}


def outside_band(index, rho, criteria):
    """Whether ``rho`` is outside the acceptance band of destination ``index``."""
    if index == 1:
        return rho > criteria['rho_cut'] or (
            criteria['strict_redistribution'] and rho <= criteria['rho_at'])
    return rho <= criteria['rho_at']


def removed_routes(kind, rho, criteria, room, dispatch_size, share, closed):
    """Routes a guard of ``kind`` removes, with the reason for each.

    ``criteria`` is a dict of thresholds, ``'unavailable'`` or ``'withheld'``;
    ``room`` is a dict of reported capacities by destination index,
    ``'unavailable'`` or ``'withheld'``.
    """
    removed = {}
    for index in (1, 2):
        if isinstance(criteria, dict):
            if outside_band(index, rho, criteria):
                removed[index] = 'acceptance'
        elif criteria == 'unavailable' and closed:
            removed[index] = 'no_acceptance_criteria'
        if index in removed:
            continue
        if isinstance(room, dict):
            if kind == 'reservation':
                if room[index] < share * dispatch_size:
                    removed[index] = 'capacity'
            elif dispatch_size > 0.0 and room[index] <= 0.0:
                removed[index] = 'no_capacity'
        elif room == 'unavailable' and kind == 'reservation' and closed:
            removed[index] = 'no_capacity_report'
    return removed


def information_states(caps, report, policy_record):
    """What the guard of a mode holds: criteria and reported room, or a state."""
    if caps.context_kind == 'mcp_only':
        criteria = 'withheld'
    elif policy_record and policy_record.get('available'):
        criteria = {key: policy_record[key]
                    for key in ('rho_at', 'rho_cut', 'strict_redistribution')}
    else:
        criteria = 'unavailable'
    if caps.context_kind == 'pirag_only':
        room = 'withheld'
    elif report is not None:
        room = {1: report['redistribution_capacity'], 2: report['recovery_capacity']}
    else:
        room = 'unavailable'
    return criteria, room


def state_name(value):
    return 'available' if isinstance(value, dict) else value


def shortfall(report, key):
    """Share of the dispatch that the reported capacity cannot take."""
    if report is None or report['dispatch_size'] <= 0:
        return 0.0
    return 1.0 - min(1.0, report[key] / report['dispatch_size'])


class OutcomeModel:
    """Step outcomes of a route in a recorded environment.

    ``policy`` and ``parameters`` are the record of the setting that was
    applied (policy fields, waste-saving coefficients, route social scores);
    ``k_ref`` and ``ea_r`` are the spoilage parameters of the episode.
    """

    def __init__(self, policy, parameters, k_ref, ea_r, mode, penalty_scale=1.0,
                 refused_share_lost=False):
        from src.models import action_selection, carbon, resilience, slca, spoilage, waste
        self._stress = action_selection.compute_thermal_stress
        self._attenuation = action_selection.compute_slca_attenuation
        self._carbon = carbon.compute_transport_carbon
        self._ari = resilience.compute_ari
        self._score = slca.slca_score
        self._rate = spoilage.arrhenius_k
        self._waste_rate = waste.compute_waste_rate
        self._save = waste.compute_save_factor
        self._inventory_baseline = waste.INV_BASELINE
        self.policy = policy
        self.saving = parameters['saving_coefficients']
        self.bases = parameters['route_social_scores']
        self.k_ref, self.ea_r = float(k_ref), float(ea_r)
        self.mode = mode
        self.penalty_scale = float(penalty_scale)
        self.refused_share_lost = bool(refused_share_lost)
        self.weights = {key: policy[key] for key in ('w_c', 'w_l', 'w_r', 'w_p')}

    def step(self, row, action, share, *, weights=None, carbon_cap=None):
        """Outcome of ``action`` with accepted ``share`` in the environment of ``row``.

        ``weights`` and ``carbon_cap`` replace the social-score weights and the
        emissions normalizer of the setting; the routing decision is not
        re-made.
        """
        policy = self.policy
        weights = self.weights if weights is None else weights
        cap = policy['carbon_cap'] if carbon_cap is None else carbon_cap
        rho = float(row['rho_outcome_environmental'])
        inventory = float(row['inventory_outcome_environmental'])
        temperature = float(row['temp_outcome_environmental'])
        humidity = float(row['rh_outcome_environmental'])
        multiplier = float(row['transport_multiplier_outcome_environmental'])
        stress = self._stress(temperature)
        surplus = max(0.0, inventory / self._inventory_baseline - 1.0)
        carbon_route = self._carbon(
            float(policy[ROUTE_KM[action]]) * multiplier, policy['carbon_per_km'], stress,
            eff_factor=1.0)
        composite = self._score(carbon_route, action, carbon_cap=cap,
                                action_bases=self.bases, **weights)['composite']
        waste_raw = float(self._waste_rate(
            self._rate(temperature, self.k_ref, self.ea_r, policy['T_ref_K'],
                       humidity / 100.0, policy['beta_humidity']), surplus))
        waste = waste_raw * (1.0 - self._save(action, self.mode, surplus,
                                              save_floor=self.saving))
        carbon = carbon_route
        if share < 1.0:
            # The refused share travels the diverted leg and then the cold chain.
            carbon_cold = self._carbon(
                float(policy['km_coldchain']) * multiplier, policy['carbon_per_km'], stress,
                eff_factor=1.0)
            carbon_refused = self.penalty_scale * carbon_route + carbon_cold
            composite_refused = self._score(
                carbon_refused, 'cold_chain', carbon_cap=cap, action_bases=self.bases,
                **weights)['composite']
            leg = float(policy[ROUTE_KM[action]]) / float(policy['km_coldchain'])
            waste_refused = min(1.0, waste_raw * (1.0 + self.penalty_scale * leg))
            if self.refused_share_lost:
                waste_refused = 1.0  # none of the refused share is sold
            composite = share * composite + (1.0 - share) * composite_refused
            carbon = share * carbon_route + (1.0 - share) * carbon_refused
            waste = share * waste + (1.0 - share) * waste_refused
        slca = composite * self._attenuation(stress, surplus)
        return {'waste': float(waste), 'carbon': float(carbon), 'slca': float(slca),
                'rho': rho, 'ari': float(self._ari(waste, slca, rho))}


def rule_action(rule, row, criteria, room, dispatch_size, share, closed, predictor):
    """Route a rule reference must choose, from what it may read."""
    rho = float(row['rho_policy_observed'])
    if rule == 'always_redistribute':
        return 1
    if rule == 'always_recovery':
        return 2
    if rule == 'risk_threshold':
        if not isinstance(criteria, dict) or rho <= criteria['rho_at']:
            return 0
        return 1 if rho <= criteria['rho_cut'] else 2
    if rule == 'capacity_informed':
        removed = removed_routes('reservation', rho, criteria, room, dispatch_size, share, closed)
        return 1 if 1 not in removed else (2 if 2 not in removed else 0)
    if rule == 'greedy_informed':
        observed = {**{key: float(row[source]) for key, source in OBSERVED.items()},
                    'transport_multiplier_outcome_environmental': 1.0}
        observed['inventory_outcome_environmental'] = max(
            observed['inventory_outcome_environmental'], 0.0)
        best, best_value = 0, predictor.step(observed, 'cold_chain', 1.0)['ari']
        for index in (1, 2):
            if isinstance(criteria, dict):
                if outside_band(index, rho, criteria):
                    continue
            elif criteria == 'unavailable' and closed:
                continue
            if isinstance(room, dict):
                expected = 1.0 if dispatch_size <= 0.0 else min(
                    1.0, max(0.0, room[index]) / dispatch_size)
            elif room == 'unavailable' and closed:
                continue
            else:
                expected = 1.0
            if expected <= 0.0:
                continue
            value = predictor.step(observed, ACTIONS[index], expected)['ari']
            if value > best_value:
                best, best_value = index, value
        return best
    raise ValueError(f'unknown rule {rule}')


def _queries_without_scenario(row, here):
    for channel in ('primary', 'cooperative'):
        evidence = ((row.get('step_channel_evidence') or {}).get(channel) or {}).get(
            'retrieval') or {}
        if not evidence.get('attempted'):
            continue
        metadata = evidence.get('query_transform_metadata') or {}
        if channel == 'primary' and metadata.get('scenario') != 'baseline':
            raise ValueError(f'{here}: a scenario identifier entered the retrieval query')
        text = ' '.join(str(metadata.get(key, '')) for key in ('base_query', 'final_query'))
        text = (text + ' ' + str(evidence.get('query', ''))).lower()
        if any(identifier in text for identifier in SCENARIO_IDENTIFIERS):
            raise ValueError(f'{here}: a scenario term entered the {channel} retrieval query')


def reconcile_endpoints(metadata, records, result, mode, parameters, where):
    """Rebuild the endpoints stored with one evaluation from its decision ledger.

    Recomputed from the records and compared with the stored values: the six
    episode metrics, the diverted and refused volume and their ratio, the two
    guard fractions, the decision latency, every step trace listed in
    ``RECONCILED_TRACES``, the quantities carried in the ledger header, and the
    context weights of the frozen policy. Raises on the first disagreement;
    returns the names of what was reconciled.

    The hierarchy weight and the threshold of the RLE are the unchanged
    definitions of the model package; the accumulation, including the credit
    on the accepted share only, is written here.
    """
    import numpy as np
    from src.models.mode_capabilities import capabilities_for
    from src.models.resilience import HIERARCHY_WEIGHT, RLE_THRESHOLD, hierarchy_weight

    summary = {'metrics': [], 'traces': [], 'header_fields': [], 'trace_values': 0,
               'context_weight_records': 0, 'largest_metric_difference': 0.0}

    def metric(name, rebuilt):
        if result.get(name) is None:
            return
        difference = abs(float(result[name]) - float(rebuilt))
        if not difference <= RECONCILIATION_TOLERANCE * max(1.0, abs(float(rebuilt))):
            raise ValueError(f'{where}: {name} is {result[name]}; the ledger gives {rebuilt}')
        summary['metrics'].append(name)
        summary['largest_metric_difference'] = max(
            summary['largest_metric_difference'], difference)

    def column(field):
        return [row[field] for row in records]

    steps = len(records)
    social = np.asarray(column('slca'), dtype=float)
    metric('ari', np.mean(column('ari')))
    metric('waste', np.mean(column('waste')))
    metric('slca', np.mean(social))
    metric('equity', max(0.0, min(1.0, float(np.mean(social)) * max(
        0.0, min(1.0, 1.0 - float(np.std(social)))))))
    carbon = numerator = denominator = diverted = refused = 0.0
    removed = changed = 0
    top_weight = max(HIERARCHY_WEIGHT.values())
    for row in records:
        carbon += row['carbon_kg']
        share = float(row.get('accepted_share', 1.0))
        rho = row['rho_outcome_environmental']
        if rho > RLE_THRESHOLD:
            numerator += rho * hierarchy_weight(row['action'], rho) * share
            denominator += rho * top_weight
        if row['action_idx'] != 0 and 'dispatch_size' in row:
            diverted += row['dispatch_size']
            refused += (1.0 - share) * row['dispatch_size']
        guard = row.get('guard') or {}
        removed += bool(guard.get('removed_routes'))
        changed += (guard.get('sampled_action_pre_guard') is not None
                    and int(guard['sampled_action_pre_guard']) != int(row['action_idx']))
    metric('carbon', carbon)
    metric('rle', numerator / denominator if denominator > 0.0 else 0.0)
    metric('diverted_volume', diverted)
    metric('refused_volume', refused)
    metric('refused_share_of_diverted_volume', refused / diverted if diverted > 0.0 else 0.0)
    metric('guard_removed_step_fraction', removed / max(steps, 1))
    metric('guard_changed_action_fraction', changed / max(steps, 1))
    latency = np.asarray(column('decision_latency_ms'), dtype=float)
    metric('mean_decision_latency_ms', np.mean(latency))
    metric('p95_decision_latency_ms', np.percentile(latency, 95))

    for name, field in RECONCILED_TRACES.items():
        if name not in result:
            continue
        if any(field not in row for row in records):
            raise ValueError(f'{where}: the ledger does not record {field} for {name}')
        recorded = np.asarray(result[name], dtype=float)
        rebuilt = np.asarray(column(field), dtype=float)
        if recorded.shape != rebuilt.shape or not (
                float(np.max(np.abs(recorded - rebuilt), initial=0.0)) <= RECONCILIATION_TOLERANCE):
            raise ValueError(f'{where}: {name} is not the series of the ledger')
        summary['traces'].append(name)
        summary['trace_values'] += int(recorded.size)

    for key in RECONCILED_HEADER:
        if key in result:
            if key not in metadata or metadata[key] != result[key]:
                raise ValueError(f'{where}: {key} differs from the header of the ledger')
            summary['header_fields'].append(key)

    learner = result.get('learner_summary') or {}
    if learner.get('final_theta') is not None:
        initial = np.asarray(learner['initial_theta'], dtype=float)
        final = np.asarray(learner['final_theta'], dtype=float)
        for row in records:
            applied = row.get('effective_context_theta')
            if applied is None:
                continue
            applied = np.asarray(applied, dtype=float)
            if applied.shape != final.shape or not (
                    float(np.max(np.abs(applied - final))) <= RECONCILIATION_TOLERANCE):
                raise ValueError(
                    f'{where} step {row["step_index"]}: the context weights of the decision '
                    'are not the final weights stored with the result')
            summary['context_weight_records'] += 1
        caps = capabilities_for(mode)
        prior = parameters.get('initial_context_matrix')
        if (prior is not None and caps.context_weight_init == 'prior'
                and caps.context_kind is not None and caps.context_adjustment
                and not caps.constant_adjustment):
            if not np.allclose(initial, np.asarray(prior, dtype=float),
                               atol=RECONCILIATION_TOLERANCE, rtol=0):
                raise ValueError(f'{where}: initial context weights are not those of the setting')
            summary['header_fields'].append('initial_context_weights')
            # With nonzero initial weights the declared sign is the initial sign.
            reversals = int(((initial != 0.0) & (final * np.sign(initial) < 0.0)).sum())
            if np.all(initial != 0.0) and int(learner.get('sign_reversal_count') or 0) != reversals:
                raise ValueError(f'{where}: sign reversal count does not follow from the weights')
    return summary


def validate_ledger(path, mode, policy, result, parameters, ledger=None):
    """Validate one evaluation ledger; return counts of what was checked.

    ``ledger`` is the ``(metadata, records)`` pair of a ledger that the caller
    has already read.
    """
    import numpy as np
    from src.models.action_selection import categorical_action_from_uniform
    from src.models.mode_capabilities import capabilities_for

    metadata, records = read_ledger(path) if ledger is None else ledger
    where = Path(path).name
    if len(records) != 288:
        raise ValueError(f'{where}: expected 288 decisions, found {len(records)}')
    caps = capabilities_for(mode)
    capacity_on = bool(policy['destination_capacity_enabled'])
    revised = bool(policy['revised_context_design'])
    without_terms = revised or not bool(policy['query_scenario_terms'])
    config = metadata.get('destination_capacity_model')
    if capacity_on != (config is not None):
        raise ValueError(f'{where}: capacity model record does not match the setting')
    if capacity_on:
        intended = {key: policy[field] for key, field in CAPACITY_PARAMETERS.items()}
        if config != intended:
            different = sorted(k for k in set(config) | set(intended)
                               if config.get(k) != intended.get(k))
            raise ValueError(f'{where}: capacity parameters differ from the setting: {different}')
    if revised != (metadata.get('context_design') == 'revised'):
        raise ValueError(f'{where}: context design record does not match the setting')
    if bool(policy['query_scenario_terms']) != (metadata.get('query_scenario_terms') is not False):
        raise ValueError(f'{where}: query setting record does not match the setting')
    if metadata.get('reward_objective', 'linear') != policy['reward_objective']:
        raise ValueError(f'{where}: learning objective record does not match the setting')
    counts = {'records': len(records), 'refused_steps': 0, 'guard_removed_steps': 0,
              'guard_changed_steps': 0, 'outcomes_checked': 0, 'capacity_checked': 0,
              'guard_checked': 0, 'context_checked': 0, 'queries_checked': 0,
              'reports_checked': 0, 'rule_decisions_checked': 0,
              'capacity_report_available_steps': 0, 'acceptance_policy_available_steps': 0,
              'override_steps': 0}
    removed_by_reason = {route: dict.fromkeys(GUARD_REASONS, 0) for route in ACTIONS[1:]}
    policy_by_role = {}
    outcomes = OutcomeModel(
        policy, parameters, result['effective_k_ref'], result['effective_Ea_R'], mode,
        config['refusal_penalty_scale'] if capacity_on else 1.0,
        config['refused_share_lost'] if capacity_on else False)
    # A rule predicts with the nominal kinetic parameters, not those of the episode.
    predictor = OutcomeModel(
        policy, parameters, policy['k_ref'], policy['Ea_R'], mode,
        policy['refusal_penalty_scale'], policy['refused_share_lost'])
    stated = published_criteria(policy)
    closed = bool(policy['missing_confirmation_closed'])
    share_reserved = float(policy['reservation_share'])
    age = int(policy['capacity_report_age_steps'])
    requests_report = caps.context_kind in ('full', 'mcp_only')
    if capacity_on:
        remaining = {'local_redistribute': config['bmax_lr'], 'recovery': config['bmax_rec']}
        maxima = dict(remaining)
    history = []
    diverted = refused = 0.0
    for row in records:
        here = f'{where} step {row["step_index"]}'
        action = ACTIONS[row['action_idx']]
        if row['action'] != action:
            raise ValueError(f'{here}: action label and index disagree')
        counts['override_steps'] += int(bool(row.get('governance_override')))
        if revised and row.get('governance_override'):
            raise ValueError(f'{here}: override fired under the revised design')
        rho = float(row['rho_outcome_environmental'])
        offline = row['scenario'] == 'cyber_outage' and float(row['hour']) >= 24.0
        share = 1.0

        # ---- capacity before the dispatch and the accepted share -------------
        if capacity_on:
            size = float(row['inventory_outcome_environmental']) / config['inventory_ref']
            if not (close(row['redistribution_capacity_before'], remaining['local_redistribute'])
                    and close(row['recovery_capacity_before'], remaining['recovery'])
                    and close(row['dispatch_size'], size)):
                raise ValueError(f'{here}: recorded capacity or dispatch size does not reconstruct')
            if action != 'cold_chain':
                if not accepted_by_destination(action, rho, config):
                    share = 0.0
                elif size > 0.0:
                    share = min(1.0, max(0.0, remaining[action]) / size)
            if not close(row['accepted_share'], share):
                raise ValueError(f'{here}: accepted share {row["accepted_share"]} != {share}')
            counts['capacity_checked'] += 1
            counts['refused_steps'] += int(share < 1.0)
            # What the destinations publish at this decision.
            history.append(dict(remaining))
            published = history[max(0, len(history) - 1 - age)]
            published = {1: published['local_redistribute'], 2: published['recovery']}
            observed_size = max(float(row['inventory_policy_observed']), 0.0) / config['inventory_ref']

        # ---- outcomes, including the refused share ---------------------------
        expected = outcomes.step(row, action, share)
        for name, key in (('waste', 'waste'), ('carbon_kg', 'carbon'), ('slca', 'slca'),
                          ('ari', 'ari')):
            if not close(row[name], expected[key]):
                raise ValueError(f'{here}: {name} {row[name]} does not reconstruct ({expected[key]})')
        counts['outcomes_checked'] += 1

        # ---- rule references: the route follows from what the rule may read ---
        if caps.fixed_rule is not None and capacity_on:
            criteria = stated if policy['acceptance_document_present'] else 'unavailable'
            room = 'unavailable' if offline else published
            must_choose = rule_action(caps.fixed_rule, row, criteria, room, observed_size,
                                      share_reserved, closed, predictor)
            if row['action_idx'] != must_choose:
                raise ValueError(f'{here}: rule chose {row["action_idx"]}, expected {must_choose}')
            counts['rule_decisions_checked'] += 1

        # ---- guard and context under the revised design ----------------------
        if revised and caps.fixed_rule is None:
            guard = row.get('guard') or {}
            report = row.get('capacity_report')
            policy_record = row.get('acceptance_policy')
            if (report is not None) != (requests_report and not offline):
                raise ValueError(f'{here}: capacity report present or absent against the design')
            if report is not None:
                if not (close(report['redistribution_capacity'], published[1])
                        and close(report['recovery_capacity'], published[2])
                        and close(report['dispatch_size'], observed_size)):
                    raise ValueError(f'{here}: capacity report differs from the reconstructed state')
                counts['reports_checked'] += 1
            counts['capacity_report_available_steps'] += int(report is not None)
            if policy_record and policy_record.get('available'):
                counts['acceptance_policy_available_steps'] += 1
                for key, value in stated.items():
                    if policy_record[key] != value:
                        raise ValueError(
                            f'{here}: retrieved criterion {key}={policy_record[key]} is not the '
                            f'value stated in the document ({value})')
            if policy_record is not None:
                role = policy_record.get('retrieved_by', row.get('role'))
                policy_by_role[role] = bool(policy_record.get('available'))
            if caps.guard in ('eligibility', 'reservation'):
                criteria, room = information_states(caps, report, policy_record)
                must_remove = removed_routes(
                    caps.guard, float(row['rho_policy_observed']), criteria, room,
                    report['dispatch_size'] if report is not None else observed_size,
                    share_reserved, closed)
                if (guard.get('kind') != caps.guard
                        or guard.get('criteria') != state_name(criteria)
                        or guard.get('capacity') != state_name(room)):
                    raise ValueError(f'{here}: guard record does not state its kind and information')
                if sorted(guard['removed_routes']) != sorted(must_remove):
                    raise ValueError(
                        f'{here}: guard removed {guard["removed_routes"]}, '
                        f'expected {sorted(must_remove)}')
                for index, reason in must_remove.items():
                    if guard['reasons'][index] != reason:
                        raise ValueError(f'{here}: guard reason {guard["reasons"][index]} != {reason}')
                    removed_by_reason[ACTIONS[index]][reason] += 1
                if row['action_idx'] in must_remove:
                    raise ValueError(f'{here}: executed a route the guard removed')
                before = np.asarray(guard['probs_pre_guard'], dtype=float)
                mask = np.array([0.0 if i in must_remove else 1.0 for i in range(3)])
                after = before * mask
                after = after / after.sum() if after.sum() > 0 else np.array([1.0, 0.0, 0.0])
                if not np.allclose(row['probs'], after, atol=1e-12, rtol=0):
                    raise ValueError(f'{here}: probabilities are not the guarded renormalization')
                uniform = row.get('policy_categorical_uniform')
                if uniform is not None:
                    if categorical_action_from_uniform(after, uniform) != row['action_idx']:
                        raise ValueError(f'{here}: executed action does not follow the recorded draw')
                    if (categorical_action_from_uniform(before, uniform)
                            != guard['sampled_action_pre_guard']):
                        raise ValueError(f'{here}: pre-guard sample does not reconstruct')
                    counts['guard_changed_steps'] += int(
                        guard['sampled_action_pre_guard'] != row['action_idx'])
                counts['guard_removed_steps'] += int(bool(must_remove))
                counts['guard_checked'] += 1
            elif guard:
                raise ValueError(f'{here}: guard recorded for a mode without one')
            psi = row.get('psi')
            if psi is not None and caps.context_kind is not None:
                if caps.constant_adjustment:
                    expected_psi = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
                else:
                    expected_psi = {2: shortfall(report, 'redistribution_capacity'), 3: 0.0,
                                    4: shortfall(report, 'recovery_capacity')}
                    if policy_record and policy_record.get('available'):
                        expected_psi[3] = 1.0 if outside_band(
                            1, float(row['rho_policy_observed']), policy_record) else 0.0
                for index, value in expected_psi.items():
                    if not close(psi[index], value):
                        raise ValueError(
                            f'{here}: context component {index} is {psi[index]}, expected {value}')
                counts['context_checked'] += 1
        if without_terms:
            _queries_without_scenario(row, here)
            counts['queries_checked'] += 1

        # ---- capacity after the dispatch ---------------------------------------
        if capacity_on:
            if action != 'cold_chain':
                diverted += size
                refused += (1.0 - share) * size
                remaining[action] = max(0.0, remaining[action] - share * size)
            demand = float(row['demand_outcome_environmental'])
            remaining['local_redistribute'] = min(
                maxima['local_redistribute'],
                remaining['local_redistribute'] + config['kappa_lr'] * demand / config['demand_ref'])
            remaining['recovery'] = min(maxima['recovery'],
                                        remaining['recovery'] + config['kappa_rec'])

    if capacity_on and not (close(result['diverted_volume'], diverted, 1e-7)
                            and close(result['refused_volume'], refused, 1e-7)):
        raise ValueError(f'{where}: diverted or refused volume does not reconstruct')
    counts['guard_removed_by_reason'] = removed_by_reason
    counts['acceptance_policy_by_role'] = policy_by_role
    return counts


def validate_task(ledger_dir, scenario, modes, parameters, results, suffix=''):
    """Validate the evaluation ledger of every mode of one task and reconcile
    the endpoints of each mode with it.

    ``parameters`` is the record of the setting actually applied: the policy
    fields, the saving coefficients and the route social scores. ``suffix`` is
    ``'.gz'`` for the compressed ledgers kept by the sensitivity tasks. Returns
    the validation counts (totals over the task's modes and counts of each
    mode) and the record of what was reconciled.
    """
    total, by_mode = {}, {}
    reconciled = {'ledgers': len(modes), 'metrics': 0, 'traces': 0, 'trace_values': 0,
                  'header_fields': 0, 'context_weight_records': 0,
                  'largest_metric_difference': 0.0, 'by_mode': {}}
    for mode in modes:
        path = Path(ledger_dir) / f'{mode}__{scenario}.jsonl{suffix}'
        ledger = read_ledger(path)
        counts = validate_ledger(path, mode, parameters['policy'], results[mode], parameters,
                                 ledger)
        by_mode[mode] = counts
        for key, value in counts.items():
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value
        summary = reconcile_endpoints(*ledger, results[mode], mode, parameters, path.name)
        reconciled['by_mode'][mode] = summary
        for key in ('metrics', 'traces', 'header_fields'):
            reconciled[key] += len(summary[key])
        for key in ('trace_values', 'context_weight_records'):
            reconciled[key] += summary[key]
        reconciled['largest_metric_difference'] = max(
            reconciled['largest_metric_difference'], summary['largest_metric_difference'])
    total['ledgers'] = len(modes)
    total['by_mode'] = by_mode
    return total, reconciled


def reconciled_structure(reconciled):
    """What was reconciled, without the size of the largest difference, which
    may differ in the last bit between the machine that ran a task and the one
    that verifies it."""
    def drop(summary):
        return {key: value for key, value in summary.items()
                if key not in ('largest_metric_difference', 'by_mode')}
    return {**drop(reconciled),
            'by_mode': {mode: drop(summary) for mode, summary in reconciled['by_mode'].items()}}
