"""Frozen-policy controls on the primary run.

For each primary cell, the evaluation episode of the full system is executed
again from its frozen checkpoint, in the same environment and with the same
random draws, with the score adjustment replaced:

live       the adjustment as computed; must reproduce the stored evaluation
fixed      the mean adjustment of the three adaptation episodes, at every step
shuffled   the adjustments of the stored evaluation in a permuted order
zero       no adjustment

This is done for both guard levels of the full system: AGRI-BRAIN (reservation
guard) and AGRI-BRAIN with the eligibility guard. Only the adjustment is
replaced. The guard, the peer messages, the learned weights and the
environment are unchanged, and nothing is adapted again. Each execution runs
through the simulator, so the capacity of the destinations follows the
decisions that are actually made. These are executions of one frozen policy,
not separately adapted modes.

Commands: run (one primary task), verify, missing.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate  # noqa: E402

import study  # noqa: E402

CONTROLS = ['live', 'fixed', 'shuffled', 'zero']
MODES = ['agribrain', 'agribrain_eligibility']
PERMUTATION_KEY = 20261112
STEPS = 288


def primary_indices(seed_sets=None):
    families = {s['id']: s['family'] for s in study.design()}
    return [task['index'] for task in study.tasks(seed_sets)
            if families[task['setting']] == 'primary']


def frame_from_archive(frame):
    """The input frame of an archived episode, with its recorded column types."""
    import pandas as pd
    table = pd.DataFrame(frame['rows'], columns=frame['columns'])
    for column, dtype in zip(frame['columns'], frame['dtypes'], strict=True):
        table[column] = (pd.to_datetime(table[column]) if dtype.startswith('datetime')
                         else table[column].astype(dtype))
    table.index = frame['index']
    table.index.name = frame['index_name']
    table.attrs.update(frame['attrs'])
    return table


def softmax(logits, temperature):
    import numpy as np
    scaled = np.asarray(logits, dtype=float) / float(temperature)
    scaled = np.exp(scaled - scaled.max())
    return scaled / scaled.sum()


def check_rows(rows, stored, control, intended):
    """Integrity of one control execution, from its own decision records."""
    import numpy as np
    worst = {'logit_equation': 0.0, 'probability': 0.0, 'allocation': 0.0, 'modifier': 0.0}
    if len(rows) != STEPS:
        raise ValueError(f'{control}: {len(rows)} decisions')
    for step, (row, original) in enumerate(zip(rows, stored, strict=True)):
        modifier = np.asarray(row['context_modifier'], dtype=float)
        base = np.asarray(row['base_logits'], dtype=float)
        logits = np.asarray(row['post_context_logits_pre_override'], dtype=float)
        worst['logit_equation'] = max(worst['logit_equation'],
                                      float(np.max(np.abs(logits - base - modifier))))
        before = softmax(logits, row['policy_temperature'])
        worst['probability'] = max(worst['probability'], float(np.max(np.abs(
            before - np.asarray(row['guard']['probs_pre_guard'], dtype=float)))))
        if row.get('governance_override'):
            raise ValueError(f'{control} step {step}: override fired')
        if row['action_idx'] in row['guard']['removed_routes']:
            raise ValueError(f'{control} step {step}: executed a route the guard removed')
        if not np.allclose(row['effective_context_theta'], original['effective_context_theta'],
                           atol=1e-12, rtol=0):
            raise ValueError(f'{control} step {step}: context weights differ from the checkpoint')
        worst['modifier'] = max(worst['modifier'],
                                float(np.max(np.abs(modifier - intended[step]))))
        if control != 'live':
            allocation = (np.asarray(row['context_feature_contributions'], dtype=float).sum(axis=1)
                          + np.asarray(row['context_nonfeature_residual'], dtype=float))
            worst['allocation'] = max(worst['allocation'],
                                      float(np.max(np.abs(allocation - modifier))))
    if max(worst.values()) > 1e-12:
        raise ValueError(f'{control}: integrity check failed: {worst}')
    return worst


def run_mode(gr, mode, record, evidence, setting, work, dest):
    """Execute the four controls of one mode; return their results and files."""
    import numpy as np
    from src.agents.coordinator import AgentCoordinator
    from src.models import action_selection

    seed, scenario = record['task']['seed'], record['task']['scenario']
    archive = json.loads(gzip.decompress(
        (evidence / 'complete_episode_evidence' / f'{mode}__{scenario}' / 'episode_3.json.gz')
        .read_bytes()))
    frame = frame_from_archive(archive['input_frame'])
    stored = validate.read_ledger(evidence / f'{mode}__{scenario}.jsonl')[1]
    live = np.array([row['context_modifier'] for row in stored], dtype=float)
    adaptation = []
    for episode in range(3):
        path = (evidence / 'adaptation_episode_ledgers' / f'{mode}__{scenario}'
                / f'episode_{episode}.jsonl.gz')
        adaptation.extend(row['context_modifier'] for row in validate.read_ledger(path)[1])
    fixed = np.mean(np.array(adaptation, dtype=float), axis=0)
    permutation = np.random.default_rng(np.random.SeedSequence(
        [PERMUTATION_KEY, seed, study.SCENARIOS.index(scenario),
         MODES.index(mode)])).permutation(STEPS)
    intended = {'live': live, 'fixed': np.tile(fixed, (STEPS, 1)),
                'shuffled': live[permutation], 'zero': np.zeros((STEPS, 3))}
    original_method = AgentCoordinator._compute_step_context
    original_theta = action_selection.THETA.copy()
    controls, files = {}, {}
    for control in CONTROLS:
        replacement = intended[control]

        def intervention(self, *args, _control=control, _replacement=replacement, **kwargs):
            computed = original_method(self, *args, **kwargs)
            if _control == 'live':
                return computed
            hour = kwargs['hour'] if 'hour' in kwargs else args[3]
            applied = np.asarray(_replacement[int(round(float(hour) * 4))], dtype=float).copy()
            self._step_context_modifier = applied.copy()
            # The replacement is not a function of the context features.
            self._step_context_feature_contributions = np.zeros((3, 5))
            self._step_context_nonfeature_residual = applied.copy()
            if self._context_log:
                self._context_log[-1]['context_modifier'] = applied.tolist()
                self._context_log[-1]['frozen_policy_control'] = _control
            return applied

        ledgers = work / f'{mode}_{control}'
        try:
            AgentCoordinator._compute_step_context = intervention
            with (work / f'{mode}_{control}.log').open('w', encoding='utf-8') as log, \
                    redirect_stdout(log), redirect_stderr(log), \
                    study.overridden(gr, setting, scenario, ledgers):
                action_selection.THETA = gr.policy_theta_for_seed(action_selection.THETA, seed)
                environment_seed = gr._stream_seed(seed, scenario, 3, 'environment')
                policy_seed = gr._stream_seed(seed, scenario, 3, 'policy')
                with gr.decision_ledger_scope(ledgers, reset=True):
                    result = gr.run_episode(
                        frame, mode, gr.Policy(), np.random.default_rng(policy_seed), scenario,
                        stoch=gr.make_stochastic_layer(
                            np.random.default_rng(environment_seed), stream_seed=environment_seed),
                        seed=seed, benchmark_seed=seed, episode_index=3,
                        learner_state_cache={
                            mode: deepcopy(archive['learner_state']['before'])},
                        learning_enabled=False)
        finally:
            AgentCoordinator._compute_step_context = original_method
            action_selection.THETA = original_theta.copy()
        if not result['learner_freeze_summary']['learners_frozen']:
            raise ValueError(f'{mode} {control}: learners were not frozen')
        ledger = ledgers / f'{mode}__{scenario}.jsonl'
        rows = validate.read_ledger(ledger)[1]
        integrity = check_rows(rows, stored, control, intended[control])
        if control == 'live':
            if result['action_trace'] != record['results'][mode]['action_trace']:
                raise ValueError(f'{mode} live: routing differs from the stored evaluation')
            error = max(abs(a['ari'] - b['ari']) for a, b in zip(rows, stored, strict=True))
            if error > 1e-10:
                raise ValueError(f'{mode} live: step ARI differs from the stored evaluation by {error}')
            integrity['stored_evaluation_step_ari'] = error
        kept = dest / f'{mode}__{control}__{scenario}.jsonl.gz'
        with ledger.open('rb') as source, gzip.GzipFile(kept, 'wb', compresslevel=5, mtime=0) as sink:
            sink.write(source.read())
        files[kept.name] = study.sha(kept)
        controls[control] = {
            **{key: result[key] for key in study.METRICS},
            'ari_trace': result['ari_trace'], 'action_trace': result['action_trace'],
            'refused_share_of_diverted_volume': result['refused_share_of_diverted_volume'],
            'guard_removed_step_fraction': result['guard_removed_step_fraction'],
            'guard_changed_action_fraction': result['guard_changed_action_fraction'],
            'integrity': integrity}
    return controls, files, {'fixed_adjustment': fixed.tolist(),
                             'permutation': permutation.tolist()}


def run_cell(output, index, development):
    from mvp.simulation import generate_results as gr

    output = Path(output).resolve()
    plan = study.read_plan(output, development)
    if index not in primary_indices(plan['seed_sets']):
        raise ValueError('Controls are defined for primary tasks only')
    dest = output / 'controls' / f'{index:04d}'
    if (dest / 'result.json').exists():
        verify_cell(output, index)
        print(f'Already complete and verified: {index}')
        return
    record = study.verify_result(output, index)
    task, setting = record['task'], record['setting']
    study.refuse_final_seeds_in_development(task['seed'], development)
    evidence = output / 'tasks' / f'{index:04d}' / record['attempt_directory'] / 'decision_ledgers'
    dest.mkdir(parents=True, exist_ok=True)
    work = dest / f'attempt_{os.getpid()}_{int(time.time() * 1000)}'
    work.mkdir()
    os.environ['RUN_TAG'] = plan['run_tag']
    os.environ['AGRIBRAIN_GIT_COMMIT'] = plan['source']['commit']
    started = time.monotonic()
    by_mode, files, inputs = {}, {}, {}
    for mode in MODES:
        by_mode[mode], mode_files, inputs[mode] = run_mode(
            gr, mode, record, evidence, setting, work, dest)
        files.update(mode_files)
    study.save(dest / 'result.json', {
        'task': task, 'plan_sha256': study.sha(output / 'plan.json'), 'source': plan['source'],
        'controls': by_mode, 'files': files, 'inputs': inputs,
        'development_only': development, 'seconds': time.monotonic() - started})
    shutil.rmtree(work)
    print(f"COMPLETE controls {index} {task['setting']} seed {task['seed']} {task['scenario']}: "
          + '; '.join(f'{mode} ' + ', '.join(
              f'{c}={by_mode[mode][c]["ari"]:.6f}' for c in CONTROLS) for mode in MODES))


def verify_cell(output, index):
    output = Path(output)
    dest = output / 'controls' / f'{index:04d}'
    record = study.load(dest / 'result.json')
    plan = study.load(output / 'plan.json')
    if (record['task'] != study.tasks(plan['seed_sets'])[index]
            or record['plan_sha256'] != study.sha(output / 'plan.json')):
        raise ValueError('Task/plan mismatch')
    if (set(record['controls']) != set(MODES)
            or len(record['files']) != len(MODES) * len(CONTROLS)):
        raise ValueError('Control set differs')
    for name, digest in record['files'].items():
        if Path(name).name != name or study.sha(dest / name) != digest:
            raise ValueError(f'Changed control evidence {index}: {name}')
    for mode, controls in record['controls'].items():
        if set(controls) != set(CONTROLS):
            raise ValueError(f'{mode}: control set differs')
        for control, values in controls.items():
            if len(values['action_trace']) != STEPS or len(values['ari_trace']) != STEPS:
                raise ValueError(f'{mode} {control}: incomplete execution')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--development-runtime', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ['run', 'verify', 'missing']:
        p = sub.add_parser(command)
        p.add_argument('--output', required=True, type=Path)
        if command != 'missing':
            p.add_argument('--index', required=True, type=int)
    args = parser.parse_args()
    development = study.configure(args.development_runtime)
    # The control executions keep their decision ledgers, not episode archives.
    os.environ['FULL_EVIDENCE_CAPTURE'] = '0'
    if args.command == 'run':
        run_cell(args.output, args.index, development)
    elif args.command == 'verify':
        study.read_plan(args.output, development)
        verify_cell(args.output, args.index)
        print('Verified')
    else:
        plan = study.read_plan(args.output, development)
        missing = []
        for index in primary_indices(plan['seed_sets']):
            try:
                verify_cell(args.output, index)
            except (OSError, ValueError, KeyError):
                missing.append(index)
        print(','.join(map(str, missing)))


if __name__ == '__main__':
    main()
