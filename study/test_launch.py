"""Tests of the scripts that submit, resume and collect the study.

The scripts run unchanged, from a copy of their folder, against stand-ins for
the scheduler and for the commands of the study. The scheduler stand-in
honours the dependencies the scripts declare: a job that needs the success of
another is cancelled when that one fails, as ``--kill-on-invalid-dep`` asks.
The stand-in for the commands keeps a completed task as a result file and
records every call, so a test can state what ran, and in which order.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCRIPTS = ['submit.sh', 'retry.sh', 'worker.sh', 'collect.sh', 'status.sh']
ENVIRONMENT = ['ensure_git_available.sh', 'publication_env.sh']
# The stand-in study: 1,010 tasks, so three blocks of 500; four pilot cells, listed out
# of order as in the design, two of them repeated; controls for the first four tasks.
TASKS = 1010
PILOT = [0, 2, 1005, 400]
REPEAT = [0, 400]
PRIMARY = [0, 1, 2, 3]
RESULTS = 'study_results'


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def result(output, index, kind='tasks'):
    return Path(output) / kind / f'{index:04d}' / 'result.json'


def control(output, index):
    """A completed control cell: its record and a ledger."""
    write(result(output, index, 'controls'), '{}\n')
    write(result(output, index, 'controls').with_name('agribrain__live__x.jsonl.gz'), 'ledger\n')


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True)
    for name in ['plan.json', 'settings.json', 'tasks.json', 'setup_environment.json']:
        write(output / name, json.dumps({'run_tag': output.name, 'counts': {'tasks': TASKS}}) + '\n')
    write(output / 'pilot_indices.txt', ''.join(f'{i}\n' for i in PILOT))
    write(output / 'repeat_indices.txt', ''.join(f'{i}\n' for i in REPEAT))


def stand_in(arguments, state):
    """Answer one command of study.py, controls.py or analyze.py."""
    script, rest = Path(arguments[0]).name, list(arguments[1:])
    command = 'analyze' if script == 'analyze.py' else rest.pop(0)
    if script == 'controls.py':
        command = 'controls' if command == 'run' else 'controls-missing'
    options = dict(zip(rest[::2], rest[1::2], strict=True))
    output = Path(options['--output'])
    index = int(options['--index']) if '--index' in options else None
    with (state / 'calls').open('a', encoding='utf-8') as log:
        log.write(json.dumps([command, index, output.name]) + '\n')
    if command == 'prepare':
        prepare(output)
    elif command == 'task':
        # A test names the tasks that fail, apart for the study and for its repeat.
        kind = 'repeat' if output.name == 'reproducibility_repeat' else 'task'
        if (state / f'fail_{kind}_{index}').exists():
            return 1
        write(result(output, index), '{}\n')
    elif command == 'missing':
        print(','.join(str(i) for i in range(TASKS) if not result(output, i).exists()))
    elif command == 'pilot-check':
        repeat = output / 'reproducibility_repeat'
        if ((state / 'fail_check').exists()
                or not all(result(output, i).exists() for i in PILOT)
                or not all(result(repeat, i).exists() for i in REPEAT)):
            return 1
        write(output / 'PILOT_PASSED.json',
              json.dumps({'plan_sha256': digest(output / 'plan.json')}) + '\n')
    elif command == 'controls-missing':
        print(','.join(str(i) for i in PRIMARY if not result(output, i, 'controls').exists()))
    elif command == 'controls':
        if not result(output, index).exists():
            return 1
        control(output, index)
    elif command == 'analyze':
        if ((state / 'fail_analysis').exists()
                or not all(result(output, i).exists() for i in range(TASKS))
                or not all(result(output, i, 'controls').exists() for i in PRIMARY)):
            return 1
        write(output / 'analysis' / 'summary.json', '{}\n')
        write(output / 'COMPLETE.json', '{}\n')
    else:
        return 2
    return 0


if __name__ == '__main__':
    # The stand-in runs as a command of its own and needs none of the tests below.
    sys.exit(stand_in(sys.argv[1:], Path(os.environ['STUDY_TEST_STATE'])))

import pytest  # noqa: E402

# Every script sources the activation of its environment. Here that file defines the
# scheduler, the interpreter and three tools as shell functions: the scripts reach the
# stand-ins without any change, nothing in the test folder has to be executable, and the
# real scheduler cannot be reached.
DOUBLES = {
    'python': r'''
python() {  # the commands of the study go to their stand-in; anything else to the interpreter
    case "${1:-}" in
        study/*.py) "$STUDY_TEST_PYTHON" "$STUDY_TEST_MODULE" "$@" ;;
        *) "$STUDY_TEST_PYTHON" "$@" ;;
    esac
}
''',
    'sbatch': r'''
sbatch() {  # record the submission and answer as sbatch --parsable does
    local count id
    count=$(ls "$STUDY_TEST_SCHEDULER" | grep -c '\.job$') || true
    if [ -e "$STUDY_TEST_STATE/fail_sbatch" ] \
            && [ "$count" -ge "$(cat "$STUDY_TEST_STATE/fail_sbatch")" ]; then
        echo 'sbatch: error: Batch job submission failed' >&2
        return 1
    fi
    id=$((1001 + count))
    printf '%s\n' "$@" > "$STUDY_TEST_SCHEDULER/$id.job"
    env | grep '^AGRI_STUDY_' > "$STUDY_TEST_SCHEDULER/$id.env" || true
    echo "$id;cluster"
}
''',
    'squeue': r'''
squeue() {  # the jobs a test declares to be queued or running, or a queue that cannot be read
    if [ -e "$STUDY_TEST_STATE/fail_squeue" ]; then
        echo 'squeue: error: Unable to contact slurm controller' >&2
        return 1
    fi
    cat "$STUDY_TEST_SCHEDULER/queued" 2> /dev/null || true
}
''',
    'sleep': r'''
sleep() { :; }
''',
    'git': r'''
git() {  # the scripts ask git for the bundle of the source only
    [ "${1:-} ${2:-}" = 'bundle create' ] || return 1
    echo bundle > "$3"
}
''',
    'tar': r'''
tar() {  # the real tar, unless the test asks that the archive of the whole study fail
    if [ -e "$STUDY_TEST_STATE/fail_full_archive" ] && [[ "$2" != *_analysis.tar.gz* ]]; then
        echo incomplete > "$2"
        echo 'tar: write error' >&2
        return 2
    fi
    command tar "$@"
}
''',
}
EXECUTED = ('task', 'pilot-check', 'controls', 'analyze')


def find_bash():
    """A POSIX bash; on Windows the launcher of the Linux subsystem is not one."""
    for candidate in [os.environ.get('STUDY_TEST_BASH'), shutil.which('bash')]:
        if candidate and 'system32' not in candidate.lower():
            return candidate
    return None


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason='the scripts of the study need bash')


def expand(specification):
    """Array indices of a specification such as 1,3-5,9."""
    indices = []
    for part in specification.split(','):
        first, _, last = part.partition('-')
        indices.extend(range(int(first), int(last or first) + 1))
    return indices


class Launcher:
    """A copy of the scripts with the stand-ins in front of the scheduler and the study."""

    def __init__(self, root):
        self.root = root
        self.repo, self.state, self.scheduler = root / 'repo', root / 'state', root / 'scheduler'
        self.output = root / RESULTS
        self.states = {}
        for folder in [self.state, self.scheduler]:
            folder.mkdir()
        for name in SCRIPTS:
            write(self.repo / 'study' / name, (HERE / name).read_text(encoding='utf-8'))
        for name in ENVIRONMENT:
            write(self.repo / 'hpc' / name, (REPO / 'hpc' / name).read_text(encoding='utf-8'))
        write(root / 'venv' / 'bin' / 'activate', ''.join(DOUBLES.values()))
        write(self.repo / 'study' / 'setup.env',
              f'export AGRI_STUDY_VENV="{(root / "venv").as_posix()}"\n')
        self.environment = {
            **{k: v for k, v in os.environ.items()
               if not k.startswith(('SLURM_', 'AGRI_STUDY_', 'STUDY_TEST_'))},
            'STUDY_TEST_PYTHON': Path(sys.executable).as_posix(),
            'STUDY_TEST_MODULE': Path(__file__).resolve().as_posix(),
            'STUDY_TEST_STATE': self.state.as_posix(),
            'STUDY_TEST_SCHEDULER': self.scheduler.as_posix(),
            'AGRI_STUDY_REPO': self.repo.as_posix(),
            'AGRI_STUDY_VENV': (root / 'venv').as_posix(),
            'AGRI_STUDY_OUTPUT': self.output.as_posix()}

    def script(self, name, *arguments, **variables):
        return subprocess.run([BASH, f'study/{name}', *arguments], cwd=self.repo,
                              env={**self.environment, **variables}, capture_output=True, text=True)

    def worker(self, kind, *arguments, element=None):
        variables = {} if element is None else {'SLURM_ARRAY_TASK_ID': str(element)}
        return self.script('worker.sh', kind, *arguments, **variables)

    def interrupted(self, missing=(), missing_repeat=(), controls=PRIMARY, passed=True):
        """A study whose jobs have ended: complete but for what is named."""
        repeat = self.output / 'reproducibility_repeat'
        prepare(self.output)
        prepare(repeat)
        for index in set(range(TASKS)) - set(missing):
            write(result(self.output, index), '{}\n')
        for index in set(REPEAT) - set(missing_repeat):
            write(result(repeat, index), '{}\n')
        for index in controls:
            control(self.output, index)
        if passed:
            write(self.output / 'PILOT_PASSED.json',
                  json.dumps({'plan_sha256': digest(self.output / 'plan.json')}) + '\n')
        (self.output / 'logs').mkdir()
        write(self.output / 'job_ids.txt', 'PILOT=901\nREPEAT=902\nCHECK=903\n')
        write(self.output / 'source.bundle', 'bundle\n')

    def jobs(self):
        """The submitted jobs by name, in the order of submission."""
        names = dict(line.split('=')[::-1] for line in
                     (self.output / 'job_ids.txt').read_text(encoding='utf-8').splitlines())
        found = {}
        for path in sorted(self.scheduler.glob('*.job'), key=lambda p: int(p.stem)):
            words = path.read_text(encoding='utf-8').splitlines()
            job = {'id': path.stem, 'needs': [], 'after': [], 'array': None, 'array_text': None}
            while words[0].startswith('--'):
                option, _, value = words.pop(0).partition('=')
                if option == '--dependency':
                    for part in value.split(','):
                        kind, *others = part.split(':')
                        job[{'afterok': 'needs', 'afterany': 'after'}[kind]] += others
                elif option == '--array':
                    job['array_text'] = value.partition('%')[0]
                    job['array'] = expand(job['array_text'])
            job['script'], job['arguments'] = words[0], words[1:]
            job['environment'] = dict(
                line.split('=', 1) for line in
                path.with_suffix('.env').read_text(encoding='utf-8').splitlines())
            found[names[path.stem]] = job
        return found

    def run_queue(self, ends=3):
        """Run the submitted jobs in order, as the scheduler would release them.

        Of an array of more than ``2 * ends`` tasks only the first and the last
        ``ends`` are executed. Returns the state of each job by name.
        """
        for name, job in self.jobs().items():
            if job['id'] in self.states:
                continue
            if any(self.states.get(other) != 'COMPLETED' for other in job['needs']):
                self.states[job['id']] = 'CANCELLED'
                continue
            assert all(other in self.states for other in job['after']), name
            elements = [None] if job['array'] is None else job['array']
            if len(elements) > 2 * ends:
                elements = elements[:ends] + elements[-ends:]
            codes = []
            for element in elements:
                variables = {**job['environment'], 'SLURM_JOB_ID': job['id'],
                             'SLURM_CPUS_PER_TASK': '2'}
                if element is not None:
                    variables['SLURM_ARRAY_TASK_ID'] = str(element)
                codes.append(self.script(Path(job['script']).name, *job['arguments'],
                                         **variables).returncode)
            self.states[job['id']] = 'FAILED' if any(codes) else 'COMPLETED'
        return {name: self.states[job['id']] for name, job in self.jobs().items()}

    def calls(self, *commands):
        """The recorded calls of the named commands: (command, index, folder)."""
        log = self.state / 'calls'
        rows = [tuple(json.loads(line)) for line in log.read_text(encoding='utf-8').splitlines()
                ] if log.exists() else []
        return [row for row in rows if row[0] in commands]

    def collected(self):
        """Whether the record of a complete collection exists and holds for both archives."""
        record = self.root / f'{RESULTS}_COLLECTED.sha256'
        if not record.exists():
            return False
        lines = [line.split() for line in record.read_text(encoding='utf-8').splitlines()]
        names = [name.lstrip('*') for _, name in lines]
        assert names == [f'{RESULTS}_analysis.tar.gz', f'{RESULTS}.tar.gz']
        for (checksum, _), name in zip(lines, names, strict=True):
            assert digest(self.root / name) == checksum
            assert (self.root / f'{name}.sha256').read_text(encoding='utf-8').split()[0] == checksum
        return True


@pytest.fixture
def launcher(tmp_path):
    return Launcher(tmp_path)


def task(index, folder=RESULTS):
    return ('task', index, folder)


def test_submission_puts_the_whole_study_behind_the_pilot_check(launcher):
    done = launcher.script('submit.sh', '--partition=compute')
    assert done.returncode == 0, done.stderr
    jobs = launcher.jobs()
    assert list(jobs) == ['PILOT', 'REPEAT', 'CHECK', 'BLOCK_0', 'BLOCK_500', 'BLOCK_1000',
                          'CONTROLS', 'COLLECT']
    identifier = {name: job['id'] for name, job in jobs.items()}
    assert {name: job['needs'] for name, job in jobs.items()} == {
        'PILOT': [], 'REPEAT': [], 'CHECK': [identifier['PILOT'], identifier['REPEAT']],
        'BLOCK_0': [identifier['CHECK']], 'BLOCK_500': [identifier['BLOCK_0']],
        'BLOCK_1000': [identifier['BLOCK_500']], 'CONTROLS': [identifier['BLOCK_1000']],
        'COLLECT': [identifier['CONTROLS']]}
    assert {name: job['array_text'] for name, job in jobs.items()} == {
        'PILOT': '0-3', 'REPEAT': '0-1', 'CHECK': None, 'BLOCK_0': '0-499', 'BLOCK_500': '0-499',
        'BLOCK_1000': '0-9', 'CONTROLS': '0-3', 'COLLECT': None}
    assert all('--partition=compute' in (launcher.scheduler / f'{job["id"]}.job').read_text(
        encoding='utf-8').splitlines() for job in jobs.values())

    # A pilot cell fails: its check and everything behind it are cancelled.
    (launcher.state / 'fail_task_400').touch()
    states = launcher.run_queue()
    assert states == {'PILOT': 'FAILED', 'REPEAT': 'COMPLETED', 'CHECK': 'CANCELLED',
                      'BLOCK_0': 'CANCELLED', 'BLOCK_500': 'CANCELLED', 'BLOCK_1000': 'CANCELLED',
                      'CONTROLS': 'CANCELLED', 'COLLECT': 'CANCELLED'}
    assert launcher.calls(*EXECUTED) == [
        task(0), task(2), task(1005), task(400), task(400),
        task(0, 'reproducibility_repeat'), task(400, 'reproducibility_repeat')]
    assert not (launcher.output / 'PILOT_PASSED.json').exists()


def test_submitted_blocks_run_after_the_check_and_address_their_tasks(launcher):
    assert launcher.script('submit.sh').returncode == 0
    states = launcher.run_queue(ends=4)
    assert [states[name] for name in ['PILOT', 'REPEAT', 'CHECK', 'BLOCK_0', 'BLOCK_500',
                                      'BLOCK_1000', 'CONTROLS']] == ['COMPLETED'] * 7
    calls = launcher.calls(*EXECUTED)
    check = calls.index(('pilot-check', None, RESULTS))
    assert sorted(calls[:check]) == sorted(
        [task(i) for i in PILOT] + [task(i, 'reproducibility_repeat') for i in REPEAT])
    after = calls[check + 1:]
    assert [call for call in after if call[0] == 'task'] == [task(i) for i in [
        0, 1, 2, 3, 496, 497, 498, 499, 500, 501, 502, 503, 996, 997, 998, 999,
        1000, 1001, 1002, 1003, 1006, 1007, 1008, 1009]]
    assert [call for call in after if call[0] == 'controls'] == [
        ('controls', i, RESULTS) for i in PRIMARY]
    # Only the ends of each block ran here, so the analysis refuses the study.
    assert states['COLLECT'] == 'FAILED' and not launcher.collected()


def test_failed_submission_stops_the_chain(launcher):
    write(launcher.state / 'fail_sbatch', '2\n')
    done = launcher.script('submit.sh')
    assert done.returncode != 0
    assert 'Submission of CHECK failed' in done.stderr
    assert list(launcher.jobs()) == ['PILOT', 'REPEAT']


def test_resumption_recovers_the_pilot_then_the_rest(launcher):
    launcher.interrupted(missing=[2, 7, 8, 9, 499, 500, 1009], missing_repeat=[400],
                         controls=[0], passed=False)
    done = launcher.script('retry.sh', '--partition=compute')
    assert done.returncode == 0, done.stderr
    jobs = launcher.jobs()
    assert list(jobs) == ['RETRY_PILOT', 'RETRY_REPEAT', 'RETRY_CHECK', 'RETRY_BLOCK_0',
                          'RETRY_BLOCK_500', 'RETRY_BLOCK_1000', 'RETRY_CONTROLS', 'RETRY_COLLECT']
    identifier = {name[6:]: job['id'] for name, job in jobs.items()}
    # Pilot cells and repeats are named by their position in their lists; pilot cells
    # are not submitted again with the blocks.
    assert {name[6:]: job['array_text'] for name, job in jobs.items()} == {
        'PILOT': '1', 'REPEAT': '1', 'CHECK': None, 'BLOCK_0': '7-9,499', 'BLOCK_500': '0',
        'BLOCK_1000': '9', 'CONTROLS': '1,2,3', 'COLLECT': None}
    check = identifier['CHECK']
    assert {name[6:]: (job['needs'], job['after']) for name, job in jobs.items()} == {
        'PILOT': ([], []), 'REPEAT': ([], []),
        'CHECK': ([identifier['PILOT'], identifier['REPEAT']], []),
        'BLOCK_0': ([check], []), 'BLOCK_500': ([check], [identifier['BLOCK_0']]),
        'BLOCK_1000': ([check], [identifier['BLOCK_500']]),
        'CONTROLS': ([check, identifier['BLOCK_1000']], []),
        'COLLECT': ([identifier['CONTROLS']], [])}

    assert set(launcher.run_queue().values()) == {'COMPLETED'}
    assert launcher.calls(*EXECUTED) == [
        task(2), task(400, 'reproducibility_repeat'), ('pilot-check', None, RESULTS),
        task(7), task(8), task(9), task(499), task(500), task(1009),
        ('controls', 1, RESULTS), ('controls', 2, RESULTS), ('controls', 3, RESULTS),
        ('pilot-check', None, RESULTS), ('analyze', None, RESULTS)]
    assert launcher.collected()


def test_resumption_releases_nothing_when_the_pilot_check_fails(launcher):
    # The check had passed before the interruption; that pass does not stand in.
    launcher.interrupted(missing=[2, 7, 499, 500, 1009], missing_repeat=[400], passed=True)
    (launcher.state / 'fail_check').touch()
    assert launcher.script('retry.sh').returncode == 0
    states = launcher.run_queue()
    assert states == {'RETRY_PILOT': 'COMPLETED', 'RETRY_REPEAT': 'COMPLETED',
                      'RETRY_CHECK': 'FAILED', 'RETRY_BLOCK_0': 'CANCELLED',
                      'RETRY_BLOCK_500': 'CANCELLED', 'RETRY_BLOCK_1000': 'CANCELLED',
                      'RETRY_COLLECT': 'CANCELLED'}
    assert launcher.calls(*EXECUTED) == [
        task(2), task(400, 'reproducibility_repeat'), ('pilot-check', None, RESULTS)]
    assert not (launcher.output / 'PILOT_PASSED.json').exists()
    assert not launcher.collected()


def test_resumption_releases_nothing_when_a_pilot_cell_fails_again(launcher):
    launcher.interrupted(missing=[2, 7, 499, 500, 1009], passed=False)
    (launcher.state / 'fail_task_2').touch()
    assert launcher.script('retry.sh').returncode == 0
    states = launcher.run_queue()
    assert states.pop('RETRY_PILOT') == 'FAILED'
    assert set(states.values()) == {'CANCELLED'}
    assert launcher.calls(*EXECUTED) == [task(2), task(2)]


def test_resumption_of_a_complete_study_checks_and_collects(launcher):
    launcher.interrupted()
    assert launcher.script('retry.sh').returncode == 0
    jobs = launcher.jobs()
    assert list(jobs) == ['RETRY_CHECK', 'RETRY_COLLECT']
    assert jobs['RETRY_CHECK']['needs'] == []
    assert jobs['RETRY_COLLECT']['needs'] == [jobs['RETRY_CHECK']['id']]
    assert set(launcher.run_queue().values()) == {'COMPLETED'}
    assert launcher.calls(*EXECUTED) == [
        ('pilot-check', None, RESULTS), ('pilot-check', None, RESULTS), ('analyze', None, RESULTS)]
    assert launcher.collected()


def test_resumption_of_a_collected_study_does_nothing(launcher):
    launcher.interrupted()
    assert launcher.script('collect.sh').returncode == 0 and launcher.collected()
    done = launcher.script('retry.sh')
    assert done.returncode == 0
    assert 'there is nothing to resume' in done.stdout
    assert not list(launcher.scheduler.glob('*.job'))
    assert launcher.collected()


def test_a_failed_task_does_not_hold_back_later_blocks(launcher):
    launcher.interrupted(missing=[7, 500, 1009])
    (launcher.state / 'fail_task_7').touch()
    assert launcher.script('retry.sh').returncode == 0
    states = launcher.run_queue()
    assert states == {'RETRY_CHECK': 'COMPLETED', 'RETRY_BLOCK_0': 'FAILED',
                      'RETRY_BLOCK_500': 'COMPLETED', 'RETRY_BLOCK_1000': 'COMPLETED',
                      'RETRY_COLLECT': 'FAILED'}
    assert launcher.calls('task') == [task(7), task(7), task(500), task(1009)]
    assert not launcher.collected()


def test_resumption_waits_for_the_earlier_jobs(launcher):
    launcher.interrupted(missing=[7])
    write(launcher.scheduler / 'queued', '77\n903\n')
    done = launcher.script('retry.sh')
    assert done.returncode == 1
    assert 'still queued or running: 903' in done.stderr
    assert not list(launcher.scheduler.glob('*.job'))
    assert launcher.calls('missing') == []


def test_resumption_stops_when_the_queue_cannot_be_read(launcher):
    launcher.interrupted(missing=[2, 7, 500], missing_repeat=[400], passed=False)
    (launcher.state / 'fail_squeue').touch()
    done = launcher.script('retry.sh')
    assert done.returncode == 1
    assert 'The queue could not be read' in done.stderr and 'Nothing was resumed' in done.stderr
    # No task was listed, nothing was submitted, nothing was written.
    assert launcher.calls('missing', 'controls-missing') == []
    assert not list(launcher.scheduler.glob('*.job'))
    assert not (launcher.output / 'missing.txt').exists()
    assert not (launcher.output / 'resumption.txt').exists()


def test_nothing_outside_the_pilot_runs_before_the_check_has_passed(launcher):
    launcher.interrupted(missing=[7], controls=[], passed=False)
    for kind, arguments, element in [('study', ['0'], 7), ('controls', [], 1)]:
        done = launcher.worker(kind, *arguments, element=element)
        assert done.returncode == 3
        assert 'The pilot check has not passed for this plan' in done.stderr
    # A pass recorded for another plan releases nothing either.
    write(launcher.output / 'PILOT_PASSED.json', json.dumps({'plan_sha256': '0' * 64}) + '\n')
    assert launcher.worker('study', '0', element=7).returncode == 3
    assert launcher.calls('task', 'controls') == []
    # Pilot cells are not held back, and the passed check releases the rest.
    assert launcher.worker('pilot', element=3).returncode == 0
    assert launcher.worker('check').returncode == 0
    assert launcher.worker('study', '0', element=7).returncode == 0
    assert launcher.worker('study', '500', element=505).returncode == 0
    assert launcher.worker('controls', element=1).returncode == 0
    assert launcher.calls('task', 'controls') == [
        task(400), task(7), task(1005), ('controls', 1, RESULTS)]


def test_a_failed_check_withdraws_an_earlier_pass(launcher):
    launcher.interrupted(missing=[7], passed=True)
    (launcher.state / 'fail_check').touch()
    assert launcher.worker('check').returncode != 0
    assert not (launcher.output / 'PILOT_PASSED.json').exists()
    assert launcher.worker('study', '0', element=7).returncode == 3
    assert launcher.calls('task') == []


def test_collection_is_recorded_only_after_both_archives(launcher):
    launcher.interrupted()
    full, compact = launcher.root / f'{RESULTS}.tar.gz', launcher.root / f'{RESULTS}_analysis.tar.gz'
    (launcher.state / 'fail_full_archive').touch()
    assert launcher.script('collect.sh').returncode != 0
    # The analysis finished and the compact archive is whole, with its checksum;
    # the archive of the study never took its name and nothing records a collection.
    assert (launcher.output / 'COMPLETE.json').exists()
    assert digest(compact) == Path(f'{compact}.sha256').read_text(encoding='utf-8').split()[0]
    assert not full.exists() and not Path(f'{full}.sha256').exists()
    assert not launcher.collected()

    (launcher.state / 'fail_full_archive').unlink()
    done = launcher.script('collect.sh')
    assert done.returncode == 0, done.stderr
    assert launcher.collected()
    with tarfile.open(compact) as archive:
        members = set(archive.getnames())
    assert {'analysis/summary.json', 'controls/0000/result.json', 'plan.json', 'settings.json',
            'tasks.json', 'setup_environment.json', 'PILOT_PASSED.json', 'COMPLETE.json',
            'job_ids.txt', 'source.bundle'} <= members
    # The compact archive carries the records of the controls, the full one their ledgers too.
    ledger = 'controls/0000/agribrain__live__x.jsonl.gz'
    assert ledger not in members and not any(name.endswith('.jsonl.gz') for name in members)
    with tarfile.open(full) as archive:
        names = set(archive.getnames())
    assert {f'{RESULTS}/tasks/1009/result.json', f'{RESULTS}/{ledger}'} <= names

    # A later collection that stops early withdraws the record of this one.
    (launcher.state / 'fail_analysis').touch()
    assert launcher.script('collect.sh').returncode != 0
    assert not launcher.collected()


def test_a_failed_reanalysis_is_not_reported_as_an_archive_failure(launcher):
    # The analysis succeeds and the archive of the study fails ...
    launcher.interrupted()
    (launcher.state / 'fail_full_archive').touch()
    assert launcher.script('collect.sh').returncode != 0
    assert (launcher.output / 'COMPLETE.json').exists()
    report = launcher.script('status.sh', launcher.output.as_posix()).stdout.splitlines()
    assert report[-1].startswith('Not complete: the archives.')
    # ... then the next collection fails in its analysis: the earlier analysis no longer counts.
    (launcher.state / 'fail_full_archive').unlink()
    (launcher.state / 'fail_analysis').touch()
    assert launcher.script('collect.sh').returncode != 0
    assert not (launcher.output / 'COMPLETE.json').exists()
    assert not launcher.collected()
    report = launcher.script('status.sh', launcher.output.as_posix()).stdout.splitlines()
    assert report[-1].startswith('Not complete: the controls or the analysis.')
    assert 'analysis complete              no' in report


@pytest.mark.parametrize('state, stage', [
    ({'missing': [2], 'passed': False}, 'Not complete: the pilot cells or their repeat.'),
    ({'missing_repeat': [400], 'passed': False}, 'Not complete: the pilot cells or their repeat.'),
    ({'passed': False}, 'Not complete: the pilot check.'),
    ({'missing': [7, 500]}, 'Not complete: the tasks.'),
    ({'controls': [0]}, 'Not complete: the controls or the analysis.'),
    ({'analysed': True}, 'Not complete: the archives.'),
    ({'collected': True}, f'Complete. Download {RESULTS}_analysis.tar.gz'),
])
def test_status_names_the_stage_that_is_not_complete(launcher, state, stage):
    analysed, collected = state.pop('analysed', False), state.pop('collected', False)
    launcher.interrupted(**state)
    if analysed:
        write(launcher.output / 'COMPLETE.json', '{}\n')
    if collected:
        assert launcher.script('collect.sh').returncode == 0
    before = sorted(p.as_posix() for p in launcher.root.rglob('*'))
    report = launcher.script('status.sh', launcher.output.as_posix())
    assert report.returncode == 0, report.stderr
    lines = report.stdout.splitlines()
    assert lines[-1].startswith(stage), report.stdout
    done = TASKS - len(state.get('missing', []))
    assert f'tasks with a result            {done} of {TASKS}' in lines
    assert f"pilot check passed             {'yes' if state.get('passed', True) else 'no'}" in lines
    # The report reads only.
    assert sorted(p.as_posix() for p in launcher.root.rglob('*')) == before


def test_collection_stops_at_a_failed_pilot_check(launcher):
    launcher.interrupted()
    (launcher.state / 'fail_check').touch()
    assert launcher.script('collect.sh').returncode != 0
    assert launcher.calls(*EXECUTED) == [('pilot-check', None, RESULTS)]
    assert not (launcher.output / 'COMPLETE.json').exists()
    assert not list(launcher.root.glob(f'{RESULTS}*.tar.gz*'))
    assert not launcher.collected()
