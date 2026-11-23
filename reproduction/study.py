"""Focused three-mode weight sensitivity; distribution derived from source 83c91ce."""
from __future__ import annotations
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source'
BASE = '83c91ce8592c1d40234e5a6bb2a8ce342412f383'
MODES = ['static', 'no_context', 'agribrain']
SEEDS = [42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515]
SCENARIOS = ['heatwave','overproduction','cyber_outage','adaptive_pricing','baseline']
GROUPS = ['base_policy','mcp_prior','retrieval_prior','w_c','w_l','w_r','w_p','eta','eta_rho']
METRICS = ['ari','waste','rle','slca','carbon','equity']
TRACES = ['ari_trace','waste_trace','rho_trace','action_trace','prob_trace','carbon_trace','slca_trace']
DEFAULT_SOCIAL = [0.30,0.20,0.25,0.25]
ENV_TRACES = ['hours','temp_outcome_environmental_trace','rh_outcome_environmental_trace',
              'rho_outcome_environmental_trace','inventory_outcome_environmental_trace',
              'demand_outcome_environmental_trace','transport_multiplier_outcome_environmental_trace']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def effective(factors):
    if set(factors)!=set(GROUPS): raise ValueError('Incorrect factor set')
    if any(not 0.8 <= float(x) <= 1.2 for x in factors.values()): raise ValueError('Factors out of bounds')
    weights=[v*factors[k] for v,k in zip(DEFAULT_SOCIAL,['w_c','w_l','w_r','w_p'])]
    total=sum(weights)
    weights=[v/total for v in weights]
    return {**dict(zip(['w_c','w_l','w_r','w_p'],weights)),
            'eta':0.5*factors['eta'],'eta_rho':0.5*factors['eta_rho']}

def design():
    settings=[{'id':'nominal','factors':dict.fromkeys(GROUPS,1.0)}]
    for group in GROUPS:
        for factor,label in [(0.8,'minus20'),(1.2,'plus20')]:
            values=dict.fromkeys(GROUPS,1.0);values[group]=factor
            settings.append({'id':f'{group}_{label}','factors':values})
    # Fixed, outcome-independent opposing profiles. Not a global uncertainty distribution.
    signs=[1,-1,1,-1,1,-1,1,-1,1]
    for amplitude in [0.10,0.20]:
        for direction,label in [(1,'A'),(-1,'B')]:
            values={k:round(1+amplitude*direction*s,6) for k,s in zip(GROUPS,signs)}
            settings.append({'id':f'joint{round(amplitude*100)}_{label}','factors':values})
    assert len(settings)==23
    for setting in settings: setting['family']='weights'
    for setting in settings: setting['effective_policy']=effective(setting['factors'])
    return settings

def tasks():
    return [{'index':i,'setting':setting['id'],'seed':seed,'scenario':scenario}
            for i,(setting,seed,scenario) in enumerate(
                (s,n,c) for s in design() for n in SEEDS for c in SCENARIOS)]

def verify_source():
    manifest=load(HERE/'SOURCE_MANIFEST.json')
    if manifest['commit']!=BASE: raise ValueError('Wrong scientific source')
    for name,digest in manifest['files'].items():
        if sha(SOURCE/name)!=digest: raise ValueError(f'Changed scientific source: {name}')
    return canonical(manifest)

def harness_hash():
    return canonical({name:sha(HERE/name) for name in ['study.py','analyze.py','PROTOCOL.md','primary_reference.json']})

def configure(development_runtime=None):
    for p in [SOURCE,SOURCE/'agribrain/backend',SOURCE/'hpc']: sys.path.insert(0,str(p))
    if development_runtime: sys.path.insert(0,str(Path(development_runtime).resolve()))
    from hpc.validate_publication_env import EXPECTED,MUST_BE_UNSET
    for key in MUST_BE_UNSET: os.environ.pop(key,None)
    os.environ.update(EXPECTED)
    os.environ['AGRIBRAIN_GIT_COMMIT']=BASE
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    return development_runtime is not None

def runtime_receipt(development):
    from hpc.capture_publication_environment import (_installed_distribution_pairs,
        _locked_versions,_core_identity,_validate_distribution_set)
    pairs=_installed_distribution_pairs()
    receipt={'python':sys.version,'platform':platform.platform(),'development_only':development,
             'distributions':sorted(f'{n}=={v}' for n,v in pairs)}
    if not development:
        if sys.version_info[:2]!=(3,11): raise ValueError('HPC study requires Python 3.11')
        if sys.prefix==sys.base_prefix: raise ValueError('HPC study requires an isolated venv')
        report,errors=_validate_distribution_set(pairs,_locked_versions(),_core_identity())
        if errors: raise ValueError('; '.join(errors))
        receipt['lock_validation']=report
    return receipt

def prepare(output,development):
    output=Path(output).resolve()
    if output==HERE or HERE in output.parents: raise ValueError('Results must be outside the package')
    source_hash=verify_source();receipt=runtime_receipt(development)
    output.mkdir(parents=True,exist_ok=False)
    save(output/'settings.json',design());save(output/'tasks.json',tasks())
    plan={'run_tag':output.name,'source_commit':BASE,'source_manifest_sha256':source_hash,
          'harness_sha256':harness_hash(),'development_only':development,
          'settings_sha256':sha(output/'settings.json'),'tasks_sha256':sha(output/'tasks.json'),
          'created_utc':datetime.now(timezone.utc).isoformat(),'modes':MODES,
          'counts':{'settings':23,'tasks':2300,'episodes':20700,'adaptation':13800,
                    'evaluations':6900,'evaluation_routing_choices':1987200}}
    save(output/'plan.json',plan);save(output/'setup_environment.json',receipt)
    print(json.dumps(plan['counts']))

def read_plan(output,development):
    output=Path(output).resolve();plan=load(output/'plan.json')
    if plan['development_only']!=development: raise ValueError('Cannot mix development and HPC evidence')
    if plan['source_manifest_sha256']!=verify_source() or plan['harness_sha256']!=harness_hash():
        raise ValueError('Source or harness changed after preparation')
    for filename,key in [('settings.json','settings_sha256'),('tasks.json','tasks_sha256')]:
        if sha(output/filename)!=plan[key]: raise ValueError('Changed design/tasks')
    if load(output/'settings.json')!=design() or load(output/'tasks.json')!=tasks():
        raise ValueError('Unrecognised design/tasks')
    return plan

@contextmanager
def overridden(gr,setting,scenario,output):
    import numpy as np
    from src.models import action_selection
    from src.models.policy import Policy
    from pirag import context_to_logits as context
    old=(gr.Policy,gr.SCENARIOS,gr.RESULTS_DIR,action_selection.THETA.copy(),context.THETA_CONTEXT.copy())
    f=setting['factors'];kwargs=effective(f)
    modified=old[4].copy()
    modified[:,context.MCP_FEATURE_INDICES]*=f['mcp_prior']
    modified[:,context.PIR_FEATURE_INDICES]*=f['retrieval_prior']
    try:
        gr.Policy=lambda:Policy(**kwargs)
        gr.SCENARIOS=[scenario];gr.RESULTS_DIR=output/'auxiliary'
        action_selection.THETA=old[3]*f['base_policy'];context.THETA_CONTEXT=modified
        assert np.array_equal(np.sign(modified),np.sign(old[4]))
        yield {'policy':kwargs,'initial_context_matrix':modified.tolist(),
               'base_policy_matrix_before_seed_perturbation':action_selection.THETA.tolist()}
    finally:
        gr.Policy,gr.SCENARIOS,gr.RESULTS_DIR,action_selection.THETA,context.THETA_CONTEXT=old

def check_results(results):
    import numpy as np
    if set(results)!=set(MODES): raise ValueError('Mode set differs')
    if len({r['latent_environment_sha256'] for r in results.values()})!=1: raise ValueError('Unpaired environments')
    for mode,r in results.items():
        if r['episode_index']!=3 or r['learning_enabled']: raise ValueError('Evaluation not frozen')
        if mode!='static' and not r['learner_freeze_summary']['learners_frozen']: raise ValueError('Learner not frozen')
        if not all(np.isfinite(r[k]) for k in METRICS): raise ValueError('Nonfinite metric')
        if len(r['action_trace'])!=288: raise ValueError('Incorrect number of decisions')
        if abs(np.mean(r['ari_trace'])-r['ari'])>1e-12: raise ValueError('ARI/trace disagreement')
        probs=np.array(r['prob_trace'])
        if probs.shape!=(288,3) or not np.isfinite(probs).all() or np.any(probs<0) or not np.allclose(probs.sum(1),1,atol=1e-10,rtol=0):
            raise ValueError('Invalid routing probabilities')
    if set(results['static']['action_trace'])!={0}: raise ValueError('Static not fixed cold chain')

def check_ledgers(directory,scenario):
    from src.models.mode_capabilities import capabilities_for
    if capabilities_for('no_context').peer_messages: raise ValueError('Wrong No-Context')
    for mode in MODES:
        path=directory/f'{mode}__{scenario}.jsonl'
        with path.open(encoding='utf-8') as f: next(f);rows=[json.loads(line) for line in f if line.strip()]
        if len(rows)!=288: raise ValueError('Incomplete ledger')
        for row in rows:
            if mode=='no_context':
                if row['mcp_tool_call_count_step'] or row['governance_override'] or any(abs(v)>1e-12 for v in row['context_modifier'] or []) or any(abs(v)>1e-12 for v in row['peer_message_bias'] or []):
                    raise ValueError('Context leaked into comparator')
                for role in ['primary','cooperative']:
                    if row.get('step_channel_evidence',{}).get(role,{}).get('retrieval',{}).get('attempted',False): raise ValueError('Retrieval leaked')
            if mode=='agribrain':
                kind=row.get('step_channel_evidence',{}).get('primary',{}).get('retrieval',{}).get('retrieval_kind')
                if kind not in [None,'standard']: raise ValueError('Incorrect retrieval configuration')

def reference_check(results,task):
    import numpy as np
    reference=load(HERE/'primary_reference.json')['values'].get(str(task['seed']),{}).get(task['scenario'])
    if task['setting']!='nominal' or reference is None: return {'available':False}
    errors={}
    for mode in ['no_context','agribrain']:
        ref=reference[mode];r=results[mode]
        if r['action_trace']!=ref['action_trace']: raise ValueError('Nominal routing differs from primary run')
        keys=METRICS+['ari_trace','prob_trace']+ENV_TRACES
        errors[mode]=max(float(np.max(np.abs(np.asarray(r[k])-np.asarray(ref[k])))) for k in keys)
        if errors[mode]>1e-10: raise ValueError(f'Nominal scientific output differs: {errors}')
    return {'available':True,'max_abs_error':errors,'all_576_adaptive_actions_equal':True,
            'environment_hash_equal':{m:results[m]['latent_environment_sha256']==reference[m]['latent_environment_sha256'] for m in ['no_context','agribrain']},
            'environment_check':'Full numerical trajectories checked at 1e-10; exact hashes may differ across numerical platforms.'}

def run_task(output,index,development):
    from mvp.simulation import generate_results as gr
    from mvp.simulation.benchmarks.episode_archive import to_json_native
    from hpc.validate_complete_episode_evidence import validate_complete_evidence
    output=Path(output).resolve();plan=read_plan(output,development)
    all_tasks=tasks()
    if not 0<=index<len(all_tasks): raise ValueError('Task index outside design')
    task=all_tasks[index];setting=next(s for s in design() if s['id']==task['setting'])
    dest=output/'tasks'/f'{index:04d}'
    if (dest/'result.json').exists():
        verify_result(output,index);print(f'Already complete and verified: {index}');return
    dest.mkdir(parents=True,exist_ok=True)
    attempt=dest/f"attempt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{os.getpid()}"
    attempt.mkdir();receipt=runtime_receipt(development)
    save(attempt/'environment.json',receipt)
    os.environ['RUN_TAG']=plan['run_tag'];os.environ['AGRIBRAIN_SOURCE_TREE_SHA256']=plan['source_manifest_sha256']
    started=time.monotonic()
    with (attempt/'simulation.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
        with overridden(gr,setting,task['scenario'],attempt) as actual:
            with gr.decision_ledger_scope(attempt/'decision_ledgers',reset=True):
                data=gr.run_all(task['seed'],modes=MODES)
        results=data['results'][task['scenario']]
        check_results(results);check_ledgers(attempt/'decision_ledgers',task['scenario'])
        counts=validate_complete_evidence(attempt/'decision_ledgers',expected_groups=3,expected_episodes=9,
            expected_adaptation_ledgers=6,expected_final_ledgers=3,manifest_path=attempt/'evidence_manifest.json')['counts']
        ref=reference_check(results,task)
        # Verify the altered prior actually reached the learner, not just a dormant constant.
        import numpy as np
        initial=results['agribrain']['learner_summary']['initial_theta']
        if not np.allclose(initial,actual['initial_context_matrix'],atol=1e-12,rtol=0):
            raise ValueError('Context prior override did not reach learner')
    retained={m:to_json_native({k:v for k,v in r.items() if not k.startswith('_')}) for m,r in results.items()}
    manifest={p.relative_to(attempt).as_posix():sha(p) for p in attempt.rglob('*') if p.is_file()}
    read_plan(output,development)
    record={'task':task,'setting':setting,'actual_parameters':actual,'plan_sha256':sha(output/'plan.json'),
            'source_commit':BASE,'attempt_directory':attempt.name,'files':manifest,'results':retained,
            'evidence_counts':counts,'primary_reproduction':ref,'development_only':development,
            'seconds':time.monotonic()-started,'bytes':sum(p.stat().st_size for p in attempt.rglob('*') if p.is_file())}
    save(dest/'result.json',record)
    print(f"COMPLETE {index} {task['setting']} {task['scenario']}: {record['seconds']:.1f}s; ARI "+
          ', '.join(f'{m}={results[m]["ari"]:.8f}' for m in MODES))

def verify_result(output,index):
    output=Path(output);dest=output/'tasks'/f'{index:04d}';r=load(dest/'result.json')
    if r['task']!=tasks()[index] or r['plan_sha256']!=sha(output/'plan.json'): raise ValueError('Task/plan mismatch')
    if r['setting']!=next(s for s in design() if s['id']==r['task']['setting']): raise ValueError('Setting mismatch')
    if Path(r['attempt_directory']).name!=r['attempt_directory']: raise ValueError('Unsafe attempt path')
    attempt=dest/r['attempt_directory']
    for name,digest in r['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('Unsafe evidence path')
        if sha(attempt/name)!=digest: raise ValueError(f'Changed evidence {index}: {name}')
    # Endpoint data live outside raw-file hashes: compare them to the immutable episode archives.
    import gzip
    manifest=load(attempt/'evidence_manifest.json')
    for artifact in manifest['artifacts']:
        ident=artifact['identity']
        if ident['mode'] not in MODES or ident['benchmark_seed']!=r['task']['seed'] or ident['scenario']!=r['task']['scenario']:
            raise ValueError('Evidence identity mismatch')
        if ident['episode_index']==3:
            original=json.loads(gzip.decompress((attempt/'decision_ledgers'/artifact['archive']).read_bytes()))['episode_result']
            for key in METRICS+TRACES+['latent_environment_sha256']:
                if original[key]!=r['results'][ident['mode']][key]: raise ValueError('Endpoint/archive mismatch')
    check_results(r['results']);return r

def check_pilot(output):
    # Nominal in three scenarios plus distinct active perturbations; all are study cells.
    indices=[0,2,3,100,300,500,700,1500,1900]
    for index in indices: verify_result(output,index)
    # A deterministic independently executed nominal repeat is required before the bulk array.
    repeat=Path(output)/'reproducibility_repeat'
    a=verify_result(output,0);b=verify_result(repeat,0)
    compare_records(a,b)
    check_influence(output,indices)
    save(Path(output)/'PILOT_PASSED.json',{'indices':indices,'repeat_verified':True,'plan_sha256':sha(Path(output)/'plan.json')})
    print('Pilot passed: primary reproduction, deterministic repeat, active perturbations and complete evidence.')

def compare_records(a,b):
    import numpy as np
    max_error=0.0
    for mode in MODES:
        for key in METRICS+TRACES:
            error=float(np.max(np.abs(np.asarray(a['results'][mode][key])-np.asarray(b['results'][mode][key]))))
            max_error=max(max_error,error)
    if max_error>1e-12: raise ValueError(f'Repeat differs by {max_error}')
    return max_error

def check_influence(output,indices):
    import numpy as np
    nominal=verify_result(output,0)['results'];rows=[]
    for index in indices:
        if index<100: continue
        r=verify_result(output,index);result=r['results']
        if r['task']['seed']!=42 or r['task']['scenario']!='heatwave': continue
        delta=float(np.max(np.abs(np.asarray(result['agribrain']['prob_trace'])-np.asarray(nominal['agribrain']['prob_trace']))))
        changed=sum(x!=y for x,y in zip(result['agribrain']['action_trace'],nominal['agribrain']['action_trace']))
        # Observed nonzero effect is evidence an override is live; zero can still be a valid result.
        rows.append({'setting':r['task']['setting'],'max_probability_change':delta,'changed_actions':changed,
                     'ari_change':result['agribrain']['ari']-nominal['agribrain']['ari']})
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-runtime',type=Path)
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ['prepare','task','verify','pilot-check','missing']:
        parser=sub.add_parser(cmd);parser.add_argument('--output',required=True,type=Path)
        if cmd in ['task','verify']: parser.add_argument('--index',required=True,type=int)
    args=p.parse_args();development=configure(args.development_runtime)
    if args.command=='prepare': prepare(args.output,development)
    elif args.command=='task': run_task(args.output,args.index,development)
    elif args.command=='verify': read_plan(args.output,development);verify_result(args.output,args.index);print('Verified')
    elif args.command=='pilot-check': read_plan(args.output,development);check_pilot(args.output)
    else:
        read_plan(args.output,development);missing=[]
        for task in tasks():
            try: verify_result(args.output,task['index'])
            except (OSError,ValueError,KeyError): missing.append(task['index'])
        print(','.join(map(str,missing)))

if __name__=='__main__': main()
