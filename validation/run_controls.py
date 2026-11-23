from pathlib import Path
import sys,os,json,gzip,time,copy,hashlib,contextlib
ROOT=Path(__import__('os').environ['AGRIBRAIN_VALIDATION_OUTPUT']).resolve()
(ROOT/'analysis').mkdir(parents=True,exist_ok=True)
SOURCE=Path(__file__).resolve().parents[1]/'reproduction/source'
EVIDENCE=Path(__import__('os').environ['AGRIBRAIN_PRIMARY_EVIDENCE']).resolve()
sys.dont_write_bytecode=True
sys.path[:0]=[str(SOURCE)]
from hpc.validate_publication_env import EXPECTED
os.environ.update(EXPECTED)
os.environ['FULL_EVIDENCE_CAPTURE']='0'
os.environ['RUN_TAG']='frozen_context_controls_20260922'
os.environ['AGRIBRAIN_GIT_COMMIT']='83c91ce8592c1d40234e5a6bb2a8ce342412f383'
from mvp.simulation import generate_results as gr
import numpy as np,pandas as pd
from src.agents.coordinator import AgentCoordinator
import src.models.action_selection as actionmod
BASE_THETA=actionmod.THETA.copy()
ORIGINAL=AgentCoordinator._compute_step_context
SCENARIOS=['heatwave','overproduction','cyber_outage','adaptive_pricing','baseline']
SEEDS=[42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515]
def loadgz(p):
 with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def rows(p):
 op=gzip.open if str(p).endswith('.gz') else open
 with op(p,'rt',encoding='utf8') as f:return [r for line in f if not (r:=json.loads(line)).get('_header')]
def run(seed,scenario,control):
 out=ROOT/'rollouts'/f'{seed}_{scenario}_{control}'
 resultpath=out/'result.json'
 if resultpath.exists():return json.loads(resultpath.read_text())
 out.mkdir(parents=True,exist_ok=True)
 ledger=EVIDENCE/f'seed_{seed}/decision_ledgers'
 archive=loadgz(ledger/f'complete_episode_evidence/agribrain__{scenario}/episode_3.json.gz')
 frame=archive['input_frame'];df=pd.DataFrame(frame['rows'],columns=frame['columns'])
 for col,dt in zip(frame['columns'],frame['dtypes']):
  df[col]=pd.to_datetime(df[col]) if dt.startswith('datetime') else df[col].astype(dt)
 df.index=frame['index'];df.index.name=frame['index_name'];df.attrs.update(frame['attrs'])
 originalrows=rows(ledger/f'agribrain__{scenario}.jsonl')
 adap=[]
 for ep in range(3):
  adap.extend(r['context_modifier'] for r in rows(ledger/f'adaptation_episode_ledgers/agribrain__{scenario}/episode_{ep}.jsonl.gz'))
 fixed=np.mean(np.array(adap,float),axis=0)
 permutation=np.random.default_rng(np.random.SeedSequence([20260922,seed,SCENARIOS.index(scenario)])).permutation(288)
 shuffled=np.array([r['context_modifier'] for r in originalrows])[permutation]
 def intervention(self,*args,**kwargs):
  live=ORIGINAL(self,*args,**kwargs)
  i=getattr(self,'_validation_control_index',0);self._validation_control_index=i+1
  applied={'fixed':fixed,'zero':np.zeros(3)}.get(control,shuffled[i] if control=='shuffled' else live)
  if control!='live':
   self._step_context_modifier=np.asarray(applied,float).copy()
   # The replacement is an experimental residual, not a feature explanation.
   self._step_context_feature_contributions=np.zeros((3,5))
   self._step_context_nonfeature_residual=np.asarray(applied,float).copy()
   if self._context_log:
    self._context_log[-1]['context_modifier']=np.asarray(applied).tolist()
    self._context_log[-1]['validation_control']=control
  return np.asarray(applied,float)
 AgentCoordinator._compute_step_context=intervention
 actionmod.THETA=gr.policy_theta_for_seed(BASE_THETA,seed)
 policy=gr.Policy()
 envseed=gr._stream_seed(seed,scenario,3,'environment');polseed=gr._stream_seed(seed,scenario,3,'policy')
 start=time.monotonic()
 with gr.decision_ledger_scope(out/'decision_ledgers',reset=True),open(out/'execution.log','w',encoding='utf8') as log,contextlib.redirect_stdout(log):
  result=gr.run_episode(df,'agribrain',policy,np.random.default_rng(polseed),scenario,
   stoch=gr.make_stochastic_layer(np.random.default_rng(envseed),stream_seed=envseed),seed=seed,benchmark_seed=seed,episode_index=3,
   learner_state_cache={'agribrain':copy.deepcopy(archive['learner_state']['before'])},learning_enabled=False)
 AgentCoordinator._compute_step_context=ORIGINAL;actionmod.THETA=BASE_THETA.copy()
 kept=rows(out/f'decision_ledgers/agribrain__{scenario}.jsonl')
 err=max(abs(r['ari']-o['ari']) for r,o in zip(kept,originalrows)) if control=='live' else None
 actionequal=result['action_trace']==archive['episode_result']['action_trace']
 if control=='live':
  assert actionequal,(seed,scenario,'actions differ')
  assert err<1e-10,(seed,scenario,err)
 assert result['learner_freeze_summary']['learners_frozen']
 payload={'seed':seed,'scenario':scenario,'control':control,'ari':result['ari'],'ari_trace':result['ari_trace'],'actions':result['action_trace'],
 'fixed_modifier':fixed.tolist(),'permutation':permutation.tolist() if control=='shuffled' else None,'seconds':time.monotonic()-start,
 'reference_actions_equal':actionequal,'reference_max_step_ari_error':err,'learners_frozen':True,
 'design':'Frozen original AGRI-BRAIN checkpoint, unchanged peer/guard configuration; injected modifier intervention only. Fixed vector fitted from all three adaptation episodes; shuffled vector permutes original evaluation adjustments within seed/scenario. No retraining.'}
 resultpath.write_text(json.dumps(payload,indent=2),encoding='utf8')
 lp=out/f'decision_ledgers/agribrain__{scenario}.jsonl'
 with open(lp,'rb') as a,gzip.open(str(lp)+'.gz','wb',compresslevel=5) as b:b.write(a.read())
 lp.unlink()
 print(seed,scenario,control,payload['ari'],round(payload['seconds'],2),flush=True)
 return payload
if __name__=='__main__':
 run(int(sys.argv[1]),sys.argv[2],sys.argv[3] if len(sys.argv)>3 else 'live')
