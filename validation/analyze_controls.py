from pathlib import Path
import json,csv,sys
import numpy as np
from scipy.stats import bootstrap,wilcoxon
ROOT=Path(__import__('os').environ['AGRIBRAIN_VALIDATION_OUTPUT']).resolve()
(ROOT/'analysis').mkdir(parents=True,exist_ok=True)
SC=['heatwave','overproduction','cyber_outage','adaptive_pricing','baseline']
rows=[json.loads(p.read_text()) for p in (ROOT/'rollouts').glob('*/result.json')]
assert len(rows)==400,len(rows)
seeds=sorted({r['seed'] for r in rows});assert len(seeds)==20
idx={(r['seed'],r['scenario'],r['control']):r for r in rows}
def ci(a):
 a=np.asarray(a,float)
 if np.ptp(a)<1e-15:return [float(a.mean()),float(a.mean())]
 c=bootstrap((a,),np.mean,n_resamples=10000,method='BCa',confidence_level=.95,rng=np.random.default_rng(20260922)).confidence_interval
 return [float(c.low),float(c.high)]
contrasts=[];means=[]
for sc in SC:
 for ctrl in ['live','fixed','shuffled','zero']:
  x=[idx[s,sc,ctrl]['ari'] for s in seeds];lo,hi=ci(x);means.append({'scenario':sc,'control':ctrl,'mean':float(np.mean(x)),'low':lo,'high':hi})
 for ctrl in ['fixed','shuffled','zero']:
  a=np.array([idx[s,sc,'live']['ari']-idx[s,sc,ctrl]['ari'] for s in seeds]);lo,hi=ci(a)
  contrasts.append({'scenario':sc,'contrast':'live_minus_'+ctrl,'mean':float(a.mean()),'low':lo,'high':hi,'positive_seeds':int(sum(a>0)),'p_two_sided':float(wilcoxon(a,alternative='two-sided',method='exact').pvalue)})
order=np.argsort([r['p_two_sided'] for r in contrasts]);previous=0
for rank,k in enumerate(order):previous=max(previous,min(1,contrasts[k]['p_two_sided']*(len(contrasts)-rank)));contrasts[k]['p_holm_15']=previous
aggregate={}
for ctrl in ['fixed','shuffled','zero']:
 a=[np.mean([idx[s,sc,'live']['ari']-idx[s,sc,ctrl]['ari'] for sc in SC]) for s in seeds];aggregate[ctrl]={'mean':float(np.mean(a)),'ci95':ci(a)}
res={'scope':'Exploratory frozen-checkpoint rollouts, not separately adapted alternative modes','seed_count':20,'scenarios':SC,'episodes':400,'steps':115200,'reference_reproductions':100,'intervention_episodes':300,'reference_action_matches':all(idx[s,sc,'live']['reference_actions_equal'] for s in seeds for sc in SC),'max_reference_step_ari_error':max(idx[s,sc,'live']['reference_max_step_ari_error'] for s in seeds for sc in SC),'means':means,'contrasts':contrasts,'aggregate':aggregate,'uncertainty':'95% BCa; 10,000 resamples of independent seed-level paired differences; exploratory two-sided Wilcoxon with Holm across 15 scenario/contrast tests; no equivalence claim'}
(ROOT/'analysis/control_summary.json').write_text(json.dumps(res,indent=2))
with (ROOT/'analysis/control_endpoints.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['seed','scenario','live','fixed','shuffled','zero'])
 for s in seeds:
  for sc in SC:w.writerow([s,sc]+[idx[s,sc,c]['ari'] for c in ['live','fixed','shuffled','zero']])
print(json.dumps(res,indent=2))
