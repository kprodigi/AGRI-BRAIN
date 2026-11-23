from pathlib import Path
import json,gzip,sys
import numpy as np
ROOT=Path(__import__('os').environ['AGRIBRAIN_VALIDATION_OUTPUT']).resolve()
(ROOT/'analysis').mkdir(parents=True,exist_ok=True)
E=Path(__import__('os').environ['AGRIBRAIN_PRIMARY_EVIDENCE']).resolve()
n=0;maxlogit=0.;maxprob=0.;maxresid=0.;override={c:0 for c in ['live','fixed','shuffled','zero']}
with gzip.open(ROOT/'analysis/frozen_control_decisions.csv.gz','wt',encoding='utf8') as export:
 import csv
 w=csv.writer(export);w.writerow(['seed','scenario','control','step','action','ari','modifier_cc','modifier_lr','modifier_recovery','override'])
 for p in sorted((ROOT/'rollouts').glob('*/result.json')):
  info=json.loads(p.read_text());ctrl=info['control'];sc=info['scenario'];seed=info['seed']
  nominal=E/f'seed_{seed}/decision_ledgers/agribrain__{sc}.jsonl'
  with nominal.open() as f:next(f);orig=[json.loads(l) for l in f]
  lp=p.parent/f'decision_ledgers/agribrain__{sc}.jsonl';op=open
  if not lp.exists():lp=Path(str(lp)+'.gz');op=gzip.open
  with op(lp,'rt',encoding='utf8') as f:next(f);rows=[json.loads(l) for l in f]
  assert len(rows)==288
  for i,r in enumerate(rows):
   m=np.array(r['context_modifier']);b=np.array(r['base_logits']);z=np.array(r['post_context_logits_pre_override']);maxlogit=max(maxlogit,float(np.max(abs(z-b-m))))
   if ctrl=='fixed':np.testing.assert_allclose(m,info['fixed_modifier'],atol=1e-12,rtol=0)
   if ctrl=='zero':assert np.max(abs(m))==0
   if ctrl=='shuffled':np.testing.assert_allclose(m,orig[info['permutation'][i]]['context_modifier'],atol=1e-12,rtol=0)
   if ctrl!='live':
    allocation=np.array(r['context_feature_contributions']).sum(axis=1)+np.array(r['context_nonfeature_residual']);maxresid=max(maxresid,float(np.max(abs(allocation-m))))
   z=z/r['policy_temperature'];probs=np.exp(z-z.max());probs/=sum(probs);maxprob=max(maxprob,float(np.max(abs(probs-np.array(r['policy_probs_pre_override'])))))
   np.testing.assert_allclose(r['effective_context_theta'],orig[i]['effective_context_theta'],atol=1e-12,rtol=0)
   override[ctrl]+=int(r['governance_override']);w.writerow([seed,sc,ctrl,i,r['action_idx'],r['ari'],*m,int(r['governance_override'])]);n+=1
assert n==115200 and maxlogit<1e-12 and maxprob<1e-12 and maxresid<1e-12
receipt={'decisions_checked':n,'max_logit_equation_error':maxlogit,'max_probability_error':maxprob,'max_intervention_allocation_error':maxresid,'control_vectors_verified':True,'context_weights_match_original':True,'overrides':override}
(ROOT/'analysis/control_integrity.json').write_text(json.dumps(receipt,indent=2));print(receipt)
