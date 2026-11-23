from pathlib import Path
import concurrent.futures,json,traceback,time
from run_controls import run,SEEDS,SCENARIOS,ROOT

def task(pair):
 seed,scenario=pair
 results=[]
 for mode in ['live','fixed','shuffled','zero']:results.append(run(seed,scenario,mode))
 return {'seed':seed,'scenario':scenario,'aris':{r['control']:r['ari'] for r in results}}
if __name__=='__main__':
 start=time.time();done=[];errors=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  pending={pool.submit(task,(s,c)):(s,c) for s in SEEDS for c in SCENARIOS}
  for future in concurrent.futures.as_completed(pending):
   try:done.append(future.result())
   except Exception:errors.append({'task':pending[future],'error':traceback.format_exc()})
   (ROOT/'analysis/rollout_progress.json').write_text(json.dumps({'complete_pairs':len(done),'errors':errors,'seconds':time.time()-start},indent=2))
   print('PROGRESS',len(done),len(errors),round(time.time()-start),flush=True)
 (ROOT/'analysis/rollout_completion.json').write_text(json.dumps({'results':done,'errors':errors,'seconds':time.time()-start},indent=2))
 if errors:raise RuntimeError(errors)
