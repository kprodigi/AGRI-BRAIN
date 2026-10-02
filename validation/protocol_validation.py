"""Supplementary interface tests. Source scientific implementation is unchanged."""
import sys,json,os,subprocess,math,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__import__('os').environ['AGRIBRAIN_VALIDATION_OUTPUT']).resolve()
(ROOT/'analysis').mkdir(parents=True,exist_ok=True)
SOURCE=Path(__file__).resolve().parents[1]/'reproduction/source'
sys.dont_write_bytecode=True
sys.path[:0]=[str(SOURCE/'agribrain/backend')]
os.environ['MCP_RATE_LIMITS']='disabled';os.environ['STRICT_VALIDATION']='1'
from pirag.mcp.registry import ToolRegistry,ToolSpec
from pirag.mcp.protocol import MCPServer,MCPMessage
from pirag.mcp.transport import InProcessTransport,StdioTransport
from pirag.mcp.tools.compliance import check_compliance

def alternative_envelope(temperature,humidity,product_type='spinach'):
 # Separately implemented test provider, same declared spinach thresholds.
 if product_type!='spinach':raise ValueError('Test provider covers spinach only')
 severity=[]
 if temperature>8:severity.append({'severity': 'critical' if temperature>11 else 'warning'})
 if not 85<=humidity<=95:severity.append({'severity':'warning'})
 return {'compliant':not severity,'violations':severity,'provider':'independent_test_provider'}

def normalized_provider(temperature_f,humidity_fraction,product_type='spinach'):
 if not all(math.isfinite(float(v)) for v in [temperature_f,humidity_fraction]):raise ValueError('Non-finite telemetry')
 return alternative_envelope((temperature_f-32)*5/9,humidity_fraction*100,product_type)

def server():
 reg=ToolRegistry()
 for name,fn,schema in [('check_compliance',check_compliance,{'temperature':'number','humidity':'number','product_type':'string'}),('alternate_envelope',alternative_envelope,{'temperature':'number','humidity':'number'}),('normalized_envelope',normalized_provider,{'temperature_f':'number','humidity_fraction':'number'})]:
  reg.register(ToolSpec(name=name,description='Validation tool',capabilities=['temperature'],fn=fn,schema=schema))
 return MCPServer(registry=reg)

def msg(name,args,idx=1):return {'jsonrpc':'2.0','id':idx,'method':'tools/call','params':{'name':name,'arguments':args}}
def decode(res):
 if 'error' in res or res.get('result',{}).get('isError'):raise ValueError(res)
 return json.loads(res['result']['content'][0]['text'])

def serve():
 t=InProcessTransport(server())
 for line in sys.stdin:
  try:r=t.send(json.loads(line))
  except Exception as e:r={'error':{'message':str(e)}}
  print(json.dumps(r),flush=True)

if __name__=='__main__' and '--serve' in sys.argv:serve();sys.exit(0)

if __name__=='__main__':
 import numpy as np
 from pirag.context_to_logits import extract_context_features,compute_context_modifier
 from src.agents.coordinator import _compose_context_attribution
 EVIDENCE=Path(__import__('os').environ['AGRIBRAIN_PRIMARY_EVIDENCE']).resolve()
 transport=InProcessTransport(server())
 proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(ROOT/'analysis/protocol_child.log','w'),text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 pipe=StdioTransport(proc_stdin=proc.stdout,proc_stdout=proc.stdin)
 stats={k:{'cases':0,'feature_matches':0,'modifier_matches':0,'probability_matches':0,'sampled_action_matches':0,'max_feature_error':0.,'max_modifier_error':0.,'max_probability_error':0.} for k in ['json_roundtrip','separate_process','alternate_provider','unit_adapter']}
 failures=[];nroles={};scene={};sample=None
 def probability(z,temp):
  v=np.asarray(z)/temp;v-=max(v);v=np.exp(v);return v/v.sum()
 for seedpath in sorted(EVIDENCE.glob('seed_*')):
  for lp in sorted((seedpath/'decision_ledgers').glob('agribrain__*.jsonl')):
   with lp.open(encoding='utf8') as f:
    next(f)
    for line in f:
     row=json.loads(line);primary=row['step_channel_evidence']['primary'];m={x['tool_name']:x['result'] for x in primary['mcp']['effective_tool_results']}
     if 'check_compliance' not in m:continue
     reads=m['check_compliance']['readings'];args={'temperature':reads['temperature'],'humidity':reads['humidity'],'product_type':'spinach'}
     variants={'json_roundtrip':decode(transport.send(msg('check_compliance',args))),
      'separate_process':decode(pipe.send(msg('check_compliance',args))),
      'alternate_provider':decode(transport.send(msg('alternate_envelope',args))),
      'unit_adapter':decode(transport.send(msg('normalized_envelope',{'temperature_f':args['temperature']*9/5+32,'humidity_fraction':args['humidity']/100,'product_type':'spinach'})))}
     rc=primary['retrieval'];rag={k:rc[k] for k in ['guards_passed','top_doc_id','top_fused_score','guard_breakdown','retrieval_kind']}
     obs=SimpleNamespace(raw={},hour=row['hour']);theta=np.array(row['effective_context_theta'])
     originaltrace={};originalmod=compute_context_modifier(m,rag,obs,theta_override=theta,retrieval_kind='standard',trace_out=originaltrace)
     stored=np.asarray(row['context_integration']['primary']['final_modifier'])
     assert np.max(abs(stored-originalmod))<1e-10
     # Primary-provider replacement only; preserve cooperative and all base-policy terms.
     ci=row['context_integration'];scope=ci['composition']['scope']
     coop=np.array(ci['cooperative']['final_modifier']) if ci.get('cooperative') else None
     def compose(v):
      if scope=='primary_context':return v
      if scope=='cooperative_blend':return np.clip(.7*v+.3*coop,-1,1)
      if scope=='cooperative_veto':return np.asarray(row['context_modifier'])
      raise ValueError(scope)
     p0=probability(np.array(row['base_logits'])+compose(originalmod),row['policy_temperature'])
     assert np.max(abs(p0-np.array(row['policy_probs_pre_override'])))<1e-10
     for key,value in variants.items():
      modified=dict(m);modified['check_compliance']=value;tr={}
      mod=compute_context_modifier(modified,rag,obs,theta_override=theta,retrieval_kind='standard',trace_out=tr)
      pp=probability(np.array(row['base_logits'])+compose(mod),row['policy_temperature'])
      errors=[float(np.max(abs(tr['effective_psi']-originaltrace['effective_psi']))),float(np.max(abs(compose(mod)-compose(originalmod)))),float(np.max(abs(pp-p0)))]
      st=stats[key];st['cases']+=1
      for error,label in zip(errors,['feature','modifier','probability']):st[label+'_matches']+=int(error<=1e-12);st['max_'+label+'_error']=max(st['max_'+label+'_error'],error)
      st['sampled_action_matches']+=int(np.searchsorted(np.cumsum(pp),row['policy_categorical_uniform'])==np.searchsorted(np.cumsum(p0),row['policy_categorical_uniform']))
     scene[row['scenario']]=scene.get(row['scenario'],0)+1;nroles[row['role']]=nroles.get(row['role'],0)+1
     if sample is None and row['scenario']=='overproduction' and row['hour']>=24:sample=row
   print('CHECKED',seedpath.name,lp.name,stats['json_roundtrip']['cases'],flush=True)
 # Deliberate malformed requests, no hidden corrections to the source implementation.
 tests=[('unknown tool',msg('missing',{})),('missing argument',msg('check_compliance',{'humidity':90})),('wrong value type',msg('check_compliance',{'temperature':'bad','humidity':90})),('invalid version',{**msg('check_compliance',{'temperature':4,'humidity':90}),'jsonrpc':'1.0'}),('NaN temperature',msg('check_compliance',{'temperature':float('nan'),'humidity':90})),('infinite temperature',msg('check_compliance',{'temperature':float('inf'),'humidity':90}))]
 checks=[]
 for label,request in tests:
  for name,tt in [('in-process',transport),('separate-process',pipe)]:
   ans=tt.send(request);rejected=bool(ans.get('error') or ans.get('result',{}).get('isError'))
   checks.append({'test':label,'path':name,'rejected':rejected,'response':ans})
 # Mechanism sweep: one retained base state; severity thresholds change only the tool feature.
 r=sample;pr=r['step_channel_evidence']['primary'];m={x['tool_name']:x['result'] for x in pr['mcp']['effective_tool_results']};rc=pr['retrieval'];rag={k:rc[k] for k in ['guards_passed','top_doc_id','top_fused_score','guard_breakdown','retrieval_kind']};obs=SimpleNamespace(raw={},hour=r['hour']);sweep=[]
 for temp in [4.,7.9,8.,8.1,10.9,11.,11.1,14.,18.]:
  variants=[check_compliance(temp,90),decode(transport.send(msg('check_compliance',{'temperature':temp,'humidity':90}))),decode(pipe.send(msg('check_compliance',{'temperature':temp,'humidity':90}))),decode(transport.send(msg('normalized_envelope',{'temperature_f':temp*9/5+32,'humidity_fraction':.9})))]
  for path,val in zip(['direct','json_roundtrip','separate_process','unit_adapter'],variants):
   mm=dict(m);mm['check_compliance']=val;tr={};v=compute_context_modifier(mm,rag,obs,theta_override=np.asarray(r['effective_context_theta']),retrieval_kind='standard',trace_out=tr)
   ci=r['context_integration'];scope=ci['composition']['scope']
   applied=v if scope=='primary_context' else (np.clip(.7*v+.3*np.asarray(ci['cooperative']['final_modifier']),-1,1) if scope=='cooperative_blend' else np.asarray(r['context_modifier']))
   pp=probability(np.array(r['base_logits'])+applied,r['policy_temperature'])
   sweep.append({'temperature':temp,'path':path,'severity':float(tr['effective_psi'][0]),'modifier':v.tolist(),'probabilities':pp.tolist()})
 proc.stdin.close();proc.wait(timeout=20)
 result={'design':'Tested project MCP-style subset; two local transport paths and separately authored test provider; explicit unit adapter is a test fixture, not automatic production unit inference. Fixed-state probability agreement, not autonomous retraining or official-client conformance.','tolerance':1e-12,'equivalence':stats,'scenarios':scene,'roles':nroles,'fault_checks':checks,'sweep':sweep,'sweep_state':{k:r[k] for k in ['hour','scenario','role','base_logits','policy_temperature']}}
 (ROOT/'analysis/protocol_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 print(json.dumps(stats,indent=2))
