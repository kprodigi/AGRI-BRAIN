import json,gzip
from pathlib import Path
from collections import defaultdict
ROOT=Path(__import__('os').environ['AGRIBRAIN_VALIDATION_OUTPUT']).resolve()
(ROOT/'analysis').mkdir(parents=True,exist_ok=True)
EVIDENCE=Path(__import__('os').environ['AGRIBRAIN_PRIMARY_EVIDENCE']).resolve()
stats=defaultdict(lambda:{'decisions':0,'primary_tool_calls':0,'peer_consumed_decisions':0,'nonzero_peer_bias_decisions':0,'cooperative_overlay_decisions':0})
for path in EVIDENCE.glob('seed_*/decision_ledgers/agribrain__*.jsonl'):
 with path.open(encoding='utf8') as f:
  next(f)
  for line in f:
   r=json.loads(line);st=stats[r['role']];st['decisions']+=1;st['primary_tool_calls']+=len(r['primary_mcp_tools_invoked_step']);st['peer_consumed_decisions']+=int(r['step_channel_evidence']['peer']['consumed_count']>0);st['nonzero_peer_bias_decisions']+=int(max(abs(v) for v in r['peer_message_bias'])>1e-12);st['cooperative_overlay_decisions']+=int(r['step_channel_evidence']['cooperative']['active'])
(ROOT/'analysis/role_activity.json').write_text(json.dumps(dict(stats),indent=2));print(json.dumps(dict(stats),indent=2))
