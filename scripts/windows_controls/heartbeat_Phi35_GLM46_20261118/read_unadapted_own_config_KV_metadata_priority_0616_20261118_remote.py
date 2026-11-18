import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
state=json.loads((D/'state.json').read_bytes());result=[]
for mid,row in state['models'].items():
 if row['status']!='needs_adaptation':continue
 roots=row.get('weight_paths') or []
 item=dict(model_id=mid,task=row['task'],status=row['status'],current_score=row.get('score'),weight_roots=roots)
 if not roots:result.append(item);continue
 root=Path(roots[0]);f=root/'config.json'
 try:
  if not f.exists():item['own_config_currently_unavailable']=True;result.append(item);continue
  cfg=json.loads(f.read_bytes());candidates=[cfg]+[cfg[k] for k in ['text_config','llm_config','language_config','language_model_config'] if isinstance(cfg.get(k),dict)]
  text=next((x for x in candidates[1:] if x.get('num_hidden_layers') is not None),cfg)
  layers=text.get('num_hidden_layers',text.get('n_layer'));hidden=text.get('hidden_size',text.get('n_embd'));heads=text.get('num_attention_heads',text.get('n_head'));kv=text.get('num_key_value_heads',heads)
  dim=text.get('head_dim',int(hidden/heads) if hidden and heads else None)
  item.update(architecture=cfg.get('architectures'),model_type=cfg.get('model_type'),layers=layers,hidden=hidden,query_heads=heads,KV_heads=kv,head_dim=dim,stored_config_dtype=cfg.get('dtype',cfg.get('torch_dtype')),own_root=str(root),own_native_KV_BF16_bytes_per_token_if_config_applies=int(layers*2*kv*dim*2) if all(isinstance(x,(int,float)) for x in [layers,kv,dim]) else None)
 except (OSError,ValueError,TypeError) as exc:item['metadata_read_error']=repr(exc)
 result.append(item)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),only_own_unadapted_config_metadata=True,no_weight_array_or_old180_report_or_tensor_read=True,no_model_dispatch_or_score_claim=True,candidates=result),ensure_ascii=False))
