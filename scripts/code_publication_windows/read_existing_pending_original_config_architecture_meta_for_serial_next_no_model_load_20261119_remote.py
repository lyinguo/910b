import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';state=json.loads((D/'state.json').read_bytes())
import yaml
files={p.stem:p for p in (R/'configs').glob('*/*.yaml')};rows=[]
for mid,e in state['models'].items():
 if e['status']!='needs_adaptation':continue
 p=files.get(mid.replace('/','--'));out=dict(model_id=mid,state_keys=list(e),state_metadata={k:e.get(k) for k in ['status','task','score','reason','config','weight_path']})
 if p:
  c=yaml.safe_load(p.read_bytes());m=c.get('model',{});kw=m.get('kwargs',{});out.update(original_config=str(p.relative_to(R)),config_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),original_model_adapter=m.get('adapter'),original_model_backend=m.get('backend'),original_model_path=m.get('model_path'),original_dtype=m.get('dtype',kw.get('dtype')),original_max_tokens=kw.get('max_new_tokens'),original_algorithm=c.get('algorithm'),original_dataset=c.get('data',{}).get('dataset_path'))
  paths=[v for k,v in {**m,**kw}.items() if k in ['model_path','weight_path','pretrained_model_name_or_path','path'] and isinstance(v,str)]
  out['own_path_metadata']=[]
  for n in paths:
   root=Path(n);meta=dict(path=n,readable_directory=root.is_dir())
   cf=root/'config.json'
   if cf.is_file() and cf.stat().st_size<1024**2:
    raw=cf.read_bytes();v=json.loads(raw);meta.update(config_SHA256=hashlib.sha256(raw).hexdigest(),architectures=v.get('architectures'),model_type=v.get('model_type'),dtype=v.get('dtype',v.get('torch_dtype')))
   out['own_path_metadata'].append(meta)
 rows.append(out)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),pending_count=len(rows),original_config_files=len(files),records=rows,no_weight_data_font_or_tokenizer_load=True,no_CPU_learned_or_NPU_or_file_mutation=True),ensure_ascii=False))
