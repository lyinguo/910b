import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,yaml
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';S=json.loads((D/'state.json').read_bytes())
ids=['zai-org/GLM-4.1V-9B-Base','google/gemma-4-E2B-it-assistant','ICTNLP/llava-mini-llama-3.1-8b','microsoft/llava-med-v1.5-mistral-7b']
out={'UTC':datetime.now(timezone.utc).isoformat(),'no_CPU_learned_or_NPU':True,'models':[]}
for mid in ids:
 e=S['models'][mid];row={'model_id':mid,'state':e,'saved_configs':[],'weight_roots':[]}
 stem=mid.replace('/','--')+'.yaml'
 for p in (R/'configs').glob('**/'+stem):
  b=p.read_bytes();row['saved_configs'].append({'path':str(p),'SHA256':hashlib.sha256(b).hexdigest(),'complete_text':b.decode(),'parsed':yaml.safe_load(b)})
 for name in e['weight_paths']:
  W=Path(name);root={'path':name,'exists':W.is_dir()}
  if W.is_dir():
   root['immediate_files']=[{'name':p.name,'bytes':p.stat().st_size} for p in W.iterdir() if p.is_file()]
   root['assets']={}
   for n in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json']:
    p=W/n
    if p.exists():b=p.read_bytes();root['assets'][n]={'SHA256':hashlib.sha256(b).hexdigest(),'value':json.loads(b)}
   p=W/'model.safetensors.index.json'
   if p.exists():b=p.read_bytes();j=json.loads(b);root['index']={'SHA256':hashlib.sha256(b).hexdigest(),'metadata':j.get('metadata'),'weight_map':j['weight_map']}
  row['weight_roots'].append(root)
 out['models'].append(row)
print(json.dumps(out,ensure_ascii=False))
