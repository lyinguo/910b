import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';s=json.loads((D/'state.json').read_bytes())['models']
ids=['microsoft/Phi-3.5-vision-instruct','microsoft/Florence-2-large-ft','dh-unibe/trocr-kurrent','DunnBC22/trocr-base-printed_captcha_ocr','yifeihu/TF-ID-base','yifeihu/TFT-ID-1.0','mlx-community/gemma-3n-E2B-it-4bit','google/deplot','RekaAI/reka-edge-2603','embedl/Cosmos-Reason2-2B-W4A16-Edge2','microsoft/OmniParser']
configs=[f for f in (R/'configs').rglob('*.yaml') if not f.name.endswith('-smoke.yaml')];o={'UTC':datetime.now(timezone.utc).isoformat(),'models':{},'prior_dossier_directories':[p.name for p in B.iterdir() if p.is_dir() and any(t in p.name.lower() for t in ['trocr','florence','phi35','phi3_5','tf_id','tft_id','reka','omnipa','deplot','cosmos','mlxe2'])]}
for mid in ids:
 v=s[mid];d={'entry':v,'roots':{},'saved_configs':{}}
 for name in v['weight_paths']:
  root=Path(name)
  try:
   values={}
   for p in sorted(root.iterdir()):
    if p.is_file():
     item={'bytes':p.stat().st_size}
     if p.name in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','quantize_config.json','quantization_config.json']:
      raw=p.read_bytes();item.update(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
     values[p.name]=item
   d['roots'][name]=values
  except OSError as e:d['roots'][name]={'currently_unreadable':str(e)}
 stem=mid.replace('/','--')
 for f in configs:
  if f.stem==stem:
   raw=f.read_bytes();d['saved_configs'][str(f)]={'SHA256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
 o['models'][mid]=d
print(json.dumps(o,ensure_ascii=False))
