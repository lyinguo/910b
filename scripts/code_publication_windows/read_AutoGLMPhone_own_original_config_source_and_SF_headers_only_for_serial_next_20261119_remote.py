import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='zai-org/AutoGLM-Phone-9B-Multilingual'
js=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
import yaml
cpath=R/'configs/Glm4vForConditionalGeneration/zai-org--AutoGLM-Phone-9B-Multilingual.yaml';cfg=yaml.safe_load(cpath.read_bytes());m=cfg['model'];root=Path(m.get('model_path',m.get('kwargs',{}).get('model_path','')));assert root.is_dir()
assets={}
for p in root.iterdir():
 if p.is_file() and p.suffix in ['.json','.py','.jinja'] and p.stat().st_size<8*1024**2:
  assets[p.name]=dict(SHA256=sha(p),bytes=p.stat().st_size)
  if p.name in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer_config.json']:assets[p.name]['value']=js(p)
headers={};total=0
for p in sorted(root.glob('*.safetensors')):
 with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];assert n<32*1024**2;header=json.loads(f.read(n))
 total+=p.stat().st_size;headers[p.name]=dict(file_bytes=p.stat().st_size,header_SHA256=hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest(),full_header=header)
state=js(D/'state.json');history=[str(p) for p in B.glob('*AutoGLM*') if p.is_dir()]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,original_config=str(cpath.relative_to(R)),original_config_SHA256=sha(cpath),full_saved_config=cfg,own_root=str(root),own_assets=assets,own_SF_headers=headers,total_SF_stored_bytes=total,own_physical_tensor_count=sum(len(v['full_header'])-('__metadata__' in v['full_header']) for v in headers.values()),state=state['models'][MID],existing_own_diagnostic_directories=history,no_full_weight_load_or_NEW_CPU_NPU_or_model_change=True),ensure_ascii=False))
