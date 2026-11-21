import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,struct
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
MID='zai-org/GLM-4.1V-9B-Thinking';root=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration/zai-org/GLM-4.1V-9B-Thinking');out={}
for rel in ['configs/Glm4vForConditionalGeneration/zai-org--GLM-4.1V-9B-Thinking.yaml','universal_eval/model/adapters/vllm_adapter.py']:
 p=R/rel;raw=p.read_bytes();out[str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),complete_text=raw.decode())
assets={}
for name in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer_config.json','chat_template.jinja','model.safetensors.index.json']:
 p=root/name
 if p.is_file():
  raw=p.read_bytes();assets[name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),complete_text=raw.decode())
headers={}
for p in root.glob('*.safetensors'):
 with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
 headers[str(p)]=dict(bytes=p.stat().st_size,header=h)
state=json.loads((D/'state.json').read_bytes())
sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import workbook_rows
rows,_,_=workbook_rows(R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');row,=[r for r in rows if r['model_id']==MID]
old_scopes=[str(p) for p in B.glob('*') if p.is_dir() and 'GLM41' in p.name and 'Thinking' in p.name]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,root=str(root),files=out,own_assets=assets,own_source_headers_only_not_weight_load_or_NN=headers,state=state['models'][MID],actual_workbook_row=row,existing_related_stage_directories=old_scopes,actual_NEW_model_forwards=0),ensure_ascii=False))
