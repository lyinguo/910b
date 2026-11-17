import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';s=json.loads((D/'state.json').read_bytes());out={}
for mid in ['huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated','unsloth/Llama-3.2-11B-Vision-Instruct','OpenGVLab/HoVLE-HD','nvidia/EGM-8B','opendatalab/MinerU2.5-Pro-2604-1.2B','HuggingFaceM4/idefics2-8b']:
 f=Path(s['models'][mid]['report_dir']);line=json.loads((f/'operator_lineage.json').read_bytes())['actual_execution'];p=Path(line['per_sample_evidence']);rows=[json.loads(x) for x in p.read_text().splitlines()]
 def compact(v):
  if isinstance(v,str):return v if v.startswith('/') else v[:140]
  if isinstance(v,(int,float,bool)) or v is None:return v
  if isinstance(v,list):return dict(list_length=len(v),first_two=[compact(x) for x in v[:2]])
  return {k:compact(x) for k,x in v.items() if not any(t in k.lower() for t in ['raw','logits','generation_config','tensor','weights','npu_parameters'])}
 roots={}
 for key in ['per_group_evidence','weight_loading_evidence','evidence_root']:
  if isinstance(line.get(key),str):roots[key]=line[key]
 owner=Path(line.get('per_group_evidence',line['weight_loading_evidence']));owner=owner.parent if owner.name=='groups' or owner.is_file() else owner
 groups=sorted(owner.glob('groups/*'))
 metadata={}
 if groups:
  for z in list(groups[0].iterdir()):
   if z.suffix=='.json' and z.stat().st_size<4*1024**2:metadata[z.name]=compact(json.loads(z.read_bytes()))
 out[mid]=dict(trace=str(p),row_count=len(rows),first=compact(rows[0]),middle=compact(rows[99]),last=compact(rows[-1]),roots=roots,first_group_files=[x.name for x in groups[0].iterdir()] if groups else [],first_group_metadata=metadata)
print(json.dumps(out,ensure_ascii=False))
