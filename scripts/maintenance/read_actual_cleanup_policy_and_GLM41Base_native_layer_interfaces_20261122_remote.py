"""Read exact cleanup state and current own native layer signatures, no inference."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,inspect,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
def rec(p):
 p=Path(p);b=p.read_bytes()
 try:v=json.loads(b)
 except ValueError:v=b.decode('utf8',errors='replace')
 return dict(SHA256=hashlib.sha256(b).hexdigest(),complete_value=v)
paths=[R/'configs/maintenance/batch_910b_immediate_disk_recovery_cleanup_v3_20261120.json',D/'immediate_disk_recovery_cleanup_watch_20261120/state.json',D/'immediate_disk_recovery_cleanup_watch_20261120/actual_failure.json'];records={str(p):rec(p) for p in paths if p.is_file()}
state=records.get(str(paths[1]),{}).get('complete_value',{});last=state.get('last_cleanup') or {}
for name in ['request','stdout','stderr']:
 p=last.get(name)
 if p and Path(p).is_file() and Path(p).stat().st_size<8*2**20:records[p]=rec(p)
from transformers.models.glm4v import modeling_glm4v as m
classes={}
for n in ['Glm4vDecoderLayer','Glm4vTextDecoderLayer','Glm4vVisionBlock','Glm4vRMSNorm','Glm4vVisionPatchEmbed','Glm4vVisionPatchMerger','Glm4vForConditionalGeneration','Glm4vModel']:
 c=getattr(m,n,None)
 if c is not None:classes[n]=dict(source_file=inspect.getfile(c),forward_signature=str(inspect.signature(c.forward)),forward_source=inspect.getsource(c.forward))
print(json.dumps(dict(cleanup_complete_records=records,installed_native_layer_interfaces=classes,Glm4v_source_file=inspect.getfile(m),Glm4v_source_SHA256=hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest(),no_cleanup_or_model_execution=True),ensure_ascii=False))
