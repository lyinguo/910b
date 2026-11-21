import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,os,hashlib
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'MetaMllama_own_original_HF_BF16_2048_frontend200_meta_sources_v1_20261120';L=B/'MetaMllama_own_original_HF_BF16_2048_frontend200_meta_sources_v1_launch_20261120'
out=dict(UTC=datetime.now(timezone.utc).isoformat(),files={})
for p in [H/'progress.json',H/'actual_completion.json',H/'review.json',L/'launch.json',L/'stdout.log',L/'stderr.log']:
 if p.is_file():
  raw=p.read_bytes();v=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
  if p.name=='review.json':x=json.loads(raw);v['summary']={k:z for k,z in x.items() if k not in ['input_records','all906_source_to_native_semantic_name_shape_mapping_verified','own_resource_sha256','native_activation_modules']}
  else:
   try:v['value']=json.loads(raw)
   except ValueError:v['complete_text']=raw.decode(errors='replace')
  out['files'][str(p)]=v
if (L/'launch.json').is_file():
 pid=json.loads((L/'launch.json').read_bytes())['PID'];q=Path('/proc')/str(pid);out['process_exists']=q.exists()
 if q.exists():out['process']=dict(PID=pid,argv=[s.decode() for s in (q/'cmdline').read_bytes().split(b'\0') if s],cwd=os.readlink(q/'cwd'),io=(q/'io').read_text(),stat=(q/'stat').read_text())
print(json.dumps(out,ensure_ascii=False))
