import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'LlavaMini_own_original_FP16_frontend200_meta_v3_actual_full_npz_interface_no_learned_NN_20261122'
out=dict(UTC=datetime.now(timezone.utc).isoformat(),files=[dict(name=p.name,bytes=p.stat().st_size) for p in H.iterdir() if p.is_file()] if H.exists() else [])
for n in ['progress.json','own_source_meta_review.json','readiness.json','actual_completion.json']:
 p=H/n
 if p.exists():out[n]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'value':json.loads(p.read_bytes())}
for n in ['_launcher_stderr.log','_launcher_stdout.log']:
 p=Path(str(H)+n)
 if p.exists():out[n]=p.read_text()
out['processes']=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  if any('prepare_llava_mini_own_original_FP16_frontend200_meta_v3_actual_full_npz_interface_20261122.py' in x for x in argv):out['processes'].append(dict(PID=int(p.name),argv=argv,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),io=(p/'io').read_text()))
 except (OSError,UnicodeError):pass
print(json.dumps(out,ensure_ascii=False))
