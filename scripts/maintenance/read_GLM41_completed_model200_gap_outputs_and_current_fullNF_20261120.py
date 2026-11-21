"""Read the original completed model outputs and actual NF process, without rescoring."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
s=json.loads((J/'state.json').read_bytes());folder=Path(s['report_dir']);report=json.loads((folder/'eval_report.json').read_bytes())
assert report['pipeline_config']['sample_count']==200 and report['metrics']['Accuracy']==.555
rows=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_bytes().splitlines() if x]
assert len(rows)==200
samples=[]
for i in [0,1,2,9,90,99,100,109,190,199]:
 samples.append(dict(index=i,original_output=rows[i]))
procs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
  if not args or not Path(args[0]).name.startswith('python') or not any(Path(v).name.startswith('verify_glm41') for v in args[1:]):continue
  procs.append(dict(PID=int(p.name),argv=args,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),io=(p/'io').read_text(),FD={x.name:os.readlink(x) for x in (p/'fd').iterdir() if x.is_symlink()}))
 except FileNotFoundError:continue
logs={}
for n in ['console.log','verification_stderr.log','verification_stdout.json']:
 p=J/'formal200'/n
 if p.is_file():
  raw=p.read_bytes();logs[n]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),complete_text=raw.decode('utf-8',errors='replace'))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),model_report=str(folder),original_summary=report.get('summary'),original_metrics=report['metrics'],output_keys=list(rows[0]),original_required_example_outputs=samples,actual_fullNF_processes=procs,actual_current_stage=s,formal200_logs=logs,no_model_math_budget_extraction_source_or_G_changed=True),ensure_ascii=False))
