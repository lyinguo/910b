"""Inspect new immutable source guard recovery; no model or old artifact mutation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_More_and_Hancom_own_original_native_serial_actual_v4_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr
x=json.loads(r.stdout);records={}
for stem in ['HancomHiLlava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_own_BF16_guard_20261122','HancomHiLlava31_own_original_Transformers_FP16_2048_source_readiness_v2_own_BF16_guard_20261122','HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v2_own_BF16_guard_real_FLOCK_20261122','HancomHiLlava31_private_exact_BF16_guard_v2_actual_launcher_20261122']:
 for n in ['state.json','readiness.json','job.json','actual_completion.json','actual_failure.json','completion.json','launch.json','stderr.log','readiness_stderr.log','exact_zero_forward_source_dtype_guard_recovery.json','fresh3/console.log','fresh3/verification_stderr.log','fresh3/verification_stdout.json']:
  p=B/stem/n
  if p.is_file():
   raw=p.read_bytes()
   try:v=json.loads(raw)
   except ValueError:v=raw.decode('utf8',errors='replace')
   records[str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),complete_value=v)
captures=[]
for home in sorted((R/'workspace/diagnostics/model_runs').glob('HancomHiLlava31_own_original_FP16_2048_*')):
 p=home/'generation_progress.jsonl';text=p.read_text() if p.is_file() else '';lines=text.splitlines();pending=bool(text and not text.endswith('\n'));lines=lines[:-1] if pending else lines;rows=[json.loads(z) for z in lines if z.strip()];counts={}
 for row in rows:counts[str(row['index'])]=max(counts.get(str(row['index']),0),int(row['step']))
 captures.append(dict(home=str(home),committed_counts=counts,total_committed_native_forwards=len(rows),last_committed_step=rows[-1] if rows else None,pending_partial_line=pending,NPZ=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in home.glob('groups/*/*.npz')]))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),original_v1_full_context_and_More_actual=x,Hancom_private_v2_current_complete_records=records,Hancom_all_capture_real_progress=captures),ensure_ascii=False))
