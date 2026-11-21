"""Read actual process, lock, trace and bounded coordination state; no model work."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
J=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
homes=[J,B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120',B/'GLM41Thinking_verified184_reviewed_publication_window_v2_actual_runner_directory_after_same_formal200_original_EX_20261120',B/'GLM41Thinking_original_VLLM_BF16_8192_LZMA_verified184_blank_G_publication_20261120',B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_wait_actual_capacity_first200_serial_20261120',B/'MetaMllama_completed_genuine3_after_GLM41_verified184_dual_release_shaped_storage_v4_serial_20261120',D/'immediate_disk_recovery_cleanup_watch_20261120']
sha=lambda b:hashlib.sha256(b).hexdigest()
files={}
for h in homes:
 for n in ['state.json','completion.json','actual_completion.json','server_phase_waiting_for_local_receipt.json','publication_receipt.json','dual_publication_completed.json','full_target_dual_publication_audit.json','reviewed_publication_admission.json']:
  p=h/n
  if p.is_file():
   b=p.read_bytes();files[str(p)]=dict(SHA256=sha(b),value=json.loads(b))
procs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
  if not args or not Path(args[0]).name.startswith('python'):continue
  relevant=any(any(k in a.lower() for k in ['glm41','glm-4.1v','metamllama','meta_mllama','meta-llama--llama-3.2','a100_alignment_evidence_runner','watch_batch_910b_immediate']) for a in args[1:])
  if not relevant:continue
  tail=(p/'stat').read_text().rsplit(')',1)[1].split()
  if tail[0]=='Z':continue
  procs.append(dict(PID=int(p.name),PPid=int(tail[1]),startticks=int(tail[19]),argv=args,cwd=os.readlink(p/'cwd')))
 except FileNotFoundError:continue
captures=[]
for p in sorted((R/'workspace/diagnostics/model_runs').glob('GLM41Thinking_own_original_BF16_8192_*')):
 tr=p/'actual_sample_trace.jsonl'
 if not tr.is_file():continue
 raw=tr.read_bytes();end=raw.rfind(b'\n');body=raw[:end+1] if end>=0 else b'';rows=[json.loads(x) for x in body.splitlines() if x]
 assert [v['index'] for v in rows]==list(range(len(rows)))
 captures.append(dict(home=str(p),completed_samples=len(rows),committed_native_forwards=sum(v['actual_native_forwards'] for v in rows),pending_nonterminated_tail_bytes=len(raw)-len(body),trace_SHA256=sha(raw),current_model_PID_matches_capture=any(p.name.endswith('_'+str(x['PID'])) and any(Path(a).name=='run.py' for a in x['argv'][1:]) for x in procs)))
nn=[p for p in procs if any(Path(a).name=='run.py' for a in p['argv'][1:])]
nf=[p for p in procs if any(Path(a).name.startswith(('verify_glm41','verify_meta_mllama')) for a in p['argv'][1:])]
agents={}
for p in [R/'AGENTS.md',R/'scripts/AGENTS.md',R/'scripts/maintenance/AGENTS.md',R/'scripts/windows_control/AGENTS.md',R/'configs/AGENTS.md']:
 if p.is_file():agents[str(p)]=p.read_text()
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_processes=procs,original_FLOCK=runner_lock_owners(),formal_model_NN_processes=nn,formal_fullNF_processes=nf,current_NN_or_NF_active=bool(nn or nf),GLM41_captures=captures,coordination_files=files,free_bytes=shutil.disk_usage(R).free,AGENTS=agents,max_concurrent_models=1,no_model_source_job_math_or_canonical_write=True),ensure_ascii=False))
