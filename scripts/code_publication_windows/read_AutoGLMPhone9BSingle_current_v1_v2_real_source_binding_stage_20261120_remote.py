import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['AutoGLMPhone9BSingle_own_after182_serial_actual_launch_20261120', 'after_AutoGLM182_own_single_Phone9B_inputs_true3_fixed200_serial_20261120', 'AutoGLMPhone9BSingle_own_original_BF16_2048_frontend200_meta_v1_20261120', 'AutoGLMPhone9BSingle_own_original_BF16_2048_native_source_readiness_v3_actual_published_GLM_fixed_input_set_20261120', 'AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120', 'AutoGLMPhone9BSingle_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120', 'after_AutoGLM182_actual_dual_audit_release_original_resources_serial_20261120', 'AutoGLMPhone9BSingle_exact_frontend_path_preNN_recovery_v2_actual_launch_20261120', 'after_AutoGLM182_own_single_Phone9B_exact_frontend_path_true3_fixed200_serial_v2_20261120', 'AutoGLMPhone9BSingle_own_original_BF16_2048_readiness_v2_exact_frontend_path_20261120', 'AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v2_exact_frontend_path_real_FLOCK_20261120']
files={};pids=set()
for n in names:
 h=B/n
 if not h.exists():files[str(h)]=None;continue
 files[str(h)]={'children':[p.name for p in sorted(h.iterdir())]}
 for p in sorted(h.rglob('*')):
  if not p.is_file() or p.suffix not in ('.json','.log') or p.name in ['own704_active_source_manifest.json','own_frontend200_existing_proof_field_binding_no_replay.json']:continue
  if p.stat().st_size>8*1024*1024:
   files[str(p)]=dict(bytes=p.stat().st_size,requires_bounded_complete_followup=True);continue
  raw=p.read_bytes();files[str(p)]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),complete_text=raw.decode('utf8'))
  if p.name in ['background_launch.json','state.json','readiness.json','actual_completion.json']:
   try:
    v=json.loads(raw)
    for k in ['PID','runner_PID','child_PID','model_PID']:
     if isinstance(v.get(k),int):pids.add(v[k])
   except json.JSONDecodeError:pass
pids.update(p['PID'] for p in project_processes() if 'AutoGLM' in str(p) or p.get('role') in ['alignment_runner','model','CPU_verifier_or_audit'])
procs={}
for pid in sorted(pids):
 p=Path('/proc')/str(pid)
 try:
  args=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode();fds={}
  for f in (p/'fd').iterdir():
   try:t=os.readlink(f)
   except FileNotFoundError:continue
   if str(R) in t:fds[f.name]=t
  procs[pid]=dict(argv=args,cwd=os.readlink(p/'cwd'),status=(p/'status').read_text(),stat=(p/'stat').read_text(),io=(p/'io').read_text(),project_FDs=fds)
 except FileNotFoundError:pass
traces=[]
for h in sorted((R/'workspace/diagnostics/model_runs').glob('AutoGLMPhone9BSingle_own_original_BF16_2048_*')):
 pid=int(h.name.rsplit('_',1)[1])
 if pid not in pids:continue
 p=h/'actual_sample_trace.jsonl';raw=p.read_bytes() if p.exists() else b'';lines=raw.splitlines();rows=[json.loads(v) for v in lines]
 traces.append(dict(home=str(h),PID=pid,sample_count=len(rows),committed_actual_forwards=sum(v.get('actual_native_forwards',v.get('actual_forward_count',0)) for v in rows),failure_files=[f.name for f in h.glob('*failed.json')],last_sample=rows[-1] if rows else None))
ns=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),files=files,actual_procs=procs,actual_project_processes=project_processes(),actual_FLOCK=runner_lock_owners(),actual_free_bytes=shutil.disk_usage(R).free,actual_model_trace=traces,npu_smi_returncode=ns.returncode,npu_smi_stdout=ns.stdout,npu_smi_stderr=ns.stderr,canonical_SHA256={str(p):sha(p) for p in [Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'),D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json']}),ensure_ascii=False))
