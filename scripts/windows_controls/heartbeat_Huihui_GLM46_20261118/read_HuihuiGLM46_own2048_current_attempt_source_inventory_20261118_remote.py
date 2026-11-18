import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pointer=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes())
ready=B/'HuihuiGLM46Flash_own_original_BF16_2048_native_source_readiness_v1_20261118/readiness.json'
rd=json.loads(ready.read_bytes()) if ready.is_file() else None
rels=['scripts/launch_huihui_glm46flash_own_original2048_frontend_ready_true3_20261118.py','scripts/continue_huihui_glm46flash_own_original_BF16_2048_20261118_true3_capacity_fresh3_fixed200.py']
if rd:rels.extend(rd['source_SHA256'])
rels.extend(str(p.relative_to(R)) for p in (R/'configs/Glm4vForConditionalGeneration').glob('huihui-ai--Huihui-GLM-4.6V-Flash-abliterated-own-original-BF16-2048-*.yaml'))
rels.extend(str(p.relative_to(R)) for p in (R/'configs/protocols').glob('huihui_glm46flash_own_original_BF16_2048_20261118_*.json'))
rows=[dict(relative=n,SHA256=sha(R/n),bytes=(R/n).stat().st_size) for n in sorted(set(rels))]
stages={};static={}
for name in ['HuihuiGLM46Flash_own_original_BF16_2048_frontend200_meta_v1_20261118','HuihuiGLM46Flash_own_original_BF16_2048_native_source_readiness_v1_20261118','HuihuiGLM46Flash_own_original_BF16_2048_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v1_20261118','HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118','HuihuiGLM46Flash_own_original_BF16_2048_approved_fresh3_fixed200_v1_real_FLOCK_20261118']:
 h=B/name;items={}
 for n in ['job.json','state.json','completion.json','actual_completion.json','actual_failure.json','review.json','readiness.json','full_shape_complete_numeric_storage_capacity_review.json','first200_measurement_approval.json','fresh3/accepted.json','fresh3/verification_stdout.json','fresh3/verification_stderr.log','fresh3/console.log','formal200/accepted.json','formal200/verification_stdout.json']:
  p=h/n
  if p.is_file():
   b=p.read_bytes();stat=dict(path=str(p),SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b));text=b.decode('utf8',errors='replace')
   try:value=json.loads(text)
   except ValueError:value=text[-24000:]
   if n=='job.json':value=dict(model_id=value['model_id'],stages=value['stages'],config=value['spec']['config'],source_SHA256=value['spec']['source_sha256'])
   items[n]=dict(stat,value=value)
   if n not in ['state.json','fresh3/console.log']:static[name+'/'+n]=stat
 stages[name]=items
active=project_processes();proc={}
for row in active:
 pid=row.get('PID');p=Path('/proc')/str(pid)
 if p.is_dir() and (row.get('role') in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier'] or 'huihui_glm46' in str(row).lower()):
  try:proc[pid]=dict(row,stat=(p/'stat').read_text(),actual_argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),actual_cwd=os.readlink(p/'cwd'),io=(p/'io').read_text())
  except OSError as e:proc[pid]=dict(row,read_error=str(e))
native={}
for h in (R/'workspace/diagnostics/model_runs').glob('HuihuiGLM46Flash_own_original_BF16_2048_*'):
 if not h.is_dir():continue
 info={}
 for n in ['actual_sample_trace.jsonl','completed_trace.json','all_own_source_actual_NPU.json','independent_report_verification.json']:
  p=h/n
  if p.is_file():
   b=p.read_bytes();meta=dict(path=str(p),bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())
   if n=='actual_sample_trace.jsonl':
    data=[json.loads(s) for s in b.decode().splitlines()];meta.update(completed_samples=len(data),committed_forward_count=sum(v['actual_native_forwards'] for v in data),last=data[-1] if data else None)
   elif n in ['completed_trace.json','independent_report_verification.json']:meta['value']=json.loads(b)
   info[n]=meta
 native[str(h)]=info
smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),pointer=pointer,pointer_SHA256=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'),canonical_SHA256={n:sha(n) for n in pointer['canonical_SHA256']},declared_Huihui_and_dependency_primary_sources=rows,static_original_metadata=static,readiness=rd,stages=stages,actual_project_processes=active,actual_proc_details=proc,actual_original_FLOCK=runner_lock_owners(),native_actual_model_evidence=native,npu_smi=dict(returncode=smi.returncode,stdout=smi.stdout,stderr=smi.stderr),free_bytes=shutil.disk_usage(R).free),ensure_ascii=False))
