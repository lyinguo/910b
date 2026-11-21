"""CAS close full handoff after complete storage diagnostics; never publish G."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def main(receipt,expected,summary):
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
 assert sha(receipt)==expected;receipt=js(receipt);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(pointer)==receipt['pointer_SHA256'];old=pointer.read_bytes();ptr=json.loads(old);H=Path(ptr['handoff_directory']);note=Path(receipt['late_actual']).parent;assert summary.resolve().is_relative_to(note)
 canon={n:sha(n) for n in ptr['canonical_SHA256']};assert canon==ptr['canonical_SHA256'];pending=js(ptr['new_primary_pending'])
 for row in pending['all_current_primary_source_and_true_consumer_configs']:assert sha(row['path'])==row['SHA256']
 homes=[B/'MetaMllama_completed_true3_complete_shaped_raw_Fortran_LZMA_storage_capacity_diagnostic_only_20261120',B/'MetaMllama_completed_true3_complete_natural_order_byteplanes_LZMA_storage_v2_diagnostic_only_20261120',B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_wait_actual_capacity_first200_serial_20261120',B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'];files={}
 for home in homes:
  for n in ['state.json','actual_completion.json','review.json','all200_complete_storage_capacity_review.json']:
   p=home/n
   if p.is_file():files[str(p)]=dict(SHA256=sha(p),complete_JSON=js(p))
 procs=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   argv=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
   if argv and Path(argv[0]).name.startswith('python') and any(any(k in s.lower() for k in ['metamllama','meta_mllama','glm41thinking','a100_alignment_evidence_runner','/run.py']) for s in argv[1:]):procs.append(dict(PID=int(p.name),argv=argv,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text()))
  except FileNotFoundError:continue
 nn=any(any(Path(s).name=='run.py' for s in p['argv'][1:]) for p in procs);nf=any(any(Path(s).name.startswith(('verify_','audit_')) for s in p['argv'][1:]) for p in procs)
 latest=js(D/'incremental_monitor/latest.json');assert not latest['source_errors'] and not latest['new_log_error_candidates'];unresolved=[]
 for err in latest['inspection_errors']:
  if err.get('job')!=str(B/'MetaMllama_own_original_SDPA_BF16_2048_true3_v1_real_FLOCK_20261120/job.json') or err.get('actual_returncode')!=1:unresolved.append(err)
 assert not unresolved,unresolved
 formal=B/'MetaMllama_completed_genuine3_formal_NF3_exact_original_instruction_no_NN_replay_capacity_serial_20261120';assert sha(formal/'formal_acceptance.json')=='e55ce8c8080c1a8696bb58725631c6b8635fb1eb01352fbafebec2f87c8c361c'
 now=datetime.now(timezone.utc);stop=datetime.fromisoformat('2026-11-20T12:31:47.234170+08:00');snapshot=dict(UTC=now.isoformat(),free_bytes=shutil.disk_usage(R).free,original_FLOCK=runner_lock_owners(),actual_processes=procs,current_model_NN_active=nn,current_formal_NF_active=nf,waiters_are_not_NN_or_fullNF=True,stage_files=files,canonical_SHA256=canon,monitor_inspection_original_failed_NF1_preserved_and_formally_closed_by_NF3=True,unknown_new_errors_not_whitelisted=True,first_proven_current_formal_stoppage_BJT=stop.isoformat(),current_stoppage_seconds=None if nn or nf else (now-stop).total_seconds(),new_complete_NPU200_in_this_heartbeat=0,new_G_or_original_protocol_qualification=0,all_reports_sources_configs_logs_small_failed_unverified_and_necessary_examples_preserved=True,no_deletion_in_this_heartbeat=True,max_concurrent_models=1)
 put(note/'final_current_storage_and_waiters_actual.json',snapshot)
 summary_text=summary.read_text(encoding='utf-8');summary_text+='\n最新只读实查 UTC='+snapshot['UTC']+'；原盘free='+str(snapshot['free_bytes'])+'；实际NN='+str(nn)+'，NF='+str(nf)+'；完整最终进程/锁/所有数值/容量/两等待器状态见同目录../final_current_storage_and_waiters_actual.json。\n';summary.write_text(summary_text,encoding='utf-8',newline='\n')
 close=js(H/'closing_addendum.json');close.update(UTC=snapshot['UTC'],latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),latest_storage_and_capacity_waiters_actual=str(note/'final_current_storage_and_waiters_actual.json'),latest_storage_and_capacity_waiters_actual_SHA256=sha(note/'final_current_storage_and_waiters_actual.json'),current_model_NN_or_NF_active=nn or nf,Meta_complete_lossless_shaped_storage_capacity_bytes=112600281096,no_new_model_NN_fullNF_200_G_or_qualification_in_storage_diagnostics=True)
 assert pointer.read_bytes()==old and {n:sha(n) for n in canon}==canon;put(H/'closing_addendum.json',close)
 ptr.update(UTC=snapshot['UTC'],closing_addendum_SHA256=sha(H/'closing_addendum.json'),latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),latest_actual_note=str(note/'final_current_storage_and_waiters_actual.json'),latest_actual_note_SHA256=sha(note/'final_current_storage_and_waiters_actual.json'),latest_actual_snapshot=str(note/'final_current_storage_and_waiters_actual.json'),latest_actual_snapshot_SHA256=sha(note/'final_current_storage_and_waiters_actual.json'),current_actual_model_process_or_NF_active=nn or nf,current_actual_model_NN_or_NF_active=nn or nf,current_actual_stage='complete_lossless_storage_verified_wait_actual_original_disk_serial_max1',current_actual_controller=str(homes[2]),current_formal_model_stage='same_passed_Meta_true3_complete_shaped_storage_verified_wait_real_capacity',first_proven_stoppage_BJT=stop.isoformat(),severity_first_line=None if nn or nf else '严重阻塞｜正式评测未运行')
 put(pointer,ptr);result=dict(passed=True,pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),new_primary_pending=ptr['new_primary_pending'],new_primary_pending_SHA256=ptr['new_primary_pending_SHA256'],source_count=ptr['current_primary_source_count'],current_model_NN_or_NF_active=nn or nf,current_canonical_unchanged=True,no_deletion_G_or_model_source_change=True);put(note/'complete_storage_capacity_wait_final_closing_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]))
