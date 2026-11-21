"""Close exact-byte storage maintenance and current GLM evidence, no model edits."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def main(receipt_path,expected,summary):
 assert sha(receipt_path)==expected;rec=js(receipt_path);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(pointer)==rec['pointer_SHA256'];old=pointer.read_bytes();ptr=json.loads(old);H=Path(ptr['handoff_directory']);note=Path(rec['late_actual']).parent;assert summary.resolve().is_relative_to(note)
 canon={p:sha(p) for p in ptr['canonical_SHA256']};assert canon==ptr['canonical_SHA256']
 cleanup=B/'user_authorized_periodic_cleanup_exact_matrix_bytes_all_paths_retained_first_serial_recovery_20261120';result=js(cleanup/'actual_completion.json');assert result['actual_returncode']==0 and result['physical_bytes_reclaimed']==65732198400 and result['all_original_matrix_paths_protected']==116 and result['all_original_complete_file_SHA_before_after_equal'] and result['no_numerical_array_pruned']
 before=js(cleanup/'all_original_paths_complete_SHA_before.json');after=js(cleanup/'all_original_paths_complete_SHA_after.json');assert before==after and len(after)==116
 child=subprocess.run([sys.executable,'-B',str(R/'scripts/maintenance/read_exact_matrix_physical_storage_application_and_GLM41_serial_20261120_remote.py')],capture_output=True,text=True);assert child.returncode==0,child.stderr;actual=json.loads(child.stdout);put(note/'current_GLM41_true_process_lock_progress_and_cleanup.json',actual)
 J=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120';assert sha(J/'job.json')=='756c7f9221f17a39e85f00cf064d640d197fa9ec42e1af0482eac85f17c7d163';state=js(J/'state.json');runner=state['PID'];owner,=actual['original_FLOCK'];assert owner['PID']==runner
 process,=[p for p in actual['actual_processes'] if p['PID']==runner];assert str(J) in process['argv'] and process['cwd']=='/root'
 live=[]
 for p in actual['actual_processes']:
  if not any(Path(s).name=='run.py' for s in p['argv']):continue
  statfields=p['stat'].split(') ',1)[1].split();assert int(statfields[1])==runner
  live.append(p)
 current_captures=[x for x in actual['GLM41_captures'] if any(x['home'].endswith('_'+str(p['PID'])) for p in live)];samples=sum(x['traces'].get('actual_sample_trace.jsonl',{}).get('rows',0) for x in current_captures)
 committed=sum(x['traces'].get('actual_sample_trace.jsonl',{}).get('last',{}).get('actual_native_forwards',0) for x in current_captures if x['traces'].get('actual_sample_trace.jsonl',{}).get('last'));model_nn=bool(samples);nf=any(any(Path(s).name.startswith('verify_') for s in p['argv'][1:]) for p in actual['actual_processes'])
 snapshot=dict(UTC=actual['UTC'],exact_physical_storage_application=result,cleanup_receipt_SHA256=sha(cleanup/'actual_completion.json'),all_original116_paths_and_complete_SHA_retained=True,no_new_numerical_array_pruning=True,original_retention_ledger_SHA256=sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'),current_GLM41_job=str(J),current_job_SHA256=sha(J/'job.json'),actual_runner_state=state,actual_runner_PID=runner,actual_model_PIDs=[p['PID'] for p in live],actual_model_NN_has_committed_samples=model_nn,current_committed_samples=samples,current_last_sample_native_forwards=committed,current_formal_NF_active=nf,loading_not_claimed_NN=not model_nn,free_bytes=actual['free_bytes'],max_concurrent_models=1,new_complete200_G_or_qualification=0,current_canonical_unchanged=True)
 put(note/'full_latest_user_periodic_cleanup_GLM41_actual.json',snapshot)
 with summary.open('a',encoding='utf-8') as f:f.write('\n最新实查='+json.dumps(snapshot,ensure_ascii=False)+'\n')
 close=js(H/'closing_addendum.json');close.update(UTC=actual['UTC'],latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),latest_user_periodic_cleanup_GLM41_actual=str(note/'full_latest_user_periodic_cleanup_GLM41_actual.json'),latest_user_periodic_cleanup_GLM41_actual_SHA256=sha(note/'full_latest_user_periodic_cleanup_GLM41_actual.json'),current_model_NN_or_NF_active=model_nn or nf,latest_storage_cleanup_no_unique_array_pruned=True,original_retention_ledger_unchanged=True)
 assert pointer.read_bytes()==old and {p:sha(p) for p in canon}==canon;put(H/'closing_addendum.json',close);ptr.update(UTC=actual['UTC'],closing_addendum_SHA256=sha(H/'closing_addendum.json'),latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),latest_actual_note=str(note/'full_latest_user_periodic_cleanup_GLM41_actual.json'),latest_actual_note_SHA256=sha(note/'full_latest_user_periodic_cleanup_GLM41_actual.json'),latest_actual_snapshot=str(note/'current_GLM41_true_process_lock_progress_and_cleanup.json'),latest_actual_snapshot_SHA256=sha(note/'current_GLM41_true_process_lock_progress_and_cleanup.json'),current_actual_model_process_or_NF_active=bool(live) or nf,current_actual_model_NN_or_NF_active=model_nn or nf,current_actual_job=str(J),current_actual_controller=str(B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'),current_actual_stage=state['phase'],current_owned_candidate='zai-org/GLM-4.1V-9B-Thinking',current_formal_model_stage='approved_NEWfresh3_fixed200_after_exact_byte_duplicate_storage_cleanup',severity_first_line=None if model_nn or nf else '严重阻塞｜正式评测尚在加载，未确认新样本前向')
 put(pointer,ptr);out=dict(passed=True,pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),new_primary_pending=ptr['new_primary_pending'],new_primary_pending_SHA256=ptr['new_primary_pending_SHA256'],source_count=ptr['current_primary_source_count'],snapshot=snapshot);put(note/'user_periodic_cleanup_GLM41_current_closing_receipt.json',out);print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]))
