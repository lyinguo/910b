"""Preserve exact185/137 authorized transitions and every new useful primary source; no model mutation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,copy,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
now=lambda:datetime.now(timezone.utc).isoformat()
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def read(p):
 if not p.is_file():return None
 b=p.read_bytes()
 try:return json.loads(b)
 except Exception:return b.decode('utf-8',errors='replace')
def main():
 plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before=pointer.read_bytes();assert hashlib.sha256(before).hexdigest()==q['expected_pointer_SHA256'];ptr=json.loads(before);H=Path(ptr['handoff_directory']);parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256']==q['expected_pending_SHA256']
 pub=B/'Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122';audit=js(pub/'full_target_dual_publication_audit.json');assert sha(pub/'full_target_dual_publication_audit.json')=='cf97094bacfa17f27c531431c7486594e0a07b57f18c02fce19e93a6751f08f7' and audit['passed'] and audit['canonical_counts']==dict(done=185,skipped_no_complete_weights=310,needs_adaptation=205)
 receipt=js(pub/'publication_receipt.json');assert receipt['completed'] and len(receipt['targets'])==1 and receipt['targets'][0]['workbook_write']['cell']=='G664'
 ret=B/'verified185_Gemma4MLX8_fullNF2_dual_audit_correct_nonexample137_exact_retention_20261122';assert sha(ret/'actual_completion.json')=='0d5c89e2143683f5891033f38d1a1d56fa0d5a8d525b7fb03c9dfd2e7cb9987e';clean=js(ret/'actual_completion.json');assert clean['actual_returncode']==0 and clean['removed_files']==137 and clean['all_protected_SHA256_unchanged']
 assert sha(ret/'removed_exact_accepted_correct_nonexample_NPZ_original_SHA256.jsonl')==clean['manifest_SHA256']=='92b0e454981636313072dab25b3e2667f8106261b069e415df9417badebcba2b'
 canonical={};transitions=[]
 for n,h in ptr['canonical_SHA256'].items():
  actual=sha(n);canonical[n]=actual
  if actual==h:continue
  if n in audit['canonical_SHA256']:
   assert h==receipt['canonical_before_sha256'][n] and actual==audit['canonical_SHA256'][n];transitions.append(dict(path=n,before=h,after=actual,reason='Authorized own fullNF200 three-D qualified185 single G/state/reg/binding dual transaction'))
  else:
   assert n==str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json') and h=='b3969dd7799de7a9d9a5cfa2621681b931c734c4c6f1233734dc3f803a80fbdd' and actual=='5f20ac2b35d58b09e5c4a3a8ca6a5dacf1446b9cb39876678b24458b0f1ef4b0' and sha(ret/'ledger_before.original')==h
   oldledger=js(ret/'ledger_before.original');current=js(n);after=copy.deepcopy(current);event=after['events'].pop();assert after==oldledger and event['receipt_SHA256']==sha(ret/'actual_completion.json') and event['removed_files']==137 and event['manifest_SHA256']==clean['manifest_SHA256'];transitions.append(dict(path=n,before=h,after=actual,reason='User-authorized exact137 correct nonexample NPZ pruning after full NF/dual target audit; all11006 protected SHA equal'))
 assert len(transitions)==5
 assert sha(R/'configs/publication/code_only.json')=='39e53cf773110d35891d79bd38c4e31b1d842da5198cf8b84d50870e838f59e5' and sha(C/'daily_20261122_actual_reviewed_commit_remote_receipts/daily_20261122_receipt.json')=='32ff728e32482f9009e8ef1fe9e22f5e836613a48edd8181fef01fa847ac1115'
 rows={r['path']:r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']};assert all(sha(n)==r['SHA256'] for n,r in rows.items())
 note=H/'late_actual_notes'/q['round_name'];note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(before);(note/'previous_closing_addendum.json').write_bytes((H/'closing_addendum.json').read_bytes())
 for r in q['local_original_records']:
  raw=base64.b64decode(r['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==r['SHA256'];dest=note/'local_original_records'/r['name'];assert dest.resolve().is_relative_to(note.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 paths=list(q['new_primary_paths'])+[str(Path(__file__).resolve().relative_to(R))]
 J=B/'Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122';paths+=list(js(J/'job.json')['spec']['source_sha256'])
 paths += [str(p.relative_to(R)) for p in (R/'scripts/tests').glob('test_unsloth_gemma_4_E4B_it_MLX_8bit_verified185_original_writer_isolated_20261122.py')]
 for n in sorted(set(paths)):
  p=R/n;assert p.is_file() and p.resolve().is_relative_to(R) and (n=='run.py' or n.startswith(('scripts/','configs/','universal_eval/','tests/')));h=sha(p)
  if str(p) in rows:assert rows[str(p)]['SHA256']==h,n
  else:
   dest=note/'primary_source_snapshots'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=n,SHA256=h,bytes=p.stat().st_size,snapshot=str(dest),role='Own original adaptation/complete NF/precise qualified publication/authorized retention/real queue consumer/Windows control or preservation implementation')
 M=R/'scripts/A100_alignment_incremental_monitor_20261103.py';assert sha(M)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
 m=subprocess.run(['python3','-B',str(M)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 sys.path[:0]=[str(R),str(R/'scripts')]
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 processes=project_processes();detail=[]
 for p in processes:
  proc=Path('/proc')/str(p['PID']);v=dict(p)
  try:
   v['proc_stat']= (proc/'stat').read_text();v['proc_status']=(proc/'status').read_text();v['proc_io']=(proc/'io').read_text();v['fd']={f.name:os.readlink(f) for f in (proc/'fd').iterdir()}
  except OSError as e:v['inspection_race']=str(e)
  detail.append(v)
 constructor=B/'XtunerLlava3_own_original_FP16_actual_CPU_loader_resize_remap_no_forward_v3_full_loading_info_schema_20261122';ci=B/'XtunerLlava3_original_actual_CPU_loader_resize_no_forward_v3_full_loading_info_schema_launcher_20261122';wait=B/'BeeSFT_completed_storage_NF_capacity_v5_verified185_actual_original_disk_waiter_serial_20261122'
 smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25)
 actual=dict(UTC=now(),processes=detail,runner_lock_owners=runner_lock_owners(),actual_free_bytes=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize,canonical_SHA256=canonical,authorized_canonical_transitions=transitions,canonical_counts=audit['canonical_counts'],last_full200_NF_completed_Beijing='2026-11-22T03:34:24.274295+08:00',last_qualified_dual_audit_Beijing='2026-11-22T03:47:22.926705+08:00',full_qualified185_publication_audit=audit,retention_actual=clean,retention_actual_SHA256=sha(ret/'actual_completion.json'),next_Xtuner_constructor_actual={str(p):read(p) for p in [constructor/'progress.json',constructor/'actual_completion.json',constructor/'actual_original_loader_resize_remap_review.json',constructor/'original_from_pretrained_loading_info.json',ci/'stderr.log',ci/'launch.json']},BeeSFT_capacity_waiter_actual={str(p):read(p) for p in [wait/'state.json',wait/'actual_completion.json']},monitor=dict(actual_returncode=m.returncode,stdout=m.stdout,stderr=m.stderr,complete_latest_and_checkpoint={str(p):js(p) for p in [D/'incremental_monitor/latest.json',D/'incremental_monitor/checkpoint.json']}),npu_smi=dict(returncode=smi.returncode,stdout=smi.stdout,stderr=smi.stderr))
 assert m.returncode==0 and smi.returncode==0 and not actual['monitor']['complete_latest_and_checkpoint'][str(D/'incremental_monitor/latest.json')]['source_errors']
 assert all(sha(R/n)==h for n,h in js(J/'job.json')['spec']['source_sha256'].items())
 model=[p for p in processes if any(Path(a).name=='run.py' for a in p['argv'])];verifiers=[p for p in processes if any(Path(a).name.startswith('verify_') for a in p['argv'])];active=bool(model or verifiers)
 actual.update(current_model_process_active=bool(model),current_full_numerical_verifier_active=bool(verifiers),current_model_NN_or_NF_active=active,CPU_constructor_frontend_waiter_not_formal_NN_or_NF=True,current_primary_source_count=len(rows));put(note/'current_serial_actual_after185_retention_next_own_source.json',actual)
 compact={k:actual[k] for k in ['UTC','actual_free_bytes','runner_lock_owners','current_model_process_active','current_full_numerical_verifier_active','current_model_NN_or_NF_active','canonical_counts','canonical_SHA256']};compact['actual_processes']=[dict(PID=p['PID'],PPid=p['PPid'],cwd=p['cwd'],argv=p['argv']) for p in processes];compact['actual_constructor_progress']=actual['next_Xtuner_constructor_actual'].get(str(constructor/'progress.json'));compact['actual_constructor_completion']=actual['next_Xtuner_constructor_actual'].get(str(constructor/'actual_completion.json'));compact['BeeSFT_waiter_state']=actual['BeeSFT_capacity_waiter_actual'].get(str(wait/'state.json'));compact['full_snapshot']=str(note/'current_serial_actual_after185_retention_next_own_source.json');compact['full_snapshot_SHA256']=sha(note/'current_serial_actual_after185_retention_next_own_source.json')
 full=q['full_handoff_template'].replace('CURRENT_SOURCE_COUNT',str(len(rows))).replace('CURRENT_ACTUAL_JSON',json.dumps(compact,ensure_ascii=False));fullpath=note/q['full_note_name'];fullpath.write_text(full,encoding='utf-8')
 pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=now(),parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-22',Code_Day_commit='43a149bb69510288451c2348159d6e7baa0d8d8e',no_second_same_day_commit=True,next_daily_review='Day23 all new useful source/config/test/Windows consumer/dependencies, fresh SHA/full diff/all outgoing/appropriate validation and actual publisher reviewed planSHA; no empty commit'))
 updates=dict(UTC=now(),canonical_SHA256=canonical,authorized_canonical_transition_snapshot=str(note/'current_serial_actual_after185_retention_next_own_source.json'),new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_primary_source_count=len(rows),current_source_count=len(rows),latest_actual_note=str(fullpath),latest_actual_note_SHA256=sha(fullpath),latest_full_actual_summary=str(fullpath),latest_full_actual_summary_SHA256=sha(fullpath),latest_actual_snapshot=str(note/'current_serial_actual_after185_retention_next_own_source.json'),latest_actual_snapshot_SHA256=sha(note/'current_serial_actual_after185_retention_next_own_source.json'),current_verified_count=185,numeric_within5pct=170,numeric_outside5pct=15,complete_weights_pending_formal=205,confirmed_missing=310,complete_original_protocol_qualified=71,current_actual_stage='verified185_Gemma4MLX8_full200_qualified_dual_delivered_exact137_retained_Xtuner_own_source_construction_and_BeeSFT_v5_actual_capacity',current_formal_model_stage='Latest qualified genuine200 Gemma4MLX8 all actual exits0; Xtuner frontend/source construction is not NEW_NPU3, BeeSFT capacity waiter not NN/NF',current_model_process_active=bool(model),current_full_numerical_verifier_active=bool(verifiers),current_model_NN_or_NF_active=active,current_actual_model_NN_or_NF_active=active,current_NN_or_NF_active=active,current_actual_model_process_or_NF_active=active,current_stop_earliest_proven_Beijing=None if active else '2026-11-22T03:52:27.024058+08:00 after authorized publication window',last_qualified_full200_NF_completed_Beijing='2026-11-22T03:34:24.274295+08:00',last_qualified_dual_audit_completed_Beijing='2026-11-22T03:47:22.926705+08:00',current_retention_ledger_SHA256=sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'),retention_latest_exact137_receipt=str(ret/'actual_completion.json'),retention_latest_exact137_receipt_SHA256=sha(ret/'actual_completion.json'))
 assert pointer.read_bytes()==before and all(sha(n)==h for n,h in canonical.items()) and all(sha(n)==r['SHA256'] for n,r in rows.items());closing=js(H/'closing_addendum.json');closing.update(updates);put(H/'closing_addendum.json',closing);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(H/'closing_addendum.json');put(pointer,ptr)
 result=dict(passed=True,UTC=now(),pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),full_actual_note=str(fullpath),full_actual_note_SHA256=sha(fullpath),actual_snapshot=str(note/'current_serial_actual_after185_retention_next_own_source.json'),actual_snapshot_SHA256=sha(note/'current_serial_actual_after185_retention_next_own_source.json'),current_model_NN_or_NF_active=active,authorized185_single_target_and137_exact_retention_only=True,no_hot_model_job_or_source_mutation=True,no_second_Day22_commit=True);put(note/'round_preservation_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
