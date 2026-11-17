import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast
P=Path(__file__).parent
Wnames=['prepare_multi16_official_910b_prompt_with_exact_preserved_nonprompt_settings_20261117.py','capture_multi16_official_update_readonly_exact_settings_and_fresh_local_book_stat_20261117.py','prepare_multi16_final_official_progress_and_all_primary_source_closure_20261117.py']
Qnames=['910b_multi16_official_update_readonly_verified_and_book_git_stat_20261117.json','910b_multi16_expected_official_prompt_20261117.txt','R_Merve_same_fresh3_resumed200_actual_progress_1304_20261117.json','R_multi16_two_card_complete_handoff_all_primary_sources_actual_20261117.json']
def enc(p):
 raw=Path(p).read_bytes();return dict(data=base64.b64encode(raw).decode(),SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
W={n:enc(P/n) for n in Wnames};Q={n:enc(P/n) for n in Qnames}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,os,subprocess,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114';sys.path[:0]=[str(R),str(R/'scripts')];W=__W__;Q=__Q__
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
pf=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before=pf.read_bytes();ptr=js(pf);H=Path(ptr['handoff_directory']);assert sha(H/'handoff.txt')=='5b72fa88e2be5662ae8b31865c4b80c089e0fa2a3566063e18f7b74d3b0e9af8' and sha(H/'facts.json')=='d27318628e0f3c82560937b6eb23bfe67cc98d2e18967f3646361a73e728ff2b';old=Path(ptr['new_primary_pending']);assert sha(old)==ptr['new_primary_pending_SHA256']=='b853eed53d253cb2aa387f3fff38dbfd3872afedd50c4eacf7b477043fdac982';pending=js(old);records=list(pending['all_current_primary_source_and_true_consumer_configs']);known={x['path']:x for x in records};note=H/'late_actual_notes/official_multi16_update_and_live_resumed200_final_source_closure_20261117';note.mkdir(parents=True,exist_ok=False)
for n,v in W.items():
 raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);p=R/'scripts/code_publication_windows'/n
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
 dest=note/'primary_source_original_bytes'/(n+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 if str(p) in known:assert known[str(p)]['SHA256']==v['SHA256']
 else:records.append(dict(path=str(p),relative=str(p.relative_to(R)),SHA256=v['SHA256'],bytes=len(raw),snapshot=str(dest)))
for n,v in Q.items():
 raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];dest=note/'local_original_bytes'/(n+'.original');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
for x in records:assert sha(x['path'])==x['SHA256'],x['path']
new=C/'pending_after_daily_20261117_multi16_two_card_official_live200_final_all_primary_v14.json';pending.update(UTC=datetime.now(timezone.utc).isoformat(),parent_pending=str(old),parent_pending_SHA256=sha(old),all_current_primary_source_and_true_consumer_configs=records,all_final_official_maintenance_sources_and_source_bytes_preserved=True,CodeDay17_already_pushed_no_second_commit=True);put(new,pending)
official=json.loads(base64.b64decode(Q['910b_multi16_official_update_readonly_verified_and_book_git_stat_20261117.json']['data']));assert official['verified'] and official['all_non_prompt_settings_and_all_keys_preserved'] and official['notification_policy_absent_and_not_reset'] and official['no_second_daily_commit']
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
J=B/'Merve_lossless_v2_completed_approved_fresh3_resumed_unstarted200_after_retention_v3_real_FLOCK_20261117';state=js(J/'state.json');done=js(J/'completion.json') if (J/'completion.json').exists() else None;trace=Path('/root/projects/comperror/workspace/diagnostics/model_runs/merve_native_vllm_slgljx7_/requests.jsonl');rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else [];index=trace.parent/'full_numeric_group_index.json';groups=js(index)['groups'] if index.exists() else [];calls=sum(js(g['path'])['actual_native_forward_calls'] for g in groups);runtime=dict(committed_samples=len(rows),committed_public_token_count=sum(len(x['generated_token_ids']) for x in rows),complete_numeric_groups=len(groups),actual_forward_calls_only_in_complete_numeric_groups=calls,error_indices=[x.get('index') for x in rows if x.get('error')],runtime_home=str(trace.parent),request_trace_SHA256=sha(trace) if trace.exists() else None)
canonical=js(H/'facts.json')['canonical_SHA256']
for p,h in canonical.items():assert sha(p)==h,p
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert npu.returncode==0
actual=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_resumed200_job=str(J),job_SHA256=sha(J/'job.json'),actual_private_state=state,actual_completion=done,actual_NPU_progress=runtime,actual_project_processes=project_processes(),actual_original_kernel_FLOCK=runner_lock_owners(),actual_npu_smi=dict(returncode=npu.returncode,stdout=npu.stdout,stderr=npu.stderr),free_project_bytes=shutil.disk_usage(R).free,official_update_readonly_verified=official,new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),primary_source_count=len(records),old948_source_bytes_preserved=True,no_new_matrix_deletion_since_exact_Dead177_170=True,canonical_SHA256=canonical,future_new_parallel_jobs_must_use_own_model_shape_capacity_not_blindly_reuse_Merve224_specialized_estimator=True)
put(note/'actual.json',actual);closing_file=H/'closing_addendum.json';raw=closing_file.read_bytes();(note/'closing_before.original').write_bytes(raw);closing=json.loads(raw);closing.update(UTC=actual['UTC'],latest_actual_snapshot_is_facts=False,latest_actual_note=str(note/'actual.json'),latest_actual_note_SHA256=sha(note/'actual.json'),new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),current_primary_source_count=len(records),official_910b_update_verified=True,official_TOML_after_SHA256=official['automation_after_SHA256'],official_prompt_after_SHA256=official['prompt_after_SHA256'],all_non_prompt_settings_and_notification_absence_preserved=True,late_actual_notes=closing.get('late_actual_notes',[])+[dict(path=str(note/'actual.json'),SHA256=sha(note/'actual.json'))]);assert closing_file.read_bytes()==raw;tmp=closing_file.with_name('closing_addendum.multi16.final.tmp');assert not tmp.exists();tmp.write_bytes(json.dumps(closing,ensure_ascii=False,indent=2).encode());os.replace(tmp,closing_file)
assert pf.read_bytes()==before;newptr=dict(ptr);newptr.update(UTC=actual['UTC'],closing_addendum_SHA256=sha(closing_file),new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),current_primary_source_count=len(records),latest_actual_note=str(note/'actual.json'),latest_actual_note_SHA256=sha(note/'actual.json'),current_same_approved_fresh3_unstarted200_home=str(J))
for key in list(newptr):
 if key in ['handoff_path','handoff_file','handoff_txt_path']:newptr[key]=str(H/'handoff.txt')
 if key in ['facts_path','facts_file']:newptr[key]=str(H/'facts.json')
 if key in ['closing_addendum_path','closing_addendum_file']:newptr[key]=str(closing_file)
 if key in ['handoff_file_SHA256','handoff_txt_SHA256']:newptr[key]=sha(H/'handoff.txt')
 if key in ['facts_file_SHA256']:newptr[key]=sha(H/'facts.json')
tmp=pf.with_name('A100_ALIGNMENT_HANDOFF_CURRENT.multi16.final.tmp');assert not tmp.exists();tmp.write_bytes(json.dumps(newptr,ensure_ascii=False,indent=2).encode());os.replace(tmp,pf)
print(json.dumps(dict(passed=True,UTC=actual['UTC'],note=str(note/'actual.json'),note_SHA256=sha(note/'actual.json'),closing_SHA256=sha(closing_file),pointer_SHA256=sha(pf),primary_source_count=len(records),new_pending=str(new),new_pending_SHA256=sha(new),actual_current_phase=state['phase'],actual_NPU_progress=runtime,actual_completion=done,actual_model_and_verifier_processes=[p for p in actual['actual_project_processes'] if p['role'] in ['model','CPU_verifier_or_audit']],actual_original_FLOCK=actual['actual_original_kernel_FLOCK'],all_original_canonical_scores_reports_sources_and_nonprompt_settings_preserved=True,no_new_matrix_deleted_since_documented170=True,pointer_keys=list(newptr)),ensure_ascii=False))
'''.replace('__W__',repr(W)).replace('__Q__',repr(Q));ast.parse(remote);(P/'archive_multi16_official_progress_final_primary_source_closure_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(prepared=True,new_useful_source_count=len(W),packet_count=len(Q),bytes=len(remote.encode()))))
