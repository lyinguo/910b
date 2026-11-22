"""Preserve immutable native Xtuner jobs and all useful primary sources after qualified185."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
now=lambda:datetime.now(timezone.utc).isoformat()
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def main():
 plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before=pointer.read_bytes();assert hashlib.sha256(before).hexdigest()==q['expected_pointer_SHA256'];ptr=json.loads(before);H=Path(ptr['handoff_directory']);closing_file=H/'closing_addendum.json';closing_before=closing_file.read_bytes();assert hashlib.sha256(closing_before).hexdigest()==q['expected_closing_SHA256'];parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256']==q['expected_pending_SHA256']
 canonical=ptr['canonical_SHA256'];assert all(sha(n)==h for n,h in canonical.items());assert ptr['current_verified_count']==185 and ptr['current_retention_ledger_SHA256']=='5f20ac2b35d58b09e5c4a3a8ca6a5dacf1446b9cb39876678b24458b0f1ef4b0'
 rows={r['path']:r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']};assert len(rows)==2751 and all(sha(n)==r['SHA256'] for n,r in rows.items())
 note=H/'late_actual_notes'/q['round_name'];note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(before);(note/'previous_closing_addendum.json').write_bytes(closing_before)
 for r in q['local_original_records']:
  raw=base64.b64decode(r['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==r['SHA256'];dest=note/'local_original_records'/r['name'];assert dest.resolve().is_relative_to(note.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 M=R/'scripts/A100_alignment_incremental_monitor_20261103.py';assert sha(M)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
 mr=subprocess.run(['python3','-B',str(M)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert mr.returncode==0
 reader=R/'scripts/maintenance/read_XtunerLlava3_private_IModel_v2_fullNF_serial_current_20261122_remote.py';r=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr;current=json.loads(r.stdout);assert current['npu_smi']['returncode']==0
 paths=list(q['new_primary_paths'])+[str(Path(__file__).relative_to(R))]
 runtime_jobs=[]
 for stem in ['XtunerLlava3_own_original_Transformers_FP16_2048_true3_v1_real_FLOCK_20261122','XtunerLlava3_own_original_Transformers_FP16_2048_true3_v2_IModel_real_FLOCK_20261122','XtunerLlava3_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v2_IModel_real_FLOCK_20261122']:
  j=B/stem/'job.json'
  if j.exists():
   job=js(j);paths+=list(job['spec']['source_sha256']);assert all(sha(R/n)==h for n,h in job['spec']['source_sha256'].items());runtime_jobs.append(dict(job=str(j),SHA256=sha(j),source_SHA256=job['spec']['source_sha256']))
   for variant in [Path(job['spec']['config']),Path(job['spec']['config']).with_stem(Path(job['spec']['config']).stem+'-smoke')]:
    if (R/variant).exists():paths.append(str(variant))
 for n in sorted(set(paths)):
  p=R/n;assert p.is_file() and p.resolve().is_relative_to(R) and (n=='run.py' or n.startswith(('scripts/','configs/','universal_eval/','tests/')));h=sha(p)
  if str(p) in rows:assert rows[str(p)]['SHA256']==h,n
  else:
   dest=note/'primary_source_snapshots'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=n,SHA256=h,bytes=p.stat().st_size,snapshot=str(dest),role='Own Xtuner full model/NF/interfaces/private source-bound serial continuation or actual YAML/protocol consumer/Windows control/readback/preservation')
 processes=current['actual_processes'];model=[p for p in processes if any(Path(a).name=='run.py' for a in p['argv'])];nf=[p for p in processes if any(Path(a).name.startswith('verify_') for a in p['argv'])];forwards=sum(x['total_committed_native_forwards'] for x in current['captures']);active=bool(nf or (model and forwards))
 # Root-cwd frozen runner is deliberately recovered from the real kernel lock owner.
 extra=[]
 for owner in current['original_FLOCK']:
  p=Path('/proc')/str(owner['PID'])
  try:extra.append(dict(owner=owner,argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x],cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),status=(p/'status').read_text(),fd={f.name:os.readlink(f) for f in (p/'fd').iterdir()}))
  except OSError as e:extra.append(dict(owner=owner,inspection_race=str(e)))
 monitor={str(p):js(p) for p in [D/'incremental_monitor/latest.json',D/'incremental_monitor/checkpoint.json']};assert not monitor[str(D/'incremental_monitor/latest.json')]['source_errors']
 actual=dict(UTC=now(),Xtuner_actual=current,actual_kernel_lock_owners_full_proc=extra,actual_runtime_jobs=runtime_jobs,current_model_process_active=bool(model),current_full_numerical_verifier_active=bool(nf),current_genuine_committed_native_forwards=forwards,current_model_NN_or_NF_active=active,current_loading_without_committed_NN=bool(model and not forwards),canonical_SHA256=canonical,canonical_counts=dict(done=185,numeric_within5pct=170,numeric_outside5pct=15,complete_weights_pending_formal=205,skipped_no_complete_weights=310,original_full_qualified=71),last_qualified_full200_NF_completed_Beijing='2026-11-22T03:34:24.274295+08:00',last_qualified_dual_audit_Beijing='2026-11-22T03:47:22.926705+08:00',monitor=dict(returncode=mr.returncode,stdout=mr.stdout,stderr=mr.stderr,complete_latest_checkpoint=monitor),all_canonical_unchanged=True,all_active_model_job_config_deadline_signal_untouched=True)
 actualpath=note/'current_Xtuner_v2_native_fullNF_real_serial_actual.json';put(actualpath,actual)
 compact={k:actual[k] for k in ['UTC','current_model_process_active','current_full_numerical_verifier_active','current_genuine_committed_native_forwards','current_model_NN_or_NF_active','current_loading_without_committed_NN','canonical_counts']};compact.update(controller=current['controller'],job=current['job'],captures=[{k:x[k] for k in ['home','committed_step_counts','total_committed_native_forwards','last_committed_step','pending_partial_line']} for x in current['captures']],actual_kernel_lock_owners_full_proc=extra,actual_free_bytes=current['free_bytes'],full_actual_snapshot=str(actualpath),full_actual_snapshot_SHA256=sha(actualpath))
 full=q['full_handoff_template'].replace('CURRENT_SOURCE_COUNT',str(len(rows))).replace('CURRENT_ACTUAL_JSON',json.dumps(compact,ensure_ascii=False));fp=note/q['full_note_name'];fp.write_text(full,encoding='utf8')
 pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=now(),parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-22',Code_Day_commit='43a149bb69510288451c2348159d6e7baa0d8d8e',no_second_same_day_commit=True,next_daily_review='Day23 all own source, actual configs/consumers/dependencies/tests/Windows controls; fresh SHA/full diff/all outgoing/checks/publisher planSHA before one normal push, no empty commit'))
 updates=dict(UTC=now(),new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_primary_source_count=len(rows),current_source_count=len(rows),latest_actual_note=str(fp),latest_actual_note_SHA256=sha(fp),latest_full_actual_summary=str(fp),latest_full_actual_summary_SHA256=sha(fp),latest_actual_snapshot=str(actualpath),latest_actual_snapshot_SHA256=sha(actualpath),current_actual_stage='verified185_delivered_Xtuner_own_native_interface_v2_serial_actual',current_formal_model_stage='Xtuner own immutable v2 IModel recovery after zero-forward v1 rejection; real committed NN/fullNF status in latest snapshot; no old job restart',current_model_process_active=bool(model),current_full_numerical_verifier_active=bool(nf),current_model_NN_or_NF_active=active,current_actual_model_NN_or_NF_active=active,current_NN_or_NF_active=active,current_actual_model_process_or_NF_active=active,current_genuine_committed_native_forwards=forwards,current_stop_earliest_proven_Beijing=None if active else '2026-11-22T03:52:27.024058+08:00',canonical_SHA256=canonical)
 assert pointer.read_bytes()==before and closing_file.read_bytes()==closing_before and all(sha(n)==h for n,h in canonical.items()) and all(sha(n)==r['SHA256'] for n,r in rows.items())
 closing=json.loads(closing_before);closing.update(updates);put(closing_file,closing);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(closing_file);put(pointer,ptr)
 result=dict(passed=True,UTC=now(),pointer_SHA256=sha(pointer),closing_SHA256=sha(closing_file),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),full_actual_note=str(fp),full_actual_note_SHA256=sha(fp),actual_snapshot=str(actualpath),actual_snapshot_SHA256=sha(actualpath),current_model_NN_or_NF_active=active,current_committed_native_forwards=forwards,no_active_job_or_source_mutation=True,no_canonical_mutation=True,no_second_Day22_commit=True);put(note/'round_preservation_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
