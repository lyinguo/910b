"""Bound late source/evidence handoff, preserving canonical and all live sources."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def actual():
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
 procs=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   argv=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
   if not argv or not Path(argv[0]).name.startswith('python'):continue
   if not any('meta_mllama' in s.lower() or 'metamllama' in s.lower() or 'meta-llama--llama-3.2-11b-vision-instruct-own-original' in s.lower() for s in argv):continue
   procs.append(dict(PID=int(p.name),argv=argv,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text()))
  except FileNotFoundError:continue
 homes=[B/'MetaMllama_own_original_SDPA_BF16_2048_true3_v1_real_FLOCK_20261120',B/'MetaMllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120',B/'MetaMllama_completed_genuine3_formal_NF2_exact_original_instruction_no_NN_replay_capacity_serial_20261120',B/'MetaMllama_completed_genuine3_formal_NF3_exact_original_instruction_no_NN_replay_capacity_serial_20261120',B/'MetaMllama_completed_genuine3_formal_NF3_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261120'];files={}
 for home in homes:
  for n in ['state.json','completion.json','actual_completion.json','formal_acceptance.json','own_passed3_all200_complete_shape_capacity_review.json']:
   p=home/n
   if p.is_file():files[str(p)]=dict(SHA256=sha(p),value=js(p))
 nf=any(any(Path(s).name.startswith('verify_meta_mllama') for s in p['argv'][1:]) for p in procs);nn=False;captures=[]
 for home in sorted((R/'workspace/diagnostics/model_runs').glob('MetaMllama_own_original_SDPA2048_*')):
  tr=home/'sample_execution.jsonl';rows=[json.loads(s) for s in tr.read_bytes().splitlines() if s] if tr.is_file() else [];progress=home/'generation_progress.jsonl';pr=[json.loads(s) for s in progress.read_bytes().splitlines() if s] if progress.is_file() else []
  current=any(str(home).endswith('_'+str(p['PID'])) and any(Path(s).name=='run.py' for s in p['argv'][1:]) for p in procs);nn=nn or (current and bool(rows or pr));captures.append(dict(home=str(home),completed_samples=len(rows),committed_native_forwards=sum(v['generation_count'] for v in rows),current_live_model_PID_matches_capture=current,last_actual_native_forward_progress=pr[-1] if pr else None))
 return dict(UTC=datetime.now(timezone.utc).isoformat(),actual_processes=procs,actual_original_FLOCK=runner_lock_owners(),current_model_NN_active=nn,current_formal_fullNF_active=nf,current_model_NN_or_NF_active=nn or nf,captures=captures,stage_files=files,free_bytes=shutil.disk_usage(R).free,max_concurrent_models=1,original_true3_NN_not_replayed=True)
def main(plan_path,expected):
 assert sha(plan_path)==expected;plan=js(plan_path);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';oldraw=pointer.read_bytes();assert hashlib.sha256(oldraw).hexdigest()==plan['expected_pointer_SHA256'];ptr=json.loads(oldraw);H=Path(ptr['handoff_directory']);assert sha(H/'handoff.txt')=='19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1'
 canon={n:sha(n) for n in ptr['canonical_SHA256']};assert canon==ptr['canonical_SHA256'] and canon[str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')]=='4ba1a4c2a3171fa17b1e4fa9e997addf6d40e2210eaebf92705d84b712f728d3'
 parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256'];pp=js(parent);rows={v['path']:v for v in pp['all_current_primary_source_and_true_consumer_configs']}
 for n,v in rows.items():assert sha(n)==v['SHA256'],n
 note=H/'late_actual_notes'/plan['late_note_name'];note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(oldraw);(note/'previous_closing_addendum.json').write_bytes((H/'closing_addendum.json').read_bytes())
 additions={v['relative']:v for v in plan['new_primary_sources']}
 homes=[B/'MetaMllama_own_original_SDPA_BF16_2048_true3_v1_real_FLOCK_20261120',B/'MetaMllama_completed_genuine3_formal_NF2_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261120',B/'MetaMllama_completed_genuine3_formal_NF3_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261120']
 for home in homes:
  if not (home/'job.json').is_file():continue
  job=js(home/'job.json')
  for n,h in job['spec']['source_sha256'].items():additions[n]=dict(relative=n,SHA256=h,role='actual Meta own job source/YAML/protocol/verifier/dependency')
  c=R/job['spec']['config'];smoke=c.with_name(c.stem+'-smoke'+c.suffix)
  if smoke.is_file():additions[str(smoke.relative_to(R))]=dict(relative=str(smoke.relative_to(R)),SHA256=sha(smoke),role='actual frozen runner generated own smoke configuration')
 for rel,v in additions.items():
  p=R/rel;assert p.is_relative_to(R) and rel.startswith(('scripts/','configs/','universal_eval/')) and p.suffix in ['.py','.json','.yaml','.yml','.mjs','.md'] and sha(p)==v['SHA256'];snap=note/'primary_source_snapshots'/rel;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=rel,SHA256=sha(p),bytes=p.stat().st_size,snapshot=str(snap),role=v['role'])
 for v in plan.get('original_records',[]):
  p=Path(v['path']);assert sha(p)==v['SHA256'];dest=note/'local_original_records'/v['name'];dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
 current=actual();put(note/'latest_actual.json',current)
 declarations=dict(UTC=current['UTC'],latest_model_id='meta-llama/Llama-3.2-11B-Vision-Instruct',own906_BF16_and_5SF_original_source_proven=True,original_saved_YAML_wrong_other_brand_root_preserved_new_YAML_own_root_only=True,original_HF_2048_outer10_inner1_default_SDPA_doubleBOS_and_threeEOS_preserved=True,genuine_true3_samples=3,genuine_true3_native_forwards=11,genuine_NPU_devices=[4,5],genuine_report='/root/projects/comperror/workspace/reports/meta-llama_Llama-3.2-11B-Vision-Instruct/image-text-to-text/20261120_121105',old_NF1_failed_exact_kwargs_instruction_guard_preserved=True,NF2_failed_reciprocal_sqrt_reference_vs_original_pow_exact_bit_difference_proven=True,NF3_only_original_power_reference_restored_no_threshold_relaxation_or_model_math_change=True,all_original_reports_sources_configs_logs_small_failed_and_unverified_values_retained=True,actual_NEW_model_forwards_in_formal_NF2_NF3_acceptances=0,current_verified_counts=ptr['canonical_counts'],exact271_cleanup_already_completed_and_ledger_4ba1_preserved_no_further_deletion=True,pruned271_and_ancestors_arrays_cannot_now_be_fully_reverified_original_proofs_retained=True,Day20_already_pushed='be09d9a8b637a6dd54f9be65db8609e89af92f64',no_second_Day20_commit=True,all_original_constraints_families_failures_user_decisions_and_precise_retention_exceptions_inherited=True,current_actual=current)
 put(note/'full_actual_declarations.json',declarations)
 pending=C/plan['pending_name'];assert not pending.exists();put(pending,dict(UTC=current['UTC'],parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-20',no_second_same_day_commit=True,next_daily_review='Day21 all useful new actual source/config/dependencies/Windows controls and full diff/outgoing/tests/planSHA, one normal push',handoff_directory=str(H)))
 formal_home=B/'MetaMllama_completed_genuine3_formal_NF3_exact_original_instruction_no_NN_replay_capacity_serial_20261120'
 stop=None
 if not current['current_model_NN_or_NF_active']:
  if (formal_home/'actual_completion.json').is_file():stop=datetime.fromisoformat(js(formal_home/'actual_completion.json')['UTC']).astimezone(timezone(timedelta(hours=8))).isoformat()
  elif (formal_home/'formal_acceptance.json').is_file():stop=datetime.fromisoformat(js(formal_home/'formal_acceptance.json')['UTC']).astimezone(timezone(timedelta(hours=8))).isoformat()
 close=js(H/'closing_addendum.json');close.update(UTC=current['UTC'],latest_actual_note=str(note/'full_actual_declarations.json'),latest_actual_note_SHA256=sha(note/'full_actual_declarations.json'),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_source_count=len(rows),current_model_NN_or_NF_active=current['current_model_NN_or_NF_active'],current_formal_NF3_home=str(formal_home),current_Meta_genuine3_report=declarations['genuine_report'],all_failed_NF_and_current_math_preserved=True,first_proven_current_formal_stoppage_BJT=stop)
 assert pointer.read_bytes()==oldraw and {n:sha(n) for n in canon}==canon
 for n,v in rows.items():assert sha(n)==v['SHA256'],n
 put(H/'closing_addendum.json',close);ptr.update(UTC=current['UTC'],closing_addendum_SHA256=sha(H/'closing_addendum.json'),new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),current_primary_source_count=len(rows),latest_actual_note=str(note/'full_actual_declarations.json'),latest_actual_note_SHA256=sha(note/'full_actual_declarations.json'),latest_actual_snapshot=str(note/'latest_actual.json'),latest_actual_snapshot_SHA256=sha(note/'latest_actual.json'),current_actual_job=str(homes[-1] if (homes[-1]/'job.json').exists() else B/'MetaMllama_completed_genuine3_formal_NF3_exact_original_instruction_no_NN_replay_capacity_serial_20261120'),current_actual_model_process_or_NF_active=bool(current['actual_processes']),current_actual_model_NN_or_NF_active=current['current_model_NN_or_NF_active'],current_actual_stage='Meta_original_SDPA2048_genuine3_completed_same_values_formal_NF3_then_capacity',current_formal_acceptance_home=str(B/'MetaMllama_completed_genuine3_formal_NF3_exact_original_instruction_no_NN_replay_capacity_serial_20261120'),severity_first_line=None if current['current_model_NN_or_NF_active'] else '严重阻塞｜正式评测未运行')
 ptr.update(current_actual_controller=str(formal_home),first_proven_stoppage_BJT=stop,current_current_owned_candidate='meta-llama/Llama-3.2-11B-Vision-Instruct',current_owned_candidate='meta-llama/Llama-3.2-11B-Vision-Instruct',current_formal_model_stage='same_genuine_own_Meta_true3_formal_fullNF3_then_complete_capacity',current_meta_source_and_capture_preserved=True)
 put(pointer,ptr);result=dict(passed=True,pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),late_actual=str(note/'full_actual_declarations.json'),late_actual_SHA256=sha(note/'full_actual_declarations.json'),canonical_and_model_sources_unchanged=True);put(note/'closing_actual_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2])
