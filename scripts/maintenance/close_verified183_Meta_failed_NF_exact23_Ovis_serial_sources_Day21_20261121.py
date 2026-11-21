"""Incremental handoff and source archive for exact23 cleanup, failed Meta NF and live Ovis."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,base64
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';old=pointer.read_bytes();assert hashlib.sha256(old).hexdigest()==q['expected_pointer_SHA256'];ptr=json.loads(old);H=Path(ptr['handoff_directory'])
 priorcanon=ptr['canonical_SHA256'];canon={p:sha(p) for p in priorcanon};LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json';changed={p for p,h in priorcanon.items() if canon[p]!=h};assert changed=={str(LED)}
 ret=B/'verified183_GLM41_full_numeric_gap_correct_nonexample_minimal_Ovis_capacity_recovery_20261121'
 assert sha(ret/'actual_completion.json')=='15e6800acd2a727af9465787cbb61e76e3257e3e01b2bee886f430a9033f5e5c'
 receipt=js(ret/'actual_completion.json');assert receipt['actual_returncode']==0 and receipt['removed_files']==23 and receipt['protected_files']==11491 and receipt['all_protected_SHA256_unchanged'] and receipt['manifest_SHA256']=='01116dda5db931e22acf9308432cba15e07018fac20c31412f29661a8fc14e3b'
 assert sha(ret/'ledger_before.original')==priorcanon[str(LED)]=='4ba1a4c2a3171fa17b1e4fa9e997addf6d40e2210eaebf92705d84b712f728d3' and canon[str(LED)]=='6f4dcf0be1dad1b0e4a1e8ec53725d76d6abff0965f5fdf87a9428f772b7179c'
 before=js(ret/'ledger_before.original');after=js(LED);assert after['events'][:-1]==before['events'] and len(after['events'])==len(before['events'])+1 and after['events'][-1]['receipt_SHA256']==sha(ret/'actual_completion.json')
 parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256']==q['expected_parent_pending_SHA256'];rows={v['path']:v for v in js(parent)['all_current_primary_source_and_true_consumer_configs']};assert all(sha(p)==v['SHA256'] for p,v in rows.items())
 note=H/'late_actual_notes'/q['late_note_name'];note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(old);(note/'previous_closing_addendum.json').write_bytes((H/'closing_addendum.json').read_bytes())
 additions={v['relative']:v for v in q['primary_sources']};selfrel=str(Path(__file__).resolve().relative_to(R));additions[selfrel]=dict(relative=selfrel,SHA256=sha(__file__),role='actual incremental source/retention/handoff consumer')
 J=B/'OvisU1_completed_storage_fullNF_passed_exact23_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261121';job=js(J/'job.json');assert sha(J/'job.json')=='11f0c684e4dfe6f39bff484c2386dca3e7371f778230faa6ca2a65ca64affc7d'
 for rel,h in job['spec']['source_sha256'].items():
  if str(R/rel) not in rows:additions[rel]=dict(relative=rel,SHA256=h,role='actual own Ovis source or runtime-approved YAML/protocol/smoke configuration')
 for rel,v in additions.items():
  p=R/rel;assert p.resolve().is_relative_to(R) and (rel=='run.py' or rel.startswith(('scripts/','configs/','universal_eval/'))) and p.suffix in ['.py','.json','.yaml','.yml','.mjs','.md'] and sha(p)==v['SHA256']
  if str(p) in rows:assert rows[str(p)]['SHA256']==v['SHA256'];continue
  dest=note/'primary_source_snapshots'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=rel,SHA256=v['SHA256'],bytes=p.stat().st_size,snapshot=str(dest),role=v['role'])
 for r in q['local_original_records']:
  b=base64.b64decode(r['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==r['SHA256'];dest=note/'local_original_records'/r['name'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 reader=R/'scripts/maintenance/read_OvisU1_current_serial_v2_and_preserved_Meta_failed200_20261121.py'
 r=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr.decode();actual=json.loads(r.stdout)
 procs=actual['actual_processes'];nn=[p for p in procs if any(Path(a).name=='run.py' for a in p['argv'])];nf=[p for p in procs if any('verify_ovisu1_' in a for a in p['argv'])];assert nn or nf
 active=[v for v in actual['captures'] if v['model_PID_matches_capture']]
 light={k:v for k,v in actual.items() if k!='files'};light['actual_file_metadata']={k:{n:w for n,w in v.items() if n in ['SHA256','bytes','exists']} for k,v in actual['files'].items()};put(note/'current_Ovis_and_original_locks_actual.json',light)
 full=q['full_handoff_text'].replace('CURRENT_ACTUAL_UTC',actual['UTC']).replace('CURRENT_OVIS_PROGRESS',json.dumps(active,ensure_ascii=False)).replace('CURRENT_ACTUAL_PROCESSES',json.dumps(procs,ensure_ascii=False)).replace('CURRENT_FREE_BYTES',str(actual['free_bytes']))
 fp=note/'full_actual_verified183_Meta_NF_failed_exact23_retention_Ovis_serial_live_Day21_20261121.txt';fp.write_text(full,encoding='utf-8')
 pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=actual['UTC'],parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[p] for p in sorted(rows)],Code_Day_already_pushed='2026-11-20',next_daily_review='Day21 all useful primary source/actual configs/dependencies/concurrentSHA/full diff/outgoing/tests/publisher consumed planSHA',handoff_directory=str(H)))
 stage='OvisU1_approved_NEWfresh3_or_fixed200_original_serial_NN' if nn else 'OvisU1_same_original_stage_independent_fullNF'
 updates=dict(UTC=actual['UTC'],canonical_SHA256=canon,current_canonical_SHA256=canon,new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_source_count=len(rows),current_primary_source_count=len(rows),latest_actual_note=str(fp),latest_actual_note_SHA256=sha(fp),latest_full_actual_summary=str(fp),latest_full_actual_summary_SHA256=sha(fp),latest_actual_snapshot=str(note/'current_Ovis_and_original_locks_actual.json'),latest_actual_snapshot_SHA256=sha(note/'current_Ovis_and_original_locks_actual.json'),current_owned_candidate='AIDC-AI/Ovis-U1-3B',current_current_owned_candidate='AIDC-AI/Ovis-U1-3B',current_actual_job=str(J),current_actual_controller=str(B/'OvisU1_completed_storage_fullNF_passed_after_exact_GLM41_23_capacity_serial_v2_20261121'),current_actual_stage=stage,current_formal_model_stage=stage,current_formal_acceptance_home=str(J/'formal200'),current_formal_acceptance_completed=False,current_actual_model_NN_or_NF_active=bool(nn or nf),current_model_NN_or_NF_active=bool(nn or nf),current_NN_or_NF_active=bool(nn or nf),current_actual_model_process_or_NF_active=bool(nn or nf),current_Ovis_progress=active,current_retention_ledger_SHA256=canon[str(LED)],latest_exact23_retention_receipt=str(ret/'actual_completion.json'),latest_exact23_retention_receipt_SHA256=sha(ret/'actual_completion.json'),latest_exact23_retention_manifest_SHA256=receipt['manifest_SHA256'],first_proven_stoppage_BJT=None,first_proven_current_formal_stoppage_BJT=None,severity_first_line=None,current_actual_recovery_BJT='2026-11-21 09:24:03+08:00',Meta_formal200_original_NF_failed=True,Meta_original_NF_failure_BJT='2026-11-21 04:07:38.308429+08:00',Meta_original200_report_score=.705,Meta_three_D_relative_error='0.09032258064516129032258064516',Meta_NF4_correct_metadata_schema_waiter_not_NN=True)
 close=js(H/'closing_addendum.json');close.update(updates)
 assert pointer.read_bytes()==old and all(sha(p)==h for p,h in canon.items()) and all(sha(p)==v['SHA256'] for p,v in rows.items())
 put(H/'closing_addendum.json',close);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(H/'closing_addendum.json');put(pointer,ptr)
 out=dict(passed=True,pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),full_actual_note=str(fp),full_actual_note_SHA256=sha(fp),current_Ovis_progress=active,canonical_scores_state_registry_binding_unchanged=True,only_exact_authorized_retention_ledger_append=True)
 put(note/'closing_actual_receipt.json',out);print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
