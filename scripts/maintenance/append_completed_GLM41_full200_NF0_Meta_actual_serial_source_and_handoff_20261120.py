"""Close actual GLM200 acceptance and current Meta serial evidence incrementally."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,base64
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
M=B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_actual_capacity_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261120'
G=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan)
 pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';old=pointer.read_bytes();assert hashlib.sha256(old).hexdigest()==q['expected_pointer_SHA256'];ptr=json.loads(old);H=Path(ptr['handoff_directory'])
 assert sha(H/'handoff.txt')=='19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1'
 canon=ptr['canonical_SHA256'];assert all(sha(p)==h for p,h in canon.items())
 parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256']==q['expected_parent_pending_SHA256']
 rows={x['path']:x for x in js(parent)['all_current_primary_source_and_true_consumer_configs']}
 assert all(sha(p)==v['SHA256'] for p,v in rows.items())
 note=H/'late_actual_notes'/q['late_note_name'];note.mkdir(exist_ok=False)
 (note/'previous_pointer.json').write_bytes(old);(note/'previous_closing_addendum.json').write_bytes((H/'closing_addendum.json').read_bytes())
 additions={x['relative']:x for x in q['new_primary_sources']}
 selfrel=str(Path(__file__).resolve().relative_to(R));additions[selfrel]=dict(relative=selfrel,SHA256=sha(__file__),role='actual completed GLM acceptance / current Meta incremental source and handoff closure')
 assert sha(M/'job.json')=='581c7c7c9a4f6a270f5c9021ee9c6824144dc7c437473b7eb3ab81fce6141757'
 for rel,h in js(M/'job.json')['spec']['source_sha256'].items():
  if str(R/rel) not in rows or rows[str(R/rel)]['SHA256']!=h:
   additions[rel]=dict(relative=rel,SHA256=h,role='actual current Meta approved fresh3/fixed200 source or generated YAML/protocol/smoke configuration')
 for rel,v in additions.items():
  p=R/rel;assert p.resolve().is_relative_to(R) and '..' not in Path(rel).parts and (rel=='run.py' or rel.startswith(('scripts/','configs/','universal_eval/'))) and p.suffix in ['.py','.json','.yaml','.yml','.mjs','.md'] and sha(p)==v['SHA256']
  if str(p) in rows:assert rows[str(p)]['SHA256']==v['SHA256'];continue
  snap=note/'primary_source_snapshots'/rel;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(p.read_bytes())
  rows[str(p)]=dict(path=str(p),relative=rel,SHA256=v['SHA256'],bytes=p.stat().st_size,snapshot=str(snap),role=v['role'])
 for v in q['original_records']:
  p=Path(v['path']);assert sha(p)==v['SHA256'];dest=note/'original_records'/v['name'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
 for v in q.get('local_original_records',[]):
  b=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==v['SHA256'];dest=note/'local_original_records'/v['name'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 reader=R/'scripts/maintenance/read_MetaMllama_current_serial_fixed200_and_GLM41_completed200_20261120.py'
 assert sha(reader)=='3349cda45e4bb0dc3632d7faaaff9dd180479f2ea9317e1d7744951158b33100'
 run=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert run.returncode==0,run.stderr.decode();actual=json.loads(run.stdout)
 active=[x for x in actual['Meta_captures'] if x['current_model_PID_matches_capture']]
 assert len(active)==1
 processes=actual['actual_processes'];owners=actual['original_FLOCK'];pids={v['PID']:v for v in processes}
 matching=[pids[v['PID']] for v in owners if v['PID'] in pids and str(M) in pids[v['PID']]['argv'] and any(Path(a).name=='A100_alignment_evidence_runner_20261103.py' for a in pids[v['PID']]['argv'])]
 assert len(matching)==1 and len([v for v in processes if v['PPid']==matching[0]['PID'] and any(Path(a).name=='run.py' for a in v['argv'])])==1
 done=js(G/'completion.json');assert done['actual_returncode']==done['verifier_returncode']==0 and done['model_returncode']==1
 proof_path=Path(done['accepted'][-1]['proof']);assert sha(proof_path)=='af10ad5191779c0732038c95b369fa6cef27ec2478e813eafa59f7c87c97c101';proof=js(proof_path)
 assert proof['passed'] and proof['sample_count']==200 and proof['metrics']['Accuracy']==.555 and proof['actual_forward_count']==55771 and proof['original_protocol_final_qualification_complete'] is False
 light={k:v for k,v in actual.items() if k!='files'}
 light['actual_current_file_metadata']={k:{n:w for n,w in v.items() if n in ['SHA256','bytes','exists']} for k,v in actual['files'].items()}
 put(note/'latest_Meta_actual_and_completed_GLM41.json',light)
 assert sum(v['status']=='done' for v in js(D/'state.json')['models'].values())==183
 full=q['full_handoff_text'].replace('ACTUAL_UTC',actual['UTC']).replace('CURRENT_META_PROGRESS',json.dumps(active,ensure_ascii=False)).replace('CURRENT_META_RUNNER_BINDING',json.dumps(matching,ensure_ascii=False)).replace('CURRENT_DISK_FREE_BYTES',str(actual['free_bytes']))
 fullpath=note/'full_actual_GLM41_full200_NF0_gap26_Meta_same_serial_fixed200_results_logs_preserved_20261120.txt';fullpath.write_text(full,encoding='utf-8')
 pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=actual['UTC'],parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-20',no_second_same_day_commit=True,next_daily_review='Day21 complete useful sources/configs/dependencies/concurrent SHA/diff/outgoing/tests/publisher consumed planSHA then exact normal push',handoff_directory=str(H)))
 stage='MetaMllama_same_original_approved_fixed200_NPU4_5_serial_model_NN'
 updates=dict(UTC=actual['UTC'],latest_actual_note=str(fullpath),latest_actual_note_SHA256=sha(fullpath),latest_full_actual_summary=str(fullpath),latest_full_actual_summary_SHA256=sha(fullpath),latest_actual_snapshot=str(note/'latest_Meta_actual_and_completed_GLM41.json'),latest_actual_snapshot_SHA256=sha(note/'latest_Meta_actual_and_completed_GLM41.json'),current_source_count=len(rows),current_primary_source_count=len(rows),current_model_NN_or_NF_active=True,current_actual_model_NN_or_NF_active=True,current_actual_model_process_or_NF_active=True,current_NN_or_NF_active=True,current_actual_stage=stage,current_formal_model_stage=stage,current_owned_candidate='meta-llama/Llama-3.2-11B-Vision-Instruct',current_current_owned_candidate='meta-llama/Llama-3.2-11B-Vision-Instruct',current_actual_job=str(M),current_actual_controller=str(B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_wait_actual_capacity_first200_serial_20261120'),current_formal_acceptance_home=str(M/'formal200'),current_formal_acceptance_completed=False,current_formal200_progress=active,next_no_NN_original_capacity_waiter=None,last_complete_result_BJT='2026-11-20 19:35:15.867201+08:00',last_complete_qualified_result_BJT='2026-11-20 06:06:06.525367+08:00',last_complete_qualified_dual_audit_BJT='2026-11-20 10:24:10+08:00',latest_complete_NPU200_measurement=dict(model_id='zai-org/GLM-4.1V-9B-Thinking',sample_count=200,actual_native_forwards=55771,Accuracy=.555,three_D_relative_error=dict(server='0.26',local='0.26',preserved_original='0.26'),proof=str(proof_path),proof_SHA256=sha(proof_path),full_NF_passed=True,original_protocol_qualified=False,no_G_written=True),severity_first_line=None,first_proven_stoppage_BJT=None,first_proven_current_formal_stoppage_BJT=None,new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_user_preservation_receipt=q['original_records'][0]['path'],current_user_preservation_receipt_SHA256=q['original_records'][0]['SHA256'],this_round_no_evidence_deleted=True)
 close=js(H/'closing_addendum.json');close.update(updates)
 assert pointer.read_bytes()==old and all(sha(p)==h for p,h in canon.items()) and all(sha(p)==v['SHA256'] for p,v in rows.items())
 put(H/'closing_addendum.json',close);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(H/'closing_addendum.json');put(pointer,ptr)
 result=dict(passed=True,pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),source_count=len(rows),pending=str(pending),pending_SHA256=sha(pending),full_actual_note=str(fullpath),full_actual_note_SHA256=sha(fullpath),current_Meta_progress=active,canonical_and_all_live_model_sources_unchanged=True)
 put(note/'closing_actual_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
