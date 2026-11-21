"""Close a reviewed handoff with exact official readback and fresh process facts only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import argparse,base64,hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)
def execute(path,expected):
 assert sha(path)==expected;plan=js(path);H=Path(plan['handoff_directory']).resolve();assert H.is_relative_to((D/'heartbeat_handoffs').resolve())
 pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';raw_pointer=pointer.read_bytes();assert hashlib.sha256(raw_pointer).hexdigest()==plan['expected_pointer_SHA256'];ptr=json.loads(raw_pointer);assert ptr['handoff_directory']==str(H)
 assert sha(H/'handoff.txt')==ptr['handoff_SHA256'] and sha(H/'facts.json')==ptr['facts_SHA256']
 for n,h in plan['canonical_SHA256'].items():assert sha(n)==h,n
 note=H/'late_actual_notes'/plan['note_name'];note.mkdir(parents=True,exist_ok=False)
 for item in plan['local_original_packets']:
  data=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(data).hexdigest()==item['SHA256'];target=note/'local_original_packets'/item['name'];assert target.resolve().is_relative_to(note.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 before=plan['official_readback']['before_preserved_fields'];after=plan['official_readback']['after_preserved_fields'];assert before==after
 assert after['id']=='910b' and after['status']=='ACTIVE' and after['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300' and after['target_thread_id']=='01a18bbd-8040-7d02-9004-beef491d836a'
 assert plan['official_readback']['after_prompt_SHA256']==plan['official_readback']['requested_prompt_normalized_SHA256']
 oldpending=Path(ptr['new_primary_pending']);assert sha(oldpending)==ptr['new_primary_pending_SHA256'];pending=js(oldpending);rows={v['path']:v for v in pending['all_current_primary_source_and_true_consumer_configs']}
 policy=str(R/'configs/publication/code_only.json');transition=plan['reviewed_Day20_publication_transition']
 receipt=Path(transition['receipt']);assert sha(receipt)==transition['receipt_SHA256']=='49bf894593e1c8bc82040382f9168dbcc53dd3d33c7245a35135e6092f127337';dr=js(receipt)
 assert dr['passed'] and dr['commit']=='be09d9a8b637a6dd54f9be65db8609e89af92f64' and dr['Code_Day']=='2026-11-20' and dr['explicit_paths']==2339 and dr['changed_files']==319
 assert rows[policy]['SHA256']=='b6974eebfc0b50bd7d817eafba8f23cc516960c1db854dd1f6d7421d927c057b' and sha(policy)=='7c60205d2f555e5502d50594b388154558a81e22d2e8002bc47244f20f09a550'
 target=note/'primary_source_snapshots/configs/publication/code_only.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(policy).read_bytes())
 rows[policy]=dict(rows[policy],SHA256=sha(policy),bytes=Path(policy).stat().st_size,snapshot=str(target),reviewed_Day20_exact_transition_receipt=str(receipt))
 ledger=str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json');cr=Path(plan['reviewed_exact_GLM154_cleanup_receipt']);assert sha(cr)=='1003b1303c34a398c3dab6513cd37d6c365f0814823befb2ea798ec2425f2fb5';cleanup=js(cr)
 assert cleanup['actual_returncode']==0 and cleanup['removed_files']==154 and cleanup['removed_bytes']==22050156035 and cleanup['all_protected_SHA256_unchanged'] and cleanup['all10_wrong_and_full_original_outer_groups_retained']
 assert ptr['canonical_SHA256'][ledger]=='7978cbde39e32b7d28ecb05cf69148e451cbe262493b39d6b07e6fdb785b9b71' and plan['canonical_SHA256'][ledger]==sha(ledger)=='06f870990721bdb4e58a409f842980b7b4578eddc23367263d041c052f8e43ce'
 assert js(ledger)['events'][-1]['receipt_SHA256']==sha(cr)
 assert all(plan['canonical_SHA256'][n]==h for n,h in ptr['canonical_SHA256'].items() if n!=ledger)
 for n,v in rows.items():assert sha(n)==v['SHA256'],n
 for v in plan.get('additional_existing_primary_sources',[]):
  p=(R/v['relative']).resolve();assert p.is_relative_to(R) and sha(p)==v['SHA256'];data=p.read_bytes();snapshot=note/'primary_source_snapshots'/v['relative'];snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(data);rows[str(p)]=dict(path=str(p),relative=v['relative'],SHA256=v['SHA256'],bytes=len(data),snapshot=str(snapshot),role=v['role'])
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 procs=project_processes();owners=runner_lock_owners();details={}
 for row in procs:
  if row.get('role') not in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier'] and plan['process_keyword'].lower() not in str(row).lower():continue
  pid=row['PID'];proc=Path('/proc')/str(pid)
  try:details[pid]=dict(row,stat=(proc/'stat').read_text(),actual_argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(),actual_cwd=os.readlink(proc/'cwd'),io=(proc/'io').read_text())
  except OSError as e:details[pid]=dict(row,read_error=str(e))
 states={}
 for name,p in plan['current_stage_files'].items():
  p=Path(p)
  if not p.is_file():states[name]=dict(path=str(p),exists=False);continue
  data=p.read_bytes();rec=dict(path=str(p),bytes=len(data),SHA256=hashlib.sha256(data).hexdigest(),exists=True)
  try:rec['value']=json.loads(data)
  except ValueError:rec.update(nonfinal_or_text=True,raw_text=data.decode('utf8',errors='replace')[-20000:])
  states[name]=rec
 fresh=dict(UTC=datetime.now(timezone.utc).isoformat(),official_readback=plan['official_readback'],actual_project_processes=procs,actual_proc_details=details,actual_original_FLOCK=owners,current_stage_files=states,free_bytes=shutil.disk_usage(R).free,all_canonical_and_preexisting_source_SHA256_unchanged=True,no_model_signal_restart_replay_or_matrix_deletion=True,approved_resource_expansion_inferred=False)
 put(note/'actual.json',fresh)
 for n,h in plan['canonical_SHA256'].items():assert sha(n)==h,n
 for n,v in rows.items():assert sha(n)==v['SHA256'],n
 newpending=Path(plan['new_pending']);assert newpending.resolve().is_relative_to((D/'code_publication_source_layout_20261114').resolve()) and not newpending.exists()
 pending.update(Code_Day_already_pushed='2026-11-20',next_daily_review='2026-11-21: all newly added current sources/config consumers and actual producer dependencies; no second Day20 commit',UTC=fresh['UTC'],parent_pending=str(oldpending),parent_pending_SHA256=sha(oldpending),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],source_count=len(rows),official_readback_and_actual_note=str(note/'actual.json'),official_readback_and_actual_note_SHA256=sha(note/'actual.json'))
 put(newpending,pending)
 closing_path=H/'closing_addendum.json';closing=js(closing_path);put(note/'prior_closing_original.json',closing)
 closing.update(current_canonical_SHA256=plan['canonical_SHA256'],Day20_commit='be09d9a8b637a6dd54f9be65db8609e89af92f64',exact_GLM154_cleanup_receipt=str(cr),new_full_current_handoff=plan['new_full_current_handoff'],UTC=fresh['UTC'],latest_actual_note=str(note/'actual.json'),latest_actual_note_SHA256=sha(note/'actual.json'),latest_pending=str(newpending),latest_pending_SHA256=sha(newpending),current_source_count=len(rows),official_readback=plan['official_readback'],latest_actual_current_stage_files=states,latest_actual_project_processes=procs,latest_actual_original_FLOCK=owners,initial_body_and_facts_snapshot_superseded_by_real_actual_exit_and_proc=True)
 put(closing_path,closing)
 ptr.update(canonical_SHA256=plan['canonical_SHA256'],Code_Day_already_pushed='2026-11-20',Day20_commit='be09d9a8b637a6dd54f9be65db8609e89af92f64',UTC=fresh['UTC'],new_primary_pending=str(newpending),new_primary_pending_SHA256=sha(newpending),current_primary_source_count=len(rows),closing_addendum_SHA256=sha(closing_path),latest_actual_note=str(note/'actual.json'),latest_actual_note_SHA256=sha(note/'actual.json'),latest_actual_snapshot=str(note/'actual.json'),latest_actual_snapshot_SHA256=sha(note/'actual.json'),latest_official_automation_readback_SHA256=plan['official_readback']['after_TOML_SHA256'])
 assert pointer.read_bytes()==raw_pointer;put(pointer,ptr)
 result=dict(passed=True,handoff_directory=str(H),latest_actual_note=str(note/'actual.json'),latest_actual_note_SHA256=sha(note/'actual.json'),closing_addendum_SHA256=sha(closing_path),new_primary_pending=str(newpending),new_primary_pending_SHA256=sha(newpending),current_primary_source_count=len(rows),pointer_SHA256=sha(pointer),all_canonical_unchanged=True)
 put(note/'closure_actual_receipt.json',result);return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('plan');ap.add_argument('SHA256');a=ap.parse_args();print(json.dumps(execute(Path(a.plan),a.SHA256),ensure_ascii=False))
