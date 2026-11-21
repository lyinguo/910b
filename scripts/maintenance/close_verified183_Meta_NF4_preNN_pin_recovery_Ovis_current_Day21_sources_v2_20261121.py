"""Archive the exact pre-NN SHA failure and its new serial consumer; leave model jobs untouched."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def main():
 plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';old=pointer.read_bytes();assert hashlib.sha256(old).hexdigest()==q['expected_pointer_SHA256'];ptr=json.loads(old);H=Path(ptr['handoff_directory']);canon=ptr['canonical_SHA256'];assert all(sha(p)==h for p,h in canon.items())
 parent=Path(ptr['new_primary_pending']);assert sha(parent)==q['expected_parent_pending_SHA256']==ptr['new_primary_pending_SHA256'];rows={v['path']:v for v in js(parent)['all_current_primary_source_and_true_consumer_configs']};assert all(sha(p)==v['SHA256'] for p,v in rows.items())
 note=H/'late_actual_notes'/q['late_note_name'];note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(old);(note/'previous_closing_addendum.json').write_bytes((H/'closing_addendum.json').read_bytes())
 additions=list(q['primary_sources'])+[dict(relative=str(Path(__file__).resolve().relative_to(R)),SHA256=sha(__file__),role='actual source archive and preNN pin failure/recovery handoff consumer')]
 for v in additions:
  p=R/v['relative'];assert p.resolve().is_relative_to(R) and v['relative'].startswith(('scripts/','configs/')) and sha(p)==v['SHA256'];assert str(p) not in rows
  dest=note/'primary_source_snapshots'/v['relative'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=v['relative'],SHA256=v['SHA256'],bytes=p.stat().st_size,snapshot=str(dest),role=v['role'])
 for v in q['local_original_records']:
  b=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==v['SHA256'];p=note/'local_original_records'/v['name'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 reader=R/'scripts/maintenance/read_OvisU1_current_serial_v2_and_preserved_Meta_failed200_20261121.py';r=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr.decode();a=json.loads(r.stdout)
 active=[v for v in a['captures'] if v['model_PID_matches_capture']];assert active and a['original_FLOCK'];put(note/'current_Ovis_actual_complete.json',a)
 full=q['full_handoff_text'].replace('CURRENT_ACTUAL_UTC',a['UTC']).replace('CURRENT_OVIS_PROGRESS',json.dumps(active,ensure_ascii=False)).replace('CURRENT_ACTUAL_PROCESSES',json.dumps(a['actual_processes'],ensure_ascii=False)).replace('CURRENT_FREE_BYTES',str(a['free_bytes']))
 fp=note/'full_actual_verified183_Meta_NF4_pin_preNN_failed_v2_recovered_Ovis_serial_Day21_20261121.txt';fp.write_text(full,encoding='utf8')
 pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=a['UTC'],parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-20',next_daily_review='Day21 all exact primary code/configs and dependencies with raw task input artifacts excluded from Git only',handoff_directory=str(H)))
 updates=dict(UTC=a['UTC'],new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_source_count=len(rows),current_primary_source_count=len(rows),latest_actual_note=str(fp),latest_actual_note_SHA256=sha(fp),latest_full_actual_summary=str(fp),latest_full_actual_summary_SHA256=sha(fp),latest_actual_snapshot=str(note/'current_Ovis_actual_complete.json'),latest_actual_snapshot_SHA256=sha(note/'current_Ovis_actual_complete.json'),Meta_NF4_v1_preNN_failed=True,Meta_NF4_v1_actual_failure_SHA256='148544c13faa7ca26efdc3daddb4f7fb658835430ff557712352fcbaff45c7c6',Meta_NF4_actual_verifier_SHA256='bf31fef6d2d4b59df491872833607fb68c512e34dc63b90156eb5a8077b0193a',Meta_NF4_v2_waiter=q['Meta_new_queue'],Meta_NF4_v2_waiter_source_SHA256=q['primary_sources'][0]['SHA256'],current_actual_stage='OvisU1_same_approved_fixed200_real_NN',current_formal_model_stage='OvisU1_same_approved_fixed200_real_NN',current_Ovis_progress=active)
 close=js(H/'closing_addendum.json');close.update(updates);assert pointer.read_bytes()==old and all(sha(p)==h for p,h in canon.items()) and all(sha(p)==v['SHA256'] for p,v in rows.items());put(H/'closing_addendum.json',close);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(H/'closing_addendum.json');put(pointer,ptr)
 out=dict(passed=True,pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),full_actual_note=str(fp),full_actual_note_SHA256=sha(fp),current_Ovis_progress=active,all_canonical_unchanged=True);put(note/'closing_actual_receipt.json',out);print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
