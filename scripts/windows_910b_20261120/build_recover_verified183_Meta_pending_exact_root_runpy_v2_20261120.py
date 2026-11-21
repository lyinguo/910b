"""Correct precise handoff source namespace: allow required root run.py only."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
old=(V/'append_verified183_Meta_actual_full_source_pending_20261120.py').read_text()
a="p.is_relative_to(R) and rel.startswith(('scripts/','configs/','universal_eval/'))";b="p.resolve().is_relative_to(R) and '..' not in Path(rel).parts and (rel=='run.py' or rel.startswith(('scripts/','configs/','universal_eval/')))";assert old.count(a)==1;s=old.replace(a,b)
needle=" current=actual();put(note/'latest_actual.json',current)"
extra=""" current=actual()
 waiter=B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'
 current['GLM41Thinking_no_NN_capacity_waiter']={n:dict(SHA256=sha(waiter/n),value=js(waiter/n)) for n in ['state.json','actual_completion.json','exact_authorized_ledger_only_continuation_review.json'] if (waiter/n).is_file()}
 put(note/'latest_actual.json',current)"""
assert s.count(needle)==1;s=s.replace(needle,extra)
name='append_verified183_Meta_actual_full_source_pending_v2_exact_root_runpy_20261120.py';ast.parse(s);(V/name).write_text(s,encoding='utf8',newline='\n')
primary={name:'scripts/maintenance/'+name,'build_recover_verified183_Meta_pending_exact_root_runpy_v2_20261120.py':'scripts/windows_910b_20261120/build_recover_verified183_Meta_pending_exact_root_runpy_v2_20261120.py','build_stage_GLM41Thinking_completed3_real_capacity_serial_v2_20261120.py':'scripts/windows_910b_20261120/build_stage_GLM41Thinking_completed3_real_capacity_serial_v2_20261120.py'}
payload={}
for fname,rel in primary.items():
 raw=(V/fname).read_bytes();ast.parse(raw);payload[rel]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
payload=json.loads(base64.b64decode('PAYLOAD'))
for rel,r in payload.items():
 raw=base64.b64decode(r['base64']);assert hashlib.sha256(raw).hexdigest()==r['SHA256'];ast.parse(raw);assert not (R/rel).exists()
for rel,r in payload.items():
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(r['base64']))
pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(pointer)=='02b13c07c1e6d61d1c0cdb7543ba2c05decaabfffebd2b2c3e6b5c37d7111c88';H=Path(json.loads(pointer.read_bytes())['handoff_directory']);oldplan=H/'late_actual_notes/Meta_full_source_pending_v53_incoming_20261120/source_pending_plan.json';assert sha(oldplan)=='626d390eb807527e03c9b0c608368284e7451899b35513b9b25f0b9cbbde18c2';plan=json.loads(oldplan.read_bytes());plan['late_note_name']+='__exact_root_runpy_recovery_v2';plan['new_primary_sources'].extend(dict(relative=n,SHA256=sha(R/n),role='actual source collector exact root run.py repair or capacity continuation consumer') for n in [*payload,'scripts/continue_GLM41Thinking_completed_true3_after_verified183_exact271_retention_wait_real_capacity_serial_v2_20261120.py','configs/queues/GLM41Thinking_completed_true3_verified183_exact271_real_capacity_serial_v2_20261120.json'])
incoming=H/'late_actual_notes/Meta_pending_v53_exact_runpy_recovery_v2_incoming_20261120';incoming.mkdir(exist_ok=False);pp=incoming/'source_pending_plan.json';pp.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\\n')
cp=subprocess.run([sys.executable,'-B',str(R/'scripts/maintenance/SOURCENAME'),str(pp),sha(pp)],cwd=R,capture_output=True,text=True);v=dict(returncode=cp.returncode,stdout=cp.stdout,stderr=cp.stderr,plan=str(pp),plan_SHA256=sha(pp));(incoming/'actual_execution.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(v,ensure_ascii=False));sys.exit(cp.returncode)
'''.replace('PAYLOAD',base64.b64encode(json.dumps(payload).encode()).decode()).replace('SOURCENAME',name)
ast.parse(remote);(V/'stage_recover_verified183_Meta_pending_exact_root_runpy_v2_20261120_remote.py').write_text(remote,encoding='utf8',newline='\n');print('Exact mandatory root run.py accepted; sibling root files/traversal are refused. No model, table or live source changes.')
