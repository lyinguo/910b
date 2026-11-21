"""Finalize bounded late evidence and actual official readback, no canonical writes."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
sources=['build_stage_verified183_Meta_final_official_readback_pending_v54_20261120.py','build_official_910b_verified183_Meta_NF3_capacity_exact271_prompt_20261120.py','verify_official_910b_MetaNF3_capacity_and_local183_readback_20261120.py','read_current183_GLM41_capacity_waiter_Meta_closing_fresh_20261120_remote.py']
payload={}
for n in sources:
 raw=(V/n).read_bytes();ast.parse(raw);rel=('scripts/maintenance/' if n.startswith('read_') else 'scripts/windows_910b_20261120/')+n;payload[rel]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
records={}
for n in ['official_910b_before_MetaNF3_capacity_exact271_20261120.toml','official_910b_after_MetaNF3_capacity_exact271_20261120.toml','official_910b_verified183_MetaNF3_capacity_exact271_requested_args_20261120.json','official_910b_MetaNF3_actual_tool_result_20261120.json','official_910b_MetaNF3_exact_settings_local183_readback_20261120.json','official_910b_MetaNF3_before_and_prompt_review_20261120.json','R_MetaMllama_serial_actual10_20261120.json','R_current183_GLM41_waiter_Meta_closing_fresh_20261120.json','full_actual_verified183_Meta_NF3_cleanup271_GLM41_serial_capacity_20261120.txt']:
 raw=(V/n).read_bytes();records[n]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n');os.replace(t,p)
payload=json.loads(base64.b64decode('PAYLOAD'));records=json.loads(base64.b64decode('RECORDS'))
for rel,r in payload.items():
 raw=base64.b64decode(r['base64']);assert hashlib.sha256(raw).hexdigest()==r['SHA256'];ast.parse(raw);assert not (R/rel).exists()
for rel,r in payload.items():
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(r['base64']))
pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(pointer)=='c8c97fd8f3f1ae744dd9b1afe7fd6f49dae8c7999646adf6f4b421def94f6c97';H=Path(json.loads(pointer.read_bytes())['handoff_directory']);incoming=H/'late_actual_notes/Meta_final_official_readback_v54_incoming_20261120';incoming.mkdir(exist_ok=False);original=[]
for n,r in records.items():
 raw=base64.b64decode(r['base64']);assert hashlib.sha256(raw).hexdigest()==r['SHA256'];p=incoming/n;p.write_bytes(raw);original.append(dict(path=str(p),name=n,SHA256=sha(p)))
plan=dict(expected_pointer_SHA256=sha(pointer),late_note_name='verified183_Meta_formal_NF3_passed_capacity_GLM41_serial_wait_official_FINAL_20261120',pending_name='pending_after_daily_20261120_verified183_Meta_NF3_capacity_GLM41_wait_official_all_primary_v54.json',new_primary_sources=[dict(relative=n,SHA256=sha(R/n),role='actual official update/readback/real local SHA-stat or final Windows handoff consumer') for n in payload],original_records=original)
pp=incoming/'final_plan.json';pp.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\\n');cp=subprocess.run([sys.executable,'-B',str(R/'scripts/maintenance/append_verified183_Meta_actual_full_source_pending_v2_exact_root_runpy_20261120.py'),str(pp),sha(pp)],cwd=R,capture_output=True,text=True);result=dict(returncode=cp.returncode,stdout=cp.stdout,stderr=cp.stderr,plan=str(pp),plan_SHA256=sha(pp));(incoming/'actual_execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\\n');assert cp.returncode==0,result
value=json.loads(cp.stdout);assert sha(pointer)==value['pointer_SHA256'];ptr=json.loads(pointer.read_bytes());closing=H/'closing_addendum.json';assert sha(closing)==value['closing_SHA256'];close=json.loads(closing.read_bytes());note=Path(value['late_actual']).parent;summary=note/'local_original_records/full_actual_verified183_Meta_NF3_cleanup271_GLM41_serial_capacity_20261120.txt';readback=note/'local_original_records/official_910b_MetaNF3_exact_settings_local183_readback_20261120.json';rb=json.loads(readback.read_bytes());assert rb['passed'] and rb['preserved_original_fields']['id']=='910b' and rb['preserved_original_fields']['status']=='ACTIVE';canon={n:sha(n) for n in ptr['canonical_SHA256']};assert canon==ptr['canonical_SHA256']
close.update(latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),official_latest_actual_readback=str(readback),official_latest_actual_readback_SHA256=sha(readback),official_latest_active_TOML_SHA256=rb['after_TOML_SHA256'],official_update_only_original_ACTIVE_name_thread_300min_preserved=True,current_capacity_waiter_is_not_NN=True);put(closing,close);ptr.update(closing_addendum_SHA256=sha(closing),latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),official_latest_actual_readback=str(readback),official_latest_actual_readback_SHA256=sha(readback),next_no_NN_original_capacity_waiter=str(D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'));put(pointer,ptr)
value.update(pointer_SHA256=sha(pointer),closing_SHA256=sha(closing),latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),official_readback_SHA256=sha(readback));(incoming/'actual_final_receipt.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(value,ensure_ascii=False))
'''.replace('PAYLOAD',base64.b64encode(json.dumps(payload).encode()).decode()).replace('RECORDS',base64.b64encode(json.dumps(records).encode()).decode())
ast.parse(remote);(V/'stage_verified183_Meta_final_official_readback_pending_v54_20261120_remote.py').write_text(remote,encoding='utf8',newline='\n');print('Final explicit source/evidence archive ready, no model or canonical workbook writes.')
