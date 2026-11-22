"""Bind actual official update/readback to exact current-source preservation."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
receipt=json.loads((V/'R_verified185_Xtuner_native_IModel_full_handoff_v77_preserved_20261122.json').read_bytes())
assert receipt['passed'] and receipt['source_count']==2777
stage={}
for n in ['verify_official_910b_verified185_Xtuner_v77_actual_readback_20261122.py','verify_official_910b_verified185_Xtuner_v77_actual_readback_v2_exact_one_terminal_LF_20261122.py']:
 stage['scripts/code_publication_windows/'+n]=(V/n).read_bytes()
for n in ['record_official910b_verified185_Xtuner_actual_readback_pending_v78_20261122.py','read_verified185_Xtuner_v2_complete_monitor_current_no_write_20261122_remote.py']:
 stage['scripts/maintenance/'+n]=(V/n).read_bytes()
stage['scripts/code_publication_windows/'+Path(__file__).name]=Path(__file__).read_bytes()
local_names=['official_910b_verified185_Xtuner_native_v77_requested_args_20261122.json','official_910b_before_verified185_Xtuner_v77_update_20261122.toml','official_910b_verified185_Xtuner_v77_actual_after_official_update_20261122.toml','official_910b_verified185_Xtuner_v77_actual_readback_20261122.json','official_910b_verified185_Xtuner_v77_actual_tool_result_20261122.json','R_verified185_Xtuner_v2_complete_monitor_current_20261122.json','R_verified185_Xtuner_native_IModel_full_handoff_v77_preserved_20261122.json']
local=[]
for n in local_names:
 b=(V/n).read_bytes();local.append(dict(name=n,SHA256=sha(b),base64=base64.b64encode(b).decode()))
q=dict(expected_pointer_SHA256=receipt['pointer_SHA256'],expected_pending_SHA256=receipt['pending_SHA256'],expected_closing_SHA256=receipt['closing_SHA256'],round_name='verified185_Xtuner_v2_actual_official910b_readback_source_preserved_v78_20261122',pending_name='pending_after_daily_20261122_verified185_Xtuner_v2_official_actual_readback_all_primary_v78_20261122.json',new_primary_paths=sorted(stage),local_original_records=local)
qb=json.dumps(q,ensure_ascii=False,indent=2).encode();name='official910b_verified185_Xtuner_v77_actual_readback_preservation_plan_v78_20261122.json';(V/name).write_bytes(qb);qh=sha(qb)
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];p=R/n\n if p.suffix=='.py':ast.parse(b)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nD=R/'workspace/diagnostics/batch_910b';T=D/'heartbeat_handoff_plans';T.mkdir(exist_ok=True);q=T/"+repr(name)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(qh)+"\nif q.exists():assert q.read_bytes()==b\nelse:q.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/record_official910b_verified185_Xtuner_actual_readback_pending_v78_20261122.py'),str(q),"+repr(qh)+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier);(V/'stage_record_official910b_verified185_Xtuner_readback_v78_20261122_remote.py').write_text(carrier,encoding='utf8')
print(json.dumps(dict(plan_SHA256=qh,explicit_useful_sources=len(stage),local_original_records=len(local),same_current_native_job_untouched=True)))
