"""Archive actual official and cleanup originals and all useful new sources, no duplicate daily commit."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
stage={}
for n in ['close_verified185_Turkish_fullNF0_capacity_official_and_cleanup_v88_20261122.py']:
    b=(V/n).read_text(encoding='utf8').encode();ast.parse(b);stage['scripts/maintenance/'+n]=b
for n in ['build_official910b_verified185_Turkish_fullNF0_capacity_Base_failed_v87_requested_args_20261122.py','verify_official910b_verified185_Turkish_completed_v87_actual_readback_20261122.py',Path(__file__).name]:
    b=(V/n).read_text(encoding='utf8').encode();ast.parse(b);stage['scripts/code_publication_windows/'+n]=b
primary=json.loads((V/'verified185_current_accepted_cleanup_metadata_reaudit_primary_plan_20261122.json').read_bytes());primary.update({n:sha(b) for n,b in stage.items()})
originals=['R_verified185_Base_failed_Turkish_same_serial_full_handoff_v87_preserved_20261122.json','R_TurkishLlava_fullNF0_current_complete_capacity_v8_20261122.json','R_verified185_current_accepted_cleanup_metadata_reaudit_actual_20261122.json','official910b_before_verified185_Turkish_completed_v87_20261122.toml','official910b_verified185_Turkish_completed_v87_requested_args_20261122.json','official910b_verified185_Turkish_completed_v87_official_tool_result_20261122.json','official910b_after_verified185_Turkish_completed_v87_20261122.toml','official910b_verified185_Turkish_completed_v87_actual_readback_receipt_20261122.json']
local=[]
for n in originals:
    b=(V/n).read_bytes();local.append(dict(name=n,SHA256=sha(b),base64=base64.b64encode(b).decode()))
clean=json.loads((V/'R_verified185_current_accepted_cleanup_metadata_reaudit_actual_20261122.json').read_bytes())
official=json.loads((V/'official910b_verified185_Turkish_completed_v87_actual_readback_receipt_20261122.json').read_bytes());assert official['passed']
q=dict(new_primary_expected_SHA256=primary,local_original_records=local,cleanup_discovery=clean['discovery_path'],cleanup_discovery_SHA256=clean['discovery_SHA256'],official_actual_readback=official)
qb=json.dumps(q,ensure_ascii=False,indent=2).encode();plan='verified185_Turkish_completed_official_cleanup_actual_close_plan_v88_20261122.json';(V/plan).write_bytes(qb)
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nq=R/'workspace/diagnostics/batch_910b/heartbeat_handoff_plans'/"+repr(plan)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(qb))+"\nif q.exists():assert q.read_bytes()==b\nelse:q.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/close_verified185_Turkish_fullNF0_capacity_official_and_cleanup_v88_20261122.py'),str(q),"+repr(sha(qb))+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier);(V/'stage_close_verified185_Turkish_completed_official_cleanup_v88_20261122_remote.py').write_text(carrier,encoding='utf8')
print(json.dumps(dict(primary_paths=len(primary),local_originals=len(local),plan_SHA256=sha(qb))))
