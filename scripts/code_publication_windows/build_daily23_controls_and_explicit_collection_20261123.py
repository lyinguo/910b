"""Stage exact useful Day23 consumers and invoke the unchanged SHA-pinned collector."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
maintenance=['collect_daily23_verified185_explicit_primary_sources_20261123.py','run_daily23_actual_R_source_tests_no_model_20261123_remote.py','run_daily23_actual_R_source_tests_no_model_v2_original_source_arg_20261123_remote.py','verify_daily23_R_source_concurrent_CAS_before_publish_20261123.py','sync_daily23_reviewed_code_policy_and_remote_receipts_only_20261123.py']
windows=['daily23_reviewed_code_publication_controls_20261123.py',Path(__file__).name]
src={}
for n in maintenance+windows:
    b=(V/n).read_text(encoding='utf8').encode();ast.parse(b)
    src[('scripts/maintenance/' if n in maintenance else 'scripts/code_publication_windows/')+n]=b
payload={n:dict(SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()) for n,b in src.items()}
(V/'daily23_actual_consumer_explicit_primary_paths_20261123.json').write_text(json.dumps({n:r['SHA256'] for n,r in payload.items()},indent=2)+'\n',encoding='utf8')
s="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/collect_daily23_verified185_explicit_primary_sources_20261123.py')],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(s);(V/'stage_collect_daily23_verified185_explicit_primary_sources_20261123_remote.py').write_text(s,encoding='utf8')
print(json.dumps(dict(primary_paths=len(payload),primary_SHA256={n:r['SHA256'] for n,r in payload.items()},explicit_inventory_only=True)))
