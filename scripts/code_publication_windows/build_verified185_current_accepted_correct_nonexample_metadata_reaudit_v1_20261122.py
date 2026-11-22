"""Recheck current accepted full200 cleanup metadata without deleting or replaying old arrays."""
from pathlib import Path
import ast, base64, hashlib, json
V = Path(__file__).resolve().parent
s = (V/'discover_verified184_current_full_accepted_and_Meta_fullNF4_nonexamples_Bee_capacity_20261121.py').read_text(encoding='utf8')
a = s.index("ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json')")
b = s.index('from A100_alignment_incremental_monitor_20261103', a)
s = s[:a]+"ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');assert all(sha(p)==h for p,h in ptr['canonical_SHA256'].items())\n"+s[b:]
s = s.replace('==184', '==185').replace('Verified184', 'Verified185')
s = s.replace('verified184_current_full_accepted_nonexample_inventory_Bee_capacity_20261121', 'verified185_current_full_accepted_correct_nonexample_metadata_reaudit_Turkish_completed_20261122')
# Exact paths are inventoried; only metadata of previously accepted reports/proofs is read.
name = 'discover_verified185_current_full_accepted_correct_nonexample_metadata_only_v1_20261122.py'
ast.parse(s)
(V/name).write_text(s, encoding='utf8', newline='\n')
own = Path(__file__).read_text(encoding='utf8').encode()
sources = {'scripts/maintenance/'+name:s.encode(), 'scripts/code_publication_windows/'+Path(__file__).name:own}
payload = {n:dict(SHA256=hashlib.sha256(b).hexdigest(), base64=base64.b64encode(b).decode()) for n,b in sources.items()}
(V/'verified185_current_accepted_cleanup_metadata_reaudit_primary_plan_20261122.json').write_text(json.dumps({n:z['SHA256'] for n,z in payload.items()}, indent=2)+'\n', encoding='utf8')
carrier = "import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/"+name+"')],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier)
(V/'stage_verified185_current_accepted_cleanup_metadata_reaudit_20261122_remote.py').write_text(carrier, encoding='utf8')
print(json.dumps(dict(metadata_only=True, no_deletion=True, primary_SHA256={n:z['SHA256'] for n,z in payload.items()})))
