"""Resolve missing discovery proof paths with the unchanged original accepted-proof selector."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
old=(V/'read_verified185_legacy_full_numeric_proof_schema_and_disk_cleanup_scope_metadata_20261122.py').read_text(encoding='utf8')
assert hashlib.sha256(old.encode()).hexdigest()=='c1ad883ded70f02d8389970f58ff346437b9e2ee954643a9b16dcce843dbc6a6'
needle="previous=js(old);proofs=[]"
repl="""previous=js(old);proofs=[]
import importlib.util
sys.path[:0]=[str(R),str(R/'scripts')]
helper=R/'scripts/retain_verified179_accepted_current_nonexample_matrices_user_request_20261117.py'
assert sha(helper)=='7ac8773a05cc87aeebd34b748089fd0dde12d8b7ee5530f0b68e3e441501e10b'
spec=importlib.util.spec_from_file_location('unchanged_original_accepted_proof_selector',helper)
selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
state=js(D/'state.json');registry=js(D/'registry.json')
assert sum(e['status']=='done' for e in state['models'].values())==185
"""
assert needle in old;s=old.replace(needle,repl)
needle="p=Path(row['proof']);q=js(p)"
repl="""p=Path(row['proof']) if row.get('proof') else selector.proof_for(row['model_id'],state['models'][row['model_id']],registry)
    assert p is not None and p.is_file(),('Original model unique proof path missing',row['model_id'])
    q=js(p)"""
assert needle in s;s=s.replace(needle,repl)
name='read_verified185_legacy_full_numeric_proof_schema_and_disk_cleanup_scope_metadata_v2_original_registry_path_20261122.py'
ast.parse(s);(V/name).write_text(s,encoding='utf8',newline='\n')
src={'scripts/maintenance/'+name:s.encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_text(encoding='utf8').encode()}
payload={n:dict(SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()) for n,b in src.items()}
(V/'verified185_legacy_numeric_proof_schema_metadata_v2_primary_plan_20261122.json').write_text(json.dumps({n:r['SHA256'] for n,r in payload.items()},indent=2)+'\n',encoding='utf8')
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/"+name+"')],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier);(V/'stage_verified185_legacy_proof_schema_cleanup_metadata_v2_read_20261122_remote.py').write_text(carrier,encoding='utf8')
print(json.dumps({n:r['SHA256'] for n,r in payload.items()}))
