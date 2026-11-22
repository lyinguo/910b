"""Stage exact read-only allocation survey for the existing project numerical directory."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent;name='read_project_numeric_physical_allocation_and_standard_archive_compression_metadata_only_20261122.py'
src={'scripts/maintenance/'+name:(V/name).read_text(encoding='utf8').encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_text(encoding='utf8').encode()}
for b in src.values():ast.parse(b)
payload={n:dict(SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()) for n,b in src.items()}
(V/'project_numeric_physical_allocation_metadata_primary_plan_20261122.json').write_text(json.dumps({n:r['SHA256'] for n,r in payload.items()},indent=2)+'\n',encoding='utf8')
s="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/"+name+"')],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(s);(V/'stage_project_numeric_physical_allocation_metadata_only_20261122_remote.py').write_text(s,encoding='utf8');print(json.dumps({n:r['SHA256'] for n,r in payload.items()}))
