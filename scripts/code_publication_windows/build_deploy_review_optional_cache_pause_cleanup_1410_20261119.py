from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
inventory=json.loads((P/'R_user_storage_detailed_categories_cache_1355_20261119.json').read_bytes())
sources=[]
mapping=[('remove_inactive_auxiliary_HTTP_cache_exact_manifest_20261119.py','scripts/maintenance/remove_inactive_auxiliary_HTTP_cache_exact_manifest_20261119.py'),('auxiliary_cache_pause_user_20261119.json','configs/retention/auxiliary_cache_pause_user_20261119.json')]
for n,r in mapping:
 b=(P/n).read_bytes();ast.parse(b) if n.endswith('.py') else json.loads(b);sources.append(dict(relative=r,SHA256=sha(b),base64=base64.b64encode(b).decode()))
data=dict(sources=sources,inventory_path=inventory['evidence_path'],inventory_SHA256=inventory['evidence_SHA256'],home='/root/projects/comperror/workspace/diagnostics/batch_910b/user_optional_HTTP_cache_pause_exact_cleanup_1410_20261119')
code='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror')
for item in DATA['sources']:
 p=R/item['relative'];assert p.resolve().is_relative_to(R);b=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==item['SHA256']
 ast.parse(b) if p.suffix=='.py' else json.loads(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
sys.path[:0]=[str(R),str(R/'scripts')]
m=importlib.import_module('maintenance.remove_inactive_auxiliary_HTTP_cache_exact_manifest_20261119')
print(json.dumps(m.run(DATA['inventory_path'],DATA['inventory_SHA256'],DATA['home'],apply=False),ensure_ascii=False))
'''
ast.parse(code);(P/'deploy_review_optional_cache_pause_exact_HTTP_cleanup_1410_20261119_remote.py').write_text(code,encoding='utf8',newline='\n')
print(json.dumps(dict(sources=[{k:v for k,v in x.items() if k!='base64'} for x in sources],inventory_source=data['inventory_path'],home=data['home']),ensure_ascii=False))
