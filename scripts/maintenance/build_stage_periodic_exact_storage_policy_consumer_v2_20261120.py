from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent;R='/root/projects/comperror'
old=(V/'discover_exact_byte_identical_completed_project_NPZ_storage_20261120.py').read_text(encoding='utf-8');needle="H.mkdir(exist_ok=False);assert not runner_lock_owners()";assert needle in old
new=old.replace("H=B/'user_periodic_cleanup_exact_byte_identical_complete_matrix_storage_discovery_20261120'","H=Path(sys.argv[1]).resolve();assert H.is_relative_to(B.resolve())")
n='discover_exact_byte_identical_completed_project_NPZ_storage_reusable_v2_20261120.py';ast.parse(new);(V/n).write_text(new,encoding='utf-8',newline='\n')
mapping={n:'scripts/maintenance/'+n,'periodic_batch_910b_exact_storage_maintenance_20261120.py':'scripts/maintenance/periodic_batch_910b_exact_storage_maintenance_20261120.py'};items={};hashes={}
for file,rel in mapping.items():
 raw=(V/file).read_bytes();ast.parse(raw);digest=hashlib.sha256(raw).hexdigest();hashes[rel]=digest;items[R+'/'+rel]=dict(SHA256=digest,base64=base64.b64encode(raw).decode())
cfg=json.loads((V/'batch_910b_periodic_exact_storage_retention_20261120.json').read_text(encoding='utf-8'));cfg.update(proactive_free_target_bytes=160*2**30,discovery_source=mapping[n],application_source='scripts/maintenance/apply_exact_byte_identical_project_matrix_physical_storage_20261120.py',exact_source_SHA256={**hashes,'scripts/maintenance/apply_exact_byte_identical_project_matrix_physical_storage_20261120.py':'8ef3c7495a26f232ac67a41cd919310c6585e64cf81ab9317c3d2bf5851e73e6'},do_not_generate_extra_unused_disk_copies_or_disable_original_model_runtime_KV_cache=True)
raw=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();rel='configs/maintenance/batch_910b_periodic_exact_storage_retention_v2_20261120.json';items[R+'/'+rel]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode());(V/Path(rel).name).write_bytes(raw)
s='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import base64,hashlib,json,subprocess
R=Path('/root/projects/comperror');items=json.loads(base64.b64decode('PAYLOAD'))
for n,x in items.items():
 raw=base64.b64decode(x['base64']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'] and not Path(n).exists()
for n,x in items.items():
 p=Path(n);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(x['base64']))
p=subprocess.run([sys.executable,'-B',str(R/'scripts/maintenance/periodic_batch_910b_exact_storage_maintenance_20261120.py')],cwd=R,capture_output=True,text=True);assert p.returncode==0,(p.stdout,p.stderr);print(p.stdout)
'''.replace('PAYLOAD',base64.b64encode(json.dumps(items).encode()).decode());ast.parse(s);(V/'stage_periodic_exact_storage_policy_consumer_v2_20261120_remote.py').write_text(s,encoding='utf-8',newline='\n');print(json.dumps({p:x['SHA256'] for p,x in items.items()},indent=2))
