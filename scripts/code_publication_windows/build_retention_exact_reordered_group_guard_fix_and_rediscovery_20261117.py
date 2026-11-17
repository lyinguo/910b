from pathlib import Path
import sys,json,base64,hashlib,ast
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
names={'scripts/retain_verified179_accepted_current_nonexample_matrices_user_request_20261117.py':('retain_verified179_accepted_current_nonexample_matrices_user_request_20261117.py','832e90494543cd095b111090d60f84910f220b2c28dee649c9047f4e9e0d76c0'),'scripts/tests/test_verified179_retention_exact_group_scope_20261117.py':('test_verified179_retention_exact_group_scope_20261117.py','410b2fa184f89bb30fc71b4eca479388df662b28f41746a4d73cf817e2e86b73')}
bundle={}
for n,(local,old) in names.items():
 b=(P/local).read_bytes();ast.parse(b);bundle[n]=dict(old_SHA256=old,SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
code='import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport os,json,hashlib,base64,subprocess\nos.chdir("/root");R=Path("/root/projects/comperror");bundle='+repr(bundle)+'\nfor n,v in bundle.items():\n p=R/n;old=p.read_bytes();assert hashlib.sha256(old).hexdigest()==v["old_SHA256"];b=base64.b64decode(v["base64"]);assert hashlib.sha256(b).hexdigest()==v["SHA256"];backup=R/"workspace/diagnostics/batch_910b/retention_user_requested_20261117_source_preflight_failure";backup.mkdir(exist_ok=True);(backup/(p.name+"."+v["old_SHA256"])).write_bytes(old);p.write_bytes(b)\nprint(json.dumps({"new_source_SHA256":{n:v["SHA256"] for n,v in bundle.items()}}),flush=True)\nsubprocess.run(["python3","-B",str(R/"scripts/tests/test_verified179_retention_exact_group_scope_20261117.py")],check=True,cwd="/root")\nsubprocess.run(["python3","-B",str(R/"scripts/retain_verified179_accepted_current_nonexample_matrices_user_request_20261117.py"),"rediscover","2e769cd86520da32bae7d1d9bc43ec820ebe4d4252fa50df119a153523abadc0"],check=True,cwd="/root")\n'
(P/'deploy_retention_reordered_group_guard_fix_and_rediscovery_20261117_remote.py').write_text(code,encoding='utf8',newline='\n')
print(json.dumps({n:v['SHA256'] for n,v in bundle.items()}))
