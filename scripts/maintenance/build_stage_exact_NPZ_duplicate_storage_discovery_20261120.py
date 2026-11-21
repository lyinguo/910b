from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent;n='discover_exact_byte_identical_completed_project_NPZ_storage_20261120.py';raw=(V/n).read_bytes();ast.parse(raw);digest=hashlib.sha256(raw).hexdigest()
s='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');p=R/'scripts/maintenance/NAME';raw=base64.b64decode('PAYLOAD');assert hashlib.sha256(raw).hexdigest()=='DIGEST';assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'user_periodic_cleanup_exact_NPZ_duplicate_storage_discovery_launch_20261120';H.mkdir(exist_ok=False)
argv=[sys.executable,'-B',str(p)]
with (H/'stdout.log').open('xb') as out,(H/'stderr.log').open('xb') as err:child=subprocess.Popen(argv,cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
x=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,argv=argv,source_SHA256='DIGEST',read_only_discovery=True,no_NN_or_model_replay=True);(H/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(x,ensure_ascii=False))
'''.replace('NAME',n).replace('PAYLOAD',base64.b64encode(raw).decode()).replace('DIGEST',digest);ast.parse(s);(V/'stage_exact_NPZ_duplicate_storage_discovery_20261120_remote.py').write_text(s,encoding='utf-8',newline='\n');print(digest)
