"""Exact new own-source installation and no-learned-forward frontend launch."""
import ast,base64,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent
name='prepare_meta_mllama_own_original_HF_BF16_2048_frontend200_meta_sources_20261120.py'
raw=(V/name).read_bytes();ast.parse(raw)
item=dict(relative='scripts/'+name,SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(R/'scripts/A100_alignment_incremental_monitor_20261103.py')=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
assert not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid()]
assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='4ba1a4c2a3171fa17b1e4fa9e997addf6d40e2210eaebf92705d84b712f728d3'
rec=json.loads(base64.b64decode('PAYLOAD'));p=R/rec['relative'];raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['SHA256'];ast.parse(raw);assert not p.exists()
with p.open('xb') as f:f.write(raw)
H=B/'MetaMllama_own_original_HF_BF16_2048_frontend200_meta_sources_v1_launch_20261120';H.mkdir(exist_ok=False)
argv=[sys.executable,'-B',str(p)]
with (H/'stdout.log').open('xb') as out,(H/'stderr.log').open('xb') as err:child=subprocess.Popen(argv,cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),start_new_session=True)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,argv=argv,source=str(p),source_SHA256=sha(p),actual_NEW_model_forwards=0,no_NPU_model_claim=True)
(H/'launch.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(v,ensure_ascii=False))
'''.replace('PAYLOAD',base64.b64encode(json.dumps(item).encode()).decode())
ast.parse(remote);p=V/'stage_MetaMllama_own_frontend_meta_sources_20261120_remote.py';p.write_text(remote,encoding='utf8',newline='\n');print(str(p))
