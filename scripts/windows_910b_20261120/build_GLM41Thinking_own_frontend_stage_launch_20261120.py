import ast,base64,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent
name='prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_20261120.py'
raw=(V/name).read_bytes();ast.parse(raw);digest=hashlib.sha256(raw).hexdigest()
remote=r'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(R/'scripts/A100_alignment_incremental_monitor_20261103.py')=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners()
assert not [r for r in project_processes() if r['PID']!=os.getpid() and r['role'] in {'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}]
canonical={str(p):sha(p) for p in [R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx',D/'state.json',D/'regression_models.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json'] if p.is_file()}
p=R/'scripts/NAME';raw=base64.b64decode('PAYLOAD');assert hashlib.sha256(raw).hexdigest()=='DIGEST';ast.parse(raw)
assert not p.exists()
with p.open('xb') as f:f.write(raw)
L=B/'GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_source_launch_v1_20261120';L.mkdir(exist_ok=False)
out=(L/'stdout.log').open('xb');err=(L/'stderr.log').open('xb');env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
child=subprocess.Popen([sys.executable,'-B',str(p)],cwd=str(R),stdout=out,stderr=err,env=env,start_new_session=True);out.close();err.close()
value=dict(UTC=datetime.now(timezone.utc).isoformat(),source=str(p),source_SHA256=sha(p),PID=child.pid,argv=[sys.executable,'-B',str(p)],cwd=str(R),stdout=str(L/'stdout.log'),stderr=str(L/'stderr.log'),source_only_native_frontend_no_learned_CPU_or_NPU=True,canonical_before=canonical)
(L/'launch.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
assert all(sha(p)==h for p,h in canonical.items());print(json.dumps(value,ensure_ascii=False))
'''.replace('NAME',name).replace('PAYLOAD',base64.b64encode(raw).decode()).replace('DIGEST',digest)
ast.parse(remote)
p=V/'stage_GLM41Thinking_own_frontend_launch_20261120_remote.py';p.write_text(remote,encoding='utf8',newline='\n');print(json.dumps(dict(remote_helper=str(p),primary_source_SHA256=digest)))
