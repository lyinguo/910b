import ast,base64,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent
prefix='glm41thinking_own_original_VLLM_BF16_8192_20261120'
rel={
 'prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_wrapper_20261120.py':'scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_wrapper_20261120.py',
 prefix+'_frontend_v2_actual_media.py':'universal_eval/model/adapters/'+prefix+'_frontend_v2_actual_media.py',
 prefix+'_native_v2_actual_media.py':'universal_eval/model/adapters/'+prefix+'_native_v2_actual_media.py',
 'verify_glm41thinking_own_original_VLLM_BF16_8192_native_v2_actual_media_20261120.py':'scripts/verify_glm41thinking_own_original_VLLM_BF16_8192_native_v2_actual_media_20261120.py',
 'prepare_glm41thinking_own_original_VLLM_BF16_8192_true3_v1_20261120.py':'scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_true3_v1_20261120.py'}
items={}
for name,path in rel.items():
 raw=(V/name).read_bytes();ast.parse(raw);items[path]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
remote=r'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert sha(R/'scripts/A100_alignment_incremental_monitor_20261103.py')=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
assert not runner_lock_owners() and not [r for r in project_processes() if r['PID']!=os.getpid()]
old=B/'GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v2_own_added_tokens_20261120/actual_completion.json';failure=json.loads(old.read_bytes());assert failure['actual_returncode']==1 and failure['no_model_inference'] and 'MediaWithBytes' in failure['error']
sources=json.loads(base64.b64decode('SOURCES'))
for rel,rec in sources.items():
 p=R/rel;raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['SHA256'];ast.parse(raw);assert not p.exists(),str(p);assert rel.startswith(('scripts/','universal_eval/model/'))
for rel,rec in sources.items():
 p=R/rel
 with p.open('xb') as f:f.write(base64.b64decode(rec['base64']))
L=B/'GLM41Thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_source_launch_20261120';L.mkdir(exist_ok=False)
argv=[sys.executable,'-B',str(R/'scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_wrapper_20261120.py')]
with (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:child=subprocess.Popen(argv,cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),start_new_session=True)
value=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,argv=argv,cwd=str(R),sources={p:r['SHA256'] for p,r in sources.items()},original_v1_v2_failures_preserved=True,exact_original_media_field_unwrap_only=True,actual_NEW_model_forwards=0)
(L/'launch.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(value,ensure_ascii=False))
'''.replace('SOURCES',base64.b64encode(json.dumps(items).encode()).decode())
ast.parse(remote);p=V/'stage_GLM41Thinking_owned_media_frontend_v3_20261120_remote.py';p.write_text(remote,encoding='utf8',newline='\n');print(str(p))
