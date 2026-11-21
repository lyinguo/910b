"""Stage only new verifier/formal consumer; keep all original failed bytes."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
names=['verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v3_exact_original_power_reference_20261120.py','continue_meta_mllama_completed_genuine3_formal_NF3_exact_instruction_no_NN_replay_20261120.py']
items={}
for name in names:
 raw=(V/name).read_bytes();ast.parse(raw);items['scripts/'+name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
cfg=dict(model_id='meta-llama/Llama-3.2-11B-Vision-Instruct',max_concurrent_models=1,actual_NEW_model_forwards_in_acceptance=0,original_true3_job_SHA256='ac1384546dbab375c0d02e2fad31e4d0dea3c8f7f2a194e48b72ea9a24086567',new_verifier_SHA256=items['scripts/'+names[0]]['SHA256'],exact_source_SHA256={n:r['SHA256'] for n,r in items.items()})
rel='configs/queues/MetaMllama_completed_genuine3_formal_NF3_exact_instruction_no_NN_replay_capacity_serial_20261120.json';raw=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();items[rel]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
assert not runner_lock_owners();assert sha(B/'MetaMllama_own_original_SDPA_BF16_2048_true3_v1_real_FLOCK_20261120/job.json')=='ac1384546dbab375c0d02e2fad31e4d0dea3c8f7f2a194e48b72ea9a24086567'
items=json.loads(base64.b64decode('PAYLOAD'))
for rel,rec in items.items():
 raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['SHA256'];assert not (R/rel).exists()
 if rel.endswith('.py'):ast.parse(raw)
 else:json.loads(raw)
for rel,rec in items.items():
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(rec['base64']))
L=B/'MetaMllama_formal_NF3_same_genuine3_no_NN_replay_source_launch_20261120';L.mkdir(exist_ok=False)
argv=[sys.executable,'-B',str(R/'scripts/continue_meta_mllama_completed_genuine3_formal_NF3_exact_instruction_no_NN_replay_20261120.py')]
with (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:child=subprocess.Popen(argv,cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),start_new_session=True)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,argv=argv,source_SHA256={n:r['SHA256'] for n,r in items.items()},no_original_source_job_report_or_model_replay=True)
(L/'launch.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(v,ensure_ascii=False))
'''.replace('PAYLOAD',base64.b64encode(json.dumps(items).encode()).decode())
ast.parse(remote);(V/'stage_meta_mllama_formal_NF3_same_genuine3_no_NN_replay_20261120_remote.py').write_text(remote,encoding='utf8',newline='\n');print('New exact formal NF3 source/consumer ready; old NN/NF failure is untouched.')
