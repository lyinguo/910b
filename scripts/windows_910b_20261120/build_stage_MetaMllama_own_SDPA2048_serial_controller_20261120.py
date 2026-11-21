"""Exact new private Meta sources/config; actual serial controller consumes config."""
import ast,base64,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent
names={
 'meta_mllama_own_original_BF16_SDPA_2048_native_v1_20261120.py':'universal_eval/model/adapters/meta_mllama_own_original_BF16_SDPA_2048_native_v1_20261120.py',
 'meta_mllama_own_original_HF_2048_MC_20261120.py':'universal_eval/algorithm/meta_mllama_own_original_HF_2048_MC_20261120.py',
 'verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v1_20261120.py':'scripts/verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v1_20261120.py',
 'meta_mllama_own_visual_value_reference_20261120.py':'scripts/meta_mllama_own_visual_value_reference_20261120.py',
 'prepare_meta_mllama_own_original_SDPA_2048_true3_v1_20261120.py':'scripts/prepare_meta_mllama_own_original_SDPA_2048_true3_v1_20261120.py',
 'continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py':'scripts/continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py'}
items={}
for name,rel in names.items():
 raw=(V/name).read_bytes();ast.parse(raw);items[rel]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
cfg=dict(model_id='meta-llama/Llama-3.2-11B-Vision-Instruct',max_concurrent_models=1,original_max_tokens=2048,own_default_SDPA=True,original_outer_batch=10,original_inner_batch=1,NPU_devices=[4,5],exact_source_SHA256={n:v['SHA256'] for n,v in items.items()})
config_rel='configs/queues/MetaMllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.json'
raw=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();items[config_rel]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
assert not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid()]
assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='4ba1a4c2a3171fa17b1e4fa9e997addf6d40e2210eaebf92705d84b712f728d3'
assert sha(B/'MetaMllama_own_original_HF_BF16_2048_frontend200_meta_sources_v1_20261120/review.json')=='1867c16c80d1ce97a46f19a7a91e13cdf1ade218c71a50ca3aa7f57158ab184d'
items=json.loads(base64.b64decode('PAYLOAD'))
for rel,rec in items.items():
 raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['SHA256'];p=R/rel;assert not p.exists()
 if p.suffix=='.py':ast.parse(raw)
 else:json.loads(raw)
for rel,rec in items.items():
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(rec['base64']))
L=B/'MetaMllama_own_SDPA2048_serial_controller_v1_launch_20261120';L.mkdir(exist_ok=False)
argv=[sys.executable,'-B',str(R/'scripts/continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py')]
with (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:child=subprocess.Popen(argv,cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),start_new_session=True)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,argv=argv,max_concurrent_models=1,own906_BF16_original2048_defaultSDPA_outer10_inner1=True,new_primary_sources={n:v['SHA256'] for n,v in items.items()},no_hot_edit_or_model_replay_or_canonical_write=True)
(L/'launch.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(v,ensure_ascii=False))
'''.replace('PAYLOAD',base64.b64encode(json.dumps(items).encode()).decode())
ast.parse(remote);(V/'stage_MetaMllama_own_SDPA2048_serial_controller_20261120_remote.py').write_text(remote,encoding='utf8',newline='\n');(V/'MetaMllama_own_SDPA2048_serial_exact_sources_20261120.json').write_text(json.dumps({n:v['SHA256'] for n,v in items.items()},indent=2)+'\n',encoding='utf8');print('Complete new private Meta stage ready for actual source/YAML validation and genuine NPU3, not preclaimed accepted.')
