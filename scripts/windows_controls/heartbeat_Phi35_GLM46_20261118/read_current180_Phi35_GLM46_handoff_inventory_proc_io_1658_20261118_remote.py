import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
pointer=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');pending=js(pointer['new_primary_pending']);old={r['path']:r for r in pending['all_current_primary_source_and_true_consumer_configs']}
rows={}
for parent,patterns in [(R/'scripts',['*phi35*0616_20261118.py','*phi35*1642_20261118.py','*glm46*0616_20261118.py','*glm46*1645_20261118.py','*glm46*1650_20261118.py','*glm46*1653_20261118.py']),(R/'universal_eval/model/adapters',['phi35*0616_20261118.py','glm46*1645_20261118.py']),(R/'universal_eval/algorithm',['phi35*0616_20261118.py','glm46*1645_20261118.py']),(R/'universal_eval/model/phi35_own_native_value_0616_20261118',['*.py']),(R/'tests',['test_phi35*0616_20261118.py']),(R/'configs/Phi3VForCausalLM',['microsoft--Phi-3.5-vision-instruct-own-original-BF16-native-4096*0616*']),(R/'configs/Glm4vForConditionalGeneration',['zai-org--GLM-4.6V-Flash-own-original-BF16-thinking4096*20261118*']),(R/'configs/protocols',['phi35_own_original_native_BF16_4096*0616*','glm46flash_own_original_BF16_thinking4096*20261118*'])]:
 for pattern in patterns:
  for p in parent.glob(pattern):
   if p.is_file() and str(p) not in old:rows[str(p)]=dict(relative=str(p.relative_to(R)),SHA256=sha(p),bytes=p.stat().st_size,role='Own model runtime, full numeric verifier, independent source/config preparation or immediate private-stage controller; live or preserved failed version')
J=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118';s=js(J/'state.json');pid=s.get('child_PID');proc={}
if pid and Path(f'/proc/{pid}').exists():
 for name in ['status','stat','io','wchan','cmdline']:
  p=Path(f'/proc/{pid}/{name}');proc[name]=p.read_bytes().decode(errors='replace').replace('\x00',' ')
 proc['cwd']=os.readlink(f'/proc/{pid}/cwd');proc['FD']={p.name:os.readlink(p) for p in Path(f'/proc/{pid}/fd').iterdir()}
out=dict(UTC=datetime.now(timezone.utc).isoformat(),pointer=pointer,pointer_SHA256=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'),parent_pending_SHA256=sha(pointer['new_primary_pending']),parent_source_count=len(old),new_primary_rows=list(rows.values()),canonical_SHA256={p:sha(p) for p in pointer['canonical_SHA256']},actual_GLM_state=s,actual_GLM_model_proc=proc,free_bytes=shutil.disk_usage(R).free)
print(json.dumps(out,ensure_ascii=False))
