import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J=B/'OvisU1_completed_storage_fullNF_passed_exact23_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261121'
H=B/'OvisU1_completed_genuine200_formal_NF2_original_raw_fallback_metadata_after_Meta_serial_no_NN_replay_20261121'
M=B/'MetaMllama_completed_genuine200_formal_NF4_original_empty_invalid_schema_v2_actual_file_SHA_after_Ovis_serial_no_NN_replay_20261121'
out={'files':{},'processes':[]}
def get(p):
 p=Path(p)
 if not p.is_file():return
 b=p.read_bytes();v={'SHA256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'complete_text':b.decode()}
 try:v['JSON']=json.loads(b)
 except Exception:pass
 out['files'][str(p)]=v
 return v.get('JSON')
j=get(J/'job.json');get(J/'completion.json');get(J/'formal200/stage_state.json')
spec=j['spec'];a=get(spec['alignment_approval_review']);get(a['capacity_review']);get(a['true3_proof'])
for p in [H,M]:
 for n in ['state.json','actual_completion.json','formal_acceptance.json','verification_stderr.log'] :get(p/n)
 if (p/'actual_completion.json').is_file(): get(p/'verification_stdout.json')
for n in ['/proc/locks'] :get(n)
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes().decode().split('\0');argv=[x for x in argv if x];low=' '.join(argv).lower()
  if any(x in low for x in ['verify_meta_mllama_completed200','continue_metamllama_completed200','continue_ovisu1_completed200','verify_ovisu1_completed200']):
   out['processes'].append({'PID':int(p.name),'argv':argv,'stat':(p/'stat').read_text(),'io':(p/'io').read_text()})
 except (OSError,ValueError):pass
print(json.dumps(out,ensure_ascii=False))