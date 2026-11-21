import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'OvisU1_completed_genuine200_formal_NF2_original_raw_fallback_metadata_after_Meta_serial_no_NN_replay_20261121'
out={'UTC':datetime.now(timezone.utc).isoformat(),'free_bytes':__import__('shutil').disk_usage(R).free,'files':{},'processes':[]}
for n in ['state.json','actual_completion.json','formal_acceptance.json','verification_stderr.log','verification_stdout.json']:
 p=H/n
 if p.is_file():
  b=p.read_bytes();out['files'][n]={'SHA256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'text':b.decode()}
out['locks']=Path('/proc/locks').read_text()
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  av=[x for x in (p/'cmdline').read_bytes().decode().split('\0') if x]
  if any(x in ' '.join(av).lower() for x in ['verify_ovisu1_completed200','continue_ovisu1_completed200']):
   fd=[]
   for a in (p/'fd').iterdir():
    try:
     t=os.readlink(a)
     if '.npz' in t or 'verification_' in t:fd.append(t)
    except OSError:pass
   out['processes'].append({'PID':int(p.name),'argv':av,'io':(p/'io').read_text(),'stat':(p/'stat').read_text(),'numerical_archive_FDs':fd})
 except OSError:pass
print(json.dumps(out,ensure_ascii=False))