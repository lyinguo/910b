import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';out={'UTC':datetime.now(timezone.utc).isoformat(),'free_bytes':shutil.disk_usage(R).free,'files':{}}
for h in ['OvisU1_completed_genuine200_formal_NF2_original_raw_fallback_metadata_after_Meta_serial_no_NN_replay_20261121','OvisU1_fullNF2_qualified184_reviewed_dual_controller_20261121','OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121','BeeSFT_completed_true3_full_LZMA_NF_capacity_after_Ovis_verified184_dual_serial_v2_POSIX_20261121']:
 for n in ['state.json','actual_completion.json','formal_acceptance.json','publication_stderr.log','publication_metadata_tests_stdout.json','publication_metadata_tests_stderr.log','publication_receipt.json','full_target_dual_publication_audit.json','dual_publication_completed.json','transaction.json']:
  p=B/h/n
  if p.is_file():
   b=p.read_bytes();v={'SHA256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
   try:v['JSON']=json.loads(b)
   except Exception:pass
   out['files'][str(p)]=v
p=B/'OvisU1_completed_genuine200_formal_NF2_original_raw_fallback_metadata_after_Meta_serial_no_NN_replay_20261121/verification_stdout.json'
if p.is_file() and p.stat().st_size:
 b=p.read_bytes();out['fullNF2_proof']={'SHA256':hashlib.sha256(b).hexdigest(),'JSON':json.loads(b)}
print(json.dumps(out,ensure_ascii=False))