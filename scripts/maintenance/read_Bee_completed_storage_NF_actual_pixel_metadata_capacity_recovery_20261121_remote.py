import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'BeeSFT_completed_true3_full_LZMA_NF_capacity_after_Ovis_verified184_dual_serial_v2_POSIX_20261121';P=B/'Bee8BSFT_own_original_frontend200_v1_no_NPU_20261116/own_original_frontend200.json';front=json.loads(P.read_bytes());out={'frontend_SHA256':hashlib.sha256(P.read_bytes()).hexdigest(),'all200_tensor_metadata_keys':sorted({k for r in front for k in r['tensor_metadata']}),'first3_tensor_metadata':[r['tensor_metadata'] for r in front[:3]],'proof':{},'v2_completion':json.loads((H/'actual_completion.json').read_bytes())}
for n in ['complete_storage_fullNF_proof.json','complete_storage_all_values_review.json','exact_old_reference_to_current184.json']:
 p=H/n;b=p.read_bytes();out['proof'][n]={'SHA256':hashlib.sha256(b).hexdigest(),'JSON':json.loads(b)}
print(json.dumps(out,ensure_ascii=False))