import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror')
rels=['scripts/continue_ovisu1_after_completed_true3_actual_BF16_RoPE_capacity_v3_20261116.py','scripts/diagnostic_OvisU1_complete_old3_standard_LZMA_capacity_no_NN_20261118.py','universal_eval/model/adapters/ovisu1_own_original_understanding_bf16_512_native_v2_actual_empty_precision_guard_20261116.py','scripts/verify_ovisu1_own_original_understanding_bf16_512_20261116.py']
print(json.dumps({'files':{r:{'SHA256':hashlib.sha256((R/r).read_bytes()).hexdigest(),'complete_text':(R/r).read_text()} for r in rels}},ensure_ascii=False))
