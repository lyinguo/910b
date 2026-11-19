import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
p=B/'GLM46Flash_own_ready_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v2_actual_D965_1653_20261118/full_shape_complete_numeric_storage_capacity_review.json'
cap=json.loads(p.read_bytes())
print(json.dumps(dict(complete_original_ZIP_parts=cap['all_complete_ZIP_members'],all200_prompt_lengths=cap['all200_original_prompt_lengths'],all200_pixel_elements=cap['all200_original_pixel_elements'],original_ratios=cap['ratios'],original_need=cap['empirical_estimate_bytes'],read_only_no_NN_or_matrix_copy=True),ensure_ascii=False))
