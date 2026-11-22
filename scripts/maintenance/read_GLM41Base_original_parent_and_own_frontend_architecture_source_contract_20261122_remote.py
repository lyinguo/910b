"""Read existing own original interfaces for a future exact probability frontend."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror')
names=['universal_eval/model/adapters/glm41v_base_vllm.py','universal_eval/model/adapters/glm41thinking_own_original_VLLM_BF16_8192_20261120_frontend_v2_actual_media.py','scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_wrapper_20261120.py','scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_20261120.py','universal_eval/model/reference_sources/glm41thinking_owned_tokenizer_config_20261120.py']
out={}
for n in names:
 p=R/n
 if p.is_file():
  b=p.read_bytes();out[n]=dict(SHA256=hashlib.sha256(b).hexdigest(),complete_text=b.decode())
print(json.dumps(dict(own_original_or_previously_verified_unweighted_source_contracts=out,no_learned_CPU_or_NPU_execution=True,no_source_mutation=True),ensure_ascii=False))
