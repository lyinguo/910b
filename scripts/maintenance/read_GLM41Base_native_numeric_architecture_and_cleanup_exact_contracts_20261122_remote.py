"""Read reusable unmodified architecture checks and the authorized cleanup consumer."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,json
R=Path('/root/projects/comperror')
names=['universal_eval/model/adapters/glm46flash_own_original_BF16_thinking4096_native_v1_1645_20261118.py','universal_eval/model/adapters/glm41thinking_own_original_VLLM_BF16_8192_20261120_native.py','universal_eval/model/adapters/glm41thinking_own_original_VLLM_BF16_8192_20261120_native_v2_actual_media.py','scripts/verify_glm46flash_own_original_BF16_thinking4096_native_v1_1645_20261118.py','scripts/verify_glm41thinking_own_original_VLLM_BF16_8192_native_v2_actual_media_20261120.py','scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_true3_v1_20261120.py','scripts/maintenance/watch_batch_910b_immediate_disk_recovery_cleanup_v1_20261120.py']
out={}
for n in names:
 p=R/n;b=p.read_bytes();out[n]=dict(SHA256=hashlib.sha256(b).hexdigest(),complete_text=b.decode())
p=R/'universal_eval/model/adapters/vllm_adapter.py';b=p.read_bytes();tree=ast.parse(b);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VLLMModel');out[str(p.relative_to(R))]=dict(SHA256=hashlib.sha256(b).hexdigest(),original_constructor_and_load_AST={n.name:ast.get_source_segment(b.decode(),n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['__init__','load']})
print(json.dumps(dict(original_architecture_sources=out,no_model_learned_CPU_or_NPU_execution=True,no_cleanup_or_source_mutation=True),ensure_ascii=False))
