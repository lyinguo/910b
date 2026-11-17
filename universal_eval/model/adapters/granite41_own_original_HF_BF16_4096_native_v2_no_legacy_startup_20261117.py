"""Own original Granite math; forbid unrelated legacy startup mock branch."""
import os,sys
from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v1_20261117 import Granite41OwnOriginalBF16
class Granite41OwnOriginalBF16NoLegacyStartup(Granite41OwnOriginalBF16):
 def load(self,model_path,**kwargs):
  assert os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')!='spawn','The original HF deployment does not need the old unrelated vLLM startup branch'
  assert all(getattr(sys.modules.get(n),'__file__',None)!='/dev/null' for n in ['flash_attn','flash_attn.ops.triton.rotary','transformer_engine']), 'Legacy mock may not participate in this HF task'
  super().load(model_path,**kwargs)
  self.lineage.update(legacy_unrelated_startup_hook_disabled=True,original_HF_no_vllm_spawn_environment=True)
