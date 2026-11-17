"""Own HF math with inert legacy import placeholders removed in this process."""
import os,sys
from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v1_20261117 import Granite41OwnOriginalBF16
class Granite41OwnOriginalBF16PrivateCleanImports(Granite41OwnOriginalBF16):
 def load(self,model_path,**kwargs):
  assert os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')!='spawn'
  removed=[]
  for root in ['flash_attn','transformer_engine']:
   module=sys.modules.get(root)
   if getattr(module,'__file__',None)=='/dev/null':
    for name in list(sys.modules):
     if name==root or name.startswith(root+'.'):removed.append(name);del sys.modules[name]
  assert all(getattr(sys.modules.get(n),'__file__',None)!='/dev/null' for n in ['flash_attn','flash_attn.ops.triton.rotary','transformer_engine'])
  super().load(model_path,**kwargs)
  assert self.model.config._attn_implementation==self.model.config.text_config._attn_implementation==self.model.config.vision_config._attn_implementation=='sdpa'
  assert all('flash_attn' not in type(module).__module__ and 'transformer_engine' not in type(module).__module__ for module in self.model.modules())
  self.lineage.update(legacy_unrelated_startup_hook_disabled=True,original_HF_no_vllm_spawn_environment=True,inert_legacy_import_registry_removed_only_in_this_process=removed,original_default_text_and_vision_SDPA_confirmed=True,no_flash_or_transformer_engine_placeholder_model_module=True)
