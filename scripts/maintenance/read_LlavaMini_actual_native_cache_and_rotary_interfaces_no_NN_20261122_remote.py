import sys
sys.dont_write_bytecode=True
from pathlib import Path
import inspect,json
sys.path[:0]=['/root/projects/comperror']
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention,LlamaRotaryEmbedding,LlamaRMSNorm,LlamaModel
from transformers.cache_utils import DynamicCache,DynamicLayer
from universal_eval.model.adapters.high_nine_llava_reference import LlavaMiniCUDAAdapter
out={}
for key,obj in [('prepare_cache',LlamaForCausalLM._prepare_cache_for_generation),('LlamaAttention',LlamaAttention.forward),('LlamaRope_init',LlamaRotaryEmbedding.__init__),('LlamaRope_forward',LlamaRotaryEmbedding.forward),('LlamaNorm',LlamaRMSNorm.forward),('DynamicCache_init',DynamicCache.__init__),('DynamicLayer_lazy',DynamicLayer.lazy_initialization),('LlamaModel_forward',LlamaModel.forward)]:out[key]=inspect.getsource(obj)
print(json.dumps(out,ensure_ascii=False))
