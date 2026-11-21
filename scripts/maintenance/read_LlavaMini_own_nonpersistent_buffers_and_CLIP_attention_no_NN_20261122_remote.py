import sys
sys.dont_write_bytecode=True
sys.path[:0]=['/root/projects/comperror']
from pathlib import Path
import json,inspect,torch
from universal_eval.model.llava_mini_own_saved_graph_FP16_20261122 import build_own_saved_graph
from transformers.models.clip.modeling_clip import CLIPAttention,CLIPVisionEmbeddings
w=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaMiniLlamaForCausalLM/ICTNLP/llava-mini-llama-3.1-8b')
m=build_own_saved_graph(json.loads((w/'config.json').read_bytes()),meta=True)
print(json.dumps(dict(buffers={n:dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device)) for n,t in m.named_buffers()},CLIPAttention=inspect.getsource(CLIPAttention.forward),CLIPEmbeddings=inspect.getsource(CLIPVisionEmbeddings.__init__)),ensure_ascii=False))
