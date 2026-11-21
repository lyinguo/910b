import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,inspect,os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
from transformers import AutoConfig,MllamaForConditionalGeneration
from transformers.integrations.sdpa_attention import sdpa_attention_forward
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/MllamaForConditionalGeneration/meta-llama/Llama-3.2-11B-Vision-Instruct')
c=AutoConfig.from_pretrained(W,local_files_only=True)
before=dict(top=c._attn_implementation,text=c.text_config._attn_implementation,vision=c.vision_config._attn_implementation)
with torch.device('meta'):m=MllamaForConditionalGeneration(c)
after=dict(top=m.config._attn_implementation,text=m.config.text_config._attn_implementation,vision=m.config.vision_config._attn_implementation)
out=dict(before=before,after=after,actual_NEW_model_forwards=0,not_historical_A100_engine_proof=True,source_functions={})
for fn in [sdpa_attention_forward,type(m.model.language_model.layers[0].self_attn).forward,type(m.model.vision_model.transformer.layers[0].self_attn).forward]:
 p=Path(inspect.getfile(fn));out['source_functions'][fn.__qualname__]=dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),function_text=inspect.getsource(fn))
print(json.dumps(out,ensure_ascii=False))
