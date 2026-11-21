import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,inspect,hashlib,importlib
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
import torch
from transformers import CLIPImageProcessor,AutoTokenizer,LlamaConfig,LlamaForCausalLM,CLIPVisionConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.cache_utils import DynamicCache,DynamicLayer
from universal_eval.model.adapters import high_nine_llava_reference as own
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaMiniLlamaForCausalLM/ICTNLP/llava-mini-llama-3.1-8b')
raw=json.loads((W/'config.json').read_bytes());tk=AutoTokenizer.from_pretrained(W,local_files_only=True)
proc=CLIPImageProcessor(size={'shortest_edge':336},crop_size={'height':336,'width':336},do_resize=True,do_center_crop=True,do_rescale=True,do_normalize=True,image_mean=own.CLIP_MEAN,image_std=own.CLIP_STD)
out=dict(torch_version=torch.__version__,tokenizer_type=str(type(tk)),tokenizer_len=len(tk),tokenizer_bos=tk.bos_token_id,tokenizer_eos=tk.eos_token_id,tokenizer_pad=tk.pad_token_id,processor_type=str(type(proc)),processor_dict=proc.to_dict(),source=[]) 
targets=[(CLIPImageProcessor,'preprocess'),(CLIPImageProcessor,'_preprocess'),(CLIPImageProcessor,'rescale_and_normalize'),(CLIPImageProcessor,'resize'),(DynamicCache,'__init__'),(DynamicLayer,'lazy_initialization'),(LlamaForCausalLM,'_prepare_cache_for_generation'),(LlamaDecoderLayer,'forward')]
for cls,name in targets:
 obj=getattr(cls,name,None)
 if obj is None:continue
 try:out['source'].append(dict(owner=cls.__name__,method=name,file=inspect.getfile(obj),text=inspect.getsource(obj)))
 except (OSError,TypeError):pass
out['own_index_prefixes']={}
idx=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map']
for n in idx:
 prefix='.'.join(n.split('.')[:4]);out['own_index_prefixes'][prefix]=out['own_index_prefixes'].get(prefix,0)+1
out['own_index_first_keys']=list(idx)[:8]
print(json.dumps(out,ensure_ascii=False))
