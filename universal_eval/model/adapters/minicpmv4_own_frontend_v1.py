"""The own MiniCPM-V-4 tokenizer, patch slicing and ChatML input, offline."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import re,torch,hashlib
from PIL import Image
from universal_eval.model.minicpmv4_own_native_20261115.tokenization_minicpmv_fast import MiniCPMVTokenizerFast
from universal_eval.model.minicpmv4_own_native_20261115.image_processing_minicpmv import MiniCPMVImageProcessor
from universal_eval.model.minicpmv4_own_native_20261115.processing_minicpmv import MiniCPMVProcessor

def tensor_sha(t):
 t=t.detach().cpu().contiguous()
 return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def flatten_tensors(v,prefix=''):
 if torch.is_tensor(v):return {prefix:v}
 if isinstance(v,dict):
  return {k:t for n,x in v.items() for k,t in flatten_tensors(x,prefix+'.'+str(n) if prefix else str(n)).items()}
 if isinstance(v,(tuple,list)):
  return {k:t for n,x in enumerate(v) for k,t in flatten_tensors(x,prefix+'.'+str(n)).items()}
 return {}

class OwnMiniCPMV4Inputs:
 def __init__(self,weight_path):
  W=Path(weight_path)
  self.tokenizer=MiniCPMVTokenizerFast.from_pretrained(W,local_files_only=True,trust_remote_code=False)
  self.image_processor=MiniCPMVImageProcessor.from_pretrained(W,local_files_only=True)
  self.processor=MiniCPMVProcessor(self.image_processor,self.tokenizer)
  assert self.image_processor.max_slice_nums==9 and self.image_processor.image_feature_size==64
  assert self.tokenizer.convert_tokens_to_ids('<|im_end|>')==73440
 def prepare(self,prompt,media_path):
  text=re.sub(r'<image\d+>\s*','',prompt).strip()
  message=[{'role':'user','content':'(<image>./</image>)\n'+text}]
  rendered=self.tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
  with Image.open(media_path) as im:image=im.convert('RGB')
  result=self.processor([rendered],[[image]],return_tensors='pt',max_length=None)
  assert len(result['input_ids'])==1 and result['input_ids'].shape[1]+2048<32768
  assert (result['attention_mask']==1).all()
  return rendered,result
