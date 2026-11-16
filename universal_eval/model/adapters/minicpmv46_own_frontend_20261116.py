"""Own original MiniCPM-V-4.6 input and saved MCQ protocol, offline only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,torch
from PIL import Image
from transformers import AutoProcessor
from tokenizers import Tokenizer
from universal_eval.model.reference_sources.minicpmv46_original_mcq_protocol import _parse_options,_strip_eval_instruction,_FEWSHOT_PREFIX

def tsha(t):
 t=t.detach().cpu().contiguous();return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def flatten(v,prefix=''):
 if torch.is_tensor(v):return {prefix:v}
 if isinstance(v,dict):return {n:t for k,x in v.items() for n,t in flatten(x,prefix+'.'+str(k) if prefix else str(k)).items()}
 if isinstance(v,(list,tuple)):return {n:t for k,x in enumerate(v) for n,t in flatten(x,prefix+'.'+str(k)).items()}
 return {}
class OwnMiniCPMV46Inputs:
 def __init__(self,W):
  self.W=Path(W);self.processor=AutoProcessor.from_pretrained(str(W),local_files_only=True,trust_remote_code=False);self.tokenizer=self.processor.tokenizer
  self.raw=Tokenizer.from_file(str(self.W/'tokenizer.json'));assert self.raw.get_vocab()==self.tokenizer.get_vocab()
 def prepare(self,prompt,media):
  opts=_parse_options(_strip_eval_instruction(prompt));model_prompt=_FEWSHOT_PREFIX+prompt if opts else prompt
  messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':model_prompt}]}]
  text=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  with Image.open(media) as im:image=im.convert('RGB')
  v=self.processor(images=[image],text=text,return_tensors='pt')
  assert v['input_ids'].shape[0]==1 and bool(v['attention_mask'].all())
  ids=v['input_ids'][0].tolist();decoded=self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
  assert self.raw.encode(decoded,add_special_tokens=False).ids==ids
  return text,v,opts,model_prompt
