"""The own Bee SFT processor and its original saved conversation contract."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,torch,numpy as np
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from universal_eval.model.bee8bsft_own_original_20261116.image_processing_bee import BeeImageProcessor
from universal_eval.model.bee8bsft_own_original_20261116.processing_bee import BeeProcessor

def tensor_sha(t):
 t=t.detach().cpu().contiguous();return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def flatten_tensors(v,prefix=''):
 if torch.is_tensor(v):return {prefix:v}
 if isinstance(v,dict):return {k:t for n,x in v.items() for k,t in flatten_tensors(x,prefix+'.'+str(n) if prefix else str(n)).items()}
 if isinstance(v,(list,tuple)):return {k:t for n,x in enumerate(v) for k,t in flatten_tensors(x,prefix+'.'+str(n)).items()}
 return {}

class OwnBeeSFTInputs:
 def __init__(self,W):
  W=Path(W);self.tokenizer=AutoTokenizer.from_pretrained(W,local_files_only=True,trust_remote_code=False)
  self.image_processor=BeeImageProcessor.from_pretrained(W,local_files_only=True)
  kw=json.loads((W/'processor_config.json').read_bytes());kw.pop('auto_map',None);kw.pop('processor_class',None)
  self.processor=BeeProcessor(self.image_processor,self.tokenizer,chat_template=(W/'chat_template.jinja').read_text(),**kw)
  self.rust=Tokenizer.from_file(str(W/'tokenizer.json'));assert self.rust.get_vocab()==self.tokenizer.get_vocab()
  assert self.tokenizer.eos_token_id==151645 and self.tokenizer.convert_tokens_to_ids('<image>')==151669
 def prepare(self,prompt,media_path):
  with Image.open(media_path) as im:image=im.convert('RGB')
  messages=[dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=prompt)])]
  # This is the preserved original BeeAdapter default, not a newly added
  # generation or thinking override to improve a measured score.
  text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  v=self.processor(images=image,text=text,return_tensors='pt')
  assert v['input_ids'].shape[0]==1 and int(v['input_ids'].max())<152000
  return text,v
 def check_pixels(self,path,v):
  with Image.open(path) as im:image=im.convert('RGB')
  raw=self.image_processor(images=image,do_rescale=False,do_normalize=False,return_tensors='pt')
  t=raw['pixel_values'];assert t.dtype==torch.uint8 and t.shape==v['pixel_values'].shape
  # Same own resize/crop layout with independent F32 normalization. A single
  # image has no inter-image batch padding; no font or replacement image used.
  a=(t.numpy().astype(np.float64)*(1/255)).astype(np.float32);want=(a-np.float32(.5))/np.float32(.5)
  assert np.array_equal(want,v['pixel_values'].numpy()),float(np.max(np.abs(want-v['pixel_values'].numpy())))
  assert torch.equal(raw['image_sizes'],v['image_sizes'])
  return True
