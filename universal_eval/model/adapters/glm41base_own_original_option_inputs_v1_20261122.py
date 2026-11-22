"""This Base checkpoint's own saved manual option prompts and direct-PIL RGB inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer,AddedToken
from transformers import AutoProcessor

def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

class OwnGLM41BaseInputs:
 def __init__(self,W):
  self.W=Path(W);self.processor=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False);self.tokenizer=self.processor.tokenizer
  self.rust=Tokenizer.from_file(str(self.W/'tokenizer.json'));cfg=json.loads((self.W/'tokenizer_config.json').read_bytes())
  for tid,r in sorted(cfg.get('added_tokens_decoder',{}).items(),key=lambda p:int(p[0])):
   tok=AddedToken(**r);self.rust.add_special_tokens([tok]) if r['special'] else self.rust.add_tokens([tok]);assert self.rust.token_to_id(r['content'])==int(tid)
  for r in cfg.get('additional_special_tokens',[]):self.rust.add_special_tokens([AddedToken(**r) if isinstance(r,dict) else AddedToken(r,special=True,normalized=False)])
  assert self.rust.get_vocab()==self.tokenizer.get_vocab()
 def prepare(self,ref,option,media_path):
  assert hashlib.sha256(Path(media_path).read_bytes()).hexdigest()==ref['image_SHA256']
  with Image.open(media_path) as im:image=im.convert('RGB')
  v=self.processor(text=option['own_manual_prompt'],images=[image],return_tensors='pt',padding=True)
  expected=option['all_input_tensor_metadata'];actual={k:t for k,t in v.items() if torch.is_tensor(t)};assert set(actual)==set(expected)
  for n,t in actual.items():assert t.device.type=='cpu' and list(t.shape)==expected[n]['shape'] and str(t.dtype)==expected[n]['dtype'] and tensor_sha(t)==expected[n]['SHA256'],(n,'Own complete input changed')
  assert actual['input_ids'].tolist()==[option['own_complete_input_ids']] and tensor_sha(v['pixel_values'].bfloat16())==option['actual_native_pixel_BF16']['SHA256']
  assert len(self.tokenizer(option['original_continuation'],add_special_tokens=False)['input_ids'])==option['option_token_count']
  return v
 def independent_pixels(self,media_path,v):
  with Image.open(media_path) as im:image=im.convert('RGB')
  ip=self.processor.image_processor;raw=ip(images=[image],do_rescale=False,do_normalize=False,return_tensors='pt');assert torch.equal(raw['image_grid_thw'],v['image_grid_thw'])
  a=raw['pixel_values'].numpy().astype(np.float32).reshape(-1,3,2,14,14);f=np.float32(1/ip.rescale_factor);mean=(np.asarray(ip.image_mean,np.float32)*f).reshape(1,3,1,1,1);std=(np.asarray(ip.image_std,np.float32)*f).reshape(1,3,1,1,1)
  assert np.array_equal(((a-mean)/std).reshape(v['pixel_values'].shape),v['pixel_values'].numpy())
