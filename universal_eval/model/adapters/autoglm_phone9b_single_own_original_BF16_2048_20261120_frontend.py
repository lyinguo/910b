"""Own saved AutoGLM-Phone-9B thinking input contract, with independent input checks."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,torch,numpy as np
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoProcessor
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def flatten_tensors(v):return {n:t for n,t in v.items() if torch.is_tensor(t)}
class OwnAutoGLMPhone9BInputs:
 def __init__(self,W):
  self.processor=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False);self.tokenizer=self.processor.tokenizer;self.rust=Tokenizer.from_file(str(Path(W)/'tokenizer.json'));assert self.rust.get_vocab()==self.tokenizer.get_vocab()
 def prepare(self,prompt,media):
  with Image.open(media) as im:image=im.convert('RGB')
  messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])]
  text=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
  return text,self.processor(text=text,images=[image],return_tensors='pt',padding=True)
 def check_pixels(self,media,v):
  with Image.open(media) as im:image=im.convert('RGB')
  ip=self.processor.image_processor;raw=ip(images=[image],do_rescale=False,do_normalize=False,return_tensors='pt');assert torch.equal(raw['image_grid_thw'],v['image_grid_thw'])
  a=raw['pixel_values'].numpy().astype(np.float32).reshape(-1,3,2,14,14);f=np.float32(1.0/ip.rescale_factor);mean=(np.asarray(ip.image_mean,np.float32)*f).reshape(1,3,1,1,1);std=(np.asarray(ip.image_std,np.float32)*f).reshape(1,3,1,1,1);assert np.array_equal(((a-mean)/std).reshape(v['pixel_values'].shape),v['pixel_values'].numpy())
  return True
