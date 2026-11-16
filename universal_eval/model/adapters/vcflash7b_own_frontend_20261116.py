"""Own saved VideoChat Flash single-image UMT448 and original Qwen ChatML input."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import torch,numpy as np,hashlib
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from universal_eval.model.vcflash7b_own_original_20261116.vision_tower_builder import UMTImageProcessor
def tensor_sha(t):
 t=t.detach().cpu().contiguous();return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def flatten_tensors(v,prefix=''):
 if torch.is_tensor(v):return {prefix:v}
 if isinstance(v,dict):return {k:t for n,x in v.items() for k,t in flatten_tensors(x,prefix+'.'+str(n) if prefix else str(n)).items()}
 if isinstance(v,(list,tuple)):return {k:t for n,x in enumerate(v) for k,t in flatten_tensors(x,prefix+'.'+str(n)).items()}
 return {}
class OwnVCFlash7Inputs:
 def __init__(self,W):
  W=Path(W);self.tokenizer=AutoTokenizer.from_pretrained(W,local_files_only=True,trust_remote_code=False);self.rust=Tokenizer.from_file(str(W/'tokenizer.json'));assert self.rust.get_vocab()==self.tokenizer.get_vocab();self.image_processor=UMTImageProcessor(size=(448,448))
 def prepare(self,prompt,path):
  text=self.tokenizer.apply_chat_template([dict(role='user',content='<image>\n'+prompt)],tokenize=False,add_generation_prompt=True)
  ids=[];rust=[]
  for i,part in enumerate(text.split('<image>')):
   if i:ids.append(-200);rust.append(-200)
   ids.extend(self.tokenizer(part).input_ids);rust.extend(self.rust.encode(part,add_special_tokens=True).ids)
  assert ids==rust and ids.count(-200)==1
  with Image.open(path) as im:image=im.convert('RGB');size=[image.height,image.width]
  pixels=self.image_processor.preprocess([image],return_tensors='pt')['pixel_values'];assert pixels.shape==(1,3,448,448)
  return text,dict(input_ids=torch.tensor([ids],dtype=torch.int64),attention_mask=torch.ones(1,len(ids),dtype=torch.int64),pixel_values=pixels,image_sizes=[size])
 def check_pixels(self,path,v):
  with Image.open(path) as im:a=np.asarray(im.convert('RGB').resize((448,448),resample=Image.Resampling.BICUBIC),dtype=np.uint8)
  a=(a.astype(np.float64)*(1/255)).astype(np.float32).transpose(2,0,1)
  mean=np.asarray([.485,.456,.406],dtype=np.float32)[:,None,None];std=np.asarray([.229,.224,.225],dtype=np.float32)[:,None,None]
  assert np.array_equal((a-mean)/std,v['pixel_values'][0].numpy())
  return True
