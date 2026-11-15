"""Own DotsOCR original OCR frontend; no learned CPU operations or external assets."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,math,re
import torch,numpy as np
from PIL import Image
from transformers import AutoTokenizer,AutoImageProcessor
from universal_eval.model.dotsocr_own_original_sources_20261115.configuration_dots import DotsVLProcessor

def tensor_sha(t):
 return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

class OwnDotsOCRInputsV2:
 def __init__(self,weight_path):
  self.W=Path(weight_path)
  self.tokenizer=AutoTokenizer.from_pretrained(self.W,local_files_only=True,trust_remote_code=False)
  self.image_processor=AutoImageProcessor.from_pretrained(self.W,local_files_only=True,trust_remote_code=False,use_fast=False)
  self.processor=DotsVLProcessor(image_processor=self.image_processor,tokenizer=self.tokenizer)
  # Original saved own multimodal chat file is selected explicitly, never a borrowed template.
  self.template=json.loads((self.W/'chat_template.json').read_bytes())['chat_template']
  assert self.tokenizer.convert_tokens_to_ids('<|imgpad|>')==151665
  self.processor.image_token='<|imgpad|>';self.processor.image_token_id=151665
  self.processor.chat_template=self.template
  self.tokenizer.chat_template=self.template
 def prepare(self,prompt,media_path):
  clean=re.sub(r'<image\d+>','',prompt).strip()
  content=[{'type':'image_url','image_url':{'url':'file://'+str(media_path)}},{'type':'text','text':clean}]
  text=self.tokenizer.apply_chat_template([{'role':'user','content':content}],chat_template=self.template,tokenize=False,add_generation_prompt=True)
  image=Image.open(media_path).convert('RGB')
  data=self.processor(text=[text],images=[image],padding=True,return_tensors='pt')
  data={k:v for k,v in data.items() if torch.is_tensor(v) and k!='mm_token_type_ids'}
  assert set(data)=={'input_ids','attention_mask','pixel_values','image_grid_thw'},set(data)
  ids=data['input_ids'];grid=data['image_grid_thw'];assert grid.shape==(1,3) and grid[0,0]==1
  count=int(grid.prod().item())//4
  assert int((ids==151665).sum())==count>0
  assert ids.shape==data['attention_mask'].shape and bool((data['attention_mask']==1).all())
  return text,data
 def independently_check_pixels(self,media_path,pixels,grid):
  # Independent PIL resize and NumPy CLIP normalization/merge-patch layout.
  image=Image.open(media_path).convert('RGB')
  gh,gw=map(int,grid[0,1:].tolist());height,width=gh*14,gw*14
  original=np.asarray(image.resize((width,height),resample=Image.Resampling.BICUBIC))
  f=(original.astype(np.float64)*(1.0/255.0)).astype(np.float32)
  f=(f-np.array(self.image_processor.image_mean,np.float32))/np.array(self.image_processor.image_std,np.float32)
  f=f.transpose(2,0,1)[None,None]
  patches=f.reshape(1,1,3,gh//2,2,14,gw//2,2,14)
  patches=patches.transpose(0,3,6,4,7,2,1,5,8).reshape(gh*gw,3*14*14)
  # Native processor retains F32 arithmetic; BF16 casting happens only after normalization.
  assert patches.shape==tuple(pixels.shape)
  assert np.array_equal(patches,pixels.numpy()),float(np.max(np.abs(patches-pixels.numpy())))
  return True
