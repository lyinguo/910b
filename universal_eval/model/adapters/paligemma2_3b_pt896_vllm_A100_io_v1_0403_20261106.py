"""Own PaliGemma2-3B-896 original vLLM caption input and Rust output semantics.

The original algorithm's final-answer suffix is part of the model input. The
Pali processor adds4096 image tokens then its own single BOS. Source files and
images remain read-only. This module performs no model scoring/publication.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from torchvision.transforms.v2 import functional as TV
from torchvision.transforms import InterpolationMode

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def fingerprint(value):
 return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

class OwnPali3Pt896IO:
 def __init__(self,root,own_resource_sha256):
  self.root=Path(root).resolve()
  for name,digest in own_resource_sha256.items():
   assert Path(name).name==name and sha(self.root/name)==digest,name
  cfg=json.loads((self.root/'preprocessor_config.json').read_text())
  assert cfg['size']=={'height':896,'width':896} and cfg['resample']==3
  assert cfg['image_seq_length']==4096 and cfg['processor_class']=='PaliGemmaProcessor'
  assert cfg['do_resize'] and cfg['do_rescale'] and cfg['do_normalize']
  assert cfg['image_mean']==cfg['image_std']==[.5]*3 and cfg['rescale_factor']==1/255
  self.tokenizer=Tokenizer.from_file(str(self.root/'tokenizer.json'))
  tc=json.loads((self.root/'tokenizer_config.json').read_text())
  assert tc.get('chat_template') is None and not (self.root/'chat_template.jinja').exists()
  self.caption_prompt='Describe this image in detail.\n\nProvide only the final answer.'
  self.image_prefix='<image>'*4096+'<bos>'
  probe=DecodeStream(ids=[2],skip_special_tokens=True);assert hasattr(probe,'step')

 def render(self,prompt):
  assert prompt==self.caption_prompt,'Original full algorithm input must include its unchanged final-answer instruction'
  rendered=self.image_prefix+prompt+'\n'
  ids=self.tokenizer.encode(rendered,add_special_tokens=False).ids
  assert len(ids)==4111 and ids[:4096]==[257152]*4096 and ids[4096]==2 and ids.count(2)==1
  return rendered,ids,0

 def pixels(self,media_path):
  path=Path(media_path).resolve();digest=sha(path)
  with Image.open(path) as source:
   image=ImageOps.exif_transpose(source)
   # Exact original vLLM ImageMediaIO: retain RGB, fill transparent modes
   # over white, otherwise convert. This does not alter the source image.
   if image.mode=='RGB':image=image.copy()
   elif image.mode in ('RGBA','LA','PA') or 'transparency' in image.info:
    rgba=image if image.mode=='RGBA' else image.convert('RGBA')
    image=Image.new('RGB',rgba.size,(255,255,255));image.paste(rgba,mask=rgba.split()[3])
   else:image=image.convert('RGB')
   rgb=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)
  resized=TV.resize(rgb,[896,896],interpolation=InterpolationMode.BICUBIC,antialias=True)
  pixels=TV.normalize(resized.float(),mean=[127.5]*3,std=[127.5]*3).bfloat16()[None]
  assert pixels.shape==(1,3,896,896)
  return pixels,digest,fingerprint(pixels),str(path)

 def decode(self,prompt_ids,output_ids,finish_reason):
  assert finish_reason in ('stop','length') and output_ids
  if finish_reason=='stop':
   assert output_ids[-1]==1
   ids=output_ids[:-1]
  else:ids=output_ids
  stream=DecodeStream(ids=list(prompt_ids),skip_special_tokens=True);text='';warnings=[]
  for index,token in enumerate(ids):
   try:fragment=stream.step(self.tokenizer,int(token))
   except (OverflowError,TypeError) as error:
    warnings.append({'output_position':index,'token_id':int(token),'kind':'original_protected_invalid_token','error':str(error)});fragment=None
   except Exception as error:
    if not str(error).startswith('Invalid prefix encountered'):raise
    warnings.append({'output_position':index,'token_id':int(token),'kind':'original_protected_invalid_UTF8_prefix','error':str(error)})
    stream=DecodeStream(skip_special_tokens=True);fragment=stream.step(self.tokenizer,int(token))
   text+=fragment or ''
  return text,warnings
