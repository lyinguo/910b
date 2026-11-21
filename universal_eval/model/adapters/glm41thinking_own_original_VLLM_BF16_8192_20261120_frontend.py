"""Own GLM4.1 Thinking original VLLM media, clean prompt and chat template."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,inspect,os,re,types,typing,torch,numpy as np
from universal_eval.model.reference_sources.glm41thinking_owned_tokenizer_config_20261120 import load_owned_rust
from transformers import AutoProcessor
from vllm.multimodal.media.image import ImageMediaIO
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def flatten_tensors(v):return {n:t for n,t in v.items() if torch.is_tensor(t)}
class OwnGLM41ThinkingInputs:
 def __init__(self,W):
  self.processor=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False);self.tokenizer=self.processor.tokenizer;self.rust=load_owned_rust(W);assert self.rust.get_vocab()==self.tokenizer.get_vocab()
  self.media=ImageMediaIO();assert self.media.image_mode=='RGB' and self.media.rgba_background_color==(255,255,255)
  src=Path('/root/projects/comperror/universal_eval/model/adapters/vllm_adapter.py');raw=src.read_bytes();assert hashlib.sha256(raw).hexdigest()=='da65f91d00696a2becf43121c793de985eb8364fd18f2a0164ff8d53541602db'
  cl=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='VLLMModel');fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='_build_message');rx=next(n for n in cl.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_IMAGE_PLACEHOLDER_RE' for x in n.targets))
  ns=dict(re=re,os=os,Optional=typing.Optional,List=list,Dict=dict,Any=typing.Any);exec(compile(ast.Module(body=[rx,fn],type_ignores=[]),str(src),'exec'),ns)
  self.original_build_message=ns['_build_message'];self.original_input=types.SimpleNamespace(_IMAGE_PLACEHOLDER_RE=ns['_IMAGE_PLACEHOLDER_RE'],_prompt_prefix='')
  ext=Path(inspect.getfile(ImageMediaIO));self.native_media_external_SHA256={str(ext):hashlib.sha256(ext.read_bytes()).hexdigest()}
  assert 'enable_thinking' not in (Path(W)/'chat_template.jinja').read_text()
 def prepare(self,prompt,media):
  message=self.original_build_message(self.original_input,prompt,str(media));clean=message[0]['content'][1]['text'];image=self.media.load_file(Path(media))
  messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=clean)])]
  text=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  return text,self.processor(text=text,images=[image],return_tensors='pt',padding=True)
 def check_pixels(self,media,v):
  image=self.media.load_file(Path(media));ip=self.processor.image_processor;raw=ip(images=[image],do_rescale=False,do_normalize=False,return_tensors='pt');assert torch.equal(raw['image_grid_thw'],v['image_grid_thw'])
  a=raw['pixel_values'].numpy().astype(np.float32).reshape(-1,3,2,14,14);f=np.float32(1.0/ip.rescale_factor);mean=(np.asarray(ip.image_mean,np.float32)*f).reshape(1,3,1,1,1);std=(np.asarray(ip.image_std,np.float32)*f).reshape(1,3,1,1,1);assert np.array_equal(((a-mean)/std).reshape(v['pixel_values'].shape),v['pixel_values'].numpy())
  return True
