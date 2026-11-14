"""Independent actual-A100 original Zelk HighNine RGB/TV/Jinja/Rust serial input replay; no model inference."""
import pathlib,hashlib,re,json,collections
import torch,numpy as np
from PIL import Image
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(x):return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':fp(x)}
def clean(s):
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',str(s));s=re.sub(r'[ \t]+',' ',s);return re.sub(r'<image\d+>\s*','',re.sub(r'\n{3,}','\n\n',s).strip())
def inputs(p,indices):
 wp=pathlib.Path(p['weight_path']);rust=Tokenizer.from_file(str(wp/'tokenizer.json'));env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True);env.globals['raise_exception']=lambda x:(_ for _ in ()).throw(ValueError(x));template=env.from_string(p['own_tokenizer_template']);texts=[];ids=[];pixels=[]
 for i in indices:
  r=p['input_identity_records'][i];messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':r['actual_original_model_prompt_after_instruction']}]}];text=template.render(messages=messages,bos_token='<bos>',eos_token='<eos>',add_generation_prompt=True);assert text==r['own_original_template_rendered_text'];texts.append(text);expanded=text.replace('<start_of_image>','\n\n<start_of_image>'+'<image_soft_token>'*256+'<end_of_image>\n\n');tokens=rust.encode(expanded,add_special_tokens=True).ids;assert tokens==r['independent_unpadded_input_ids'] and tokens[:2]==[2,2];ids.append(tokens)
  with Image.open(r['media_path']) as im:rgb=np.asarray(im.convert('RGB')).copy()
  x=F.resize(torch.from_numpy(rgb).permute(2,0,1)[None],[896,896],interpolation=InterpolationMode.BILINEAR,antialias=True).float();pixels.append((x-127.5)/127.5);assert fp(pixels[-1].bfloat16())==r['pixel_values_sha256']
 length=max(map(len,ids));assert len(indices)==1;tokens=torch.tensor([v+[0]*(length-len(v)) for v in ids]);mask=torch.tensor([[1]*len(v)+[0]*(length-len(v)) for v in ids]);return {'input_ids':tokens,'attention_mask':mask,'token_type_ids':(tokens==262144).long(),'pixel_values':torch.cat(pixels)},texts,rust
def arrays(path,records):
 out={}
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(records)
  for name,r in records.items():
   b=np.ascontiguousarray(z[name]);assert b.dtype==np.uint8 and hashlib.sha256(b.tobytes()).hexdigest()==r['sha256'];t=torch.from_numpy(b.copy()).view({'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32,'torch.int64':torch.int64,'torch.bool':torch.bool}[r['dtype']]).reshape(r['shape']);assert tr(t)==r;out[name]=t
 return out
