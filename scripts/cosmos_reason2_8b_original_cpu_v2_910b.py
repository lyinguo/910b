"""Independent original Cosmos Reason2 image/Jinja/Rust/group/position replay."""
import pathlib,hashlib,json,math,re,collections
import torch,numpy as np
from PIL import Image
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(x):return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':fp(x)}
def clean(s):
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',str(s));s=re.sub(r'[ \t]+',' ',s);return re.sub(r'<image\d+>\s*','',re.sub(r'\n{3,}','\n\n',s).strip())
def pixels(image):
 w,h=image.size;hh=round(h/32)*32;ww=round(w/32)*32
 if hh*ww>16777216:
  scale=math.sqrt(h*w/16777216);hh=max(32,math.floor(h/scale/32)*32);ww=max(32,math.floor(w/scale/32)*32)
 elif hh*ww<65536:
  scale=math.sqrt(65536/(h*w));hh=math.ceil(h*scale/32)*32;ww=math.ceil(w*scale/32)*32
 rgb=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1)
 resized=F.resize(rgb,[hh,ww],interpolation=InterpolationMode.BICUBIC,antialias=True).float();x=(resized-127.5)/127.5
 gh,gw=hh//16,ww//16;tiles=x.unfold(1,16,16).unfold(2,16,16)
 order=[(y+dy,z+dx) for y in range(0,gh,2) for z in range(0,gw,2) for dy in range(2) for dx in range(2)]
 patches=torch.stack([tiles[:,y,z] for y,z in order]);patches=patches[:,:,None].expand(-1,-1,2,-1,-1).reshape(gh*gw,1536).contiguous()
 return patches,[1,gh,gw]
def input_tensors(p,indices):
 wp=pathlib.Path(p['weight_path']);rust=Tokenizer.from_file(str(wp/'tokenizer.json'));env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True);env.globals['raise_exception']=lambda x:(_ for _ in ()).throw(ValueError(x));template=env.from_string(json.loads((wp/'chat_template.json').read_text())['chat_template']);ids=[];xs=[];grids=[];texts=[]
 for i in indices:
  r=p['input_records'][i]
  with Image.open(r['media_path']) as im:x,grid=pixels(im)
  prompt=r['model_prompt'];messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}];text=template.render(messages=messages,add_generation_prompt=True);texts.append(text)
  expanded=text.replace('<|image_pad|>','<|image_pad|>'*(grid[0]*grid[1]*grid[2]//4));tokens=rust.encode(expanded,add_special_tokens=True).ids;ids.append(tokens);xs.append(x);grids.append(grid)
 length=max(map(len,ids));tokens=torch.tensor([v+[151643]*(length-len(v)) for v in ids]);mask=torch.tensor([[1]*len(v)+[0]*(length-len(v)) for v in ids]);types=(tokens==151655).long()
 return {'input_ids':tokens,'attention_mask':mask,'mm_token_type_ids':types,'pixel_values':torch.cat(xs),'image_grid_thw':torch.tensor(grids)},texts,rust
def positions(inp):
 ids=inp['input_ids'];mask=inp['attention_mask'];types=inp['mm_token_type_ids'];grids=iter(inp['image_grid_thw'].tolist());out=torch.zeros(3,*ids.shape,dtype=torch.int64);delta=[]
 for b in range(len(ids)):
  active=types[b][mask[b].bool()].tolist();parts=[];start=0
  for kind,g in __import__('itertools').groupby(active):
   n=len(list(g))
   if kind==0:parts.append(torch.arange(start,start+n).repeat(3,1));start+=n
   else:
    t,h,w=next(grids);assert kind==1 and t==1 and n==h*w//4;hh,ww=h//2,w//2
    co=torch.tensor([[start]*n,[start+y for y in range(hh) for x in range(ww)],[start+x for y in range(hh) for x in range(ww)]],dtype=torch.int64);parts.append(co);start+=max(hh,ww)
  value=torch.cat(parts,1);out[:,b,mask[b].bool()]=value;delta.append(int(value.max()+1-len(active)))
 text=mask.cumsum(-1)-1;text.masked_fill_(mask==0,0)
 return torch.cat((text[None],out)),torch.tensor(delta)[:,None]
def decode_arrays(path,records):
 result={};dtypes={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32,'torch.int64':torch.int64,'torch.int32':torch.int32,'torch.bool':torch.bool}
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(records)
  for k,r in records.items():
   b=np.ascontiguousarray(z[k]);assert b.dtype==np.uint8 and hashlib.sha256(b.tobytes()).hexdigest()==r['sha256'];t=torch.from_numpy(b.copy()).view(dtypes[r['dtype']]).reshape(r['shape']);assert tr(t)==r;result[k]=t
 return result
