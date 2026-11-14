"""Independent CPU image/token/M-RoPE replay for the frozen MinerU serial protocol."""
import pathlib,json,hashlib,math,re,itertools
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF
from torchvision.transforms import InterpolationMode
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment
INSTRUCTION='\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) in the format <answer>X</answer>.'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fp(t)}
def clean(s):
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',str(s));s=re.sub(r'[ \t]+',' ',s);return re.sub(r'<image\d+>\s*','',re.sub(r'\n{3,}','\n\n',s).strip())
def manual_pixels(image,resource):
 image=image.convert('RGB');w,h=image.size;factor=resource['patch_size']*resource['merge_size'];hn=round(h/factor)*factor;wn=round(w/factor)*factor
 if hn*wn>resource['max_pixels']:
  scale=math.sqrt(h*w/resource['max_pixels']);hn=max(factor,math.floor(h/scale/factor)*factor);wn=max(factor,math.floor(w/scale/factor)*factor)
 elif hn*wn<resource['min_pixels']:
  scale=math.sqrt(resource['min_pixels']/(h*w));hn=math.ceil(h*scale/factor)*factor;wn=math.ceil(w*scale/factor)*factor
 # Preserve the real uint8 bicubic antialias kernel, then the original fused FP32 normalization.
 x=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).unsqueeze(0)
 x=tvF.resize(x,[hn,wn],interpolation=InterpolationMode.BICUBIC,antialias=True).float()[0]
 scale=1/255;mean=(torch.tensor(resource['image_mean'],dtype=torch.float32)*(1/scale)).view(3,1,1);std=(torch.tensor(resource['image_std'],dtype=torch.float32)*(1/scale)).view(3,1,1);x=(x-mean)/std
 patch=resource['patch_size'];merge=resource['merge_size'];gh=hn//patch;gw=wn//patch
 blocks=x.unfold(1,patch,patch).unfold(2,patch,patch).permute(1,2,0,3,4).contiguous()
 order=[(r,c) for br in range(gh//merge) for bc in range(gw//merge) for r in range(br*merge,(br+1)*merge) for c in range(bc*merge,(bc+1)*merge)]
 r,c=zip(*order);ordered=blocks[list(r),list(c)];pixels=ordered.unsqueeze(2).expand(-1,-1,resource['temporal_patch_size'],-1,-1).reshape(gh*gw,-1).contiguous();grid=torch.tensor([[1,gh,gw]],dtype=torch.int64)
 return pixels,grid,torch.tensor(order,dtype=torch.int64)
def manual_inputs(root,prompt,media):
 root=pathlib.Path(root);pc=json.loads((root/'preprocessor_config.json').read_text());tc=json.loads((root/'tokenizer_config.json').read_text());template=(root/'chat_template.jinja').read_text();messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':re.sub(r'<image\d+>\s*','',prompt).strip()}]}]
 environment=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
 chat=environment.from_string(template).render(messages=messages,add_generation_prompt=True,bos_token=None,eos_token=tc['eos_token'],pad_token=tc['pad_token'])
 with Image.open(media) as image:pixels,grid,vision_position=manual_pixels(image,pc)
 n=int(grid.prod())//(pc['merge_size']**2);expanded=chat.replace('<|image_pad|>','<|image_pad|>'*n);rust=Tokenizer.from_file(str(root/'tokenizer.json'));ids=torch.tensor([rust.encode(expanded,add_special_tokens=True).ids],dtype=torch.int64);mm=(ids==151655).to(torch.int64)
 return {'pixel_values':pixels,'image_grid_thw':grid,'input_ids':ids,'attention_mask':torch.ones_like(ids),'mm_token_type_ids':mm},chat,vision_position
def manual_mrope(ids,grid):
 ids=np.asarray(ids,dtype=np.int64).reshape(-1);loc=np.flatnonzero(ids==151655);assert len(loc)>0 and np.array_equal(loc,np.arange(loc[0],loc[-1]+1));start=int(loc[0]);t,h,w=[int(v) for v in grid];h//=2;w//=2;assert t==1 and len(loc)==h*w
 pos=np.zeros((3,1,len(ids)),dtype=np.int64);pos[:,:,:start]=np.arange(start)[None,None,:]
 pos[0,0,start:start+len(loc)]=start;pos[1,0,start:start+len(loc)]=np.repeat(np.arange(h),w)+start;pos[2,0,start:start+len(loc)]=np.tile(np.arange(w),h)+start
 end=start+len(loc);pos[:,:,end:]=np.arange(start+max(h,w),start+max(h,w)+len(ids)-end)[None,None,:]
 return torch.from_numpy(pos),int(pos.max())+1-len(ids)
