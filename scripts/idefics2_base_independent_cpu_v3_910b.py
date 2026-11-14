"""Independent replay of Idefics2-base original single-image mc_score inputs."""
import pathlib,json,hashlib,re
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer,pre_tokenizers
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fp(t)}
def options(prompt):
 found=[]
 for x in re.findall(r'(?:^|\n)\s*([A-Ea-e])\.\s+',prompt or '',re.M):
  if x.upper() not in found:found.append(x.upper())
 return found or ['A','B','C','D']
def independent_inputs(wp,prompt,media):
 wp=pathlib.Path(wp);raw=json.loads((wp/'preprocessor_config.json').read_text())
 assert raw['do_image_splitting'] is False and raw['resample']==2 and raw['size']=={'longest_edge':980,'shortest_edge':378}
 with Image.open(media) as im:im=im.convert('RGB');arr=np.array(im,copy=True)
 h,w=arr.shape[:2];aspect=w/h
 if w>=h and w>980:w=980;h=int(w/aspect)
 elif h>w and h>980:h=980;w=int(h*aspect)
 h=max(h,378);w=max(w,378)
 # Original Transformers5.14.1 AutoProcessor selects TorchvisionBackend. Resize the
 # integer image with its native antialiased bilinear kernel before FP32 normalization.
 u8=torch.from_numpy(arr).permute(2,0,1)
 resized=F.resize(u8,[h,w],InterpolationMode.BILINEAR,antialias=True)
 pixels=((resized.float()-torch.tensor(127.5,dtype=torch.float32))/torch.tensor(127.5,dtype=torch.float32))[None,None]
 mask=torch.ones((1,1,h,w),dtype=torch.float32)
 prefix='<image>\n'+prompt.rstrip('\n')+'\nAnswer:'
 expanded=prefix.replace('<image>','<fake_token_around_image>'+'<image>'*64+'<fake_token_around_image>')
 raw_tok=json.loads((wp/'tokenizer.json').read_text());assert raw_tok['normalizer']=={'type':'Sequence','normalizers':[{'type':'Prepend','prepend':'▁'},{'type':'Replace','pattern':{'String':' '},'content':'▁'}]} and raw_tok['pre_tokenizer'] is None
 rust=Tokenizer.from_file(str(wp/'tokenizer.json'));rust.normalizer=None;rust.pre_tokenizer=pre_tokenizers.Metaspace(replacement='▁',prepend_scheme='first',split=False);ids=rust.encode(expanded,add_special_tokens=True).ids
 candidates={}
 for opt in options(prompt):
  encoded=rust.encode(opt.strip(),add_special_tokens=False).ids
  if not encoded:encoded=rust.encode(' '+opt.strip(),add_special_tokens=False).ids
  assert encoded;candidates[opt]=encoded[0]
 return {'prefix':prefix,'expanded':expanded,'input_ids':torch.tensor([ids],dtype=torch.int64),'attention_mask':torch.ones((1,len(ids)),dtype=torch.int64),'pixel_values':pixels,'pixel_attention_mask':mask,'candidate_first_token_ids':candidates,'resized_hw':[h,w]}
def independent_vision_positions(pixel_mask):
 m=pixel_mask.bool().reshape(-1,*pixel_mask.shape[-2:])
 # Independent block reshape instead of native double unfold.
 nh=m.shape[-2]//14;nw=m.shape[-1]//14
 blocks=m[:,:nh*14,:nw*14].reshape(m.shape[0],nh,14,nw,14)
 patch=blocks.all(2).all(-1)
 positions=torch.zeros((patch.shape[0],nh*nw),dtype=torch.int64)
 boundaries=torch.arange(1/70,1,1/70,dtype=torch.float32)
 for b in range(len(patch)):
  h=int(patch[b,:,0].sum());w=int(patch[b,0,:].sum())
  hc=(torch.arange(h,dtype=torch.float32)*(1/h)).clamp(max=1-1e-6).bfloat16()
  wc=(torch.arange(w,dtype=torch.float32)*(1/w)).clamp(max=1-1e-6).bfloat16()
  # CPU comparison count independently implements bucketize(right=True).
  hi=(hc.float()[:,None]>=boundaries[None]).sum(-1);wi=(wc.float()[:,None]>=boundaries[None]).sum(-1)
  idx=(hi[:,None]*70+wi[None]).reshape(-1);positions[b,patch[b].reshape(-1)]=idx
 return patch,positions

def original_clean_prompt(text):
 text=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',str(text));text=re.sub(r'[ \t]+',' ',text);text=re.sub(r'\n{3,}','\n\n',text).strip();return re.sub(r'<image\d+>\s*','',text)
