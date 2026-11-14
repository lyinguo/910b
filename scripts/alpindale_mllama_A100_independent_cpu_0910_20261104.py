"""Independent same-model actual A100 serial1 IO; own full vocabulary and original AA template."""
import pathlib,json,math,datetime
import torch,numpy as np
from PIL import Image
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from scripts.egm8b_original_independent_cpu_910b import sha,fp,tr,clean

RATIOS=[(h,w) for h in range(1,5) for w in range(1,5) if h*w<=4]
def pixels(image):
 w,h=image.size;options=[(a*560,b*560) for a,b in RATIOS];scales=[min(a/h,b/w) for a,b in options];up=[s for s in scales if s>=1];scale=min(up) if up else max(scales);choices=[(i,v) for i,v in enumerate(options) if scales[i]==scale];idx,(ch,cw)=min(choices,key=lambda iv:iv[1][0]*iv[1][1]);th=min(max(h,560),ch);tw=min(max(w,560),cw);sh=th/h;sw=tw/w
 if sw<sh:nh=min(math.floor(h*sw) or 1,th);nw=tw
 else:nh=th;nw=min(math.floor(w*sh) or 1,tw)
 original=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1);resized=F.resize(original,[int(nh),int(nw)],interpolation=InterpolationMode.BILINEAR,antialias=True);canvas=torch.zeros((3,ch,cw),dtype=torch.uint8);canvas[:,:nh,:nw]=resized;rows,cols=RATIOS[idx];tiles=torch.stack([canvas[:,a*560:(a+1)*560,b*560:(b+1)*560] for a in range(rows) for b in range(cols)]).float();mean=torch.tensor([.48145466,.4578275,.40821073],dtype=torch.float32)*255.;std=torch.tensor([.26862954,.26130258,.27577711],dtype=torch.float32)*255.;tiles=(tiles-mean[None,:,None,None])/std[None,:,None,None];result=torch.zeros((4,3,560,560),dtype=torch.float32);result[:len(tiles)]=tiles
 return result,idx+1,rows*cols,{'canvas':[ch,cw],'resized':[int(nh),int(nw)],'ratio':[rows,cols]}

def inputs(p,indices):
 wp=pathlib.Path(p['weight_path']);rust=Tokenizer.from_file(str(wp/'tokenizer.json'));template=p['original_A100_chat_template'];env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True);env.globals.update(raise_exception=lambda s:(_ for _ in ()).throw(ValueError(s)),strftime_now=lambda fmt:datetime.datetime.now().strftime(fmt));t=env.from_string(template);texts=[];seq=[];xs=[];ar=[];nt=[];geometry=[]
 for i in indices:
  r=p['input_records'][i];text=t.render(messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':r['model_prompt']}]}],add_generation_prompt=True,bos_token='<|begin_of_text|>',eos_token='<|eot_id|>',pad_token='<|finetune_right_pad_id|>');texts.append(text);ids=rust.encode(text,add_special_tokens=False).ids;assert ids[0]==128000 and ids.count(128000)==1 and ids.count(128256)==1;seq.append(ids)
  with Image.open(r['media_path']) as im:x,ratio,n,g=pixels(im)
  xs.append(x);ar.append([ratio]);nt.append(n);geometry.append(g)
 assert len(indices)==1,'Original A100 native language batch is serial1'
 length=max(map(len,seq));ids=torch.tensor([x+[128004]*(length-len(x)) for x in seq]);mask=torch.tensor([[1]*len(x)+[0]*(length-len(x)) for x in seq]);aspect=torch.tensor([[[1]*n+[0]*(4-n)] for n in nt]);cross=torch.zeros((len(indices),length,1,4),dtype=torch.int64)
 for b,n in enumerate(nt):start=ids[b].tolist().index(128256);cross[b,start:,0,:n]=1
 return {'input_ids':ids,'attention_mask':mask,'cross_attention_mask':cross,'pixel_values':torch.stack(xs)[:,None],'aspect_ratio_ids':torch.tensor(ar),'aspect_ratio_mask':aspect},texts,geometry
