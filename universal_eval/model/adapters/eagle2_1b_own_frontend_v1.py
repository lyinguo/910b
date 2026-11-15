"""Eagle2-1B own image-only path: exact saved fast tiling and own chat template.

This module never loads model weights or a generic/stub processor initializer.
The independent preparation script executes the original image math for comparison.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

class OwnEagle2Inputs:
    def __init__(self,weights):
        self.W=Path(weights);self.cfg=json.loads((self.W/'preprocessor_config.json').read_bytes())
        self.template=json.loads((self.W/'chat_template.json').read_bytes())['chat_template']
        self.tokenizer=AutoTokenizer.from_pretrained(self.W,local_files_only=True,trust_remote_code=False)
        self.tokenizer.padding_side='left'
        assert len(self.tokenizer)==151674 and self.tokenizer.eos_token_id==151645
        assert self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')==151667
        assert self.cfg['max_dynamic_tiles']==12 and self.cfg['min_dynamic_tiles']==1 and self.cfg['do_resize'] is False and self.cfg['use_thumbnail']
        assert self.cfg['image_mean']==self.cfg['image_std']==[.5,.5,.5] and self.cfg['do_pad'] is False
    def grid(self,width,height):
        ratios=set((x,y) for n in range(1,13) for x in range(1,n+1) for y in range(1,n+1) if 1<=x*y<=12)
        best=(1,1);best_factor=float('-inf');aspect=width/height
        for x,y in sorted(ratios,key=lambda p:p[0]*p[1]):
            a=x/y;factor=min(x*y*448*448/(width*height),.6)*min(a/aspect,aspect/a)
            if factor>best_factor:best,best_factor=(x,y),factor
        return best
    def pixels(self,path):
        image=Image.open(path).convert('RGB');src=torch.from_numpy(np.array(image,dtype=np.uint8,copy=True)).permute(2,0,1)
        columns,rows=self.grid(image.width,image.height);big=F.resize(src,[rows*448,columns*448],interpolation=InterpolationMode.BICUBIC,antialias=True)
        tiles=[big[:,y*448:(y+1)*448,x*448:(x+1)*448] for y in range(rows) for x in range(columns)]
        if len(tiles)>1:tiles.append(F.resize(src,[448,448],interpolation=InterpolationMode.BICUBIC,antialias=True))
        values=torch.stack(tiles).to(torch.float32).sub(127.5).div(127.5)
        return values,values.to(torch.bfloat16)
    def prepare(self,prompt,path):
        f32,bf16=self.pixels(path);messages=[dict(role='user',content=[dict(type='image',image=str(path)),dict(type='text',text=prompt)])]
        text=self.tokenizer.apply_chat_template(messages,chat_template=self.template,tokenize=False,add_generation_prompt=True)
        assert text.count('<image-1>')==1 and '<video-' not in text
        text=text.replace('<image-1>','<image 1><img>'+'<IMG_CONTEXT>'*(len(bf16)*256)+'</img>')
        enc=self.tokenizer([text],padding=True,return_tensors='pt')
        assert int((enc.input_ids==151667).sum())==len(bf16)*256
        return dict(text=text,input_ids=enc.input_ids,attention_mask=enc.attention_mask,pixels_F32=f32,pixels_BF16=bf16)
