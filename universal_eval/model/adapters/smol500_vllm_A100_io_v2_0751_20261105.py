"""Own Smol500 original full-chat, PIL10.4 pass order and Rust output IO."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def fingerprint(tensor):return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def original_PIL104_resize(image,size):
    """Original uint8 LANCZOS horizontal pass followed by vertical pass.

    Pillow13 may reverse these passes. Explicit one-axis passes preserve the
    original Pillow10.4 processing, verified against all200 actual A100 inputs.
    Original images, kernel, dimensions and intermediate uint8 rounding remain.
    """
    width,height=size
    if image.width!=width:image=image.resize((width,image.height),Image.Resampling.LANCZOS)
    if image.height!=height:image=image.resize((image.width,height),Image.Resampling.LANCZOS)
    return image

class OwnSmol500IO:
    suffix='\n\nPlease answer with only the letter of the correct option (A, B, C, D, or E). Do not explain.'
    def __init__(self,root,own_resource_sha256):
        self.root=Path(root).resolve()
        for name,digest in own_resource_sha256.items():
            assert Path(name).name==name and sha(self.root/name)==digest,name
        cfg=json.loads((self.root/'preprocessor_config.json').read_text())
        assert cfg['size']=={'longest_edge':2048} and cfg['max_image_size']=={'longest_edge':512}
        assert cfg['resample']==1 and cfg['rescale_factor']==1/255 and cfg['image_mean']==cfg['image_std']==[.5,.5,.5]
        assert cfg['do_image_splitting'] and cfg['do_normalize'] and cfg['do_resize'] and cfg['do_rescale']
        self.tokenizer=Tokenizer.from_file(str(self.root/'tokenizer.json'))
        self.template_text=json.loads((self.root/'chat_template.json').read_text())['chat_template']
        self.template=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True).from_string(self.template_text)
        assert self.tokenizer.get_vocab_size(with_added_tokens=True)==49280
        assert self.tokenizer.token_to_id('<image>')==49190 and self.tokenizer.token_to_id('<end_of_utterance>')==49279
    @staticmethod
    def geometry(width,height):
        ratio=width/height
        if width>=height:rw=2048;rh=int(rw/ratio)
        else:rh=2048;rw=int(rh*ratio)
        rh=max(1,rh+rh%2);rw=max(1,rw+rw%2);ratio=rw/rh
        if rw>=rh:aw=math.ceil(rw/512)*512;ah=math.ceil(int(aw/ratio)/512)*512
        else:ah=math.ceil(rh/512)*512;aw=math.ceil(int(ah*ratio)/512)*512
        return (rw,rh),(aw,ah),ah//512,aw//512
    def render(self,prompt,grid):
        enhanced=prompt.strip()+self.suffix
        conversation=[{'role':'user','content':[{'type':'image'},{'type':'text','text':enhanced}]}]
        rendered=self.template.render(messages=conversation,add_generation_prompt=True)
        assert rendered.count('<image>')==1
        rows,cols=grid;fake='<fake_token_around_image>';im='<image>'
        tiles=''.join(fake+f'<row_{r+1}_col_{c+1}>'+im*64+('\n' if c==cols-1 else '') for r in range(rows) for c in range(cols))
        expanded=rendered.replace(im,tiles+'\n'+fake+'<global-img>'+im*64+fake)
        ids=self.tokenizer.encode(expanded,add_special_tokens=True).ids
        soft=[i for i,x in enumerate(ids) if x==49190]
        assert ids[0]==1 and len(soft)==(rows*cols+1)*64
        assert ids.count(1)==1
        return rendered,expanded,ids,soft
    def pixels(self,media_path):
        path=Path(media_path).resolve();digest=sha(path)
        with Image.open(path) as original:image=original.convert('RGB')
        first,aligned,rows,cols=self.geometry(*image.size)
        image=original_PIL104_resize(image,first);image=original_PIL104_resize(image,aligned)
        patches=[np.asarray(image.crop((c*512,r*512,(c+1)*512,(r+1)*512)),dtype=np.uint8).transpose(2,0,1) for r in range(rows) for c in range(cols)]
        patches.append(np.asarray(original_PIL104_resize(image,(512,512)),dtype=np.uint8).transpose(2,0,1))
        raw=np.stack(patches).astype(np.float64)*(1/255)
        values=(raw.astype(np.float32)-np.float32(.5))/np.float32(.5)
        pixels=torch.from_numpy(values.copy()).bfloat16();mask=torch.ones((len(patches),512,512),dtype=torch.int64)
        assert pixels.shape==(rows*cols+1,3,512,512)
        return pixels,mask,[rows,cols],digest,str(path)
    def decode(self,prompt_ids,output_ids,finish_reason):
        assert finish_reason in ('stop','length') and output_ids
        if finish_reason=='stop':assert output_ids[-1]==49279;decode_ids=output_ids[:-1]
        else:decode_ids=output_ids
        stream=DecodeStream(ids=list(prompt_ids),skip_special_tokens=True);text='';warnings=[]
        for index,token_id in enumerate(decode_ids):
            try:fragment=stream.step(self.tokenizer,int(token_id))
            except (OverflowError,TypeError) as error:
                warnings.append({'output_position':index,'token_id':int(token_id),'kind':'original_protected_invalid_token','error':str(error)});fragment=None
            except Exception as error:
                if not str(error).startswith('Invalid prefix encountered'):raise
                warnings.append({'output_position':index,'token_id':int(token_id),'kind':'original_protected_invalid_UTF8_prefix','error':str(error)})
                stream=DecodeStream(skip_special_tokens=True);fragment=stream.step(self.tokenizer,int(token_id))
            text+=fragment or ''
        return text,warnings
