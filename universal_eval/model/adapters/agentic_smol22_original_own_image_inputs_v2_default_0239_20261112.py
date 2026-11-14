"""Own Agentic SmolVLM2-2.2B image-only input math. No optional video dependency or borrowed resources."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def original_pil_resize(image,size):
    width,height=size
    # Original AA Pillow10.4 two-pass order; current Pillow changes pass order.
    if image.width!=width:image=image.resize((width,image.height),Image.Resampling.LANCZOS)
    if image.height!=height:image=image.resize((image.width,height),Image.Resampling.LANCZOS)
    return image

def image_only_arrays(path):
    with Image.open(path) as img:image=img.convert('RGB')
    w,h=image.size;ratio=w/h
    if w>=h:
        w=1152;h=int(w/ratio);h+=h%2
    else:
        h=1152;w=int(h*ratio);w+=w%2
    w=max(w,1);h=max(h,1)
    image=original_pil_resize(image,(w,h));ratio=w/h
    if w>=h:
        w=math.ceil(w/384)*384;h=math.ceil(int(w/ratio)/384)*384
    else:
        h=math.ceil(h/384)*384;w=math.ceil(int(h*ratio)/384)*384
    image=original_pil_resize(image,(w,h))
    rows,cols=h//384,w//384
    if rows>1 or cols>1:
        tiles=[image.crop((c*384,r*384,(c+1)*384,(r+1)*384)) for r in range(rows) for c in range(cols)]
        tiles.append(original_pil_resize(image,(384,384)))
    else:tiles=[image];rows=cols=0
    raw=np.stack([np.asarray(x,dtype=np.uint8).transpose(2,0,1) for x in tiles])
    scaled=(raw.astype(np.float64)*(1/255)).astype(np.float32)
    value=(scaled-np.float32(.5))/np.float32(.5)
    return dict(pixel_values=value[None],pixel_attention_mask=np.ones((1,len(tiles),384,384),dtype=np.int64)),rows,cols

def expanded_image_string(rows,cols):
    fake='<fake_token_around_image>';image='<image>';glob='<global-img>';seq=81
    if rows==cols==0:return fake+glob+image*seq+fake
    text=''
    for r in range(rows):
        for c in range(cols):text+=fake+f'<row_{r+1}_col_{c+1}>'+image*seq
        text+='\n'
    return text+'\n'+fake+glob+image*seq+fake

class AgenticSmol22OwnImageInputs:
    def __init__(self,weights,original_template_path):
        self.W=Path(weights);cfg=js(self.W/'config.json');pc=js(self.W/'processor_config.json');ip=js(self.W/'preprocessor_config.json')
        assert cfg['scale_factor']==3 and cfg['image_token_id']==49190 and cfg['text_config']['hidden_size']==2048 and cfg['text_config']['num_hidden_layers']==24
        assert pc['image_seq_len']==81
        self.rust=Tokenizer.from_file(str(self.W/'tokenizer.json'));self.kw=js(self.W/'tokenizer_config.json')
        default=js(original_template_path);assert default['template_origin'].startswith('own original actual SmolVLMProcessor')
        template=default['actual_chat_template'];assert hashlib.sha256(template.encode()).hexdigest()==default['actual_chat_template_SHA256']
        if isinstance(template,list):template=next(t['template'] for t in template if t['name']=='default')
        self.template=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols']).from_string(template)
        self.special=js(self.W/'special_tokens_map.json')
        self.special={k:(v.get('content') if isinstance(v,dict) else v) for k,v in self.special.items()}
        assert self.rust.token_to_id('<image>')==49190
    def prepare(self,prompt,path):
        messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])]
        text=self.template.render(messages=messages,add_generation_prompt=True,**self.special)
        images,rows,cols=image_only_arrays(path);expanded=text.replace('<image>',expanded_image_string(rows,cols),1)
        ids=np.asarray([self.rust.encode(expanded,add_special_tokens=True).ids],dtype=np.int64)
        return dict(images,input_ids=ids,attention_mask=np.ones_like(ids)),dict(actual_text=text,expanded_text=expanded,grid=[rows,cols],image_count=images['pixel_values'].shape[1])
