"""Independent original grouped Idefics2 input and label replay; no NPU execution."""
import pathlib,json,re,collections
import torch,numpy as np
from PIL import Image
from tokenizers import Tokenizer,pre_tokenizers
from jinja2.sandbox import ImmutableSandboxedEnvironment
from torchvision.transforms.v2 import functional as V
from torchvision.transforms import InterpolationMode
from idefics2_base_independent_cpu_v3_910b import sha,fp,tr,original_clean_prompt,independent_vision_positions
INSTRUCTION='\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) in the format <answer>X</answer>.'
def independent_pixels(images):
 batch=[];sizes=[]
 for image in images:
  a=torch.from_numpy(np.array(image.convert('RGB'),copy=True)).permute(2,0,1);h,w=a.shape[-2:];hy=h//2;wx=w//2;parts=[a[:,:hy,:wx],a[:,:hy,wx:],a[:,hy:,:wx],a[:,hy:,wx:],a];values=[];sh=[]
  for part in parts:
   h,w=part.shape[-2:];aspect=w/h
   if w>=h and w>980:w=980;h=int(w/aspect)
   elif h>w and h>980:h=980;w=int(h*aspect)
   h=max(h,378);w=max(w,378);resized=V.resize(part,[h,w],InterpolationMode.BILINEAR,antialias=True);values.append((resized.float()-127.5)/127.5);sh.append([h,w])
  batch.append(values);sizes.append(sh)
 h=max(x.shape[-2] for group in batch for x in group);w=max(x.shape[-1] for group in batch for x in group);pixels=torch.zeros((len(images),5,3,h,w));mask=torch.zeros((len(images),5,h,w))
 for i,group in enumerate(batch):
  for j,value in enumerate(group):hh,ww=value.shape[-2:];pixels[i,j,:,:hh,:ww]=value;mask[i,j,:hh,:ww]=1
 return pixels,mask,sizes
def own_rust(wp):
 wp=pathlib.Path(wp);raw=json.loads((wp/'tokenizer.json').read_text());assert raw['normalizer']=={'type':'Sequence','normalizers':[{'type':'Prepend','prepend':'▁'},{'type':'Replace','pattern':{'String':' '},'content':'▁'}]} and raw['pre_tokenizer'] is None
 tok=Tokenizer.from_file(str(wp/'tokenizer.json'));tok.normalizer=None;tok.pre_tokenizer=pre_tokenizers.Metaspace(replacement='▁',prepend_scheme='first',split=False);return tok
def independent_group(wp,protocol,data,indices):
 wp=pathlib.Path(wp);dataset=pathlib.Path(protocol['dataset_path']);rust=own_rust(wp);config=json.loads((wp/'tokenizer_config.json').read_text());template=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True).from_string(protocol['own_chat_template']);images=[];ids=[];texts=[]
 def token(v):return v['content'] if isinstance(v,dict) else v
 for i in indices:
  with Image.open(dataset.parent/data[i]['media_path']) as image:images.append(image.convert('RGB'))
  prompt=original_clean_prompt(data[i]['prompt'])+INSTRUCTION;messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
  text=template.render(messages=messages,add_generation_prompt=True,bos_token=token(config['bos_token']),eos_token=token(config['eos_token']),pad_token=token(config['pad_token']));texts.append(text)
  expanded=text.replace('<image>',('<fake_token_around_image>'+'<image>'*64+'<fake_token_around_image>')*5).replace('<fake_token_around_image><fake_token_around_image>','<fake_token_around_image>');expanded=re.sub(r'<fake_token_around_image>(?=[^\s<])','<fake_token_around_image> ',expanded);ids.append(rust.encode(expanded,add_special_tokens=True).ids)
 longest=max(map(len,ids));inputs=torch.tensor([[0]*(longest-len(row))+row for row in ids]);attention=torch.tensor([[0]*(longest-len(row))+[1]*len(row) for row in ids]);pixels,mask,sizes=independent_pixels(images)
 return {'input_ids':inputs,'attention_mask':attention,'pixel_values':pixels,'pixel_attention_mask':mask},texts,sizes
def groups_for(data,dataset,count):
 groups=[]
 for start in range(0,count,10):
  grouped=collections.OrderedDict()
  for i in range(start,min(start+10,count)):
   with Image.open(pathlib.Path(dataset).parent/data[i]['media_path']) as image:key=(image.width//64,image.height//64)
   grouped.setdefault(key,[]).append(i)
  groups.extend(grouped.values())
 return groups
def independent_choice(raw):
 if not isinstance(raw,str):return None
 raw=raw.strip()
 for rule in [r'<([A-E])>',r'\[([A-E])\]',r'\(([A-E])\)',r'\*\*([A-E])\*\*',r'([A-E])']:
  match=re.fullmatch(rule,raw,re.I)
  if match:return match[1].upper()
 match=re.fullmatch(r'<([A-E])>\s*([A-E])\s*</([A-E])>',raw,re.I)
 if match and len({s.upper() for s in match.groups()})==1:return match[1].upper()
 for pattern in [r'<answer>\s*([A-E])\s*</answer>',r'\\boxed\{\s*([A-E])\s*\}']:
  values=re.findall(pattern,raw,re.I)
  if values:return values[-1].upper()
 prefixes=[r'(?i:(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*:?|:)\s*)',r'(?i:(?:correct\s+)?option\s+(?:is\s+)?\s*)',r'(?i:(?:choose|select|pick)\s+)']
 for i,prefix in enumerate(prefixes):
  values=re.findall(prefix+r'[\'"(\[]\s*([A-Ea-e])\s*\.?\s*[\'"\])]',raw)
  if values:return values[-1].upper()
  values=re.findall(prefix+(r'([A-Ea-e])' if i==1 else r'([A-E])')+r'(?=[\s.,:;!?]|$)',raw)
  if values:return values[-1].upper()
 match=re.match(r'^([A-Ea-e])\s*[.)](?=\s|$)',raw)
 if match:return match[1].upper()
 values=re.findall(r'(?m)^\s*([A-Ea-e])\s*$',raw);return values[-1].upper() if values else None
