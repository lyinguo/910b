"""Read-only own Ferret code/resources and the original Oct18 option protocol.

Own code is imported unchanged from the exact existing checkpoint. No learned
value is obtained from the declared CLIP config directory. All 576 learned
tensors must come from Ferret's two original shards.
"""
import functools,hashlib,importlib.util,json,pathlib,re,sys,types
import numpy as np
import torch
from PIL import Image

MODEL_ID='jadechoghari/Ferret-UI-Gemma2b'
WEIGHTS=pathlib.Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/FerretGemmaForCausalLM')/MODEL_ID
CLIP_CONFIG=pathlib.Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/openai/clip-vit-large-patch14-336/config.json')
CLIP_SHA='51b1c14aabcdf639c4a0370eeda1010b773bbe1df78319c7d0f5882c22ac0ac0'
REFERENCE=pathlib.Path('/mnt/afs/comp_error/workspace/reports/codex_paired_ten_ferret_full2/jadechoghari_Ferret-UI-Gemma2b/image-text-to-text/20261018_174452')
FROZEN=pathlib.Path('/mnt/afs/comp_error/workspace/diagnostics/paired_ten_20261018/runner/universal_eval/model/adapters')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fp(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def record(x):return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':fp(x)}

def own_modules():
 name='ferret_original_exact_checkpoint_0106'
 if name not in sys.modules:
  package=types.ModuleType(name);package.__path__=[str(WEIGHTS)];sys.modules[name]=package
  for part in ['constants','conversation','mm_utils','clip_encoder','ferret_arch','modeling']:
   spec=importlib.util.spec_from_file_location(name+'.'+part,WEIGHTS/(part+'.py'));module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 return {part:sys.modules[name+'.'+part] for part in ['constants','conversation','mm_utils','clip_encoder','ferret_arch','modeling']}

def processor_and_tokenizer():
 from transformers import AutoTokenizer
 from transformers.models.clip.image_processing_pil_clip import CLIPImageProcessorPil
 m=own_modules();original_resize=m['clip_encoder'].CLIPImageProcessor_Ferret.resize
 class OwnFerretPIL(CLIPImageProcessorPil):
  # Exactly the frozen parent PIL backend and own checkpoint resize function.
  def resize(self,image,size=None,resample=None,**kwargs):
   if size is not None and hasattr(size,'get') and size.get('shortest_edge') is not None and (size.get('height') is None or size.get('width') is None):size={'height':size.get('shortest_edge'),'width':size.get('shortest_edge')}
   return original_resize(self,image,size=size,resample=resample,**kwargs)
 processor=OwnFerretPIL.from_pretrained(WEIGHTS,local_files_only=True)
 tokenizer=AutoTokenizer.from_pretrained(WEIGHTS,trust_remote_code=True,use_fast=False,local_files_only=True)
 assert tokenizer.bos_token_id==2 and tokenizer.vocab_size==256000 and len(tokenizer)==256001
 return processor,tokenizer,m

def blueprint():
 from transformers import CLIPVisionModel
 assert sha(CLIP_CONFIG)==CLIP_SHA
 m=own_modules();raw=json.loads((WEIGHTS/'config.json').read_text());cfg=m['modeling'].FerretGemmaConfig(**raw)
 assert cfg.hidden_size==2048 and cfg.num_hidden_layers==18 and cfg.num_attention_heads==8 and cfg.num_key_value_heads==1 and cfg.head_dim==256 and cfg.vocab_size==256001
 assert cfg.mm_vision_tower=='openai/clip-vit-large-patch14-336' and cfg.mm_patch_merge_type=='spatial' and cfg.image_aspect_ratio=='anyres'
 cfg.mm_vision_tower=str(CLIP_CONFIG.parent) # Only resolves the very same declared component's existing config offline.
 cfg._attn_implementation='eager'
 with torch.device('meta'):
  model=m['modeling'].FerretGemmaForCausalLM(cfg)
  vt=model.get_vision_tower();vt.vision_tower=CLIPVisionModel(vt.cfg_only);vt.is_loaded=True
 cfg.mm_vision_tower=raw['mm_vision_tower']
 return model,m

def target_key(key,targets):
 if key in targets:return key
 candidate=key.replace('.vision_tower.vision_tower.vision_model.','.vision_tower.vision_tower.')
 assert candidate in targets,('No semantic native prefix mapping',key,candidate)
 return candidate

def parse_options(text):
 # Frozen FerretGemmaDiscMCVLM rule, including contiguous A..E and suffix stripping.
 clean=re.sub(r'\n+Answer the question by providing only the option letter.*$','',text,flags=re.S)
 clean=re.sub(r'\n+Provide only the final answer\.?\s*$','',clean,flags=re.S)
 marks=list(re.finditer(r'(?m)^\s*([A-E])\s*[.)]\s+',clean));options=[];expected='A'
 for i,mark in enumerate(marks):
  letter=mark.group(1)
  if letter!=expected:continue
  value=clean[mark.end():marks[i+1].start() if i+1<len(marks) else len(clean)].strip()
  if not value:continue
  options.append((letter,value));expected=chr(ord(expected)+1)
 return clean,options

def prepare(prompt,media_path,processor,tokenizer,modules):
 clean,options=parse_options(prompt);assert options,'Original generation fallback required: no contiguous options'
 image=Image.open(media_path).convert('RGB');image_size=image.size
 cfg=types.SimpleNamespace(**json.loads((WEIGHTS/'config.json').read_text()))
 func=functools.partial(processor.preprocess,return_tensors='pt',do_resize=True,do_center_crop=False,size=[336,336])
 pixels=modules['mm_utils'].process_images([image],processor,cfg,image_process_func=func)[0]
 conv=modules['conversation'].conv_templates['ferret_gemma_instruct'].copy();conv.append_message(conv.roles[0],'<image>\n'+clean);conv.append_message(conv.roles[1],None);base=conv.get_prompt()
 base_ids=modules['mm_utils'].tokenizer_image_token(base,tokenizer,-200)
 merged_base=len(base_ids)-1+576*pixels.shape[0]
 assert merged_base<=4096,'Original fallback required: merged prompt too long'
 candidates=[]
 for letter,value in options:
  continuation=f'{letter}. {value}<end_of_turn>';n=len(tokenizer(continuation,add_special_tokens=False)['input_ids'])
  ids=modules['mm_utils'].tokenizer_image_token(base+continuation,tokenizer,-200)
  labels=torch.tensor(ids,dtype=torch.long);labels[:-n]=-100;labels[0]=-100
  candidates.append({'letter':letter,'option':value,'continuation':continuation,'continuation_token_count':n,'input_ids':ids,'labels':labels.tolist(),'merged_length_before_original_4096_truncation':len(ids)-1+576*pixels.shape[0]})
 return {'clean':clean,'base_prompt':base,'base_ids':base_ids,'image_size':list(image_size),'pixel_values':pixels,'merged_base':merged_base,'candidates':candidates}

def independent_pixels(image,grid):
 # Independently select the original stretch grid (no padding) and hand execute
 # PIL default RGB resize + own CLIP BICUBIC resize/rescale/normalization.
 w,h=image.size;scores=[]
 for j,(gw,gh) in enumerate(grid):
  ratio=min(gw/w,gh/h);effective=min(int(w*ratio)*int(h*ratio),w*h);scores.append((-effective,gw*gh-effective,j,gw,gh))
 _,_,_,gw,gh=min(scores);stretched=image.resize((gw,gh))
 pieces=[image.resize((336,336))]+[stretched.crop((x,y,x+336,y+336)) for y in range(0,gh,336) for x in range(0,gw,336)]
 pc=json.loads((WEIGHTS/'preprocessor_config.json').read_text());arrays=[]
 for piece in pieces:
  piece=piece.resize((336,336),resample=Image.Resampling.BICUBIC)
  x=(np.asarray(piece).astype(np.float64)*pc['rescale_factor']).astype(np.float32)
  x=(x-np.array(pc['image_mean'],np.float32))/np.array(pc['image_std'],np.float32);arrays.append(x.transpose(2,0,1).copy())
 return torch.from_numpy(np.stack(arrays))
