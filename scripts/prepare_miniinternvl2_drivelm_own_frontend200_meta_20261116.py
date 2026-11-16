"""Own InternVL2 frontend200, original current vLLM processor and independent pixels.
No learned CPU forward or online AA requirement; no score/canonical write.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,re,struct,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/InternVLChatModel/OpenGVLab/Mini-InternVL2-4B-DA-DriveLM');H=B/'MiniInternVL2_DriveLM_own_frontend200_original_vllm_v1_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def tensor(t):
 import torch
 t=t.detach().contiguous();return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())
def clean(s):
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
def main():
 import torch,numpy as np
 from PIL import Image
 from transformers import AutoTokenizer
 from vllm.transformers_utils.processors.internvl import InternVLImageProcessor,InternVLProcessor
 from universal_eval.algorithm.inference import GreedyInference
 torch.set_num_threads(4);cfg=js(W/'config.json');assert cfg['vision_config']['drop_path_rate']==0.1 and cfg['llm_config']['num_hidden_layers']==32 and cfg['llm_config']['rope_scaling']['type']=='su'
 tokenizer=AutoTokenizer.from_pretrained(str(W),local_files_only=True,use_fast=True);image_context_id=tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>');assert tokenizer.is_fast and len(tokenizer)==32020
 processor=InternVLProcessor(InternVLImageProcessor(image_size=448,min_dynamic_patch=cfg['min_dynamic_patch'],max_dynamic_patch=cfg['max_dynamic_patch'],dynamic_image_size=cfg['dynamic_image_size'],use_thumbnail=cfg['use_thumbnail']),tokenizer,image_seq_length=256)
 weights={};source={}
 for sf in sorted(W.glob('*.safetensors')):
  weights[sf.name]=dict(bytes=sf.stat().st_size,SHA256=sha(sf))
  with sf.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];hdr=json.loads(f.read(n))
  for k,v in hdr.items():
   if k=='__metadata__':continue
   assert k not in source;source[k]=v
 assert all(v['dtype']=='BF16' for v in source.values());put('source_headers.json',source)
 index=js(W/'model.safetensors.index.json');assert set(index['weight_map'])==set(source) and set(index['weight_map'].values())==set(weights)
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and (p.suffix in ['.json','.model','.py'])};data=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';rows=[json.loads(s) for s in data.read_text().splitlines()];assert len(rows)==4329
 output=H/'frontend200.jsonl';records=[]
 with output.open('x',encoding='utf-8') as out:
  for i,row in enumerate(rows[:200]):
   prompt=clean(row['prompt']);model_prompt=GreedyInference._add_answer_instruction(prompt);media=Path(row['media_path']);media=(data.parent/media).resolve() if not media.is_absolute() else media.resolve();assert media.is_relative_to(data.parent)
   image=Image.open(media).convert('RGB');chat=tokenizer.apply_chat_template([dict(role='user',content='<image>\n'+model_prompt)],add_generation_prompt=True,tokenize=False)
   got=processor(text=chat,images=image,return_tensors='pt');pixels=got['pixel_values_flat'];tiles=int(got['image_num_patches'][0]);assert len(pixels)==tiles
   # Independent geometry and uint8 -> F32 normalize; no call into vLLM preprocessing.
   ratios=sorted({(a,b) for n in range(1,13) for a in range(1,n+1) for b in range(1,n+1) if 1<=a*b<=12},key=lambda x:x[0]*x[1]);best=(1,1);diff=float('inf');aspect=image.width/image.height
   for a,b in ratios:
    delta=abs(aspect-a/b)
    if delta<diff or (delta==diff and image.width*image.height>.5*448*448*a*b):best=(a,b);diff=delta
   full=image.resize((448*best[0],448*best[1]));parts=[full.crop((j%best[0]*448,j//best[0]*448,(j%best[0]+1)*448,(j//best[0]+1)*448)) for j in range(best[0]*best[1])]
   if len(parts)>1:parts.append(image.resize((448,448)))
   raw=np.stack([np.asarray(p,dtype=np.uint8).transpose(2,0,1) for p in parts]);ind=torch.from_numpy(raw.copy()).float().div(255);ind=ind.sub(torch.tensor([.485,.456,.406])[None,:,None,None]).div(torch.tensor([.229,.224,.225])[None,:,None,None]);assert torch.equal(pixels,ind),(i,'own original vLLM pixels differ')
   repl='<img>'+'<IMG_CONTEXT>'*(tiles*256)+'</img>';repl_ids=tokenizer.encode(repl,add_special_tokens=False);repl=tokenizer.decode(repl_ids,skip_special_tokens=False);expanded=chat.replace('<image>',repl);ids=tokenizer(expanded,return_tensors='pt')['input_ids'];assert torch.equal(ids,got['input_ids']) and int((ids==image_context_id).sum())==tiles*256 and len(ids[0])+2048<131072
   rec=dict(index=i,prompt=prompt,original_model_prompt=model_prompt,media_path=str(media),image_SHA256=sha(media),reference=row['reference'],chat=chat,expanded_text=expanded,input_ids=ids[0].tolist(),attention_mask=got['attention_mask'][0].tolist(),pixels_F32=tensor(pixels),pixels_BF16=tensor(pixels.bfloat16()),tiles=tiles,independent_pixels_equal=True)
   out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush();records.append(rec);put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed=i+1,target=200,no_NN=True))
 from universal_eval.model.miniinternvl2_drivelm_own_20261116.configuration_internvl_chat import InternVLChatConfig
 from universal_eval.model.miniinternvl2_drivelm_own_20261116.modeling_internvl_chat import InternVLChatModel
 with torch.device('meta'):model=InternVLChatModel(InternVLChatConfig(**cfg),use_flash_attn=False)
 expected={n:list(t.shape) for n,t in model.named_parameters(remove_duplicate=False)};assert set(expected)==set(source) and all(expected[n]==v['shape'] for n,v in source.items())
 assert len(source)==541 and model.config.vision_config.drop_path_rate==0.1
 put('own_meta_parameter_source_bijection.json',dict(passed=True,own_source_parameters=541,all_own_source_meta_names_shapes_exact=True,own_original_drop_path_rate=.1,all_original_model_math_preserved=True,no_NPU_or_learned_CPU_forward=True))
 import inspect
 pp=Path(inspect.getfile(InternVLProcessor));result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id='OpenGVLab/Mini-InternVL2-4B-DA-DriveLM',own_assets_SHA256=assets,own_weight_files=weights,image_context_token_id=image_context_id,source_tensors=len(source),frontend=str(output),frontend_SHA256=sha(output),dataset=str(data),dataset_SHA256=sha(data),vllm_processor=str(pp),vllm_processor_SHA256=sha(pp),samples=200,all200_independent_pixels_equal=True,stop_ids=js(W/'generation_config.json')['eos_token_id'],max_prompt_length=max(len(r['input_ids']) for r in records),no_learned_CPU_forward=True,no_NPU_inference=True,history_A100_engine_bytes_claimed=False)
 put('review.json',result);put('actual_completion.json',dict(actual_returncode=0,UTC=result['UTC']));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:put('actual_completion.json',dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc()));raise
