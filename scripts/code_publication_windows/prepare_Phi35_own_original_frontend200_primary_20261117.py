import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast
P=Path(__file__).parent
packet=json.loads((P/'R_Phi35_own_runtime_native_full_source_headers_20261117.json').read_bytes())
code='''"""Own Phi3.5 original assets and independent HD pixels/tokens; no learned NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,struct,re,math,traceback
import numpy as np
import torch
from PIL import Image,ImageOps
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
MID='microsoft/Phi-3.5-vision-instruct';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Phi3VForCausalLM/microsoft/Phi-3.5-vision-instruct');H=B/'Phi35_own_original_BF16_4096_native_candidate_frontend200_v1_no_NN_20261117'
EXPECTED=__EXPECTED__
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tm(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tsha(t))
def put(p,v):
 with Path(p).open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
def independent_pixels(image,pp):
 im=image.convert('RGB');transposed=im.width<im.height
 if transposed:im=im.transpose(Image.Transpose.TRANSPOSE)
 ratio=im.width/im.height;scale=1
 while scale*math.ceil(scale/ratio)<=pp['num_crops']:scale+=1
 scale-=1;nw=scale*336;nh=int(nw/ratio);im=im.resize((nw,nh),resample=Image.Resampling.BILINEAR)
 pad=math.ceil(nh/336)*336-nh;im=ImageOps.expand(im,(0,pad//2,0,pad-pad//2),fill=(255,255,255))
 if transposed:im=im.transpose(Image.Transpose.TRANSPOSE)
 raw=torch.from_numpy(np.array(im,dtype=np.uint8,copy=True)).permute(2,0,1).contiguous().float().div(255.)
 mean=torch.tensor(pp['image_mean'],dtype=raw.dtype).view(3,1,1);std=torch.tensor(pp['image_std'],dtype=raw.dtype).view(3,1,1);normalized=(raw-mean)/std
 h,w=normalized.shape[-2:];global_image=torch.nn.functional.interpolate(normalized[None].float(),size=(336,336),mode='bicubic').to(normalized.dtype)
 local=normalized.reshape(1,3,h//336,336,w//336,336).permute(0,2,4,1,3,5).reshape(-1,3,336,336).contiguous();crops=torch.cat((global_image,local))
 if len(crops)<pp['num_crops']+1:crops=torch.cat((crops,torch.zeros(pp['num_crops']+1-len(crops),3,336,336,dtype=crops.dtype)))
 return crops[None],torch.tensor([[h,w]],dtype=torch.long),int((h//336*w//336+1)*144+1+(h//336+1)*12)
def main():
 assert os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') is None
 torch.set_num_threads(4);H.mkdir(exist_ok=False)
 resources={}
 for name,record in EXPECTED['own_assets'].items():assert sha(name)==record['SHA256'];resources[name]=record
 headers={}
 for name,record in EXPECTED['headers'].items():
  assert sha(name)==record['SHA256'] and Path(name).stat().st_size==record['bytes'];resources[name]=dict(SHA256=record['SHA256'],bytes=record['bytes'])
  for key,value in record['tensors'].items():assert value['dtype']=='BF16' and key not in headers;headers[key]=dict(value,shard=Path(name).name)
 assert len(headers)==592
 for name,record in EXPECTED['sources'].items():assert sha(name)==record['SHA256'];resources[name]=dict(SHA256=record['SHA256'])
 config=js(W/'config.json');pp=js(W/'preprocessor_config.json');assert config['hidden_size']==3072 and config['num_hidden_layers']==32 and config['vocab_size']==32064 and config['torch_dtype']=='bfloat16' and pp['num_crops']==4
 from transformers import AutoProcessor
 processor=AutoProcessor.from_pretrained(str(W),local_files_only=True,trust_remote_code=True)
 tokenizer=processor.tokenizer;assert tokenizer.eos_token_id==2
 dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';rows=[json.loads(t) for t in dataset.read_text().splitlines() if t.strip()];assert len(rows)==4329
 index=[]
 with (H/'own_original_chat_frontend200.jsonl').open('x',encoding='utf8') as out:
  for i,row in enumerate(rows[:200]):
   prompt=row['prompt'];media=Path(row['media_path']);assert media.is_file();clean=re.sub(r'<image\\d+>\\s*','',prompt).strip()
   # Original generic vLLM _build_message emits image then text. The own
   # tokenizer accepts the processor string representation below; native
   # engine renderer token equality remains an explicit later NN guard.
   text=tokenizer.apply_chat_template([dict(role='user',content='<|image_1|>\\n'+clean)],tokenize=False,add_generation_prompt=True)
   with Image.open(media) as image:
    inp=processor(text=text,images=image.convert('RGB'),return_tensors='pt');pixels,sizes,num=independent_pixels(image,pp)
   assert torch.equal(inp['pixel_values'],pixels),(i,'own independent original HD pixels')
   assert torch.equal(inp['image_sizes'],sizes) and int((inp['input_ids']==-1).sum())==num
   chunks=[tokenizer(chunk).input_ids for chunk in re.split(r'<\\|image_\\d+\\|>',text)];assert len(chunks)==2
   expected=chunks[0]+[-1]*num+chunks[1];assert inp['input_ids'].tolist()==[expected] and torch.equal(inp['attention_mask'],torch.ones_like(inp['input_ids']))
   value=dict(index=i,original_dataset_row=row,original_prompt=prompt,original_native_message=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='file://'+str(media))),dict(type='text',text=clean)])],own_chat_template_string=text,own_HF_input_ids=expected,own_HF_attention_mask=inp['attention_mask'].tolist()[0],own_native_image_token_replacement_ID=32044,own_native_positive_image_ID_replacement=[32044 if n==-1 else n for n in expected],pixel_values_F32=tm(pixels),pixel_values_original_BF16_cast=tm(pixels.bfloat16()),image_sizes=sizes.tolist(),num_image_tokens=num,image_SHA256=sha(media),independent_pixels_and_token_composition_bitwise_equal=True,native_renderer_actual_token_equality_not_yet_claimed=True)
   out.write(json.dumps(value,ensure_ascii=False)+'\\n');index.append(dict(index=i,pixel_values_F32=tm(pixels),image_sizes=sizes.tolist(),num_image_tokens=num,prompt_SHA256=hashlib.sha256(prompt.encode()).hexdigest()))
 for name,record in resources.items():assert sha(name)==record['SHA256'],name
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id=MID,own_full_BF16_source_count=592,own_whole_weight_bytes=sum(v['bytes'] for v in EXPECTED['headers'].values()),own_original_4096_outer10_Greedy_and_config_values_unchanged=True,complete200_own_HD_pixels_and_HF_chat_token_composition_verified=True,own_frontend=str(H/'own_original_chat_frontend200.jsonl'),own_frontend_SHA256=sha(H/'own_original_chat_frontend200.jsonl'),dataset=str(dataset),dataset_SHA256=sha(dataset),all_own_assets_and_runtime_sources=resources,input_metadata=index,no_learned_CPU_or_NPU_model_forward=True,no_adapter_or_independent_full_NPU_verifier_or_model_dispatch_readiness_claim=True,original_native_chat_renderer_and_full592_NPU_source_mapping_still_required=True,original_model_math_precision_budget_beam_cache_prompt_unchanged=True)
 put(H/'review.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['input_metadata','all_own_assets_and_runtime_sources']},ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put(H/'actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_learned_NN=True))
  raise
'''
minimal={k:packet[k] for k in ['own_assets','headers','sources']}
minimal['sources']={n:{'SHA256':v['SHA256']} for n,v in minimal['sources'].items()}
code=code.replace('__EXPECTED__',repr(minimal));ast.parse(code)
name='scripts/prepare_phi35_own_original_BF16_4096_HD_frontend200_v1_no_NN_20261117.py';h=hashlib.sha256(code.encode()).hexdigest();(P/Path(name).name).write_bytes(code.encode())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';p=R/__NAME__;raw=base64.b64decode(__DATA__);assert hashlib.sha256(raw).hexdigest()==__HASH__;ast.parse(raw);assert not p.exists();p.write_bytes(raw);H=B/'Phi35_own_HD_frontend200_v1_no_NN_actual_launch_20261117';H.mkdir(exist_ok=False)
with (H/'stdout.json').open('xb') as out,(H/'stderr.log').open('xb') as err:
 c=subprocess.run(['python3','-B',str(p)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),timeout=1800)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=c.returncode,source=str(p),source_SHA256=__HASH__,stdout=(H/'stdout.json').read_text(),stderr=(H/'stderr.log').read_text(),no_learned_model_forward=True);(H/'actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False));raise SystemExit(c.returncode)
'''.replace('__NAME__',repr(name)).replace('__HASH__',repr(h)).replace('__DATA__',repr(base64.b64encode(code.encode()).decode()))
ast.parse(remote);(P/'deploy_run_Phi35_own_original_frontend200_v1_no_NN_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(source=name,SHA256=h)))
