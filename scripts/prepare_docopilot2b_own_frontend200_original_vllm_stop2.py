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
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/InternVLChatModel/OpenGVLab/Docopilot-2B');H=B/'Docopilot2B_own_frontend200_original_vllm_stop2_1535_20261115'
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
 from universal_eval.model.docopilot2b_own_native_20261115.tokenization_internlm2 import InternLM2Tokenizer
 from vllm.transformers_utils.processors.internvl import InternVLImageProcessor,InternVLProcessor
 from universal_eval.algorithm.inference import GreedyInference
 torch.set_num_threads(4);cfg=js(W/'config.json');assert cfg['vision_config']['drop_path_rate']==0.1 and cfg['llm_config']['num_hidden_layers']==24 and cfg['llm_config']['rope_scaling']==dict(factor=2.0,type='dynamic')
 tokenizer=InternLM2Tokenizer.from_pretrained(str(W),local_files_only=True);assert len(tokenizer)==92553
 from transformers import GenerationConfig
 from vllm import SamplingParams
 generation=GenerationConfig.from_pretrained(str(W),local_files_only=True).to_diff_dict();assert generation.get('eos_token_id') is None and tokenizer.eos_token_id==2
 sampling=SamplingParams(temperature=0,top_p=1,max_tokens=2048);sampling.update_from_generation_config(generation,tokenizer.eos_token_id);assert sampling.eos_token_id==2 and sampling.all_stop_token_ids=={2}
 put('original_vllm_stop_defaults.json',dict(passed=True,own_generation_config=generation,own_tokenizer_EOS=tokenizer.eos_token_id,effective_stop_ids=[2],no_added_im_end_or_early_EOS=True,max_tokens=2048,HF_use_cache_config=cfg['llm_config']['use_cache'],saved_runtime='vLLM persistent engine KV; native port uses explicit tuple KV',historical_engine_bytes_captured=False))
 processor=InternVLProcessor(InternVLImageProcessor(image_size=448,min_dynamic_patch=cfg['min_dynamic_patch'],max_dynamic_patch=cfg['max_dynamic_patch'],dynamic_image_size=cfg['dynamic_image_size'],use_thumbnail=cfg['use_thumbnail']),tokenizer,image_seq_length=256)
 sf=W/'model.safetensors';weight=dict(bytes=sf.stat().st_size,SHA256=sha(sf));
 with sf.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];hdr=json.loads(f.read(n))
 source={k:v for k,v in hdr.items() if k!='__metadata__'};assert all(v['dtype']=='BF16' for v in source.values());put('source_headers.json',source)
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
   repl='<img>'+'<IMG_CONTEXT>'*(tiles*256)+'</img>';repl_ids=tokenizer.encode(repl,add_special_tokens=False);repl=tokenizer.decode(repl_ids,skip_special_tokens=False);expanded=chat.replace('<image>',repl);ids=tokenizer(expanded,return_tensors='pt')['input_ids'];assert torch.equal(ids,got['input_ids']) and int((ids==92546).sum())==tiles*256 and len(ids[0])+2048<32768
   rec=dict(index=i,prompt=prompt,original_model_prompt=model_prompt,media_path=str(media),image_SHA256=sha(media),reference=row['reference'],chat=chat,expanded_text=expanded,input_ids=ids[0].tolist(),attention_mask=got['attention_mask'][0].tolist(),pixels_F32=tensor(pixels),pixels_BF16=tensor(pixels.bfloat16()),tiles=tiles,independent_pixels_equal=True)
   out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush();records.append(rec);put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed=i+1,target=200,no_NN=True))
 import inspect
 pp=Path(inspect.getfile(InternVLProcessor));result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id='OpenGVLab/Docopilot-2B',own_assets_SHA256=assets,own_weight_files={'model.safetensors':weight},source_tensors=len(source),frontend=str(output),frontend_SHA256=sha(output),dataset=str(data),dataset_SHA256=sha(data),vllm_processor=str(pp),vllm_processor_SHA256=sha(pp),samples=200,all200_independent_pixels_equal=True,stop_ids=[tokenizer.eos_token_id],max_prompt_length=max(len(r['input_ids']) for r in records),no_learned_CPU_forward=True,no_NPU_inference=True,history_A100_engine_bytes_claimed=False)
 put('review.json',result);put('actual_completion.json',dict(actual_returncode=0,UTC=result['UTC']));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:put('actual_completion.json',dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc()));raise
