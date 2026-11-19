"""Own GLM4.6V input/source preparation only; no learned CPU or NPU forwards."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,inspect,json,math,os,struct,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='zai-org/AutoGLM-Phone-9B';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration')/MID
H=B/'AutoGLMPhone9BSingle_own_original_BF16_2048_frontend200_meta_v1_20261120'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(name,value):
 p=H/name;tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode());os.replace(tmp,p)
def now():return datetime.now(timezone.utc).isoformat()
def main():
 import numpy as np,torch,yaml
 from PIL import Image
 from tokenizers import Tokenizer
 from torchvision.transforms.v2.functional import pil_to_tensor,resize
 from torchvision.transforms import InterpolationMode
 from transformers import AutoProcessor,AutoConfig,AutoModelForImageTextToText
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 adapter_source=R/'universal_eval/model/adapters/transformers_vlm.py';adapter_text=adapter_source.read_text()
 assert 'enable_thinking: Optional[bool] = None' in adapter_text and 'if self._enable_thinking is not True:' in adapter_text and '_ct_kwargs["enable_thinking"] = False' in adapter_text
 assert 'Glm4v' in adapter_text and 'return [self.predict(prompt, media_path) for prompt, media_path in batch]' in adapter_text
 adapter_SHA=sha(adapter_source)
 def tm(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
 original=R/'configs/Glm4vForConditionalGeneration/zai-org--AutoGLM-Phone-9B.yaml'
 assert sha(original)=='a004f7d4f8c9bfeb19314f61697a293a7faae06d5f8b406cfab8e5d74fbba010'
 cfg=yaml.safe_load(original.read_bytes());assert cfg['model']['name']==MID and cfg['algorithm']['batch_size']==10 and cfg['model']['kwargs']['max_new_tokens']==2048 and 'enable_thinking' not in cfg['model']['kwargs']
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.jinja','.py','.txt']}
 files={};headers={}
 for p in sorted(W.glob('*.safetensors')):
  files[p.name]=dict(bytes=p.stat().st_size,SHA256=sha(p))
  with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
  for name,record in header.items():
   if name=='__metadata__':continue
   assert record['dtype']=='BF16' and name not in headers;headers[name]=dict(record,source_file=p.name)
 assert len(files)==5 and sum(v['bytes'] for v in files.values())==20585644112 and len(headers)==704
 put('own704_physical_source_headers.json',headers);put('own_whole_source_files.json',files)
 processor=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False)
 ip=processor.image_processor;assert type(processor).__name__=='Glm4vProcessor' and type(ip).__name__=='Glm4vImageProcessor'
 assert ip.patch_size==14 and ip.temporal_patch_size==ip.merge_size==2
 rust=Tokenizer.from_file(str(W/'tokenizer.json'));assert rust.get_vocab()==processor.tokenizer.get_vocab()
 dataset=Path(cfg['data']['dataset_path']);assert str(dataset)=='/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl' and sha(dataset)=='2f5184c8d02695ef95ef16efe0c48640a0d99c680ab06f6847ec5d1372b06451';ds=DatasetLoader().load(str(dataset));ds=TaskDataRegistry._resolve_media_paths(ds,str(dataset.parent));ds=DataPreprocessor().preprocess(ds,task_type='image-text-to-text',apply_template=cfg['data']['apply_template'],filter_empty=cfg['data']['filter_empty'])
 assert len(ds)==4329
 rows=[]
 with (H/'own_original_frontend200.jsonl').open('x',encoding='utf8') as out:
  for i,row in enumerate(ds[:200]):
   model_prompt=GreedyInference._add_answer_instruction(row['prompt'])
   messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=model_prompt)])]
   text=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
   assert text==processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False) and text.endswith('<|assistant|>\n') and '/nothink' not in text and '<think>' not in text
   with Image.open(row['media_path']) as im:image=im.convert('RGB')
   actual=processor(text=text,images=[image],return_tensors='pt',padding=True)
   height,width=image.height,image.width;factor=28
   if height<factor or width<factor:
    scale=max(factor/height,factor/width);height,width=int(height*scale),int(width*scale)
   assert max(height,width)/min(height,width)<=200
   rh=round(height/factor)*factor;rw=round(width/factor)*factor
   min_pixels=ip.size['shortest_edge'];max_pixels=ip.size['longest_edge']
   if 2*rh*rw>max_pixels:
    beta=math.sqrt(2*height*width/max_pixels);rh=max(factor,math.floor(height/beta/factor)*factor);rw=max(factor,math.floor(width/beta/factor)*factor)
   elif 2*rh*rw<min_pixels:
    beta=math.sqrt(min_pixels/(2*height*width));rh=math.ceil(height*beta/factor)*factor;rw=math.ceil(width*beta/factor)*factor
   uint8=resize(pil_to_tensor(image)[None],size=[rh,rw],interpolation=InterpolationMode.BICUBIC,antialias=True)[0]
   mean=np.asarray(ip.image_mean,dtype=np.float32).reshape(3,1,1)*np.float32(1./ip.rescale_factor)
   std=np.asarray(ip.image_std,dtype=np.float32).reshape(3,1,1)*np.float32(1./ip.rescale_factor)
   values=torch.from_numpy((uint8.numpy().astype(np.float32)-mean)/std)
   gh,gw=rh//14,rw//14
   patches=values[None,None].repeat(1,2,1,1,1).reshape(1,1,2,3,gh//2,2,14,gw//2,2,14).permute(0,1,4,7,5,8,3,2,6,9).reshape(gh*gw,1176)
   assert torch.equal(actual['pixel_values'],patches),(i,'Own independent complete resize/fused F32 normalization/patch layout',float((actual['pixel_values']-patches).abs().max()))
   assert actual['image_grid_thw'].tolist()==[[1,gh,gw]]
   expanded=text.replace(processor.image_token,processor.image_token*(gh*gw//4))
   ids=rust.encode(expanded,add_special_tokens=True).ids;assert actual['input_ids'].tolist()==[ids] and torch.equal(actual['attention_mask'],torch.ones_like(actual['input_ids']))
   value=dict(index=i,prompt=row['prompt'],actual_model_prompt=model_prompt,own_chat_template_string=text,media_path=row['media_path'],image_SHA256=sha(row['media_path']),reference=row['reference'],own_complete_input_ids=ids,all_actual_input_tensor_metadata={n:tm(t) for n,t in actual.items() if torch.is_tensor(t)},own_all_resize_fused_F32_normalization_patch_layout_and_Rust_tokens_bitwise_equal=True,original_existing_adapter_default_ignored_by_own_template_and2048_outer10_inner1_preserved=True)
   if i in [0,99,199]:np.savez_compressed(H/('own_complete_input_example_'+str(i)+'.npz'),**{n:t.detach().cpu().numpy() for n,t in actual.items() if torch.is_tensor(t)})
   out.write(json.dumps(value,ensure_ascii=False)+'\n');rows.append(value);put('progress.json',dict(UTC=now(),completed_samples=i+1,total_samples=200,no_learned_CPU_or_NPU=True))
 cfg_obj=AutoConfig.from_pretrained(W,local_files_only=True,trust_remote_code=False)
 with torch.device('meta'):model=AutoModelForImageTextToText.from_config(cfg_obj)
 params=dict(model.named_parameters(remove_duplicate=False));meta={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in params.items()}
 diff=dict(physical_not_meta=sorted(set(headers)-set(meta)),meta_not_physical=sorted(set(meta)-set(headers)),shape_mismatch={n:dict(physical=headers[n]['shape'],meta=meta[n]['shape']) for n in set(headers)&set(meta) if headers[n]['shape']!=meta[n]['shape']})
 put('own_meta_only_parameter_layout.json',dict(actual_class=type(model).__name__,meta=meta,mapping_review=diff,no_learned_forward=True))
 assert sha(adapter_source)==adapter_SHA
 for name,h in assets.items():assert sha(W/name)==h,name
 for name,record in files.items():assert sha(W/name)==record['SHA256'],name
 imports={inspect.getfile(type(x)):sha(inspect.getfile(type(x))) for x in [processor,ip,model,cfg_obj]}
 result=dict(UTC=now(),passed=True,model_id=MID,own_original_saved_YAML=str(original),own_original_saved_YAML_SHA256=sha(original),own_physical_BF16_sources=704,own_whole_weight_bytes=20585644112,own_weight_files=files,own_assets_SHA256=assets,own_source_headers=str(H/'own704_physical_source_headers.json'),own_source_headers_SHA256=sha(H/'own704_physical_source_headers.json'),own_frontend200=str(H/'own_original_frontend200.jsonl'),own_frontend200_SHA256=sha(H/'own_original_frontend200.jsonl'),dataset=str(dataset),dataset_SHA256=sha(dataset),all200_own_original_input_pixels_tokens_refs_verified=True,actual_original_existing_thinking_false_preserved=True,original_generic_adapter_SHA256=adapter_SHA,original_generic_adapter_path=str(adapter_source),max_new_tokens=2048,outer_batch=10,inner_serial=1,own_original_generate_greedy_overrides=dict(do_sample=False),own_original_generation_EOS=js(W/'generation_config.json')['eos_token_id'],actual_installed_meta_class=type(model).__name__,meta_mapping=diff,meta_map_complete_without_conversion=not any(diff.values()),installed_source_SHA256=imports,original_generic_adapter_initializer_not_executed=True,no_learned_CPU_or_NPU_inference=True,no_NPU_adapter_fullNF_or_dispatch_readiness_claim=True,A100_historical_engine_or_tensor_capture_claimed=False)
 put('review.json',result);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_model_inference=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_model_inference=True))
  raise
