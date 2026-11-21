"""Verify this checkpoint's original native chat inputs; no learned CPU/NPU run."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,contextlib,hashlib,inspect,json,math,os,re,struct,traceback,types
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
MID='zai-org/GLM-4.1V-9B-Thinking';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration')/MID
H=B/'GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v1_20261120'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def now():return datetime.now(timezone.utc).isoformat()
def main():
 os.chdir(str(R));os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 import numpy as np,torch,yaml
 from PIL import Image
 from tokenizers import Tokenizer
 from torchvision.transforms.v2.functional import pil_to_tensor,resize
 from torchvision.transforms import InterpolationMode
 from transformers import AutoProcessor,AutoConfig,AutoModelForImageTextToText
 from vllm.engine.arg_utils import EngineArgs
 from vllm.renderers.registry import renderer_from_config
 from vllm.renderers.params import ChatParams
 from vllm.multimodal.media.image import ImageMediaIO
 from vllm.v1.engine.input_processor import InputProcessor
 from vllm import SamplingParams
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 original=R/'configs/Glm4vForConditionalGeneration/zai-org--GLM-4.1V-9B-Thinking.yaml';assert sha(original)=='1d45b618d8f46d022ec0577ef460fa4c68346c08aadd35d225074ee7b5c00dbf'
 cfg=yaml.safe_load(original.read_bytes());assert cfg['model']['name']==MID and cfg['algorithm']['batch_size']==10 and cfg['model']['kwargs']['max_tokens']==8192
 assert cfg['model']['adapter']=='universal_eval/model/adapters/vllm_adapter.py:VLLMModel' and 'chat_template_kwargs' not in cfg['model']['kwargs'] and 'dtype' not in cfg['model']['kwargs'] and 'max_model_len' not in cfg['model']['kwargs']
 source=R/'universal_eval/model/adapters/vllm_adapter.py';source_sha=sha(source);assert source_sha=='da65f91d00696a2becf43121c793de985eb8364fd18f2a0164ff8d53541602db'
 tree=ast.parse(source.read_bytes());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VLLMModel')
 fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='_build_message');regex=next(n for n in cl.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_IMAGE_PLACEHOLDER_RE' for x in n.targets))
 # Execute only the original pure input method and regex. No generic model
 # constructor, fallback decoder, engine, synthetic library, or learned forward.
 ns=dict(re=re,os=os,Optional=__import__('typing').Optional,List=list,Dict=dict,Any=object)
 exec(compile(ast.Module(body=[regex,fn],type_ignores=[]),str(source),'exec'),ns)
 original_input=types.SimpleNamespace(_IMAGE_PLACEHOLDER_RE=ns['_IMAGE_PLACEHOLDER_RE'],_prompt_prefix='')
 build_message=lambda prompt,media:ns['_build_message'](original_input,prompt,media)
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.jinja','.py','.txt']};files={};headers={}
 for p in sorted(W.glob('*.safetensors')):
  files[p.name]=dict(bytes=p.stat().st_size,SHA256=sha(p))
  with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
  for k,v in header.items():
   if k=='__metadata__':continue
   assert k not in headers and v['dtype']=='BF16';headers[k]=dict(v,source_file=p.name)
 assert len(files)==4 and sum(x['bytes'] for x in files.values())==20585644104 and len(headers)==704
 put('own704_physical_source_headers.json',headers);put('own_whole_source_files.json',files)
 processor=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False);ip=processor.image_processor
 assert type(processor).__name__=='Glm4vProcessor' and type(ip).__name__=='Glm4vImageProcessor' and ip.patch_size==14 and ip.temporal_patch_size==ip.merge_size==2
 rust=Tokenizer.from_file(str(W/'tokenizer.json'));assert rust.get_vocab()==processor.tokenizer.get_vocab();assert 'enable_thinking' not in (W/'chat_template.jinja').read_text()
 dataset=Path(cfg['data']['dataset_path']);assert sha(dataset)=='2f5184c8d02695ef95ef16efe0c48640a0d99c680ab06f6847ec5d1372b06451'
 def load(path,filter_empty):
  ds=DatasetLoader().load(str(path));ds=TaskDataRegistry._resolve_media_paths(ds,str(path.parent));return DataPreprocessor().preprocess(ds,task_type='image-text-to-text',apply_template=False,filter_empty=filter_empty)
 ds=load(dataset,True);all_unfiltered=load(dataset,False);assert len(ds)==len(all_unfiltered)==4329 and ds==all_unfiltered
 target=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';assert sha(target)==sha(dataset);target_ds=load(target,False);assert len(target_ds)==4329
 assert [(x['prompt'],x['reference']) for x in ds]==[(x['prompt'],x['reference']) for x in target_ds]
 args=EngineArgs(model=str(W),trust_remote_code=True,tensor_parallel_size=1,gpu_memory_utilization=.9,allowed_local_media_path=str(R/'workspace/datasets'),enforce_eager=True)
 native_config=args.create_engine_config();assert str(native_config.model_config.dtype)=='torch.bfloat16' and native_config.model_config.max_model_len==65536
 renderer=renderer_from_config(native_config);native_input=InputProcessor(native_config,renderer);media=ImageMediaIO()
 assert media.image_mode=='RGB' and media.rgba_background_color==(255,255,255)
 def tm(t):
  t=t.detach().cpu().contiguous();return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
 stop_sets=set();native_types=set();rgba=[]
 with (H/'own_original_frontend200.jsonl').open('x',encoding='utf8') as out:
  for i,(row,target_row) in enumerate(zip(ds[:200],target_ds[:200])):
   assert sha(row['media_path'])==sha(target_row['media_path']);model_prompt=GreedyInference._add_answer_instruction(row['prompt'])
   messages=build_message(model_prompt,target_row['media_path']);clean_prompt=messages[0]['content'][1]['text'];assert clean_prompt==ns['_IMAGE_PLACEHOLDER_RE'].sub('',model_prompt).strip()
   _,prompts=renderer.render_chat([messages],ChatParams(chat_template_kwargs={'add_generation_prompt':True,'continue_final_message':False}),skip_mm_cache=True)
   engine=prompts[0];request=native_input.process_inputs('own_glm41thinking_input_'+str(i),engine,SamplingParams(temperature=0.,top_p=1.,max_tokens=8192),('generate',))
   sampling=request.sampling_params;assert sampling.max_tokens==8192 and sampling.temperature==0. and sampling.top_p==1. and not sampling.ignore_eos
   stops=sorted(sampling.all_stop_token_ids);assert stops==[151329,151336,151338,151348];stop_sets.add(tuple(stops))
   with Image.open(target_row['media_path']) as im:
    if im.mode in ['RGBA','LA'] or 'transparency' in im.info:rgba.append(i)
   image=media.load_file(Path(target_row['media_path']))
   hf_messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=clean_prompt)])]
   text=processor.apply_chat_template(hf_messages,add_generation_prompt=True,tokenize=False)
   assert text.endswith('<|assistant|>\n') and '/nothink' not in text
   actual=processor(text=text,images=[image],return_tensors='pt',padding=True)
   height,width=image.height,image.width;factor=28
   if height<factor or width<factor:
    scale=max(factor/height,factor/width);height,width=int(height*scale),int(width*scale)
   assert max(height,width)/min(height,width)<=200;rh=round(height/factor)*factor;rw=round(width/factor)*factor
   minimum=ip.size['shortest_edge'];maximum=ip.size['longest_edge']
   if 2*rh*rw>maximum:
    beta=math.sqrt(2*height*width/maximum);rh=max(factor,math.floor(height/beta/factor)*factor);rw=max(factor,math.floor(width/beta/factor)*factor)
   elif 2*rh*rw<minimum:
    beta=math.sqrt(minimum/(2*height*width));rh=math.ceil(height*beta/factor)*factor;rw=math.ceil(width*beta/factor)*factor
   u8=resize(pil_to_tensor(image)[None],size=[rh,rw],interpolation=InterpolationMode.BICUBIC,antialias=True)[0]
   f=np.float32(1./ip.rescale_factor);mean=(np.asarray(ip.image_mean,np.float32)*f).reshape(3,1,1);std=(np.asarray(ip.image_std,np.float32)*f).reshape(3,1,1)
   values=torch.from_numpy((u8.numpy().astype(np.float32)-mean)/std);gh,gw=rh//14,rw//14
   patches=values[None,None].repeat(1,2,1,1,1).reshape(1,1,2,3,gh//2,2,14,gw//2,2,14).permute(0,1,4,7,5,8,3,2,6,9).reshape(gh*gw,1176)
   assert torch.equal(actual['pixel_values'],patches),(i,'Independent own full resized F32 pixels/patches')
   assert actual['image_grid_thw'].tolist()==[[1,gh,gw]]
   expanded=text.replace(processor.image_token,processor.image_token*(gh*gw//4));ids=rust.encode(expanded,add_special_tokens=True).ids
   assert actual['input_ids'].tolist()==[ids] and request.prompt_token_ids==ids and engine['prompt_token_ids']==ids
   assert torch.equal(actual['attention_mask'],torch.ones_like(actual['input_ids']))
   item=engine['mm_kwargs']['image'][0];npix=item['pixel_values'].data;ngrid=item['image_grid_thw'].data
   if ngrid.ndim==1:ngrid=ngrid.unsqueeze(0)
   assert ngrid.tolist()==actual['image_grid_thw'].tolist() and npix.shape==actual['pixel_values'].shape
   assert npix.dtype in [torch.bfloat16,torch.float32] and torch.equal(npix.bfloat16(),actual['pixel_values'].bfloat16()),(i,'Original native model BF16 input must match completely')
   native_types.add(str(npix.dtype))
   v=dict(index=i,prompt=row['prompt'],actual_model_prompt=model_prompt,original_native_clean_prompt=clean_prompt,original_native_message=messages,own_chat_template_string=text,media_path=target_row['media_path'],own_original_AFS_media_path=row['media_path'],image_SHA256=sha(target_row['media_path']),reference=row['reference'],own_complete_input_ids=ids,all_actual_input_tensor_metadata={n:tm(t) for n,t in actual.items() if torch.is_tensor(t)},actual_native_pixel_metadata=tm(npix),actual_native_pixel_BF16=tm(npix.bfloat16()),actual_native_grid_metadata=tm(ngrid),actual_native_sampling=dict(max_tokens=sampling.max_tokens,temperature=sampling.temperature,top_p=sampling.top_p,all_stop_token_ids=stops),all_independent_own_pixels_tokens_native_media_and_BF16_full_values_equal=True,original_native_template_default_no_thinking_override=True)
   if i in [0,99,199]:np.savez_compressed(H/('own_complete_input_example_'+str(i)+'.npz'),**{n:t.detach().cpu().numpy() for n,t in actual.items() if torch.is_tensor(t)})
   out.write(json.dumps(v,ensure_ascii=False)+'\n');out.flush();put('progress.json',dict(UTC=now(),completed_samples=i+1,total_samples=200,no_learned_CPU_or_NPU=True))
 cfg_obj=AutoConfig.from_pretrained(W,local_files_only=True,trust_remote_code=False)
 with torch.device('meta'):model=AutoModelForImageTextToText.from_config(cfg_obj)
 meta={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in model.named_parameters(remove_duplicate=False)}
 diff=dict(physical_not_meta=sorted(set(headers)-set(meta)),meta_not_physical=sorted(set(meta)-set(headers)),shape_mismatch={n:dict(physical=headers[n]['shape'],meta=meta[n]['shape']) for n in set(headers)&set(meta) if headers[n]['shape']!=meta[n]['shape']})
 assert not any(diff.values()) and len(meta)==704;put('own_meta_only_parameter_layout.json',dict(actual_class=type(model).__name__,meta=meta,mapping_review=diff,no_learned_forward=True))
 assert sha(source)==source_sha and sha(original)=='1d45b618d8f46d022ec0577ef460fa4c68346c08aadd35d225074ee7b5c00dbf'
 for n,h in assets.items():assert sha(W/n)==h
 for n,r in files.items():assert sha(W/n)==r['SHA256']
 result=dict(UTC=now(),passed=True,model_id=MID,own_original_saved_YAML=str(original),own_original_saved_YAML_SHA256=sha(original),own_physical_BF16_sources=704,own_whole_weight_bytes=20585644104,own_weight_files=files,own_assets_SHA256=assets,own_source_headers=str(H/'own704_physical_source_headers.json'),own_source_headers_SHA256=sha(H/'own704_physical_source_headers.json'),own_frontend200=str(H/'own_original_frontend200.jsonl'),own_frontend200_SHA256=sha(H/'own_original_frontend200.jsonl'),dataset=str(dataset),dataset_SHA256=sha(dataset),actual_target_dataset=str(target),original_filter_true_and_false_all4329_prompt_ref_order_and_200_images_identical=True,all200_own_original_input_pixels_tokens_refs_verified=True,actual_original_no_thinking_override_preserved=True,original_generic_adapter_SHA256=source_sha,original_generic_adapter_path=str(source),original_build_message_AST_SHA256=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),max_new_tokens=8192,outer_batch=10,inner_serial=1,own_original_generation_EOS=js(W/'generation_config.json')['eos_token_id'],native_dtype=str(native_config.model_config.dtype),native_context_length=native_config.model_config.max_model_len,actual_native_pixel_dtypes=sorted(native_types),actual_native_stop_sets=[list(x) for x in sorted(stop_sets)],native_white_background_transparency_indices=rgba,actual_installed_meta_class=type(model).__name__,meta_mapping=diff,meta_map_complete_without_conversion=True,original_generic_adapter_initializer_not_executed=True,no_learned_CPU_or_NPU_inference=True,no_NPU_adapter_fullNF_or_dispatch_readiness_claim=True,A100_historical_engine_or_tensor_capture_claimed=False,current_native_input_processing_skip_mm_cache_to_obtain_full_values_only=True,installed_source_SHA256={inspect.getfile(type(x)):sha(inspect.getfile(type(x))) for x in [processor,ip,renderer,native_input,media,model,cfg_obj]})
 put('review.json',result);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_model_inference=True));return result
if __name__=='__main__':
 try:
  with contextlib.redirect_stdout(sys.stderr):result=main()
  print(json.dumps(result,ensure_ascii=False))
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_model_inference=True))
  raise
