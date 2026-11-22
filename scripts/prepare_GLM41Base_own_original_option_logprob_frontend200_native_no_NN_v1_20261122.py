"""Own Base option-text logprob input proof; no checkpoint loading or learned work."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,contextlib,hashlib,inspect,json,math,os,re,struct,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='zai-org/GLM-4.1V-9B-Base';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration')/MID;H=B/'GLM41Base_own_original_option_logprob_native_frontend200_meta_no_NN_v1_20261122'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,x):
 p=H/n;tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)
def now():return datetime.now(timezone.utc).isoformat()
def main():
 os.chdir(R);os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');H.mkdir(exist_ok=False)
 import numpy as np,torch,yaml
 from PIL import Image
 from tokenizers import Tokenizer,AddedToken
 from transformers import AutoConfig,AutoProcessor,AutoModelForImageTextToText
 from torchvision.transforms.v2.functional import pil_to_tensor,resize
 from torchvision.transforms import InterpolationMode
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 from vllm.engine.arg_utils import EngineArgs
 from vllm.renderers.registry import renderer_from_config
 from vllm.v1.engine.input_processor import InputProcessor
 from vllm import SamplingParams
 torch.set_num_threads(4)
 original=R/'configs/Glm4vForConditionalGeneration/zai-org--GLM-4.1V-9B-Base.yaml';assert sha(original)=='606ac2677328d9b7dfe02dc0e5ef546ea1d56bce4d9c69485278a2e21fb35860';cfg=yaml.safe_load(original.read_bytes())
 assert cfg['model']['name']==MID and cfg['model']['adapter']=='universal_eval/model/adapters/glm41v_base_discriminative.py:GLM41VBaseDiscriminativeModel' and cfg['model']['kwargs']==dict(tensor_parallel_size=1,max_tokens=2048,gpu_memory_utilization=.9,trust_remote_code=True) and cfg['algorithm']['batch_size']==10
 disc=R/'universal_eval/model/adapters/glm41v_base_discriminative.py';parent=R/'universal_eval/model/adapters/glm41v_base_vllm.py';assert sha(disc)=='c95d7286200f93a000f7232cd807ec64e80bb4e9ac5911f06976258ed328184d' and sha(parent)=='c435cdcfa181637ed69181ab1d9aa5ca0aaaca17efb6f0015be74e400f10e86c'
 pt=ast.parse(parent.read_bytes());dt=ast.parse(disc.read_bytes());selected=[n for n in pt.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['_EVAL_INSTRUCTION_SUFFIXES','_OPTION_LINE_RE'] for t in n.targets)]+[n for n in pt.body if isinstance(n,ast.FunctionDef) and n.name=='_strip_eval_instruction']+[n for n in dt.body if isinstance(n,ast.FunctionDef) and n.name=='_parse_options']
 cls=next(n for n in dt.body if isinstance(n,ast.ClassDef) and n.name=='GLM41VBaseDiscriminativeModel');render=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_render_prompt');selected.append(render)
 ns=dict(re=re,List=list,Tuple=tuple,Optional=__import__('typing').Optional);exec(compile(ast.Module(body=selected,type_ignores=[]),str(disc),'exec'),ns)
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.jinja','.py','.txt']};index=js(W/'model.safetensors.index.json');files={};headers={}
 for n in sorted(set(index['weight_map'].values())):
  p=W/n
  with p.open('rb') as f:h=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
  for key,v in h.items():
   if key=='__metadata__':continue
   assert key not in headers and index['weight_map'][key]==n;headers[key]=dict(shape=v['shape'],dtype=v['dtype'],source_file=n,data_offsets=v['data_offsets'])
  files[n]=dict(bytes=p.stat().st_size,SHA256=sha(p))
 assert len(files)==4 and len(headers)==704 and all(x['dtype']=='BF16' for x in headers.values()) and sum(r['bytes'] for r in files.values())==20585644104
 put('own_whole_source_files.json',files);put('own704_physical_source_headers.json',headers)
 processor=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False);ip=processor.image_processor;rust=Tokenizer.from_file(str(W/'tokenizer.json'));tc=js(W/'tokenizer_config.json')
 for tid,rec in sorted(tc.get('added_tokens_decoder',{}).items(),key=lambda p:int(p[0])):
  tok=AddedToken(**rec);rust.add_special_tokens([tok]) if rec['special'] else rust.add_tokens([tok]);assert rust.token_to_id(rec['content'])==int(tid)
 for rec in tc.get('additional_special_tokens',[]):
  tok=AddedToken(**rec) if isinstance(rec,dict) else AddedToken(rec,special=True,normalized=False)
  rust.add_special_tokens([tok])
 assert rust.get_vocab()==processor.tokenizer.get_vocab() and ip.patch_size==14 and ip.temporal_patch_size==ip.merge_size==2
 dataset=Path(cfg['data']['dataset_path']);assert sha(dataset)=='2f5184c8d02695ef95ef16efe0c48640a0d99c680ab06f6847ec5d1372b06451'
 def load(path,filter_empty):
  a=DatasetLoader().load(str(path));a=TaskDataRegistry._resolve_media_paths(a,str(path.parent));return DataPreprocessor().preprocess(a,task_type='image-text-to-text',apply_template=False,filter_empty=filter_empty)
 original_data=load(dataset,True);assert original_data==load(dataset,False) and len(original_data)==4329;target=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';assert sha(target)==sha(dataset);data=load(target,False);assert [(x['prompt'],x['reference']) for x in data]==[(x['prompt'],x['reference']) for x in original_data]
 args=EngineArgs(model=str(W),trust_remote_code=True,tensor_parallel_size=1,gpu_memory_utilization=.9,allowed_local_media_path=str(R/'workspace/datasets'),enforce_eager=True);native=args.create_engine_config();assert str(native.model_config.dtype)=='torch.bfloat16' and native.model_config.max_model_len==65536
 renderer=renderer_from_config(native);input_processor=InputProcessor(native,renderer);total_options=0;counts=[];all_stops=set()
 def tm(t):
  t=t.detach().cpu().contiguous();return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
 with (H/'own_original_option_frontend200.jsonl').open('x',encoding='utf8') as out:
  for i,(row,oldrow) in enumerate(zip(data[:200],original_data[:200])):
   assert sha(row['media_path'])==sha(oldrow['media_path']);model_prompt=GreedyInference._add_answer_instruction(row['prompt']);clean=ns['_strip_eval_instruction'](model_prompt or '').strip();options=ns['_parse_options'](clean);assert 2<=len(options)<=5,(i,'Original no-options fallback must not be silently replaced')
   with Image.open(row['media_path']) as im:image=im.convert('RGB')
   minimum=ip.size['shortest_edge'];maximum=ip.size['longest_edge'];height,width=image.height,image.width;factor=28
   if height<factor or width<factor:scale=max(factor/height,factor/width);height,width=int(height*scale),int(width*scale)
   assert max(height,width)/min(height,width)<=200;rh=round(height/factor)*factor;rw=round(width/factor)*factor
   if 2*rh*rw>maximum:beta=math.sqrt(2*height*width/maximum);rh=max(factor,math.floor(height/beta/factor)*factor);rw=max(factor,math.floor(width/beta/factor)*factor)
   elif 2*rh*rw<minimum:beta=math.sqrt(minimum/(2*height*width));rh=math.ceil(height*beta/factor)*factor;rw=math.ceil(width*beta/factor)*factor
   u8=resize(pil_to_tensor(image)[None],size=[rh,rw],interpolation=InterpolationMode.BICUBIC,antialias=True)[0];f=np.float32(1/ip.rescale_factor);mean=(np.asarray(ip.image_mean,np.float32)*f).reshape(3,1,1);std=(np.asarray(ip.image_std,np.float32)*f).reshape(3,1,1);v=torch.from_numpy((u8.numpy().astype(np.float32)-mean)/std);gh,gw=rh//14,rw//14
   pixels=v[None,None].repeat(1,2,1,1,1).reshape(1,1,2,3,gh//2,2,14,gw//2,2,14).permute(0,1,4,7,5,8,3,2,6,9).reshape(gh*gw,1176);records=[]
   for letter,option in options:
    cont=' '+option;k=len(processor.tokenizer(cont,add_special_tokens=False)['input_ids']);assert k==len(rust.encode(cont,add_special_tokens=False).ids)
    text=ns['_render_prompt'](None,clean,row['media_path'],letter,cont);actual=processor(text=text,images=[image],return_tensors='pt',padding=True);assert torch.equal(actual['pixel_values'],pixels) and actual['image_grid_thw'].tolist()==[[1,gh,gw]]
    ids=rust.encode(text.replace(processor.image_token,processor.image_token*(gh*gw//4)),add_special_tokens=True).ids;assert actual['input_ids'].tolist()==[ids] and torch.equal(actual['attention_mask'],torch.ones_like(actual['input_ids']))
    item=dict(prompt=text,multi_modal_data=dict(image=image));eng,=renderer.render_cmpl([item],skip_mm_cache=True);sp=SamplingParams(temperature=0.,max_tokens=1,prompt_logprobs=0);request=input_processor.process_inputs('own_glm41base_'+str(i)+'_'+letter,eng,sp,('generate',));sampling=request.sampling_params
    assert eng['prompt_token_ids']==request.prompt_token_ids==ids and sampling.max_tokens==1 and sampling.temperature==0. and sampling.prompt_logprobs==0 and not sampling.ignore_eos;stops=sorted(sampling.all_stop_token_ids);assert stops==[151329,151336,151338,151348];all_stops.add(tuple(stops))
    mm=eng['mm_kwargs']['image'][0];npix=mm['pixel_values'].data;ngrid=mm['image_grid_thw'].data;ngrid=ngrid.unsqueeze(0) if ngrid.ndim==1 else ngrid;assert ngrid.tolist()==[[1,gh,gw]] and npix.shape==pixels.shape and torch.equal(npix.bfloat16(),pixels.bfloat16())
    # Preserve the original actual plp[-(k+2):-2] slice, including tokenizer boundary behavior.
    score_positions=list(range(len(ids)))[-(k+2):-2];assert len(score_positions)==k and min(score_positions)>=1
    records.append(dict(letter=letter,original_option_text=option,original_continuation=cont,own_manual_prompt=text,own_complete_input_ids=ids,option_token_count=k,original_prompt_logprob_positions=score_positions,original_score_token_ids=[ids[j] for j in score_positions],all_input_tensor_metadata={n:tm(t) for n,t in actual.items() if torch.is_tensor(t)},actual_native_pixel_metadata=tm(npix),actual_native_pixel_BF16=tm(npix.bfloat16()),actual_native_sampling=dict(max_tokens=sampling.max_tokens,temperature=sampling.temperature,prompt_logprobs=sampling.prompt_logprobs,all_stop_token_ids=stops),original_option_text_probability_protocol=True))
    if i in [0,99,199]:np.savez_compressed(H/('own_complete_input_example_'+str(i)+'_'+letter+'.npz'),**{n:t.detach().cpu().numpy() for n,t in actual.items() if torch.is_tensor(t)})
   total_options+=len(records);counts.append(len(records));out.write(json.dumps(dict(index=i,prompt=row['prompt'],actual_model_prompt=model_prompt,original_clean_prompt=clean,reference=row['reference'],media_path=row['media_path'],own_original_AFS_media_path=oldrow['media_path'],image_SHA256=sha(row['media_path']),own_original_PIL_direct_RGB=True,all_independent_own_full_pixels_tokens_and_native_BF16_values_equal=True,options=records),ensure_ascii=False)+'\n');out.flush();put('progress.json',dict(UTC=now(),completed_samples=i+1,total_samples=200,total_option_inputs=total_options,no_learned_CPU_or_NPU=True))
 ownconfig=AutoConfig.from_pretrained(W,local_files_only=True,trust_remote_code=False)
 with torch.device('meta'):model=AutoModelForImageTextToText.from_config(ownconfig)
 meta={n:list(t.shape) for n,t in model.named_parameters(remove_duplicate=False)};assert len(meta)==len(headers)==704 and all(meta[n]==headers[n]['shape'] for n in headers)
 put('own_meta_only_parameter_layout.json',dict(actual_class=type(model).__name__,parameters=meta,no_learned_forward=True));assert sha(original)=='606ac2677328d9b7dfe02dc0e5ef546ea1d56bce4d9c69485278a2e21fb35860' and sha(disc)=='c95d7286200f93a000f7232cd807ec64e80bb4e9ac5911f06976258ed328184d' and sha(parent)=='c435cdcfa181637ed69181ab1d9aa5ca0aaaca17efb6f0015be74e400f10e86c'
 for n,h in assets.items():assert sha(W/n)==h
 result=dict(UTC=now(),passed=True,model_id=MID,own_original_saved_YAML=str(original),own_original_saved_YAML_SHA256=sha(original),own_original_adapter_SHA256=sha(disc),own_original_parent_adapter_SHA256=sha(parent),own_weight_path=str(W),own_weight_files=files,own_assets_SHA256=assets,own_physical_BF16_sources=704,own_whole_weight_bytes=20585644104,own_frontend200=str(H/'own_original_option_frontend200.jsonl'),own_frontend200_SHA256=sha(H/'own_original_option_frontend200.jsonl'),own_source_headers=str(H/'own704_physical_source_headers.json'),own_source_headers_SHA256=sha(H/'own704_physical_source_headers.json'),all4329_original_filter_true_false_prompt_ref_order_and200_image_bytes_same=True,all200_own_native_and_independent_full_pixels_tokens_and_BF16_values_verified=True,total_native_original_option_requests=total_options,option_counts=counts,original_option_mean_logprob_slices_unchanged=True,original_probability_not_replaced_by_generation=True,all200_have_original_contiguous_options_no_fallback_triggered=True,original_PIL_direct_RGB_not_other_model_white_background=True,outer_batch=10,original_fallback_max_tokens_2048_preserved_but_not_triggered=True,actual_original_option_sampling_max_tokens=1,prompt_logprobs=0,native_dtype=str(native.model_config.dtype),native_context_length=native.model_config.max_model_len,actual_native_stop_sets=[list(x) for x in sorted(all_stops)],actual_installed_meta_class=type(model).__name__,no_checkpoint_loading_or_learned_CPU_or_NPU_forward=True,no_full_NPU_adapter_fullNF_NEW3_or_dispatch_readiness_claim=True,A100_historical_engine_or_tensor_capture_claimed=False,current_native_skip_mm_cache_for_complete_input_values_only=True,installed_external_source_SHA256={inspect.getfile(type(x)):sha(inspect.getfile(type(x))) for x in [processor,ip,renderer,input_processor,model,ownconfig]},source_SHA256=sha(__file__))
 put('review.json',result);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_model_inference=True));return result
if __name__=='__main__':
 try:
  with contextlib.redirect_stdout(sys.stderr):result=main()
  print(json.dumps(result,ensure_ascii=False))
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_model_inference=True))
  raise
