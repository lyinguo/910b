"""Own Meta input/source validation only; no learned CPU or NPU forward."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,contextlib,hashlib,inspect,json,os,struct,traceback,types
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
MID='meta-llama/Llama-3.2-11B-Vision-Instruct'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/MllamaForConditionalGeneration')/MID
H=B/'MetaMllama_own_original_HF_BF16_2048_frontend200_meta_sources_v1_20261120'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def main():
 os.chdir(R);os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 import torch,numpy as np,yaml
 from PIL import Image
 from transformers import AutoConfig,AutoProcessor,MllamaForConditionalGeneration,GenerationConfig
 from safetensors import safe_open
 from tokenizers import Tokenizer
 from scripts.mllama11b_original_independent_cpu_910b import inputs as independent_inputs
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 original=R/'configs/MllamaForConditionalGeneration/meta-llama--Llama-3.2-11B-Vision-Instruct.yaml'
 assert sha(original)=='0735e7b22b702418180f9a1132fe71d03af241183fa214ab1c6f0b81a99eda0b'
 cfg=yaml.safe_load(original.read_bytes());assert cfg['model']['name']==MID and cfg['model']['kwargs']==dict(device='auto',max_new_tokens=2048,torch_dtype='auto') and cfg['algorithm']['batch_size']==10
 assert cfg['model']['adapter']=='universal_eval/model/adapters/transformers_vlm.py:TransformersVLM'
 assert not Path(cfg['model']['model_path']).is_dir() and W.is_dir()
 source=R/'universal_eval/model/adapters/transformers_vlm.py';assert sha(source)=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3'
 tree=ast.parse(source.read_bytes());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TransformersVLM')
 fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='_build_messages')
 batchfn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='batch_predict')
 assert 'return [self.predict(prompt, media_path) for prompt, media_path in batch]' in ast.get_source_segment(source.read_text(),batchfn)
 ns=dict(Optional=__import__('typing').Optional,List=list,Dict=dict,Any=object,os=os)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
 fake=type('MllamaForConditionalGeneration',(),{})();original_input=types.SimpleNamespace(_model=fake)
 processor=AutoProcessor.from_pretrained(W,local_files_only=True);assert type(processor).__name__=='MllamaProcessor'
 rust=Tokenizer.from_file(str(W/'tokenizer.json'));assert rust.get_vocab()==processor.tokenizer.get_vocab()
 assert len(processor.tokenizer)==128257 and processor.image_token_id==128256 and processor.tokenizer.padding_side=='left'
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.jinja','.py','.txt']}
 dataset=Path(cfg['data']['dataset_path']);assert sha(dataset)=='2f5184c8d02695ef95ef16efe0c48640a0d99c680ab06f6847ec5d1372b06451'
 def load(p,filtered):
  ds=DatasetLoader().load(str(p));ds=TaskDataRegistry._resolve_media_paths(ds,str(p.parent));return DataPreprocessor().preprocess(ds,task_type='image-text-to-text',apply_template=False,filter_empty=filtered)
 ds=load(dataset,True);unfiltered=load(dataset,False);assert ds==unfiltered and len(ds)==4329
 target=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';assert sha(target)==sha(dataset);td=load(target,False)
 assert [(x['prompt'],x['reference']) for x in ds]==[(x['prompt'],x['reference']) for x in td]
 def tm(t):
  t=t.detach().cpu().contiguous();return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
 input_records=[]
 with (H/'own_original_frontend200.jsonl').open('x',encoding='utf8') as out:
  for i,(r,tr) in enumerate(zip(ds[:200],td[:200])):
   assert sha(r['media_path'])==sha(tr['media_path']);prompt=GreedyInference._add_answer_instruction(r['prompt'])
   messages=ns['_build_messages'](original_input,prompt,tr['media_path']);text=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
   assert text.count('<|image|>')==1
   with Image.open(tr['media_path']) as im:image=im.convert('RGB');actual=processor(text=text,images=[image],return_tensors='pt',padding=True)
   row=dict(index=i,prompt=r['prompt'],model_prompt=prompt,media_path=tr['media_path'],original_AFS_media_path=r['media_path'],reference=r['reference'],image_SHA256=sha(tr['media_path']),own_chat_template_string=text,complete_input_ids=actual['input_ids'].tolist()[0],actual_tensor_metadata={n:tm(t) for n,t in actual.items() if torch.is_tensor(t)})
   p=dict(weight_path=str(W),input_records=[row]);reference,texts,geometry=independent_inputs(p,[0])
   assert texts==[text] and set(reference)==set(actual)
   for n,v in reference.items():assert torch.equal(v,actual[n]),(i,n,'Independent own complete input differs')
   assert row['complete_input_ids'][:2]==[128000,128000]
   row.update(independent_all_pixels_tokens_masks_aspects_equal=True,geometry=geometry[0],original_max_new_tokens=2048,outer_batch=10,inner_original_serial=1)
   if i in [0,99,199]:np.savez_compressed(H/('own_complete_input_example_'+str(i)+'.npz'),**{n:v.numpy() for n,v in actual.items()})
   input_records.append(row);out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();put('progress.json',dict(UTC=now(),stage='own_frontend',samples=i+1,total=200,actual_NEW_model_forwards=0))
 put('own_input_records200.json',input_records)
 native_cfg=AutoConfig.from_pretrained(W,local_files_only=True);native_cfg._attn_implementation='eager';native_cfg.text_config._attn_implementation='eager';native_cfg.vision_config._attn_implementation='eager'
 with torch.device('meta'):model=MllamaForConditionalGeneration(native_cfg)
 params=dict(model.named_parameters());assert len(params)==906
 files={};header={};mapped=[]
 def target_name(k):
  if k.startswith('language_model.model.'):return k.replace('language_model.model.','model.language_model.',1)
  if k.startswith('language_model.lm_head.'):return k.replace('language_model.lm_head.','lm_head.',1)
  if k.startswith(('vision_model.','multi_modal_projector.')):return 'model.'+k
  raise AssertionError(k)
 for shard in sorted(W.glob('*.safetensors')):
  fullsha=sha(shard)
  with shard.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];hh=json.loads(f.read(n))
  files[shard.name]=dict(bytes=shard.stat().st_size,sha256=fullsha)
  with safe_open(str(shard),framework='pt',device='cpu') as sf:
   for k in sf.keys():
    m=target_name(k);v=hh[k];assert k not in header and m in params and v['dtype']=='BF16' and list(params[m].shape)==v['shape']
    x=sf.get_tensor(k);assert x.dtype==torch.bfloat16 and torch.isfinite(x).all();h=tm(x)['SHA256']
    header[k]=dict(v,file=shard.name);mapped.append(dict(source_key=k,target_key=m,shape=v['shape'],source_dtype='BF16',full_sha256=h,original_CPU_BF16_cast_sha256=h))
  put('progress.json',dict(UTC=now(),stage='own_full_source_SHA_and_meta',sources=len(mapped),files=len(files),actual_NEW_model_forwards=0))
 assert len(files)==5 and len(mapped)==906 and {v['target_key'] for v in mapped}==set(params)
 put('own_whole_source_files.json',files);put('own906_physical_source_headers.json',header);put('own906_source_to_native_map.json',mapped)
 model.generation_config=GenerationConfig.from_pretrained(W,local_files_only=True)
 gen,_=model._prepare_generation_config(None,max_new_tokens=2048,do_sample=False,temperature=0.)
 assert gen.eos_token_id==[128001,128008,128009] and gen.pad_token_id==128004 and gen.max_new_tokens==2048 and gen.use_cache and gen.num_beams==1 and not gen.do_sample
 put('own_resolved_generation.json',gen.to_dict())
 assert sha(original)=='0735e7b22b702418180f9a1132fe71d03af241183fa214ab1c6f0b81a99eda0b' and sha(source)=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3'
 for n,h in assets.items():assert sha(W/n)==h
 review=dict(UTC=now(),passed=True,model_id=MID,weight_path=str(W),original_config=str(original),original_config_SHA256=sha(original),original_wrong_unreadable_weight_path=cfg['model']['model_path'],own_named_root_rebinding_only_no_other_checkpoint=True,original_adapter=str(source),original_adapter_SHA256=sha(source),own_resource_sha256=assets,all_source_file_sha256=files,all906_source_to_native_semantic_name_shape_mapping_verified=mapped,input_records=input_records,native_image_processor=json.loads(json.dumps(processor.image_processor.to_dict())),original_resolved_generation=gen.to_dict(),native_activation_modules={n:type(m).__name__ for n,m in model.named_modules() if n.endswith(('act_fn','activation_fn'))},dataset_path=str(target),dataset_SHA=sha(target),original_AFS_dataset=str(dataset),original_filter_true_false_all4329_order_and_first200_images_identical=True,frontend200=str(H/'own_original_frontend200.jsonl'),frontend200_SHA256=sha(H/'own_original_frontend200.jsonl'),original_max_new_tokens=2048,original_outer10_inner_serial1=True,original_runtime_precision_frozen_auto_BF16=True,original_two_BOS_preserved=True,meta_only=True,actual_NEW_model_forwards=0,no_NPU_dispatch_readiness_claim=True,historical_A100_engine_not_captured=True,independent_input_source=str(R/'scripts/mllama11b_original_independent_cpu_910b.py'),independent_input_source_SHA256=sha(R/'scripts/mllama11b_original_independent_cpu_910b.py'),installed_class_sources={inspect.getfile(type(o)):sha(inspect.getfile(type(o))) for o in [processor,processor.image_processor,model]})
 put('review.json',review);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,actual_NEW_model_forwards=0,no_model_inference=True));return {k:v for k,v in review.items() if k not in ['all906_source_to_native_semantic_name_shape_mapping_verified','input_records','own_resource_sha256']}
if __name__=='__main__':
 try:
  with contextlib.redirect_stdout(sys.stderr):result=main()
  print(json.dumps(result,ensure_ascii=False))
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_model_inference=True))
  raise
