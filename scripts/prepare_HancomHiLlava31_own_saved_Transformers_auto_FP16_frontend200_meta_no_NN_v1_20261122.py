"""Own checkpoint/frontend/original processor contract; no learned forward or dispatch."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import collections,contextlib,hashlib,inspect,json,os,struct,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='HancomInSpaceAI/HiLLaVA-llama_3_1-8b';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaLlamaForCausalLM')/MID
H=B/'HancomHiLlava31_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122'
sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 import numpy as np,torch
 from PIL import Image
 from tokenizers import Tokenizer
 from transformers import AutoTokenizer,AutoProcessor,CLIPImageProcessor,LlavaProcessor,LlamaConfig,CLIPVisionConfig,LlavaConfig,LlavaForConditionalGeneration
 from universal_eval.model.adapters.transformers_vlm import TransformersVLM
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz
 from torchvision.transforms.v2 import functional as tv
 from torchvision.transforms import InterpolationMode
 from batch_910b_support import workbook_rows
 import xml.etree.ElementTree as ET
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 original=R/'universal_eval/model/adapters/transformers_vlm.py';saved=R/'configs/LlavaLlamaForCausalLM/HancomInSpaceAI--HiLLaVA-llama_3_1-8b.yaml'
 assert sha(original)=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3' and sha(saved)=='624d81316384e7bff93ba2fd7024f3e7b47051198eb3deff2313aa400a83f84e'
 import yaml
 old=yaml.safe_load(saved.read_text());assert old['model']['kwargs']=={'device':'auto','max_new_tokens':2048,'torch_dtype':'auto'} and old['algorithm']['batch_size']==10
 raw=js(W/'config.json');assert raw['architectures']==['LlavaLlamaForCausalLM'] and raw['num_hidden_layers']==32 and raw['hidden_size']==4096 and raw['num_key_value_heads']==8 and raw['rope_theta']==500000 and raw['rope_scaling']==dict(factor=8.0,high_freq_factor=4.0,low_freq_factor=1.0,original_max_position_embeddings=8192,rope_type='llama3')
 index=js(W/'model.safetensors.index.json')['weight_map'];headers={};files={};dtype_counts=collections.Counter();flat={}
 for n in sorted(set(index.values())):
  file=W/n
  with file.open('rb') as stream:
   header_length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(header_length))
  h={k:v for k,v in header.items() if k!='__metadata__'};headers[n]=h
  for k,v in h.items():
   assert k not in flat and index[k]==n;flat[k]=v;dtype_counts[v['dtype']]+=1
  files[n]=dict(bytes=file.stat().st_size,SHA256=sha(file))
 assert len(flat)==len(index)==686 and dict(dtype_counts)=={'BF16':686} and sum(v['bytes'] for v in files.values())==16709592504
 put('own_weight_files_and_headers.json',dict(own_weight_files=files,headers=headers,source_dtype_counts=dict(dtype_counts),source_tensor_count=686))
 # Exact saved TransformersVLM legacy configuration reconstruction, no source/math edit.
 fields=['vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','rms_norm_eps','rope_theta','max_position_embeddings','tie_word_embeddings','attention_bias','hidden_act','initializer_range','pretraining_tp']
 text=LlamaConfig(**{k:raw[k] for k in fields if k in raw})
 vision=CLIPVisionConfig(hidden_size=1024,num_hidden_layers=24,num_attention_heads=16,intermediate_size=4096,patch_size=14,image_size=336,num_channels=3,projection_dim=768,layer_norm_eps=1e-5,hidden_act='quick_gelu')
 config=LlavaConfig(text_config=text,vision_config=vision,image_token_index=int(raw.get('image_token_index',32000)),vision_feature_layer=-2,vision_feature_select_strategy='default',image_aspect_ratio=raw['image_aspect_ratio'],mm_hidden_size=1024)
 before_dtype=torch.get_default_dtype()
 try:
  torch.set_default_dtype(torch.float16)
  with torch.device('meta'):meta=LlavaForConditionalGeneration(config)
 finally:torch.set_default_dtype(before_dtype)
 params=dict(meta.named_parameters());mapped={};mapping={}
 for k,v in flat.items():
  n=k
  if k.startswith('model.layers.'):n='model.language_model.layers.'+k[len('model.layers.'):]
  elif k=='model.embed_tokens.weight':n='model.language_model.embed_tokens.weight'
  elif k=='model.norm.weight':n='model.language_model.norm.weight'
  elif k.startswith('model.mm_projector.'):
   idx,sfx=k[len('model.mm_projector.'):].split('.');n='model.multi_modal_projector.'+{'0':'linear_1','2':'linear_2'}[idx]+'.'+sfx
  elif k.startswith('model.vision_tower.vision_tower.vision_model.'):n='model.vision_tower.'+k[len('model.vision_tower.vision_tower.vision_model.'):]
  assert n not in mapped;mapping[k]=n;mapped[n]=v
 exact=set(mapped)==set(params) and all(list(params[k].shape)==v['shape'] for k,v in mapped.items() if k in params)
 put('own_original_legacy_mapping_meta_review.json',dict(exact_all686_original_mapping_names_shapes=exact,mapped_count=len(mapped),meta_parameter_count=len(params),source_to_native_mapping=mapping,unmapped_source_keys=sorted(set(mapped)-set(params)),missing_parameter_keys=sorted(set(params)-set(mapped)),mismatched_shapes={k:dict(own=v['shape'],meta=list(params[k].shape)) for k,v in mapped.items() if k in params and list(params[k].shape)!=v['shape']},meta_buffer_layout={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in meta.named_buffers()},meta_not_actual_CPU_or_NPU_loaded_dtype_proof=True))
 # Source/meta mismatch never licenses random parameters or borrowed vision weights.
 assert exact,'Original saved legacy mapping not exact for current native class; retain evidence, 0NN'
 tokenizer=AutoTokenizer.from_pretrained(W,local_files_only=True,trust_remote_code=True);rust=Tokenizer.from_file(str(W/'tokenizer.json'))
 processor_error=None
 try:proc=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=True)
 except Exception as e:processor_error=type(e).__name__+': '+str(e);proc=tokenizer
 original_processor_type=type(proc).__name__;image_processor=None
 if not hasattr(proc,'image_processor') and not hasattr(proc,'feature_extractor'):
  image_processor=CLIPImageProcessor(do_resize=True,size={'shortest_edge':336},resample=3,do_center_crop=True,crop_size={'height':336,'width':336},do_rescale=True,rescale_factor=1/255.,do_normalize=True,image_mean=[.48145466,.4578275,.40821073],image_std=[.26862954,.26130258,.27577711])
  tok=getattr(proc,'tokenizer',None) or proc
  if '<image>' not in tok.get_vocab():tok.add_special_tokens({'additional_special_tokens':['<image>']})
  proc=LlavaProcessor(image_processor=image_processor,tokenizer=proc,patch_size=14,vision_feature_select_strategy='patch',num_additional_image_tokens=0)
 if getattr(proc,'patch_size',None) is None:proc.patch_size=14
 if getattr(proc,'num_additional_image_tokens',None) is None:proc.num_additional_image_tokens=0
 if getattr(proc,'vision_feature_select_strategy',None) is None:proc.vision_feature_select_strategy='full'
 tok=getattr(proc,'tokenizer',None) or proc
 if '<image>' not in tok.get_vocab():tok.add_special_tokens({'additional_special_tokens':['<image>']})
 image_id=tok.convert_tokens_to_ids('<image>');assert isinstance(image_id,int) and image_id>=0
 proc.image_token_id=image_id;meta.config.image_token_id=image_id;meta.config.image_token_index=image_id
 actor=TransformersVLM(device='auto',max_new_tokens=2048,torch_dtype='auto');actor._model=meta;actor._model_name=MID;actor._model_path=str(W);actor._original_model_path=old['model']['model_path'];actor._processor=proc;actor._tokenizer=tok
 dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';afs=Path(old['data']['dataset_path']);assert sha(dataset)==sha(afs)
 def rows(p,flt):
  x=DatasetLoader().load(str(p));x=TaskDataRegistry._resolve_media_paths(x,str(p.parent));return DataPreprocessor().preprocess(x,task_type='image-text-to-text',apply_template=False,filter_empty=flt)
 records=rows(dataset,False);filtered=rows(dataset,True);orig=rows(afs,True);assert len(records)==len(filtered)==len(orig)==4329
 assert [(r['prompt'],r['reference']) for r in records]==[(r['prompt'],r['reference']) for r in filtered]==[(r['prompt'],r['reference']) for r in orig]
 mean=np.asarray([.48145466,.4578275,.40821073],dtype=np.float32)[:,None,None]*np.float32(255);std=np.asarray([.26862954,.26130258,.27577711],dtype=np.float32)[:,None,None]*np.float32(255)
 front=[];tkraw=js(W/'tokenizer.json');added_image=next((v for v in tkraw.get('added_tokens',[]) if v['content']=='<image>'),None)
 if added_image is None:rust.add_special_tokens(['<image>'])
 assert rust.token_to_id('<image>')==image_id
 for i,r in enumerate(records[:200]):
  assert sha(r['media_path'])==sha(orig[i]['media_path'])
  prompt=GreedyInference._add_answer_instruction(r['prompt']);messages=actor._build_messages(prompt,r['media_path']);assert len(messages)==1 and isinstance(messages[0]['content'],str)
  text=(proc.apply_chat_template(messages,add_generation_prompt=True,tokenize=False) if getattr(proc,'chat_template',None) is not None else tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=False))
  independent_text=js(W/'tokenizer_config.json')['bos_token']+'<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'+messages[0]['content'].strip()+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n';assert text==independent_text
  with Image.open(r['media_path']) as im:image=im.convert('RGB')
  if (getattr(proc,'pad_token',None) or getattr(tok,'pad_token',None)) is None:
   eos=(getattr(proc,'eos_token',None) or getattr(tok,'eos_token',None));assert eos is not None;setattr(proc,'pad_token',eos);setattr(tok,'pad_token',eos)
  actual=proc(images=[image],text=text,return_tensors='pt',padding=True);pixels=actual['pixel_values'];assert pixels.dtype==torch.float32 and list(pixels.shape)==[1,3,336,336]
  independent_ids=rust.encode(text.replace('<image>','<image>'*576),add_special_tokens=True).ids;assert actual['input_ids'][0].tolist()==independent_ids and independent_ids.count(image_id)==576
  assert actual['attention_mask'].tolist()==[[1]*len(independent_ids)]
  t=torch.from_numpy(np.asarray(image,dtype=np.uint8).transpose(2,0,1).copy());h,w=t.shape[-2:];nh,nw=(336,int(336*w/h)) if h<=w else (int(336*h/w),336)
  u=tv.resize(t,[nh,nw],interpolation=InterpolationMode.BICUBIC,antialias=True);top=int((nh-336)/2.);left=int((nw-336)/2.);u=u[:,top:top+336,left:left+336].numpy().copy();ind=((u.astype(np.float32)-mean)/std)[None];assert np.array_equal(ind,pixels.numpy()),('Independent pixels mismatch',i)
  digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
  front.append(dict(index=i,prompt=r['prompt'],reference=r['reference'],media_path=r['media_path'],media_file_SHA256=sha(r['media_path']),actual_model_prompt=prompt,text=text,input_ids=independent_ids,attention_mask=actual['attention_mask'][0].tolist(),image_token_id=image_id,original_full576_image_expansion=True,pixels_F32_shape=list(pixels.shape),pixels_F32_SHA256=digest(ind),independent_own_Rust_tokens_and_TV2_uint8_numpy_F32_all_values_exact=True))
  if i in [0,99,199]:write_full_standard_npz(H/('input_example_%05d.npz'%i),dict(input_ids=np.asarray(independent_ids,dtype=np.int64),pixels_F32=ind,resized_uint8=u))
  put('progress.json',dict(UTC=now(),phase='own_frontend_only_no_learned_CPU_or_NPU',completed=i+1,total=200))
 put('own_original_frontend200.json',front)
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
 wr,_,_=workbook_rows(book);row,=[v for v in wr if v['model_id']==MID];assert row==dict(row=338,model_id=MID,task='image-text-to-text',score=None)
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:server_D=ET.fromstring(z.read('xl/worksheets/sheet1.xml')).find('.//s:c[@r="D338"]/s:v',ns).text
 oldref=Path(binding['local_A100_reference_snapshot']);localref=Path(binding['current_user_workbook_D_reference_snapshot']);assert sha(oldref)==binding['local_A100_reference_snapshot_sha256'] and sha(localref)==binding['current_user_workbook_D_reference_snapshot_sha256']
 three=dict(server=server_D,local=js(localref)['models'][MID]['A100_D'],preserved_original=js(oldref)['models'][MID]['A100_D']);assert all(Decimal(v)==Decimal('.715') for v in three.values())
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix=='.json'}
 value=dict(passed=True,UTC=now(),model_id=MID,own_weight_path=str(W),own_weight_files=files,source_tensor_count=686,source_dtype_counts=dict(dtype_counts),own_assets_SHA256=assets,own_frontend200=str(H/'own_original_frontend200.json'),own_frontend200_SHA256=sha(H/'own_original_frontend200.json'),own_original_legacy_mapping_meta_review=str(H/'own_original_legacy_mapping_meta_review.json'),own_original_legacy_mapping_meta_review_SHA256=sha(H/'own_original_legacy_mapping_meta_review.json'),original_saved_YAML_SHA256=sha(saved),original_adapter_SHA256=sha(original),original_processor_type=original_processor_type,actual_frontend_processor_type=type(proc).__name__,original_processor_fallback_error=processor_error,own_tokenizer_image_ID=image_id,own_tokenizer_length=len(tok),own_checkpoint_embedding_and_head_rows=128256,original_saved_resize_token_embeddings_required=image_id>=128256,original_saved_resize_must_be_audited_before_any_NPU_dispatch=True,original_auto_on_nonCPU_requests_FP16=True,actual_loaded_CPU_or_NPU_dtype_not_yet_proved=True,original_generation_budget2048_outer10_Greedy_preserved=True,original_default_system_date_26_Jul_2024_preserved=True,original_saved_legacy_reconstruction_omits_raw_llama31_rope_scaling=True,own_BF16_storage_original_auto_nonCPU_FP16_request=True,current_row=338,three_D_values=three,dataset_SHA256=sha(dataset),all4329_order_and_all200_own_images_tokens_pixels_verified=True,historical_shadow_engine_not_captured_and_not_claimed=True,no_foreign_checkpoint_or_vision_source_used=True,ready_for_NPU=False,no_learned_CPU_or_NPU_forward=True,next_required='Own actual original head-resize provenance and full native graph/cache/observations/independent verifier/YAML binding before genuine NEW3')
 put('readiness.json',value);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_learned_CPU_or_NPU_forward=True));return value
if __name__=='__main__':
 try:
  with contextlib.redirect_stdout(sys.stderr):value=main()
  print(json.dumps(value,ensure_ascii=False))
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_learned_CPU_or_NPU_forward=True,all_original_source_reports_logs_unchanged=True))
  raise
