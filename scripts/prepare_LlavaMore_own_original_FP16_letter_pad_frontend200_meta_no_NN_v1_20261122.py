"""Own original discriminative-letter pad/system frontend and complete source/meta; no NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import collections,contextlib,hashlib,json,os,struct,traceback,zipfile,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaLlamaForCausalLM')/MID
H=B/'LlavaMore_own_original_FP16_discriminative_letter_pad_frontend200_meta_no_NN_v1_20261122'
sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
now=lambda:datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def main():
 import numpy as np,torch,yaml
 from PIL import Image
 from tokenizers import Tokenizer
 from transformers import AutoTokenizer,AutoProcessor,CLIPImageProcessor,LlavaProcessor,LlamaConfig,CLIPVisionConfig,LlavaConfig,LlavaForConditionalGeneration
 from universal_eval.model.adapters.llava_more_vlm import LlavaMoreVLM,_strip_eval_instruction,_parse_options,_LLAMA_3_1_SYSTEM,_PAD_COLOR
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz
 from torchvision.transforms.v2 import functional as tv
 from torchvision.transforms import InterpolationMode
 from batch_910b_support import workbook_rows
 import xml.etree.ElementTree as ET
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 source=R/'universal_eval/model/adapters/llava_more_vlm.py';base=R/'universal_eval/model/adapters/transformers_vlm.py';saved=R/'configs/LlavaLlamaForCausalLM/aimagelab--LLaVA_MORE-llama_3_1-8B-finetuning.yaml'
 assert sha(source)=='819af5fbdbda709c179b3b5f1256f19162f43b94d49182f3054a7e0b29af50a1' and sha(base)=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3' and sha(saved)=='a1052e5e48c71e6b2de7277674c588866596afa4bd5d5236e486826524e37e39'
 old=yaml.safe_load(saved.read_text());assert old['model']['kwargs']==dict(device='auto',max_new_tokens=2048,torch_dtype='auto',image_mode='pad',mode='discriminative_letter') and old['algorithm']['batch_size']==10
 raw=js(W/'config.json');assert raw['architectures']==['LlavaLlamaForCausalLM'] and raw['num_hidden_layers']==32 and raw['hidden_size']==4096 and raw['num_key_value_heads']==8 and raw['rope_theta']==500000 and raw['rope_scaling']['rope_type']=='llama3' and raw['max_position_embeddings']==131072
 index=js(W/'model.safetensors.index.json')['weight_map'];headers={};files={};flat={};dtypes=collections.Counter()
 for name in sorted(set(index.values())):
  p=W/name;assert p.is_file()
  with p.open('rb') as f:count=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(count))
  h={k:v for k,v in h.items() if k!='__metadata__'};assert h and all(index[k]==name for k in h) and not (set(flat)&set(h))
  headers[name]=h;flat.update(h);dtypes.update(v['dtype'] for v in h.values());files[name]=dict(bytes=p.stat().st_size,SHA256=sha(p),header_bytes=count)
  put('progress.json',dict(UTC=now(),phase='own_full_original_SF_SHA_header_only_no_NN',completed_shards=len(files),total_shards=len(set(index.values()))))
 assert set(flat)==set(index) and len(flat)==686 and dict(dtypes)=={'F16':686} and sum(v['bytes'] for v in files.values())==16709591824
 put('own_weight_files_and_headers.json',dict(own_weight_files=files,headers=headers,source_dtype_counts=dict(dtypes),source_tensor_count=686))
 # Exact saved inherited legacy conversion deliberately omits flat raw rope_scaling.
 fields=['vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','rms_norm_eps','rope_theta','max_position_embeddings','tie_word_embeddings','attention_bias','hidden_act','initializer_range','pretraining_tp']
 text=LlamaConfig(**{k:raw[k] for k in fields if k in raw});assert text.rope_parameters['rope_type']=='default'
 vision=CLIPVisionConfig(hidden_size=1024,num_hidden_layers=24,num_attention_heads=16,intermediate_size=4096,patch_size=14,image_size=336,num_channels=3,projection_dim=768,layer_norm_eps=1e-5,hidden_act='quick_gelu')
 config=LlavaConfig(text_config=text,vision_config=vision,image_token_index=int(raw.get('image_token_index',32000)),vision_feature_layer=-2,vision_feature_select_strategy='default',image_aspect_ratio=raw['image_aspect_ratio'],mm_hidden_size=1024)
 before=torch.get_default_dtype()
 try:
  torch.set_default_dtype(torch.float16)
  with torch.device('meta'):meta=LlavaForConditionalGeneration(config)
 finally:torch.set_default_dtype(before)
 params=dict(meta.named_parameters());mapped={};mapping={}
 for key,val in flat.items():
  n=key
  if key.startswith('model.layers.'):n='model.language_model.layers.'+key[len('model.layers.'):]
  elif key=='model.embed_tokens.weight':n='model.language_model.embed_tokens.weight'
  elif key=='model.norm.weight':n='model.language_model.norm.weight'
  elif key.startswith('model.mm_projector.'):
   a,b=key[len('model.mm_projector.'):].split('.');n='model.multi_modal_projector.'+{'0':'linear_1','2':'linear_2'}[a]+'.'+b
  elif key.startswith('model.vision_tower.vision_tower.vision_model.'):n='model.vision_tower.'+key[len('model.vision_tower.vision_tower.vision_model.'):]
  assert n not in mapped;mapping[key]=n;mapped[n]=val
 exact=set(mapped)==set(params) and all(list(params[k].shape)==v['shape'] for k,v in mapped.items() if k in params)
 put('own_original_legacy_mapping_meta_review.json',dict(exact_all686_original_mapping_names_shapes=exact,source_to_native_mapping=mapping,mapped_count=len(mapped),meta_parameter_count=len(params),missing_parameter_keys=sorted(set(params)-set(mapped)),unmapped_source_keys=sorted(set(mapped)-set(params)),mismatched_shapes={k:dict(own=v['shape'],meta=list(params[k].shape)) for k,v in mapped.items() if k in params and list(params[k].shape)!=v['shape']},meta_buffer_layout={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in meta.named_buffers()},original_saved_legacy_field_list=fields,raw_llama31_scaling_present_but_original_saved_conversion_omits_it=True,native_plain_RoPE_theta500000_preserved=True,meta_not_actual_loaded_dtype_or_NN=True))
 assert exact,'Own source/meta mapping mismatch; retain all evidence, no random or borrowed source'
 tokenizer=AutoTokenizer.from_pretrained(W,local_files_only=True,trust_remote_code=True);rust=Tokenizer.from_file(str(W/'tokenizer.json'));processor_error=None
 try:proc=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=True)
 except Exception as e:processor_error=type(e).__name__+': '+str(e);proc=tokenizer
 original_processor_type=type(proc).__name__
 if not hasattr(proc,'image_processor') and not hasattr(proc,'feature_extractor'):
  image_processor=CLIPImageProcessor(do_resize=True,size={'shortest_edge':336},resample=3,do_center_crop=True,crop_size={'height':336,'width':336},do_rescale=True,rescale_factor=1/255.,do_normalize=True,image_mean=[.48145466,.4578275,.40821073],image_std=[.26862954,.26130258,.27577711]);tok=getattr(proc,'tokenizer',None) or proc
  if '<image>' not in tok.get_vocab():tok.add_special_tokens({'additional_special_tokens':['<image>']})
  proc=LlavaProcessor(image_processor=image_processor,tokenizer=proc,patch_size=14,vision_feature_select_strategy='patch',num_additional_image_tokens=0)
 if getattr(proc,'patch_size',None) is None:proc.patch_size=14
 if getattr(proc,'num_additional_image_tokens',None) is None:proc.num_additional_image_tokens=0
 if getattr(proc,'vision_feature_select_strategy',None) is None:proc.vision_feature_select_strategy='full'
 tok=getattr(proc,'tokenizer',None) or proc
 if '<image>' not in tok.get_vocab():tok.add_special_tokens({'additional_special_tokens':['<image>']})
 if rust.token_to_id('<image>') is None:rust.add_special_tokens(['<image>'])
 assert rust.get_vocab()==tok.get_vocab();image_id=tok.convert_tokens_to_ids('<image>');assert image_id==rust.token_to_id('<image>')==128256 and len(tok)==128257
 proc.image_token_id=image_id
 actor=LlavaMoreVLM(device='auto',max_new_tokens=2048,torch_dtype='auto',image_mode='pad',mode='discriminative_letter');actor._processor=proc;actor._tokenizer=tok;actor._model_name=MID;actor._model_path=str(W)
 assert tuple(_PAD_COLOR)==(122,116,104) and actor._system_prompt==_LLAMA_3_1_SYSTEM
 dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';afs=Path(old['data']['dataset_path']);assert sha(dataset)==sha(afs)
 def rows(p,flt):
  x=DatasetLoader().load(str(p));x=TaskDataRegistry._resolve_media_paths(x,str(p.parent));return DataPreprocessor().preprocess(x,task_type='image-text-to-text',apply_template=False,filter_empty=flt)
 records=rows(dataset,False);filtered=rows(dataset,True);orig=rows(afs,True);assert len(records)==len(filtered)==len(orig)==4329
 assert [(r['prompt'],r['reference']) for r in records]==[(r['prompt'],r['reference']) for r in filtered]==[(r['prompt'],r['reference']) for r in orig]
 mean=np.asarray([.48145466,.4578275,.40821073],dtype=np.float32)[:,None,None]*np.float32(255);std=np.asarray([.26862954,.26130258,.27577711],dtype=np.float32)[:,None,None]*np.float32(255)
 front=[]
 for i,r in enumerate(records[:200]):
  assert sha(r['media_path'])==sha(orig[i]['media_path'])
  prompt=GreedyInference._add_answer_instruction(r['prompt']);clean=_strip_eval_instruction(prompt).replace('<image>','',1).lstrip('\n').strip();options=_parse_options(clean);assert 2<=len(options)<=5 and [a for a,b in options]==list('ABCDE'[:len(options)])
  rendered=actor._build_text(clean);ind_text=str(tok.bos_token)+'<|start_header_id|>system<|end_header_id|>\n\n'+_LLAMA_3_1_SYSTEM+'<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\n'+clean+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n';assert rendered==ind_text
  image=actor._prepare_image(r['media_path']);assert image is not None and image.mode=='RGB' and image.size[0]==image.size[1]
  with Image.open(r['media_path']) as im:original_pixels=np.asarray(im.convert('RGB'),dtype=np.uint8)
  ih,iw=original_pixels.shape[:2];side=max(ih,iw);canvas=np.empty((side,side,3),dtype=np.uint8);canvas[:]=[122,116,104];top,left=(side-ih)//2,(side-iw)//2;canvas[top:top+ih,left:left+iw]=original_pixels;assert np.array_equal(canvas,np.asarray(image))
  actual=proc(text=rendered,images=[image],return_tensors='pt');pixels=actual['pixel_values'];assert pixels.dtype==torch.float32 and list(pixels.shape)==[1,3,336,336]
  ids=rust.encode(rendered.replace('<image>','<image>'*576),add_special_tokens=True).ids;assert actual['input_ids'][0].tolist()==ids and ids.count(image_id)==576 and actual['attention_mask'].tolist()==[[1]*len(ids)]
  t=torch.from_numpy(canvas.transpose(2,0,1).copy());u=tv.resize(t,[336,336],interpolation=InterpolationMode.BICUBIC,antialias=True).numpy().copy();ind=((u.astype(np.float32)-mean)/std)[None];assert np.array_equal(ind,pixels.numpy()),('Independent own padded pixels mismatch',i)
  half=ind.astype(np.float16);assert np.array_equal(half,pixels.to(torch.float16).numpy())
  letters={}
  for ch,opt in options:
   a=set();b=set()
   for form in (' '+ch,ch):a.update(tok(form,add_special_tokens=False)['input_ids']);b.update(rust.encode(form,add_special_tokens=False).ids)
   assert a==b and a;letters[ch]=sorted(a)
  digest=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
  front.append(dict(index=i,prompt=r['prompt'],reference=r['reference'],media_path=r['media_path'],media_file_SHA256=sha(r['media_path']),actual_algorithm_prompt=prompt,original_clean_prompt=clean,original_system_prompt=_LLAMA_3_1_SYSTEM,text=rendered,original_options=options,original_letter_candidate_ID_sets=letters,input_ids=ids,attention_mask=actual['attention_mask'][0].tolist(),image_token_id=image_id,original_full576_image_expansion=True,original_pad_RGB=[122,116,104],pixels_F32_shape=list(pixels.shape),pixels_F32_SHA256=digest(ind),original_actual_FP16_inputs_SHA256=digest(half),independent_own_Rust_tokens_and_TV2_uint8_numpy_pixels_and_pad_all_values_exact=True,original_branch_discriminative_letter_one_forward_cacheFalse=True,original_full_F32_last_logits_variant_all_token_ID_max_not_generation=True))
  if i in [0,99,199]:write_full_standard_npz(H/('input_example_%05d.npz'%i),dict(input_ids=np.asarray(ids,dtype=np.int64),pixels_F32=ind,pixels_FP16=half,padded_uint8=canvas,resized_uint8=u))
  put('progress.json',dict(UTC=now(),phase='own_frontend_only_no_learned_CPU_or_NPU',completed=i+1,total=200))
 put('own_original_frontend200.json',front)
 book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
 row,=[x for x in workbook_rows(book)[0] if x['model_id']==MID];assert row==dict(row=159,model_id=MID,task='image-text-to-text',score=None)
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:server_D=ET.fromstring(z.read('xl/worksheets/sheet1.xml')).find('.//s:c[@r="D159"]/s:v',ns).text
 oldref=Path(binding['local_A100_reference_snapshot']);localref=Path(binding['current_user_workbook_D_reference_snapshot']);assert sha(oldref)==binding['local_A100_reference_snapshot_sha256'] and sha(localref)==binding['current_user_workbook_D_reference_snapshot_sha256']
 three=dict(server=server_D,local=js(localref)['models'][MID]['A100_D'],preserved_original=js(oldref)['models'][MID]['A100_D']);assert all(Decimal(v)==Decimal('.79') for v in three.values())
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix=='.json'}
 put('original_no_forward_scoring_and_source_boundary.json',dict(original_mode='discriminative_letter',original_max_new_tokens=2048,original_outer_batch=10,actual_per_sample_forward_when_options_present=1,original_use_cache=False,original_letter_score_math='full native last logits .float then max over ALL IDs from each original spaced/unspaced uppercase option letter; final Python max preserves original option order on ties',all200_original_options_present=True,original_fallback_generation_not_executed_by_this_input_only_stage=True,original_system_prompt=_LLAMA_3_1_SYSTEM,original_pad_RGB=[122,116,104],original_prepare_input_F32_to_FP16=True,source_function_text={n:inspect.getsource(getattr(LlavaMoreVLM,n)) for n in ['_pad_image','_prepare_image','_build_text','_prepare_inputs','_letter_scores','_predict_letter','predict']},no_learned_CPU_or_NPU_forward=True))
 value=dict(passed=True,UTC=now(),model_id=MID,own_weight_path=str(W),own_weight_files=files,source_tensor_count=686,source_dtype_counts=dict(dtypes),own_assets_SHA256=assets,own_frontend200=str(H/'own_original_frontend200.json'),own_frontend200_SHA256=sha(H/'own_original_frontend200.json'),own_original_legacy_mapping_meta_review=str(H/'own_original_legacy_mapping_meta_review.json'),own_original_legacy_mapping_meta_review_SHA256=sha(H/'own_original_legacy_mapping_meta_review.json'),original_saved_YAML_SHA256=sha(saved),original_adapter_SHA256=sha(source),original_inherited_Transformers_adapter_SHA256=sha(base),original_processor_type=original_processor_type,actual_frontend_processor_type=type(proc).__name__,original_processor_fallback_error=processor_error,own_tokenizer_image_ID=image_id,own_tokenizer_length=len(tok),own_checkpoint_embedding_and_head_rows=128256,original_saved_resize_token_embeddings_required=True,original_saved_resize_must_be_audited_before_any_NPU_dispatch=True,original_auto_on_nonCPU_requests_FP16=True,actual_loaded_CPU_or_NPU_dtype_not_yet_proved=True,original_probability_not_generation_protocol_preserved=True,original_raw_llama31_scaling_not_in_saved_legacy_graph=True,original_generation_budget2048_outer10_preserved_and_not_used_to_change_mode=True,current_row=159,three_D_values=three,dataset_SHA256=sha(dataset),all4329_order_and_all200_own_images_tokens_pixels_verified=True,no_foreign_checkpoint_vision_vocabulary_or_input_reuse=True,historical_A100_engine_not_captured_and_not_claimed=True,ready_for_NPU=False,no_learned_CPU_or_NPU_forward=True,next_required='Own original constructor full head/resize provenance, native source loading/complete observations and independent full-numeric verifier/standalone YAML/unique runner genuine NEW3. Do not dispatch on this frontend result.')
 put('readiness.json',value);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_learned_CPU_or_NPU_forward=True));return value
if __name__=='__main__':
 try:
  with contextlib.redirect_stdout(sys.stderr):value=main()
  print(json.dumps(value,ensure_ascii=False))
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_learned_CPU_or_NPU_forward=True,all_original_sources_reports_logs_retained=True))
  raise
