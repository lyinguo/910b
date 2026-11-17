"""Own Granite4.1 vision saved HF BF16/4096 inputs and source META only.

No learned CPU or NPU model forward, no downloaded or foreign resource.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,struct,math,time,traceback,ast
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='ibm-granite/granite-vision-4.1-4b';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Granite4VisionForConditionalGeneration')/MID;H=B/'Granite41_own_original_HF_BF16_4096_frontend200_source_META_v3_exact_own_wrapper_layout_20261117';DATA=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def clean(s):
 import re
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
def best_resolution(h,w,pinpoints):
 choices=[]
 for i,(th,tw) in enumerate(pinpoints):
  scale=min(tw/w,th/h);effective=min(int(w*scale)*int(h*scale),w*h);wasted=tw*th-effective;choices.append((-effective,wasted,i,th,tw))
 _,_,_,th,tw=min(choices);return th,tw
def independent_features(h,w,bh,bw):
 ph,pw=12*(bh//384),12*(bw//384)
 if w/h>pw/ph:
  nh=int(round(h*(pw/w),7));ph-=((ph-nh)//2)*2
 else:
  nw=int(round(w*(ph/h),7));pw-=((pw-nw)//2)*2
 return 144+ph*pw+ph
def main():
 import torch,yaml
 from PIL import Image
 from torchvision.transforms.v2 import functional as VF
 from torchvision.transforms import InterpolationMode
 from universal_eval.algorithm.inference import GreedyInference
 from universal_eval.model.granite41_own_original_source_20261117.configuration import Granite4VisionConfig
 from universal_eval.model.granite41_own_original_source_20261117.processing import Granite4VisionProcessor
 from universal_eval.model.granite41_own_original_source_20261117.modeling import Granite4VisionForConditionalGeneration
 torch.set_num_threads(4);start=time.monotonic();protected=[D/'state.json',D/'registry.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')];before={str(p):sha(p) for p in protected}
 saved=R/'configs/Granite4VisionForConditionalGeneration/ibm-granite--granite-vision-4.1-4b.yaml';deployment=yaml.safe_load(saved.read_text());assert deployment['model']['adapter']=='universal_eval/model/adapters/transformers_vlm.py:TransformersVLM' and deployment['model']['kwargs']==dict(device='auto',max_new_tokens=4096,torch_dtype='bfloat16');assert deployment['algorithm']==dict(name='greedy_inference',batch_size=10,kwargs=dict(extract_answer=True))
 original=R/'universal_eval/model/adapters/transformers_vlm.py';assert sha(original)=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3'
 own_remote={}
 for name in ['configuration.py','processing.py','modeling.py','downsampling.py']:
  a=W/name;b=R/'universal_eval/model/granite41_own_original_source_20261117'/name;assert a.read_bytes()==b.read_bytes();own_remote[str(a)]=sha(a)
 config=Granite4VisionConfig.from_pretrained(W,local_files_only=True);processor=Granite4VisionProcessor.from_pretrained(W,local_files_only=True)
 assert config.text_config.num_hidden_layers==40 and config.text_config.hidden_size==2560 and config.text_config.vocab_size==100352 and config.vision_config.num_hidden_layers==27 and config.image_token_index==100266
 assert config.text_config.attention_multiplier==.015625 and config.text_config.embedding_multiplier==12 and config.text_config.logits_scaling==10 and config.text_config.residual_multiplier==.22
 assert processor.downsample_rate=='4/8' and processor.patch_size==16 and processor.vision_feature_select_strategy=='full' and processor.num_additional_image_tokens==0 and processor.image_token_id==100266
 ip=processor.image_processor;assert type(ip).__name__=='LlavaNextImageProcessor' and type(ip).__module__.endswith('.image_processing_llava_next'),(type(ip).__module__,type(ip).__name__)
 assert (ip.size.height,ip.size.width)==(384,384) and (ip.crop_size.height,ip.crop_size.width)==(384,384) and tuple(ip.image_mean)==tuple(ip.image_std)==(.5,.5,.5) and ip.rescale_factor==1/255 and ip.resample==3 and ip.do_resize and ip.do_center_crop and ip.do_rescale and ip.do_normalize and ip.do_pad
 assert js(W/'generation_config.json')['eos_token_id']==100257
 headers={};weight_files={}
 for f in sorted(W.glob('*.safetensors')):
  with f.open('rb') as inp:prefix=inp.read(8);n=struct.unpack('<Q',prefix)[0];raw=inp.read(n)
  part={k:dict(v,shard=f.name) for k,v in json.loads(raw).items() if k!='__metadata__'};assert not set(part)&set(headers);headers.update(part);weight_files[f.name]=dict(bytes=f.stat().st_size,SHA256=sha(f),header_SHA256=hashlib.sha256(prefix+raw).hexdigest(),tensor_count=len(part));put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_original_source_file_SHA_no_NN',files_done=len(weight_files)))
 put('own_checkpoint_files_full_SHA.json',weight_files)
 assert len(weight_files)==2 and sum(x['bytes'] for x in weight_files.values())==7994560248 and len(headers)==1083 and {x['dtype'] for x in headers.values()}=={'BF16'}
 with torch.device('meta'):model=Granite4VisionForConditionalGeneration(config)
 native=model.state_dict();mapping={}
 for n,x in headers.items():
  candidate=n if n in native else n.replace('.vision_tower.vision_model.','.vision_tower.',1)
  assert candidate in native and candidate not in mapping.values() and x['shape']==list(native[candidate].shape),(n,candidate)
  mapping[n]=candidate
 extra=set(headers)-set(mapping);missing=set(native)-set(mapping.values());assert not extra and missing==set(['lm_head.weight']),(sorted(extra),sorted(missing))
 assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
 put('own_source_to_native_META_name_map.json',dict(mapping=mapping,only_optional_vision_model_wrapper_prefix_removed=True,own_source_count=len(headers),native_state_count=len(native),all_own_sources_mapped_exactly_once=True,no_tensor_value_dtype_or_model_math_changed=True))
 put('source_tensor_headers.json',headers);put('own_source_native_META_mapping.json',dict(meta_only_no_learned_values_or_forward=True,own_source_tensors=1083,native_state_tensors=len(native),exact_extra_sources=sorted(extra),own_tied_head_source='model.language_model.embed_tokens.weight' if 'lm_head.weight' in missing else None,native_state={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in native.items()}))
 rows=[json.loads(s) for s in DATA.read_text().splitlines()];assert len(rows)==4329;records=[];out=H/'frontend200.jsonl'
 with out.open('x',encoding='utf8') as stream:
  for i,row in enumerate(rows[:200]):
   prompt=clean(row['prompt']);mp=GreedyInference._add_answer_instruction(prompt);media=Path(row['media_path']);media=media if media.is_absolute() else DATA.parent/media;media=media.resolve();assert media.is_relative_to(DATA.parent);image=Image.open(media).convert('RGB');h,w=image.height,image.width;message=[dict(role='user',content=[dict(type='image'),dict(type='text',text=mp)])];text=processor.apply_chat_template(message,add_generation_prompt=True,tokenize=False);assert text.count(processor.image_token)==1
   inputs=processor(text=text,images=[image],return_tensors='pt',padding=True)
   bh,bw=best_resolution(h,w,ip.image_grid_pinpoints);sw=bw/w;sh=bh/h
   if sw<sh:nh=min(math.ceil(h*sw),bh);nw=bw
   else:nh=bh;nw=min(math.ceil(w*sh),bw)
   pixels=VF.pil_to_tensor(image);resized=VF.resize(pixels,[nh,nw],interpolation=InterpolationMode.BICUBIC,antialias=True);dx=bw-nw;dy=bh-nh;padded=torch.nn.functional.pad(resized,(dx//2,dx-dx//2,dy//2,dy-dy//2));patches=[VF.resize(pixels,[384,384],interpolation=InterpolationMode.BICUBIC,antialias=True)]+[padded[:,y:y+384,x:x+384] for y in range(0,bh,384) for x in range(0,bw,384)];independent=torch.stack(patches).to(torch.float32).sub(127.5).div(127.5)[None]
   assert torch.equal(inputs['pixel_values'],independent),(i,'Own independent uint8 bicubic/zero pad/row-major tiles/F32 fused normalize differs')
   assert torch.equal(inputs['image_sizes'],torch.tensor([[h,w]]))
   features=independent_features(h,w,bh,bw);assert features==processor._get_number_of_features(h,w,384,384) and (inputs['input_ids']==100266).sum().item()==features
   expanded=text.replace(processor.image_token,processor.image_token*features);tokens=processor.tokenizer([expanded],padding=True,return_tensors='pt');assert torch.equal(tokens['input_ids'],inputs['input_ids']) and torch.equal(tokens['attention_mask'],inputs['attention_mask']);assert inputs['input_ids'].shape[1]+4096<=131072
   data={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()) for n,t in inputs.items()};rec=dict(index=i,prompt=prompt,reference=row['reference'],media_path=str(media),image_SHA256=sha(media),original_model_prompt=mp,formatted_text=text,input_ids=inputs['input_ids'].tolist(),attention_mask=inputs['attention_mask'].tolist(),inputs=data,actual_image_shape=[h,w],selected_resolution=[bh,bw],resized_shape=[nh,nw],tiles=len(patches),actual_image_tokens=features,all_independent_pixel_tile_token_mask_replays_equal=True,original_max_new_tokens=4096,original_outer_batch10_inner_serial1=True,historical_A100_tensor_capture_claimed=False)
   stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();records.append(rec);put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_original_frontend_no_learned_model',completed=i+1,target=200))
 assert before=={str(p):sha(p) for p in protected}
 review=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,samples=200,own_weight_files=weight_files,own_source_count=1083,own_original_remote_source_SHA256=own_remote,original_saved_YAML=str(saved),original_saved_YAML_SHA256=sha(saved),original_adapter_source_SHA256=sha(original),actual_processor_class=type(processor).__module__+'.'+type(processor).__name__,actual_image_processor_class=type(ip).__module__+'.'+type(ip).__name__,actual_processor_input_dtype='torch.float32',original_model_execution_dtype='BF16',original_max_new_tokens=4096,original_outer_batch_size=10,original_within_batch_execution='serial1',all200_independent_pixel_tile_token_mask_replays_equal=True,frontend_jsonl_SHA256=sha(out),dataset_SHA256=sha(DATA),own_source_native_META_mapping=str(H/'own_source_native_META_mapping.json'),learned_CPU_or_NPU_model_forward=False,actual_NPU_source_mapping_numeric_small3_approval_200_G_still_pending=True,canonical_SHA256=before,elapsed_seconds=time.monotonic()-start)
 put('review.json',review);put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,no_NN=True));print(json.dumps(review,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:
  put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,no_NN=True,error=traceback.format_exc()));raise
