"""Own Gemma4 E4B checkpoint, original adapter template and independent patches.
No learned CPU/NPU forward or score. The saved original Gemma4 adapter already
sets enable_thinking=False; this records that original branch, never adds a new
thinking/budget/stop change to improve an evaluation result.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,struct,math,time,traceback,ast
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='prithivMLmods/gemma-4-E4B-it-Uncensored-MAX';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration')/MID;H=B/'PrithivGemma4E4B_own_original_BF16_adapter_template_false_pixel_patch_frontend200_v1_20261116';DATA=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(t,p)
def clean(s):
 import re
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
def main():
 import torch,numpy as np
 from PIL import Image
 from torchvision.transforms.v2 import functional as VF
 from torchvision.transforms import InterpolationMode
 from transformers import AutoConfig,AutoProcessor,AutoModelForImageTextToText
 from universal_eval.algorithm.inference import GreedyInference
 torch.set_num_threads(4);started=time.monotonic();protected=[D/'state.json',D/'registry.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')];before={str(p):sha(p) for p in protected}
 config=AutoConfig.from_pretrained(str(W),local_files_only=True);processor=AutoProcessor.from_pretrained(str(W),local_files_only=True);assert config.dtype==config.text_config.dtype==config.vision_config.dtype==torch.bfloat16 and config.text_config.num_hidden_layers==42 and config.vision_config.num_hidden_layers==16
 assert processor.chat_template and processor.image_processor.patch_size==16 and processor.image_processor.max_soft_tokens==280 and processor.image_processor.pooling_kernel_size==3 and not processor.image_processor.do_normalize
 assert js(W/'generation_config.json')['eos_token_id']==[1,106,50]
 original=R/'universal_eval/model/adapters/gemma4_vllm.py';tree=ast.parse(original.read_text());constructor=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');kwargs=[n for n in ast.walk(constructor) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='_chat_template_kwargs' for t in n.targets)];assert len(kwargs)==1 and ast.literal_eval(kwargs[0].value)=={'enable_thinking':False}
 headers={};source_files={};tensor_files={}
 index=js(W/'model.safetensors.index.json')['weight_map'];shards=sorted(set(index.values()));assert len(shards)==4
 for name in shards:
  weight=W/name
  with weight.open('rb') as f:prefix=f.read(8);length=struct.unpack('<Q',prefix)[0];raw=f.read(length)
  own={k:v for k,v in json.loads(raw).items() if k!='__metadata__'};assert not set(headers)&set(own)
  assert set(own)=={n for n,fn in index.items() if fn==name}
  headers.update(own);tensor_files.update({n:name for n in own})
  source_files[name]=dict(bytes=weight.stat().st_size,SHA256=sha(weight),header_SHA256=hashlib.sha256(prefix+raw).hexdigest(),tensor_count=len(own))
 assert len(headers)==2130 and set(index)==set(headers) and {v['dtype'] for v in headers.values()}=={'BF16'}
 put('source_tensor_files.json',tensor_files);put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_four_shards_full_file_SHA_and_exact_index_complete_no_NN',tensor_count=2130))
 import yaml
 saved=R/'configs/Gemma4ForConditionalGeneration/prithivMLmods--gemma-4-E4B-it-Uncensored-MAX.yaml';deployment=yaml.safe_load(saved.read_text());assert deployment['model']['name']==MID and deployment['model']['adapter']=='universal_eval/model/adapters/gemma4_vllm.py:Gemma4VLLMModel' and deployment['model']['kwargs']['max_tokens']==2048 and deployment['model']['kwargs']['max_model_len']==4096 and deployment['algorithm']['batch_size']==10
 with torch.device('meta'):model=AutoModelForImageTextToText.from_config(config)
 native=model.state_dict();unused={f'model.language_model.layers.{i}.self_attn.{n}' for i in range(24,42) for n in ['k_proj.weight','v_proj.weight','k_norm.weight']};missing=set(native)-set(headers);extra=set(headers)-set(native)
 assert extra==unused and missing=={'lm_head.weight'},dict(extra=sorted(extra),missing=sorted(missing))
 assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
 for n in set(headers)&set(native):assert headers[n]['shape']==list(native[n].shape),n
 put('source_tensor_headers.json',headers);put('own_source_native_meta_mapping.json',dict(native_state_tensors=len(native),all_source_tensors=2130,expected_unused_shared_KV_source=sorted(unused),own_tied_head_source='model.language_model.embed_tokens.weight',meta_state={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in native.items()},intrinsic_nonpersistent_buffers={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in model.named_buffers() if n not in native}))
 rows=[json.loads(s) for s in DATA.read_text().splitlines()];assert len(rows)==4329;records=[];output=H/'frontend200.jsonl'
 with output.open('x',encoding='utf-8') as dst:
  for i,row in enumerate(rows[:200]):
   prompt=clean(row['prompt']);model_prompt=GreedyInference._add_answer_instruction(prompt);assert '<answer>' in model_prompt;media=Path(row['media_path']);media=media if media.is_absolute() else DATA.parent/media;media=media.resolve();assert media.is_relative_to(DATA.parent)
   image=Image.open(media).convert('RGB');message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':model_prompt}]}];text=processor.apply_chat_template(message,add_generation_prompt=True,tokenize=False,enable_thinking=False)
   inputs=processor(text=text,images=[image],return_tensors='pt',add_special_tokens=False)
   h,w=image.height,image.width;factor=math.sqrt(2520*16**2/(h*w));th=int(math.floor(h*factor/48))*48;tw=int(math.floor(w*factor/48))*48
   assert th or tw
   if not th:th=48;tw=min(int(math.floor(w/h))*48,280*48)
   elif not tw:tw=48;th=min(int(math.floor(h/w))*48,280*48)
   assert th*tw<=2520*16**2
   pix=VF.pil_to_tensor(image)
   if (th,tw)!=(h,w):pix=VF.resize(pix,[th,tw],interpolation=InterpolationMode.BICUBIC,antialias=True)
   pix=pix.to(torch.float32).mul(1/255);ph,pw=th//16,tw//16;patch=pix.reshape(3,ph,16,pw,16).permute(1,3,2,4,0).reshape(ph*pw,768);position=torch.stack(torch.meshgrid(torch.arange(pw),torch.arange(ph),indexing='xy'),dim=-1).reshape(ph*pw,2);soft=ph*pw//9
   patch=torch.nn.functional.pad(patch,(0,0,0,2520-ph*pw));position=torch.nn.functional.pad(position,(0,0,0,2520-ph*pw),value=-1)
   assert torch.equal(inputs['pixel_values'],patch[None]) and torch.equal(inputs['image_position_ids'],position[None]),(i,'Original independent uint8 BICUBIC / F32 rescale / patch xy / zero and -1 padding differs')
   assert (inputs['input_ids']==config.image_token_id).sum().item()==soft and inputs['input_ids'].shape[1]+2048<=4096
   actual={}
   for n,t in inputs.items():
    assert torch.is_tensor(t);t=t.to(torch.bfloat16) if t.is_floating_point() else t;actual[n]=dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
   rec=dict(index=i,prompt=prompt,reference=row['reference'],media_path=str(media),image_SHA256=sha(media),original_model_prompt=model_prompt,formatted_text=text,inputs=actual,input_ids=inputs['input_ids'].tolist(),own_original_thinking_false_from_saved_adapter=True,independent_pixel_patch_position_replay_equal=True,actual_image_shape=[h,w],resized_image_shape=[th,tw],actual_soft_tokens=soft,generation=dict(max_new_tokens=2048,do_sample=False),historical_A100_tensor_capture_claimed=False)
   dst.write(json.dumps(rec,ensure_ascii=False)+'\n');dst.flush();records.append(rec);put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_original_CPU_patch_frontend_and_independent_replay_no_NN',completed=i+1,target=200))
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.model','.jinja']};assert before=={str(p):sha(p) for p in protected}
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,samples=200,own_checkpoint_tensor_count=2130,own_original_execution_dtype='BF16',own_weight_files=source_files,own_runtime_assets_SHA256=assets,dataset_SHA256=sha(DATA),frontend_jsonl_SHA256=sha(output),all200_independent_F32_patch_pixel_position_replays_equal=True,original_adapter_source=str(original),original_adapter_source_SHA256=sha(original),original_adapter_enable_thinking_false_preserved_not_newly_added=True,original_configured_outer_batch_size=10,original2048_budget_and4096_request_limit_preserved=True,source_native_mapping=str(H/'own_source_native_meta_mapping.json'),native_shared_unused_source_tensors=54,learned_CPU_or_NPU_forward=False,new_NPU_score=None,canonical_SHA256=before,elapsed_seconds=time.monotonic()-started)
 put('review.json',result);put('actual_completion.json',dict(actual_returncode=0,UTC=result['UTC'],no_NN=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(parents=True,exist_ok=True)
 try:main()
 except BaseException:
  put('actual_completion.json',dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),no_NN=True,error=traceback.format_exc()));raise
