"""Own original Ovis-U1 image-understanding frontend, source headers, and active meta bijection."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,traceback,struct,re,math,ast
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/OvisU1/AIDC-AI/Ovis-U1-3B');H=B/'OvisU1_own_original_understanding_frontend200_active_meta_v2_exact_mixed_source_headers_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def tr(t):
 import torch
 return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
def clean(s):
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
def main():
 import torch,numpy as np
 from PIL import Image
 from universal_eval.model.ovisu1_own_understanding_20261116.configuration_ovis_u1 import OvisU1Config,Qwen3ConversationFormatter
 from universal_eval.model.ovisu1_own_understanding_20261116.understanding_v2_tokenizer_class_alias import OwnOvisUnderstandingV2
 from universal_eval.algorithm.inference import GreedyInference
 torch.set_num_threads(4);cfg=json.loads((W/'config.json').read_bytes());cfg.update(_name_or_path=str(W),multimodal_max_length=4096)
 with torch.device('meta'):model=OwnOvisUnderstandingV2(OvisU1Config(**cfg),W)
 assert model.llm.config._attn_implementation=='sdpa' and model.llm.config.num_hidden_layers==28 and model.llm.config.hidden_size==2048 and model.llm.config.vocab_size==151936 and model.llm.config.num_key_value_heads==8
 assert model.llm.lm_head.weight is model.llm.model.embed_tokens.weight
 headers={};weights={}
 for sf in sorted(W.glob('*.safetensors')):
  weights[sf.name]=dict(bytes=sf.stat().st_size,SHA256=sha(sf))
  with sf.open('rb') as f:size=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(size))
  for k,v in head.items():
   if k!='__metadata__':assert k not in headers;headers[k]=v
 index=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map'];assert set(index)==set(headers) and set(index.values())==set(weights)
 put('own_weight_files_SHA256.json',weights)
 expected_F32_active={'visual_tokenizer.head.0.weight','visual_tokenizer.head.1.bias','visual_tokenizer.head.1.weight'}
 assert {k for k,v in headers.items() if not k.startswith('visual_generator.') and v['dtype']=='F32'}==expected_F32_active
 assert all(v['dtype']=='F32' if k.startswith('visual_generator.') or k in expected_F32_active else v['dtype']=='BF16' for k,v in headers.items())
 expected={n:list(t.shape) for n,t in model.named_parameters(remove_duplicate=False)};active={n:v for n,v in headers.items() if not n.startswith('visual_generator.')};inactive={n:v for n,v in headers.items() if n.startswith('visual_generator.')}
 assert set(expected)-set(active)=={'llm.lm_head.weight'} and not set(active)-set(expected) and all(expected[n]==v['shape'] for n,v in active.items())
 assert expected['llm.lm_head.weight']==active['llm.model.embed_tokens.weight']['shape']==[151936,2048] and len(active)==487 and len(inactive)==573
 put('own_active_and_unused_source_headers.json',dict(active=active,original_unexecuted_Yak_generator=inactive));put('active_meta_bijection_review.json',dict(passed=True,own_source_tensors=len(headers),active_physical_parameters=len(active),active_alias_paths=len(expected),unused_original_generator_parameters=len(inactive),only_own_lm_head_tie_alias=True,no_model_forward=True))
 data=R/'workspace/datasets/any-to-any-mmbench/eval.jsonl';rows=[json.loads(x) for x in data.read_text().splitlines()];assert len(rows)>=200
 ip=model.visual_tokenizer.image_processor;assert type(ip).__name__=='CLIPImageProcessor' and ip.do_center_crop is False and ip.resample==3 and ip.patch_size==14 and ip.temporal_patch_size==1 and ip.hidden_stride==2
 records=[];out=H/'frontend200.jsonl'
 with out.open('x',encoding='utf-8') as stream:
  for i,row in enumerate(rows[:200]):
   prompt=clean(row['prompt']);mp=GreedyInference._add_answer_instruction(prompt);media=Path(row['media_path']);media=(data.parent/media).resolve() if not media.is_absolute() else media.resolve();assert media.is_relative_to(data.parent)
   image=Image.open(media).convert('RGB');query='<image>\n'+mp;formatted,ids,pix,grid=model.preprocess_inputs(query,[image]);mask=torch.ne(ids,model.text_tokenizer.pad_token_id)
   t,h,w=grid[0].tolist();assert t==1 and h%2==w%2==0;rh,rw=h*14,w*14
   # Independent original PIL bicubic -> uint8 -> F64 rescale/F32 normalize, original patch permutation.
   resized=np.asarray(image.resize((rw,rh),resample=Image.Resampling.BICUBIC),dtype=np.uint8).transpose(2,0,1)
   scaled=(resized.astype(np.float64)*ip.rescale_factor).astype(np.float32);mean=np.asarray(ip.image_mean,dtype=np.float32)[:,None,None];std=np.asarray(ip.image_std,dtype=np.float32)[:,None,None];normal=(scaled-mean)/std
   patches=normal[None].reshape(t,1,3,h//2,2,14,w//2,2,14).transpose(0,3,6,4,7,2,1,5,8).reshape(t*h*w,588);ind=torch.from_numpy(patches.copy())
   assert torch.equal(pix,ind),(i,'Own CLIP pixels differ from independent original normalization')
   text_again,raw_ids,_=model.get_conversation_formatter().format([{'from':'human','value':query}],generation_preface='',enable_thinking=False);assert formatted==text_again
   assert raw_ids.count(-200)==1 and ids.tolist().count(-300)==1 and pix.dtype==torch.float32 and len(ids)<4096
   rec=dict(index=i,prompt=prompt,original_model_prompt=mp,query=query,formatted_prompt=formatted,media_path=str(media),image_SHA256=sha(media),reference=row['reference'],input_ids=ids.tolist(),attention_mask=mask.tolist(),grid_thws=grid.tolist(),pixels_F32=tr(pix),pixels_BF16=tr(pix.bfloat16()),merged_visual_token_count=h*w//4,independent_pixels_equal=True)
   stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();records.append(rec);put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed=i+1,target=200,no_NN=True))
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.py','.json','.txt','.model']};result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id='AIDC-AI/Ovis-U1-3B',samples=200,own_weight_files=weights,own_assets_SHA256=assets,own_source_tensors=1060,active_source_tensors=487,active_source_storage_dtype_counts={'BF16':484,'F32':3},unused_Yak_generator_source_tensors=573,unused_Yak_source_storage_dtype='F32',original_requested_inference_dtype='BF16',own_lm_head_is_exact_tied_embedding=True,frontend=str(out),frontend_SHA256=sha(out),dataset=str(data),dataset_SHA256=sha(data),all200_independent_pixels_equal=True,original_multimodal_max_length=4096,original_max_new_tokens=512,original_outer_batch_size=4,actual_inner_batch_size=1,original_enable_thinking_default=False,actual_LM_backend='sdpa',tokenizer_type=type(model.text_tokenizer).__name__,processor_type=type(ip).__name__,no_NPU_inference=True,no_learned_CPU_forward=True,no_A100_historical_engine_or_tensor_capture_claimed=True)
 put('review.json',result);put('actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_model_forward=True));raise
