"""Own original source/meta and complete pixel/token frontend; no learned forward.

The original wrapper ignores its declared max_new_tokens on the query path.
Retain reasoning=True and the source's 768/.5/.9 defaults. Do not infer the
effective loaded model dtype from a generic auto option or a meta constructor.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib,json,math,os,re,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
MID='moondream/moondream3-preview';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/HfMoondream')/MID
PARENT=B/'Moondream3_own_original_source_index_meta_preflight_0610_20261119';PKG=R/'universal_eval/model/moondream3_own_original_source_20261119'
H=B/'Moondream3_own_original_source_native_BF16_pixel_frontend200_v2_0610_20261119'
DATA=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(name,value):
 p=H/name;tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)
def clean(s):
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
def independent_crops(image,backend):
 import numpy as np
 from PIL import Image
 a=np.asarray(image.convert('RGB'));height,width=a.shape[:2];margin=56;window=266;h=height-112;w=width-112
 if min(h,w)<=window:tiles=(1,1)
 else:
  mh,mw=math.ceil(h/window),math.ceil(w/window)
  if mh*mw>12:
   ratio=(12/(mh*mw))**.5;tiles=(max(1,math.floor(mh*ratio)),max(1,math.floor(mw*ratio)))
  else:
   nh=max(math.floor((12*h/w)**.5),mh);nw=max(math.floor((12*w/h)**.5),mw)
   if nh*nw>12:
    if nw>nh:nw=12//nh
    else:nh=12//nw
   tiles=(max(1,nh),max(1,nw))
 nh,nw=tiles;target=(nh*window+2*margin,nw*window+2*margin)
 if backend.HAS_VIPS:
  v=backend.pyvips.Image.new_from_array(a);resized=v.resize(target[1]/width,vscale=target[0]/height).numpy();global_=v.resize(378/width,vscale=378/height).numpy()
 else:
  v=Image.fromarray(a);resized=np.asarray(v.resize((target[1],target[0]),resample=Image.Resampling.LANCZOS));global_=np.asarray(v.resize((378,378),resample=Image.Resampling.LANCZOS))
 output=np.zeros((1+nh*nw,378,378,3),dtype=np.uint8);output[0]=global_
 for ordinal in range(nh*nw):
  iy,ix=divmod(ordinal,nw);part=resized[iy*window:iy*window+378,ix*window:ix*window+378];output[ordinal+1,:part.shape[0],:part.shape[1]]=part
 return output,tiles
def independent_BF16_normalization(crops):
 import numpy as np
 def rounded(x):
  u=np.asarray(x,dtype=np.float32).view(np.uint32);return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
 def expand(x):return (x.astype(np.uint32)<<16).view(np.float32)
 u=rounded(crops.transpose(0,3,1,2).astype(np.float32));u=rounded(expand(u)/np.float32(255));u=rounded(expand(u)-np.float32(.5));return rounded(expand(u)/np.float32(.5))
def main():
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
 H.mkdir(exist_ok=False);start=time.monotonic();sys.path[:0]=[str(R),str(R/'scripts')]
 import torch,numpy as np
 from PIL import Image
 from tokenizers import Tokenizer
 from universal_eval.algorithm.inference import GreedyInference
 torch.set_num_threads(4)
 ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());before={p:sha(p) for p in ptr['canonical_SHA256']};assert before==ptr['canonical_SHA256']
 parent=json.loads((PARENT/'review.json').read_bytes());assert parent['passed'] and parent['no_model_NPU_or_learned_CPU_forward']
 original_sources=json.loads((PARENT/'own_original_source_migration.json').read_bytes())['sources']
 for rec in original_sources:assert sha(rec['own_original_path'])==sha(rec['primary_path'])==rec['SHA256']
 package='universal_eval.model.moondream3_own_original_source_20261119';hf=importlib.import_module(package+'.hf_moondream');vis=importlib.import_module(package+'.vision');crop=importlib.import_module(package+'.image_crops')
 original=Tokenizer.from_pretrained
 def local_tokenizer(name,*args,**kwargs):
  assert name=='moondream/starmie-v1' and not args and not kwargs;return Tokenizer.from_file(str(W/'tokenizer.json'))
 try:
  Tokenizer.from_pretrained=local_tokenizer;cfg=hf.HfConfig(**json.loads((W/'config.json').read_bytes()));cfg.dtype=torch.float16
  with torch.device('meta'):model=hf.HfMoondream(cfg)
 finally:Tokenizer.from_pretrained=original
 meta=model.state_dict();headers=json.loads((PARENT/'own_index_exact_shards_and_headers.json').read_bytes())['all_tensor_headers']
 missing=sorted(set(headers)-set(meta));extra=set(meta)-set(headers);aliases={}
 for name in extra:
  same=[key for key in headers if id(meta[key])==id(meta[name])]
  # state_dict tensors detach to new wrappers; parameter identity is authoritative.
  if not same:
   parameters=dict(model.named_parameters(remove_duplicate=False));same=[key for key in headers if key in parameters and name in parameters and parameters[key] is parameters[name]]
  assert len(same)==1,(name,same);aliases[name]=same[0]
 assert not missing and all(list(meta[k].shape)==headers[k]['shape'] for k in headers)
 meta_dtype=sorted({str(meta[k].dtype) for k in headers});assert meta_dtype==['torch.bfloat16']
 put('actual_original_HF_class_meta_precision_and_aliases.json',dict(actual_physical_meta_dtype=meta_dtype,requested_config_dtype=str(cfg.dtype),source_count=len(headers),extra_true_aliases=aliases,missing=missing,
  supersedes_v1_unproven_runtime_FP16_inference=True,meta_is_not_an_effective_loaded_runtime_dtype_proof=True,own_vision_source_always_produces_BF16_pixels=True,no_NN=True))
 index=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map'];shards={}
 for name in sorted(set(index.values())):
  f=W/name;shards[name]=dict(bytes=f.stat().st_size,SHA256=sha(f));put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_selected_index_full_shard_SHA_no_weight_array_or_NN',completed_shards=len(shards),target=4))
 rows=[json.loads(line) for line in DATA.read_text().splitlines()];assert len(rows)==4329;tokenizer=Tokenizer.from_file(str(W/'tokenizer.json'));records=[];assert tokenizer.get_vocab_size()==51200
 with (H/'frontend200.jsonl').open('x',encoding='utf8') as out:
  for i,row in enumerate(rows[:200]):
   image_path=Path(row['media_path']);image_path=image_path if image_path.is_absolute() else DATA.parent/image_path;image_path=image_path.resolve();assert image_path.is_file()
   image=Image.open(image_path).convert('RGB');actual,tiling=vis.prepare_crops(image,model.model.config.vision,device='cpu');assert actual.dtype==torch.bfloat16
   independent,tiles=independent_crops(image,crop);assert tuple(tiling)==tiles
   reference=independent_BF16_normalization(independent);a=actual.contiguous().view(torch.uint16).numpy();assert np.array_equal(a,reference),(i,'original per-operation BF16 rounding or pixels differ')
   prompt=clean(row['prompt']);model_prompt=GreedyInference._add_answer_instruction(prompt);wrapper=f'<image>\n\nQuestion: {model_prompt.strip()}\n\nAnswer:';question=hf.extract_question(wrapper);assert question==model_prompt.strip()
   ids=tokenizer.encode(question).ids;native_prefix=model.model.config.tokenizer.templates['query']['prefix'];reasoning_ids=[*native_prefix,*ids,model.model.config.tokenizer.thinking_id];assert native_prefix==[1,15381,2] and reasoning_ids[-1]==4
   if i in [0,99,199]:np.savez_compressed(H/f'necessary_input_example_{i:05d}.npz',original_RGB_crops_uint8=independent,actual_original_BF16_pixel_bits=a,independent_original_BF16_pixel_bits=reference,own_original_reasoning_prefill_token_ids=np.array(reasoning_ids,dtype=np.int64))
   record=dict(index=i,prompt=prompt,reference=row['reference'],media_path=str(image_path),image_SHA256=sha(image_path),original_greedy_model_prompt=model_prompt,original_HF_wrapper_text=wrapper,original_query_question=question,
    reasoning=True,own_original_reasoning_prefill_token_ids=reasoning_ids,original_answer_prefix=[3],pixel_shape=list(actual.shape),pixel_dtype='torch.bfloat16',pixel_SHA256=hashlib.sha256(a.tobytes()).hexdigest(),tiling=list(tiling),
    independent_full_pixel_bitwise_replay_equal=True,original_native_query_defaults=dict(max_tokens=768,temperature=.5,top_p=.9),saved_declared_512_ignored_by_original_query_path=True,not_an_A100_tensor_capture=True)
   out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush();records.append(record);put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_frontend200_full_pixel_and_token_control_no_NN',samples=len(records),target=200))
 assert before=={p:sha(p) for p in before}
 review=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,scope='Own original source/metadata and CPU frontend only; no learned CPU or NPU inference, no score or formal qualification',model_id=MID,samples=200,
  own_original_selected_full_shards=shards,own_source_and_original_HF_meta_count=len(headers),original_HF_meta_physical_dtype=meta_dtype,
  prior_v1_FP16_runtime_inference_explicitly_superseded=True,effective_loaded_NPU_dtype_not_yet_proven=True,original_flex_decoding_backend_not_yet_proven=True,
  all200_independent_full_BF16_pixel_bitwise_replays_equal=True,resize_backend='pyvips' if crop.HAS_VIPS else 'original_PIL_LANCZOS',
  frontend_SHA256=sha(H/'frontend200.jsonl'),dataset_SHA256=sha(DATA),own_tokenizer_SHA256=sha(W/'tokenizer.json'),
  original_reasoning_True_and_native_768_temperatureHalf_topP_point9_preserved=True,original_outer_batch10_preserved_in_saved_YAML=True,
  no_model_adapter_YAML_fullNF_true3_or200_claimed=True,no_G_or_canonical_write=True,no_model_source_or_precision_or_budget_or_stop_tuning=True,elapsed_seconds=time.monotonic()-start)
 put('review.json',review);put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,no_NN=True,frontend_only=True));print(json.dumps(review,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.is_dir() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_NN=True))
  raise
