"""Own OS-Atlas resources, real meta architecture, original MCScore CPU inputs.
No learned CPU forward, no original generic load, no grade or canonical write.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,struct,traceback,ast,inspect,re,zipfile
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
MID='OS-Copilot/OS-Atlas-Base-4B';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/InternVLChatModel')/MID
H=B/'OSAtlas4B_own_original_probability_frontend200_source_v2_eval_original_point1_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
def rec(t):
 import torch
 t=t.detach().contiguous();return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
def clean(s):
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
def main():
 import torch,numpy as np
 from PIL import Image
 from transformers import AutoTokenizer
 from universal_eval.model.osatlas4b_own_original_mcscore_20261116.configuration_internvl_chat import InternVLChatConfig
 from universal_eval.model.osatlas4b_own_original_mcscore_20261116.modeling_internvl_chat import InternVLChatModel
 from universal_eval.model.adapters.high_nine_osatlas_reference import OSAtlasMCScore
 from universal_eval.model.scoring import parse_options
 from safetensors import safe_open
 torch.set_num_threads(4);rawcfg=js(W/'config.json');cfg=InternVLChatConfig(**rawcfg)
 assert cfg.vision_config.drop_path_rate==0.1 and cfg.llm_config.num_hidden_layers==32 and cfg.llm_config.vocab_size==32022 and cfg.llm_config.hidden_size==3072
 assert cfg.llm_config.use_cache is False and cfg.template=='phi3-chat' and cfg.max_dynamic_patch==6
 with torch.device('meta'):model=InternVLChatModel(cfg,use_flash_attn=False)
 model.eval();assert not model.training and all(not m.training for m in model.modules())
 expected={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in model.named_parameters(remove_duplicate=False)}
 for layer in model.language_model.model.layers:layer.self_attn._init_rope()
 assert all(not t.is_meta for t in model.buffers())
 tokenizer=AutoTokenizer.from_pretrained(str(W),trust_remote_code=True,local_files_only=True)
 assert len(tokenizer)==32022
 wrapper=OSAtlasMCScore(device='cpu',max_new_tokens=2048,torch_dtype='auto')
 wrapper._model=model;wrapper._tokenizer=tokenizer
 image_context_id=tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
 assert 0<=image_context_id<32022 and model.num_image_token==256
 sources={};weights={}
 for sf in sorted(W.glob('*.safetensors')):
  weights[sf.name]=dict(bytes=sf.stat().st_size,SHA256=sha(sf))
  with sf.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];hdr=json.loads(f.read(n))
  with safe_open(str(sf),framework='pt',device='cpu') as f:
   for name,v in hdr.items():
    if name=='__metadata__':continue
    assert name not in sources and v['dtype']=='BF16' and name in expected and v['shape']==expected[name]['shape']
    # Inspect shape/dtype using slices only; no learned CPU model or matmul.
    assert list(f.get_slice(name).get_shape())==v['shape']
    sources[name]=dict(file=sf.name,**v)
 assert set(sources)==set(expected)
 idx=js(W/'model.safetensors.index.json') if (W/'model.safetensors.index.json').exists() else None
 if idx is not None:assert idx['weight_map']=={n:v['file'] for n,v in sources.items()}
 put('source_headers.json',sources);put('meta_preflight.json',dict(passed=True,source_tensor_count=len(sources),own_exact_parameter_bijection=True,all_parameters_meta=True,actual_meta_module=inspect.getfile(InternVLChatModel),source_AST_review=str(B/'OSAtlas4B_own_original_probability_source_layout_AST_review_20261116/review.json'),own_use_cache=False,own_attention_class=type(model.language_model.model.layers[0].self_attn).__name__,own_tokenizer_class=type(tokenizer).__name__,head_shape=expected['language_model.lm_head.weight']['shape'],own_original_drop_path_rate=cfg.vision_config.drop_path_rate,own_original_evaluation_mode=True,no_learned_CPU_forward=True))
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.model','.py','.jinja']}
 data=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';rows=[json.loads(s) for s in data.read_text().splitlines()];assert len(rows)==4329
 from tokenizers import Tokenizer
 rust=Tokenizer.from_file(str(W/'tokenizer.json'))
 records=[];output=H/'frontend200.jsonl'
 with output.open('x',encoding='utf-8') as out:
  for i,row in enumerate(rows[:200]):
   prompt=clean(row['prompt']);media=Path(row['media_path']);media=(data.parent/media).resolve() if not media.is_absolute() else media.resolve();assert media.is_relative_to(data.parent)
   image=Image.open(media).convert('RGB');pixels,tiles=wrapper._preprocess_internvl(image)
   ratios=sorted({(a,b) for n in range(1,7) for a in range(1,n+1) for b in range(1,n+1) if 1<=a*b<=6},key=lambda x:x[0]*x[1]);best=(1,1);diff=float('inf');aspect=image.width/image.height
   for a,b in ratios:
    d=abs(aspect-a/b)
    if d<diff or (d==diff and image.width*image.height>.5*448*448*a*b):best=(a,b);diff=d
   full=image.resize((448*best[0],448*best[1]));parts=[full.crop((j%best[0]*448,j//best[0]*448,(j%best[0]+1)*448,(j//best[0]+1)*448)) for j in range(best[0]*best[1])]
   if len(parts)>1:parts.append(image.resize((448,448)))
   raw=np.stack([np.asarray(p,dtype=np.uint8).transpose(2,0,1) for p in parts]);ind=torch.from_numpy(raw.copy()).float().div(255);ind=ind.sub(torch.tensor([.485,.456,.406])[None,:,None,None]).div(torch.tensor([.229,.224,.225])[None,:,None,None]);assert torch.equal(pixels,ind),(i,'original probability pixels differ')
   text_prompt=prompt.rstrip()+OSAtlasMCScore._SCORE_TAIL;query=wrapper._build_internvl_query(text_prompt,image_size=None)
   independent='<|system|>\nYou are an AI assistant whose name is Phi-3.<|end|><|user|>\n<image>\n'+text_prompt+'<|end|><|assistant|>\n'
   assert query==independent
   expanded=query.replace('<image>','<img>'+'<IMG_CONTEXT>'*(256*tiles)+'</img>',1)
   ids=tokenizer(expanded,return_tensors='pt')['input_ids'];rust_ids=rust.encode(expanded,add_special_tokens=True).ids
   assert ids[0].tolist()==rust_ids and int((ids==image_context_id).sum())==tiles*256
   options=parse_options(prompt) or ['A','B','C','D'];variants={}
   for option in options:
    cand=str(option).strip().upper();variants[cand]=[]
    for variant in [cand,' '+cand,'\n'+cand]:
     encoded=tokenizer.encode(variant,add_special_tokens=False);other=rust.encode(variant,add_special_tokens=False).ids;assert encoded==other and encoded
     variants[cand].append(dict(text=variant,input_ids=encoded,scored_id=encoded[0]))
   reci=dict(index=i,prompt=prompt,original_model_prompt=prompt,media_path=str(media),image_SHA256=sha(media),reference=row['reference'],query=query,expanded_text=expanded,input_ids=ids[0].tolist(),attention_mask=None,pixels_F32=rec(pixels),pixels_BF16=rec(pixels.bfloat16()),tiles=tiles,options=options,option_variants=variants,image_context_token_id=image_context_id,independent_pixels_equal=True,Rust_token_ids_equal=True,original_algorithm='mc_score',original_raw_output=None,actual_forward_count_expected=1,original_use_cache=False)
   out.write(json.dumps(reci,ensure_ascii=False)+'\n');out.flush();records.append(reci)
   put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed=i+1,target=200,no_NN=True))
 original={str(R/n):sha(R/n) for n in ['configs/InternVLChatModel/OS-Copilot--OS-Atlas-Base-4B.yaml','universal_eval/model/adapters/high_nine_osatlas_reference.py','universal_eval/model/adapters/high_nine_internvl_reference.py','universal_eval/model/adapters/transformers_vlm.py','universal_eval/algorithm/mc_score.py','universal_eval/model/scoring.py']}
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,own_assets_SHA256=assets,own_weight_files=weights,own_source_tensor_count=len(sources),own_head_size=32022,own_use_cache=False,own_original_algorithm='mc_score',own_tokenizer_class=type(tokenizer).__name__,original_source_SHA256=original,source_headers=str(H/'source_headers.json'),source_headers_SHA256=sha(H/'source_headers.json'),meta_preflight=str(H/'meta_preflight.json'),meta_preflight_SHA256=sha(H/'meta_preflight.json'),frontend=str(output),frontend_SHA256=sha(output),dataset=str(data),dataset_SHA256=sha(data),samples=200,all200_independent_pixels_equal=True,all200_Rust_input_and_candidate_token_ids_equal=True,max_prompt_length=max(len(r['input_ids']) for r in records),max_tile_count=max(r['tiles'] for r in records),no_generic_original_load_called=True,no_learned_CPU_forward=True,no_NPU_inference=True,no_A100_historical_runtime_or_tensor_capture_claimed=True)
 put('review.json',result);put('actual_completion.json',dict(actual_returncode=0,UTC=result['UTC']));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:put('actual_completion.json',dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc()));raise
