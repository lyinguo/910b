"""Prepare own current MiniCPM-V-4 sources and inputs; never execute learned CPU math."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,struct,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/MiniCPMV/openbmb/MiniCPM-V-4')
H=B/'MiniCPMV4_own_native_frontend200_source_meta_preflight_user_continue_20261115'
PKG=R/'universal_eval/model/minicpmv4_own_native_20261115'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False)
def main():
 H.mkdir(exist_ok=False);PKG.mkdir(exist_ok=False);(PKG/'__init__.py').write_text('"""Own original MiniCPM-V-4 source namespace; no downloaded code."""\n')
 records=[]
 for p in sorted(W.glob('*.py')):
  raw=p.read_bytes();dest=PKG/p.name;dest.write_bytes(raw);assert sha(dest)==sha(p)
  records.append(dict(source=str(p),copy=str(dest),SHA256=sha(dest),all_original_function_AST_unchanged=True))
 put(H/'own_source_copy_review.json',records)
 sys.path[:0]=[str(R),str(R/'scripts')]
 import torch,numpy as np
 from safetensors import safe_open
 from universal_eval.model.minicpmv4_own_native_20261115.configuration_minicpm import MiniCPMVConfig
 from universal_eval.model.minicpmv4_own_native_20261115.modeling_minicpmv import MiniCPMV
 from universal_eval.model.adapters.minicpmv4_own_frontend_v1 import OwnMiniCPMV4Inputs,tensor_sha,flatten_tensors
 torch.set_num_threads(4)
 cfg=MiniCPMVConfig(**json.loads((W/'config.json').read_bytes()));cfg._attn_implementation='eager';cfg.vision_config._attn_implementation='eager'
 with torch.device('meta'):model=MiniCPMV(cfg)
 params=dict(model.named_parameters(remove_duplicate=False));index=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map'];assert set(index)==set(params),(set(index)-set(params),set(params)-set(index))
 weights={};files={}
 for name in sorted(set(index.values())):
  p=W/name;files[name]={'bytes':p.stat().st_size,'SHA256':sha(p)}
  with safe_open(str(p),framework='pt',device='cpu') as f:
   assert set(f.keys())=={k for k,v in index.items() if v==name}
   for key in f.keys():
    t=f.get_tensor(key);assert t.dtype==torch.bfloat16 and t.shape==params[key].shape
    weights[key]=dict(source_file=name,runtime_name=key,shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=tensor_sha(t),cast_SHA256=tensor_sha(t))
 put(H/'own_source_tensor_manifest.json',weights);put(H/'own_weight_file_SHA256.json',files)
 inputs=OwnMiniCPMV4Inputs(W);dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';rows=[json.loads(s) for s in dataset.read_text().splitlines()];assert len(rows)==4329
 front=[]
 for i,r in enumerate(rows[:200]):
  path=Path(r['media_path']);path=path if path.is_absolute() else dataset.parent/path;path=str(path.resolve());ref=r['reference']
  # Match the original stored Greedy instruction, without adding a new instruction.
  from universal_eval.algorithm.inference import GreedyInference
  prompt=GreedyInference._add_answer_instruction(r['prompt'])
  rendered,v=inputs.prepare(prompt,path)
  tensors=flatten_tensors(v)
  pixels=v['pixel_values'][0]
  assert all(x.dtype==torch.float32 for x in pixels)
  # Independent original normalization: channel values must follow uint8 -> /255 -> (x-.5)/.5.
  for px in pixels:
   assert torch.isfinite(px).all() and px.min()>=-1 and px.max()<=1
  front.append(dict(index=i,source_prompt=r['prompt'],actual_model_prompt=prompt,media_path=path,media_file_SHA256=sha(path),reference=ref,text=rendered,tensor_metadata={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t)) for n,t in tensors.items()},image_sizes=v['image_sizes'],image_bound=[t.tolist() for t in v['image_bound']],input_ids=v['input_ids'][0].tolist(),patches=len(pixels)))
 put(H/'own_frontend200.json',front)
 out=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id='openbmb/MiniCPM-V-4',own_source_tensor_count=len(weights),own_weight_files=files,own_frontend200=str(H/'own_frontend200.json'),own_frontend200_SHA256=sha(H/'own_frontend200.json'),max_input_length=max(len(x['input_ids']) for x in front),max_patches=max(x['patches'] for x in front),own_meta_constructed_without_learned_CPU_or_NPU_forward=True,original_functions_unchanged=True,still_requires_exact_Greedy_preparation_review=True)
 put(H/'readiness.json',out);print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
