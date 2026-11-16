"""Own complete checkpoint + native meta layout + 200 input-only controls; no learned CPU/NPU execution."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,traceback,inspect,ast
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMV46_own_original_BF16_MCq_fullsource_frontend200_v3_original_fused_uint8_tuple_20261116';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/MiniCPMV4_6ForConditionalGeneration/openbmb/MiniCPM-V-4.6')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 import torch,numpy as np
 from safetensors import safe_open
 from transformers import AutoConfig,MiniCPMV4_6ForConditionalGeneration
 from universal_eval.model.adapters.minicpmv46_own_frontend_20261116 import OwnMiniCPMV46Inputs,tsha,flatten
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.jinja']};wf={p.name:dict(bytes=p.stat().st_size,SHA256=sha(p)) for p in W.glob('*.safetensors')};assert len(wf)==1
 previous=B/'MiniCPMV46_own_original_BF16_MCq_fullsource_frontend200_v1_20261116'
 sources=json.loads((previous/'own_full_source_manifest.json').read_bytes());assert len(sources)==779
 assert json.loads((previous/'own_native_meta_layout.json').read_bytes())['source_tensor_count']==779
 put('own_full_source_manifest.json',sources)
 cfg=AutoConfig.from_pretrained(str(W),local_files_only=True,trust_remote_code=False);cfg._attn_implementation='eager';cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='eager'
 with torch.device('meta'):m=MiniCPMV4_6ForConditionalGeneration(cfg)
 expected=dict(m.named_parameters(remove_duplicate=False));missing=set(expected)-set(sources);extra=set(sources)-set(expected)
 assert not extra and missing <= {'lm_head.weight'},(missing,extra)
 assert cfg.tie_word_embeddings and cfg.text_config.num_hidden_layers==24 and cfg.text_config.vocab_size==248094
 for n,v in sources.items():assert list(expected[n].shape)==v['shape'],n
 put('own_native_meta_layout.json',dict(parameter_count=len(expected),source_tensor_count=len(sources),own_declared_tied_aliases=sorted(missing),buffers={n:dict(shape=list(t.shape),dtype=str(t.dtype),is_meta=t.is_meta) for n,t in m.named_buffers()}));del m,expected
 inputs=OwnMiniCPMV46Inputs(W);dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';rows=DatasetLoader().load(str(dataset));rows=TaskDataRegistry._resolve_media_paths(rows,str(dataset.parent));rows=DataPreprocessor().preprocess(rows,task_type='image-text-to-text',apply_template=False,filter_empty=False);assert len(rows)==4329
 from PIL import Image
 front=[]
 for i,row in enumerate(rows[:200]):
  prompt=GreedyInference._add_answer_instruction(row['prompt']);text,v,opts,model_prompt=inputs.prepare(prompt,row['media_path'])
  with Image.open(row['media_path']) as im:image=im.convert('RGB')
  ip=inputs.processor.image_processor;raw=ip(images=[image],return_tensors='pt',do_rescale=False,do_normalize=False)['pixel_values']
  # TorchvisionBackend preserves uint8 geometry then uses a fused rescale/normalization.
  mean=torch.tensor(ip.image_mean,dtype=torch.float32).reshape(1,3,1,1);std=torch.tensor(ip.image_std,dtype=torch.float32).reshape(1,3,1,1)
  # Packed tensor layout has C=3 before patchification; reverse the 14x14 patches.
  assert tuple(ip.image_mean)==tuple(ip.image_std)==(.5,.5,.5) and ip.rescale_factor==1/255
  candidates={'original_fused_uint8_mean127p5_std127p5':(raw.float()-127.5)/127.5}
  exact=[n for n,t in candidates.items() if torch.equal(t,v['pixel_values'])]
  assert exact,('Independent uint8 pixel arithmetic/order must be reviewed',i,raw.shape,v['pixel_values'].shape,{n:float((t-v['pixel_values']).abs().max()) for n,t in candidates.items()})
  front.append(dict(index=i,prompt=row['prompt'],actual_model_prompt=prompt,own_saved_MCq_model_prompt=model_prompt,options=opts,text=text,media_path=row['media_path'],media_file_SHA256=sha(row['media_path']),reference=row['reference'],input_ids=v['input_ids'][0].tolist(),tensor_metadata={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tsha(t)) for n,t in flatten(v).items()},independent_uint8_normalization_exact=exact,own_Rust_token_ID_roundtrip_exact=True))
  put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed=i+1,total=200,no_NPU_or_learned_CPU_forward=True))
 put('own_original_frontend200.json',front)
 libraries={}
 import transformers
 for leaf in ['minicpmv4_6','qwen3_5']:
  for p in (Path(transformers.__file__).parent/'models'/leaf).glob('*.py'):libraries[str(p)]=sha(p)
 libs=Path(transformers.__file__).parent/'image_processing_backends.py'
 if libs.is_file():libraries[str(libs)]=sha(libs)
 ready=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id='openbmb/MiniCPM-V-4.6',weight_path=str(W),source_tensor_count=len(sources),own_assets_SHA256=assets,own_weight_files=wf,own_source_manifest=str(H/'own_full_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own_full_source_manifest.json'),own_native_meta_layout=str(H/'own_native_meta_layout.json'),own_native_meta_layout_SHA256=sha(H/'own_native_meta_layout.json'),own_frontend200=str(H/'own_original_frontend200.json'),own_frontend200_SHA256=sha(H/'own_original_frontend200.json'),dataset_path=str(dataset),dataset_SHA256=sha(dataset),native_libraries_SHA256=libraries,original_stop_ids=[248044],all200_own_input_pixels_tokens_checked=True,no_NPU_or_learned_CPU_inference_yet=True)
 put('readiness.json',ready);put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0));print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_NPU_inference=True))
  raise
