"""Own GLM-OCR resources and all original fixed200 inputs; no learned inference."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,traceback,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'GLMOCRZai_own_original_BF16_2048_frontend200_meta_v2_explicit_original_unused_MTP_no_NPU_20261116';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/GlmOcrForConditionalGeneration/zai-org/GLM-OCR');sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def tensor_sha(t):
 import torch
 return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 import torch,numpy as np
 from safetensors import safe_open
 from PIL import Image
 from tokenizers import Tokenizer
 from transformers import AutoProcessor,GlmOcrConfig,GlmOcrForConditionalGeneration
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 cfg=GlmOcrConfig(**json.loads((W/'config.json').read_bytes()));cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
 with torch.device('meta'):model=GlmOcrForConditionalGeneration(cfg)
 params=dict(model.named_parameters(remove_duplicate=False));expected={};weights={};unused={}
 for p in sorted(W.glob('*.safetensors')):
  weights[p.name]=dict(bytes=p.stat().st_size,SHA256=sha(p))
  with safe_open(str(p),framework='pt',device='cpu') as f:
   for k in f.keys():
    t=f.get_tensor(k);assert t.dtype==torch.bfloat16,(k,t.dtype)
    if k not in params:
     assert k.startswith('model.language_model.layers.16.'),('Unreviewed unused source',k)
     unused[k]=dict(shape=list(t.shape),source_dtype=str(t.dtype),source_file=p.name,SHA256=tensor_sha(t),not_executed_by_original_installed_HF_16_layer_greedy=True);continue
    assert list(t.shape)==list(params[k].shape),(k,t.shape,params[k].shape)
    expected[k]=dict(source_key=k,source_file=p.name,runtime_name=k,shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=tensor_sha(t),cast_SHA256=tensor_sha(t))
 assert set(expected)==set(params) and len(expected)+len(unused)==526 and cfg.text_config.num_hidden_layers==16 and cfg.text_config.num_nextn_predict_layers==1,(len(expected),len(params),set(params)-set(expected))
 put('own_source_manifest.json',expected);put('own_unused_MTP_original_source_manifest.json',unused)
 processor=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False);assert type(processor).__name__=='Glm46VProcessor' and type(processor.image_processor).__name__=='Glm46VImageProcessor'
 rust=Tokenizer.from_file(str(W/'tokenizer.json'));assert rust.get_vocab()==processor.tokenizer.get_vocab();assert cfg.text_config.num_hidden_layers==16 and cfg.text_config.vocab_size==59392 and cfg.vision_config.depth==24
 ds=R/'workspace/datasets/image-to-text-ocr/eval.jsonl';rows=DatasetLoader().load(str(ds));rows=TaskDataRegistry._resolve_media_paths(rows,str(ds.parent));rows=DataPreprocessor().preprocess(rows,task_type='image-text-to-text',apply_template=False,filter_empty=False);assert len(rows)>=200
 front=[];ip=processor.image_processor
 for i,row in enumerate(rows[:200]):
  prompt=GreedyInference._add_answer_instruction(row['prompt']);messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])]
  # This kwarg is the saved generic adapter's GlmOcr branch, not a new choice.
  text=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
  with Image.open(row['media_path']) as im:image=im.convert('RGB')
  v=processor(text=text,images=[image],return_tensors='pt',padding=True)
  raw=ip(images=[image],do_rescale=False,do_normalize=False,return_tensors='pt')
  assert torch.equal(raw['image_grid_thw'],v['image_grid_thw'])
  a=raw['pixel_values'].numpy().astype(np.float32).reshape(-1,3,2,14,14)
  mean=np.asarray([x*255 for x in ip.image_mean],np.float32).reshape(1,3,1,1,1);std=np.asarray([x*255 for x in ip.image_std],np.float32).reshape(1,3,1,1,1)
  want=((a-mean)/std).reshape(v['pixel_values'].shape);assert np.array_equal(want,v['pixel_values'].numpy()),('Own fused normalization must be exact',i,float(np.abs(want-v['pixel_values'].numpy()).max()))
  _,replacements=processor._process_images([image],return_tensors='pt');expanded,_=processor.get_text_with_replacements([text],replacements,[],[])
  ids=rust.encode(expanded[0],add_special_tokens=True).ids;assert ids==v['input_ids'][0].tolist()
  front.append(dict(index=i,prompt=row['prompt'],actual_model_prompt=prompt,media_path=row['media_path'],media_file_SHA256=sha(row['media_path']),reference=row['reference'],query=text,input_ids=ids,tensor_metadata={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t)) for n,t in v.items() if torch.is_tensor(t)},independent_fused_uint8_F32_normalization_exact=True,own_Rust_ID_exact=True))
  put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed_samples=i+1,total=200,no_NPU_or_learned_CPU=True))
 put('own_original_frontend200.json',front)
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.jinja','.txt','.py'] and p.name not in weights}
 libs={inspect.getfile(type(x)):sha(inspect.getfile(type(x))) for x in [processor,ip,model]}
 ready=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id='zai-org/GLM-OCR',weight_path=str(W),source_tensor_count=len(expected),original_physical_source_tensor_count=526,original_unused_MTP_source_tensor_count=len(unused),own_unused_MTP_source_manifest=str(H/'own_unused_MTP_original_source_manifest.json'),own_unused_MTP_source_manifest_SHA256=sha(H/'own_unused_MTP_original_source_manifest.json'),unused_MTP_not_misclaimed_as_executed=True,own_weight_files=weights,own_assets_SHA256=assets,own_source_manifest=str(H/'own_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own_source_manifest.json'),own_frontend200=str(H/'own_original_frontend200.json'),own_frontend200_SHA256=sha(H/'own_original_frontend200.json'),dataset_path=str(ds),dataset_SHA256=sha(ds),all200_own_original_tokens_pixels_boundaries_exact=True,original_stop_ids=[59246,59253],original_BF16_2048_outer10_inner1_CER_protocol_preserved=True,original_adapter=str(R/'universal_eval/model/adapters/transformers_vlm.py'),original_adapter_SHA256=sha(R/'universal_eval/model/adapters/transformers_vlm.py'),installed_architecture_sources_SHA256=libs,no_NPU_or_learned_CPU_inference=True)
 put('readiness.json',ready);put('actual_completion.json',dict(UTC=ready['UTC'],actual_returncode=0,no_NPU_inference=True));print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_NPU_inference=True))
  raise
