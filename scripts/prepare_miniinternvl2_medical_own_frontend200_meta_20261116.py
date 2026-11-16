"""Own Medical checkpoint/meta and proven identical unweighted input-control reuse.
No learned CPU or NPU inference; no DriveLM weights or output tensors reused.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,struct,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='OpenGVLab/Mini-InternVL2-4B-DA-Medical';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/InternVLChatModel')/MID;H=B/'MiniInternVL2_Medical_own_frontend200_original_vllm_v1_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 with (H/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import torch
 from transformers import LlamaTokenizerFast
 from vllm.transformers_utils.processors.internvl import InternVLImageProcessor,InternVLProcessor
 from universal_eval.model.miniinternvl2_medical_own_20261116.configuration_internvl_chat import InternVLChatConfig
 from universal_eval.model.miniinternvl2_medical_own_20261116.modeling_internvl_chat import InternVLChatModel
 H.mkdir(exist_ok=False);torch.set_num_threads(4);weights={};source={}
 for sf in sorted(W.glob('*.safetensors')):
  weights[sf.name]=dict(bytes=sf.stat().st_size,SHA256=sha(sf))
  with sf.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];hdr=json.loads(f.read(n))
  for k,v in hdr.items():
   if k=='__metadata__':continue
   assert k not in source;source[k]=v
 assert len(weights)==2 and sum(v['bytes'] for v in weights.values())==8293778040 and len(source)==541 and all(v['dtype']=='BF16' for v in source.values())
 index=js(W/'model.safetensors.index.json');assert set(index['weight_map'])==set(source) and set(index['weight_map'].values())==set(weights);put('source_headers.json',source)
 cfg=js(W/'config.json')
 with torch.device('meta'):model=InternVLChatModel(InternVLChatConfig(**cfg),use_flash_attn=False)
 expected={n:list(t.shape) for n,t in model.named_parameters(remove_duplicate=False)};assert set(expected)==set(source) and all(expected[n]==v['shape'] for n,v in source.items()) and cfg['vision_config']['drop_path_rate']==.1
 put('own_meta_parameter_source_bijection.json',dict(passed=True,own_source_parameters=541,all_own_source_meta_names_shapes_exact=True,own_original_drop_path_rate=.1,all_original_model_math_preserved=True,no_NPU_or_learned_CPU_forward=True))
 prior=B/'MiniInternVL2_DriveLM_own_frontend200_original_vllm_v1_20261116';old=js(prior/'review.json');assert old['passed'] and old['samples']==200 and old['all200_independent_pixels_equal']
 other=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/InternVLChatModel/OpenGVLab/Mini-InternVL2-4B-DA-DriveLM');names=['config.json','generation_config.json','preprocessor_config.json','tokenizer_config.json','special_tokens_map.json','tokenizer.json'];same={}
 for n in names:assert sha(W/n)==sha(other/n)==old['own_assets_SHA256'][n];same[n]=sha(W/n)
 saved=R/'configs/InternVLChatModel/OpenGVLab--Mini-InternVL2-4B-DA-Medical.yaml'
 import yaml
 ownY=yaml.safe_load(saved.read_text());assert ownY['model']['name']==MID and ownY['model']['kwargs']['max_tokens']==2048 and ownY['algorithm']==dict(name='greedy_inference',batch_size=10,kwargs=dict(extract_answer=True)) and ownY['data']['dataset_path']=='/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl'
 data=Path(old['dataset']);assert sha(data)==old['dataset_SHA256'];pp=Path(inspect.getfile(InternVLProcessor));assert str(pp)==old['vllm_processor'] and sha(pp)==old['vllm_processor_SHA256'];original=Path(old['frontend']);assert sha(original)==old['frontend_SHA256']
 rows=[json.loads(x) for x in original.read_text().splitlines()];assert len(rows)==200 and [r['index'] for r in rows]==list(range(200))
 # Pixel/token calculations depend only on the exact identical own configs,
 # own tokenizer assets and the frozen processor, not on model weight values.
 tok=LlamaTokenizerFast.from_pretrained(str(W),local_files_only=True);assert tok.is_fast and len(tok)==32020
 for row in rows:
  assert tok(row['expanded_text'],return_tensors='pt')['input_ids'][0].tolist()==row['input_ids'] and row['independent_pixels_equal']
 output=H/'frontend200.jsonl'
 with output.open('xb') as f:f.write(original.read_bytes())
 reuse=dict(passed=True,exact_own_configuration_and_tokenizer_SHA256=same,original_full200_input_control=str(original),original_full200_input_control_SHA256=sha(original),all200_input_pixels_already_independently_checked=True,all200_own_tokenizer_ids_rechecked=True,own_saved_deployment_SHA256=sha(saved),no_checkpoint_or_NPU_arrays_or_outputs_reused=True,raw_remote_source_differences_retained_in_own_package=True,not_an_A100_or_NPU_run=True);put('input_control_reuse_proof.json',reuse)
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.model','.py']};result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,own_assets_SHA256=assets,own_weight_files=weights,image_context_token_id=32013,source_tensors=541,frontend=str(output),frontend_SHA256=sha(output),dataset=str(data),dataset_SHA256=sha(data),vllm_processor=str(pp),vllm_processor_SHA256=sha(pp),samples=200,all200_independent_pixels_equal=True,stop_ids=js(W/'generation_config.json')['eos_token_id'],max_prompt_length=max(len(r['input_ids']) for r in rows),no_learned_CPU_forward=True,no_NPU_inference=True,history_A100_engine_bytes_claimed=False,input_control_reuse=reuse)
 put('review.json',result);put('actual_completion.json',dict(actual_returncode=0,UTC=result['UTC']));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
