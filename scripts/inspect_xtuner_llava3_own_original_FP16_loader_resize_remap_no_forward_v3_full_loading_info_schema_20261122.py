"""Actual own CPU source construction only; preserve original resize/remap, no learned forward."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import collections,contextlib,hashlib,inspect,json,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='xtuner/llava-llama-3-8b-v1_1-hf';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaLlamaForCausalLM')/MID
H=B/'XtunerLlava3_own_original_FP16_actual_CPU_loader_resize_remap_no_forward_v3_full_loading_info_schema_20261122';FRONT=B/'XtunerLlava3_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_original_tokenizer_fallback_20261122'
sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
now=lambda:datetime.now(timezone.utc).isoformat()
def jsonable(value):
 if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
 if isinstance(value,set):return [jsonable(v) for v in sorted(value,key=repr)]
 if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
 if value is None or isinstance(value,(str,int,float,bool)):return value
 raise TypeError(('Unsupported actual loading metadata type; preserve failure',type(value).__name__))
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(jsonable(v),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 import torch,numpy as np
 from safetensors import safe_open
 from transformers import LlamaConfig,CLIPVisionConfig,LlavaConfig,LlavaForConditionalGeneration,GenerationConfig
 from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz
 from scripts.egm8b_original_independent_cpu_910b import fp
 torch.set_num_threads(4);H.mkdir(exist_ok=False)
 ready=js(FRONT/'readiness.json');assert ready['passed'] and ready['no_learned_CPU_or_NPU_forward'] and ready['own_frontend200_SHA256']==sha(FRONT/'own_original_frontend200.json')
 assert sha(R/'universal_eval/model/adapters/transformers_vlm.py')=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3'
 assert ready['own_tokenizer_image_ID']==128256 and ready['own_tokenizer_length']==128257 and ready['own_checkpoint_embedding_and_head_rows']==128256
 assert all((W/n).stat().st_size==r['bytes'] for n,r in ready['own_weight_files'].items())
 raw=js(W/'config.json');fields=['vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','rms_norm_eps','rope_theta','max_position_embeddings','tie_word_embeddings','attention_bias','hidden_act','initializer_range','pretraining_tp']
 tc=LlamaConfig(**{k:raw[k] for k in fields if k in raw});vc=CLIPVisionConfig(hidden_size=1024,num_hidden_layers=24,num_attention_heads=16,intermediate_size=4096,patch_size=14,image_size=336,num_channels=3,projection_dim=768,layer_norm_eps=1e-5,hidden_act='quick_gelu')
 cfg=LlavaConfig(text_config=tc,vision_config=vc,image_token_index=32000,vision_feature_layer=-2,vision_feature_select_strategy='default',image_aspect_ratio='pad',mm_hidden_size=1024)
 original_resize=LlavaForConditionalGeneration.resize_token_embeddings;resize_source=inspect.getsource(original_resize);(H/'original_resize_token_embeddings_source.py.txt').write_text(resize_source,encoding='utf-8')
 put('progress.json',dict(UTC=now(),phase='own_original_F16_from_pretrained_CPU_source_construction_no_forward'))
 rng_initial=torch.get_rng_state().clone()
 model,info=LlavaForConditionalGeneration.from_pretrained(W,config=cfg,torch_dtype=torch.float16,low_cpu_mem_usage=True,local_files_only=True,output_loading_info=True)
 def no_forward(*args,**kwargs):raise AssertionError('Learned model forward prohibited in source construction diagnostic')
 handle=model.register_forward_pre_hook(no_forward)
 put('original_from_pretrained_loading_info_collection_types.json',{k:type(v).__name__ for k,v in info.items()});put('original_from_pretrained_loading_info.json',info)
 assert model.get_input_embeddings().weight.shape==(128256,4096) and model.lm_head.weight.shape==(128256,4096)
 assert len(dict(model.named_parameters()))==686
 mapping=js(FRONT/'own_original_legacy_mapping_meta_review.json')['source_to_native_mapping'];params=dict(model.named_parameters())
 pre_head=fp(model.lm_head.weight);pre_embedding=fp(model.get_input_embeddings().weight)
 rng_before_resize=torch.get_rng_state().clone();put('progress.json',dict(UTC=now(),phase='original_saved_resize_token_embeddings_128257_CPU_no_forward'))
 # The saved adapter performs this ORIGINAL default resize BEFORE its legacy source remap.
 model.config.image_token_id=model.config.image_token_index=128256
 model.resize_token_embeddings(128257)
 rng_after_resize=torch.get_rng_state().clone();assert model.get_input_embeddings().weight.shape==model.lm_head.weight.shape==(128257,4096)
 added_embedding=model.get_input_embeddings().weight[128256:].detach().clone();added_head=model.lm_head.weight[128256:].detach().clone();params=dict(model.named_parameters());loaded=[];dtype_counts=collections.Counter()
 for n,r in ready['own_weight_files'].items():
  with safe_open(str(W/n),framework='pt',device='cpu') as sf:
   for key in sf.keys():
    source=sf.get_tensor(key);name=mapping[key];target=params[name];assert torch.isfinite(source).all() and target.dtype==torch.float16
    cast=source.to(target.dtype)
    if target.shape==cast.shape:target.data.copy_(cast)
    else:assert key in ['model.embed_tokens.weight','lm_head.weight'] and target.shape==(128257,4096) and cast.shape==(128256,4096);target.data[:128256].copy_(cast)
    actual=target[:128256] if key in ['model.embed_tokens.weight','lm_head.weight'] else target
    assert fp(actual)==fp(cast)
    loaded.append(dict(source_key=key,target_key=name,source_shape=list(source.shape),source_dtype=str(source.dtype),source_SHA256=fp(source),original_cast_dtype=str(cast.dtype),original_cast_SHA256=fp(cast),actual_CPU_original_source_rows_SHA256=fp(actual),not_NPU_loading_proof=True));dtype_counts[str(source.dtype)]+=1
  put('progress.json',dict(UTC=now(),phase='original_own686_source_remap_CPU_no_forward',completed=len(loaded),total=686))
 assert len(loaded)==686 and dtype_counts=={'torch.float16':682,'torch.float32':4}
 assert torch.equal(added_embedding,model.get_input_embeddings().weight[128256:]) and torch.equal(added_head,model.lm_head.weight[128256:])
 assert all(t.device.type=='cpu' and t.dtype==torch.float16 and torch.isfinite(t).all() for t in model.parameters())
 path=H/'original_added_embedding_head_rows_and_RNG_full_values.npz';write_full_standard_npz(path,dict(added_embedding=added_embedding.numpy(),added_lm_head=added_head.numpy(),RNG_before_from_pretrained=rng_initial.numpy(),RNG_before_original_resize=rng_before_resize.numpy(),RNG_after_original_resize=rng_after_resize.numpy()))
 put('own_all686_actual_CPU_source_cast_remap_SHA256.json',loaded)
 put('original_actual_CPU_buffer_layout.json',{n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=fp(t)) for n,t in model.named_buffers()})
 generation=GenerationConfig.from_pretrained(W,local_files_only=True);assert generation.eos_token_id==128001
 model.generation_config=generation
 gen,unused=model._prepare_generation_config(None,max_new_tokens=2048,do_sample=False,temperature=0.)
 value=dict(passed=True,UTC=now(),model_id=MID,own_frontend200_SHA256=ready['own_frontend200_SHA256'],own_full686_CPU_source_cast_rows_SHA256=str(H/'own_all686_actual_CPU_source_cast_remap_SHA256.json'),own_full686_CPU_source_cast_rows_SHA256_file_SHA=sha(H/'own_all686_actual_CPU_source_cast_remap_SHA256.json'),native_head_and_embedding_rows_after_original_resize=128257,original_checkpoint_head_rows_preserved=128256,own_actual_source_dtype_counts=dict(dtype_counts),actual_CPU_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),actual_NPU_loaded_dtype_not_proved=True,original_extra_head_and_image_embedding_row_values=str(path),original_extra_head_and_image_embedding_row_values_SHA256=sha(path),original_added_embedding_SHA256=fp(added_embedding),original_added_lm_head_SHA256=fp(added_head),original_before_resize_embedding_SHA256=pre_embedding,original_before_resize_lm_head_SHA256=pre_head,original_resize_method_SHA256=hashlib.sha256(resize_source.encode()).hexdigest(),original_resize_before_manual_remap_order_preserved=True,original_own_first128256_source_rows_bitwise_exact_after_remap=True,source_constructor_derived_extra_rows_not_foreign_checkpoint_or_score_tuning=True,original_generation_config=gen.to_dict(),original_only_EOS128001_not_borrowed_Mini_EOS128009=True,own_graph_buffer_layout=js(H/'original_actual_CPU_buffer_layout.json'),no_learned_CPU_or_NPU_forward=True,no_A100_or_score_claim=True,ready_for_NPU=False,next_required='Independent provenance/recompute of original derived row and native NPU full graph/cache/observations/full verifier/standalone YAML, then genuine NEW3 only')
 put('actual_original_loader_resize_remap_review.json',value);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_learned_CPU_or_NPU_forward=True));handle.remove();return value
if __name__=='__main__':
 try:
  with contextlib.redirect_stdout(sys.stderr):v=main()
  print(json.dumps(v,ensure_ascii=False))
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_learned_CPU_or_NPU_forward=True,original_source_data_reports_logs_unchanged=True))
  raise
