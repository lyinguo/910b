"""Own Unified12B input/META controls only. Never a learned CPU/NPU score."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,traceback,ast,re
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
R=Path('/root/projects/comperror');sys.path.insert(0,str(R));D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';OLD=B/'Gemma4Unified12Bit_own_original_source_assets_META_only_next_candidate_20261116';H=B/'Gemma4Unified12Bit_own_original_HF_false_thinking_frontend200_exact_names_20261116';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4UnifiedForConditionalGeneration/google/gemma-4-12B-it');DATA=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';MID='google/gemma-4-12B-it'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def clean(s):
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
def fp(t):return hashlib.sha256(t.contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
def main():
 import torch,yaml,inspect
 from PIL import Image
 from transformers import AutoProcessor
 from universal_eval.algorithm.inference import GreedyInference
 from universal_eval.model.reference_sources.gemma4_unified_owned_checkpoint_layout import owned_mapping,independent_merged_patches
 torch.set_num_threads(4);assert js(OLD/'actual_completion.json')['actual_returncode']==0
 own=js(OLD/'own_tensor_headers_and_shards.json');assert own['own_tensor_count']==677 and own['whole_weight_bytes']==23919549408 and own['weights'][str(W/'model.safetensors')]['SHA256']=='5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d'
 meta=js(OLD/'genuine_installed_META_contract.json');mapped,alias=owned_mapping(own['headers'],{n:v['shape'] for n,v in meta['meta'].items()});assert len(mapped)==677 and len(set(mapped.values()))==677 and {v['dtype'] for v in own['headers'].values()}=={'BF16'}
 put('exact_own_677_source_678_native_META_mapping.json',dict(passed=True,model_id=MID,native_to_own_source=mapped,own_tied_head_alias=alias,all_own_source_names_used_once=True,all_shapes_exact=True,no_tensor_values_from_other_checkpoint=True,no_learned_CPU_or_NPU=True))
 saved=R/'configs/Gemma4UnifiedForConditionalGeneration/google--gemma-4-12B-it.yaml';cfg=yaml.safe_load(saved.read_bytes());assert cfg['task_type']=='image-text-to-text' and cfg['algorithm']==dict(name='greedy_inference',batch_size=1,kwargs=dict(extract_answer=True)) and cfg['model']['kwargs']['enable_thinking'] is False and cfg['model']['kwargs']['max_new_tokens']==2048
 source=R/'universal_eval/model/adapters/gemma4_unified_vlm.py';text=source.read_text();tree=ast.parse(text);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Gemma4UnifiedVLLMModel');predict=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='predict');original_predict=ast.get_source_segment(text,predict);assert 'enable_thinking=self._enable_thinking' in original_predict and 'temperature=0.0' in original_predict
 p=AutoProcessor.from_pretrained(str(W),local_files_only=True);ip=p.image_processor;assert type(p).__name__=='Gemma4UnifiedProcessor' and ip.patch_size==16 and ip.pooling_kernel_size==3 and ip.max_soft_tokens==280 and not ip.do_normalize and ip.do_rescale
 source_files={str(inspect.getfile(type(m))):sha(inspect.getfile(type(m))) for m in [p,p.image_processor,p.tokenizer]};assets={str(f):sha(f) for f in W.iterdir() if f.is_file() and f.suffix in ['.json','.model','.jinja']};rows=[json.loads(l) for l in DATA.read_text().splitlines()];assert len(rows)==4329
 with (H/'frontend200.jsonl').open('x',encoding='utf-8') as out:
  for i,row in enumerate(rows[:200]):
   prompt=clean(row['prompt']);model_prompt=GreedyInference._add_answer_instruction(prompt);media=Path(row['media_path']);media=(media if media.is_absolute() else DATA.parent/media).resolve();assert media.is_relative_to(DATA.parent);image=Image.open(media).convert('RGB');messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':model_prompt}]}]
   inputs=p.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
   patches,pos,geometry=independent_merged_patches(image);assert torch.equal(inputs['pixel_values'],patches) and torch.equal(inputs['image_position_ids'],pos),(i,'Independent merged-patch/position differs')
   formatted=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);expanded=formatted.replace(p.image_token,p.boi_token+p.image_token*geometry['valid_merged_patches']+p.eoi_token);ids=p.tokenizer(expanded,add_special_tokens=False)['input_ids']
   assert inputs['input_ids'].tolist()==[ids] and inputs['attention_mask'].tolist()==[[1]*len(ids)] and inputs['mm_token_type_ids'].tolist()==[[int(t==258880) for t in ids]],(i,'Own independent Rust IDs/mask/media differs')
   assert sum(t==258880 for t in ids)==geometry['valid_merged_patches'] and len(ids)+2048<=262144
   tensors={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=fp(t)) for n,t in inputs.items()};rec=dict(index=i,prompt=prompt,original_model_prompt=model_prompt,reference=row['reference'],media_path=str(media),source_image_SHA256=sha(media),formatted_text=formatted,expanded_text=expanded,input_ids=ids,inputs=tensors,geometry=geometry,original_enable_thinking_False_restored_not_newly_added=True,independent_F32_merged_patch_and_position_replay_equal=True,independent_own_Rust_token_mask_media_replay_equal=True)
   out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush();put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_original_input200_no_NN',completed=i+1,target=200))
 assert all(sha(n)==h for n,h in assets.items())
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,actual_returncode=0,model_id=MID,own_source_count=677,own_META_names_and_tied_head_closed=True,samples=200,frontend_SHA256=sha(H/'frontend200.jsonl'),own_677_source_META_mapping_SHA256=sha(H/'exact_own_677_source_678_native_META_mapping.json'),own_original_saved_YAML=str(saved),own_original_saved_YAML_SHA256=sha(saved),own_original_predict_source=str(source),own_original_predict_source_SHA256=sha(source),original_adapter_initializer_not_executed=True,original_processor_source_SHA256=source_files,own_assets_SHA256=assets,dataset_SHA256=sha(DATA),all200_F32_pixels_positions_Rust_IDs_mask_media_equal=True,original_dtype_BF16_budget2048_serial1_and_saved_enable_thinking_False_preserved=True,no_learned_CPU_or_NPU_score_or_new_qualification=True,no_real3_or_approval_or_200_claimed=True)
 put('review.json',result);put('actual_completion.json',dict(actual_returncode=0,UTC=result['UTC'],review_SHA256=sha(H/'review.json'),no_NN=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  put('actual_completion.json',dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc(),no_learned_CPU_or_NPU=True));raise
