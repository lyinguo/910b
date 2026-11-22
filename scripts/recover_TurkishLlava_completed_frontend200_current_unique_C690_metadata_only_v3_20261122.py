"""Bind completed own input proof to actual unique C row; never replay frontend200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import hashlib,json,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='ytu-ce-cosmos/Turkish-LLaVA-v0.1';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaLlamaForCausalLM')/MID;F=B/'TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_exact_own_template_20261122';H=B/'TurkishLlava_completed_own_frontend200_current_unique_C690_metadata_only_v3_20261122'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 from batch_910b_support import workbook_rows
 from tokenizers import Tokenizer
 H.mkdir(exist_ok=False);old=js(F/'actual_completion.json');assert old['actual_returncode']==1 and old['no_learned_CPU_or_NPU_forward'] and 'row==dict(row=689' in old['error'];assert js(F/'progress.json')['completed']==200 and sha(R/'scripts/prepare_TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_exact_own_template_20261122.py')=='1d3d39f619aff6e837fe2755a9414d8d5a3edaac5a049062098178ac4ab14e22'
 front=js(F/'own_original_frontend200.json');assert len(front)==200 and [r['index'] for r in front]==list(range(200)) and sha(F/'own_original_frontend200.json')=='4fe9c7ee3d26954dcb56d64f8bd78ba3445756c56754a8c96c554668d5c79f54' and all(r['independent_own_Rust_tokens_and_TV2_uint8_numpy_F32_all_values_exact'] and r['image_token_id']==128257 and r['original_full576_image_expansion'] for r in front)
 source=js(F/'own_weight_files_and_headers.json');meta=js(F/'own_original_legacy_mapping_meta_review.json');assert source['source_tensor_count']==686 and source['source_dtype_counts']=={'BF16':686} and meta['exact_all686_original_mapping_names_shapes']
 for n,r in source['own_weight_files'].items():assert (W/n).stat().st_size==r['bytes'] and sha(W/n)==r['SHA256']
 saved=R/'configs/LlavaLlamaForCausalLM/ytu-ce-cosmos--Turkish-LLaVA-v0.1.yaml';adapter=R/'universal_eval/model/adapters/transformers_vlm.py';assert sha(saved)=='2a5e440def37d2d882d587d8077de0e2ea280de46d708213b966944ba30e88b7' and sha(adapter)=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3'
 book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878';rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row==dict(row=690,model_id=MID,task='image-text-to-text',score=None)
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:server=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 current=Path(binding['current_user_workbook_D_reference_snapshot']);original=Path(binding['local_A100_reference_snapshot']);assert sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256'] and sha(original)==binding['local_A100_reference_snapshot_sha256'];three=dict(server=server,local=js(current)['models'][MID]['A100_D'],preserved_original=js(original)['models'][MID]['A100_D']);assert all(Decimal(str(v))==Decimal('.705') for v in three.values())
 tk=Tokenizer.from_file(str(W/'tokenizer.json'));tk.add_special_tokens(['<image>']);assert tk.token_to_id('<image>')==128257 and tk.get_vocab_size()==128258;assert sha(W/'tokenizer_config.json')=='bbbb28058cdc9ab5bf89b2c9436fb999e8c46fab543b8d5827475e9eff3dc5c0';assert 'Today Date:' not in js(W/'tokenizer_config.json')['chat_template'];assert js(W/'generation_config.json')['eos_token_id']==128009
 for name in ['own_original_frontend200.json','own_weight_files_and_headers.json','own_original_legacy_mapping_meta_review.json']:(H/name).write_bytes((F/name).read_bytes())
 examples={str(p):sha(p) for p in F.glob('input_example_*.npz')};assert len(examples)==3;assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix=='.json'}
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,own_weight_path=str(W),own_weight_files=source['own_weight_files'],source_tensor_count=686,source_dtype_counts={'BF16':686},own_assets_SHA256=assets,own_frontend200=str(H/'own_original_frontend200.json'),own_frontend200_SHA256=sha(H/'own_original_frontend200.json'),own_original_legacy_mapping_meta_review=str(H/'own_original_legacy_mapping_meta_review.json'),own_original_legacy_mapping_meta_review_SHA256=sha(H/'own_original_legacy_mapping_meta_review.json'),original_saved_YAML_SHA256=sha(saved),original_adapter_SHA256=sha(adapter),own_tokenizer_image_ID=128257,own_tokenizer_length=128258,own_checkpoint_embedding_and_head_rows=128257,original_saved_resize_token_embeddings_required=True,original_saved_resize_must_be_audited_before_any_NPU_dispatch=True,original_auto_on_nonCPU_requests_FP16=True,actual_loaded_CPU_or_NPU_dtype_not_yet_proved=True,original_generation_budget2048_outer10_Greedy_preserved=True,own_original_chat_template_has_no_injected_system_or_date=True,original_saved_legacy_reconstruction_and_own_null_rope_scaling_same=True,own_BF16_storage_original_auto_nonCPU_FP16_request=True,current_row=690,original_state_legacy_row689_not_modified=True,three_D_values=three,dataset_SHA256=sha(R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'),all4329_order_and_all200_own_images_tokens_pixels_verified=True,all200_completed_in_original_v2_before_metadata_only_failure=True,original_v2_nonzero_completion_preserved=True,actual_frontend200_replay_count=0,necessary_full_input_example_SHA256=examples,historical_shadow_engine_not_captured_and_not_claimed=True,no_foreign_checkpoint_or_vision_source_used=True,ready_for_NPU=False,no_learned_CPU_or_NPU_forward=True,next_required='Own original CPU resize128258 source provenance, full native graph/cache/observations/independent verifier and standalone YAML before genuine NEW3')
 put('readiness.json',result);put('actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,metadata_binding_only=True,actual_frontend200_replay_count=0,no_learned_CPU_or_NPU_forward=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  if H.exists():put('actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),metadata_only=True,actual_frontend200_replay_count=0,no_learned_CPU_or_NPU_forward=True))
  raise
