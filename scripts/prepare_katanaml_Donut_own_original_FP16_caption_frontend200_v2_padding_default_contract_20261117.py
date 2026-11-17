"""Own Donut source/META and input200 only; no learned CPU/NPU or scores."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,traceback,inspect,time
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'DonutInvoices_own_original_auto_FP16_caption200_source_META_v2_reference_padding_contract_20261117';MID='katanaml-org/invoices-donut-model-v1';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/VisionEncoderDecoderModel')/MID;DATA=R/'workspace/datasets/image-to-text-short/eval.jsonl'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 t=H/(n+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,H/n)
def main():
 import torch
 from PIL import Image
 from transformers import AutoConfig,DonutProcessor,VisionEncoderDecoderModel
 from tokenizers import Tokenizer
 from universal_eval.model.reference_sources.donut_invoice_own_input_reference import pixels
 torch.set_num_threads(4);H.mkdir(exist_ok=False);start=time.monotonic()
 OLD=B/'DonutInvoices_own_original_auto_FP16_caption200_source_META_v1_20261117';meta_path=OLD/'own_source_cast_and_META_mapping.json';meta=json.loads(meta_path.read_bytes());assert not meta['META_missing_names'] and not meta['source_extra_names'] and not meta['shape_mismatches'] and meta['own_head_and_word_embedding_equal'];assert len(meta['tensors'])==meta['META_state_count']==484
 own_assets=meta['own_assets'];whole=meta['own_weight'];tensors=meta['tensors'];expected={x['key']:x['shape'] for x in tensors};missing=[];extra=[];mismatch=[]
 for name,record in own_assets.items():assert sha(W/name)==record['SHA256']
 assert (W/'pytorch_model.bin').stat().st_size==whole['bytes']==809346841
 put('original_completed_own_META_reuse_review.json',dict(passed=True,UTC=now(),original_own_META=str(meta_path),original_own_META_SHA256=sha(meta_path),actual_own_source_tensors=484,no_META_or_source_tensor_cast_repeated=True,no_learned_CPU_or_NPU_forward=True,raw_Rust_padding768_guard_failure_preserved=True,reference_only_fix='Original HF TokenizersBackend call defaults disable padding/truncation; independent raw own Rust reference now explicitly uses those same defaults. No deployed model mask/IDs/vocabulary changed.'))
 processor=DonutProcessor.from_pretrained(W,local_files_only=True);p=processor.image_processor.to_dict();assert p['size']=={'height':1280,'width':960};tokenizer=processor.tokenizer;rust=Tokenizer.from_file(str(W/'tokenizer.json'));rust.no_padding();rust.no_truncation();dataset=[json.loads(x) for x in DATA.read_text().splitlines()];assert len(dataset)>=200
 count=0;output=H/'frontend200.jsonl';put('progress.json',dict(UTC=now(),PID=os.getpid(),phase='own_independent_input200_no_NN',completed=0))
 with output.open('x',encoding='utf-8') as out:
  for i,row in enumerate(dataset[:200]):
   media=(DATA.parent/row['media_path']).resolve();assert media.is_file() and media.is_relative_to(R/'workspace/datasets') and row['prompt'] and row['reference']
   with Image.open(media) as image:
    image=image.convert('RGB');actual=processor.image_processor(images=image,return_tensors='pt')['pixel_values'];independent=pixels(image,p)
   assert actual.dtype==independent.dtype==torch.float32 and torch.equal(actual,independent),('original_uint8_bilinear_thumbnail_fused_normalization_mismatch',i,float((actual-independent).abs().max()))
   ids=tokenizer(row['prompt'],add_special_tokens=False,return_tensors='pt').input_ids;ri=rust.encode(row['prompt'],add_special_tokens=False).ids;assert ids.tolist()==[ri] and ids.shape[1]>0 and int(ids.max())<57566
   out.write(json.dumps(dict(index=i,prompt=row['prompt'],reference=row['reference'],source_media_path=row['media_path'],resolved_media_path=str(media),image_SHA256=sha(media),original_prompt_decoder_ids=ri,decoder_prefix_length=ids.shape[1],original_configured_max_new_tokens=2048,original_position_limit=1536,original_poscap_effective_max_new_tokens=min(2048,1536-ids.shape[1]-1),CPU_F32_pixels=dict(shape=list(actual.shape),dtype=str(actual.dtype),SHA256=fp(actual)),original_FP16_pixels=dict(shape=list(actual.shape),dtype='torch.float16',SHA256=fp(actual.half())),own_reference_control_byte_exact=True),ensure_ascii=False)+'\n');out.flush();count+=1
   if count%10==0:put('progress.json',dict(UTC=now(),PID=os.getpid(),phase='own_independent_input200_no_NN',completed=count))
 review=dict(passed=not (missing or extra or mismatch),UTC=now(),model_id=MID,samples=count,own_weight=whole,own_assets=own_assets,own_source_tensor_count=len(tensors),META_state_count=len(expected),META_missing_names=missing,source_extra_names=extra,shape_mismatches=mismatch,all200_independent_own_uint8_F32_and_FP16_input_controls_passed=True,all200_own_Rust_decoder_prompt_IDs_equal=True,reused_original_completed_META_no_repeat=True,independent_reference_only_padding_contract_corrected=True,original_runtime_auto_maps_to_FP16_from_own_saved_adapter=True,original_saved_source_SHA256={n:sha(R/n) for n in ['universal_eval/model/adapters/transformers_vlm.py','universal_eval/model/adapters/transformers_vlm_donut_poscap.py']},own_image_processor=p,frontend200_SHA256=sha(output),dataset_SHA256=sha(DATA),original_A100_selected_metric='ROUGE-L',original_A100_selected_score_workbook_metadata='0.003904',original_A100_selected_sample_count=8,original_A100_selected_run='20260828_115000__8b04820e89da',workbook_D_preserved='0.0039',MLU_selected_metric='BLEU-4',historical_full200_A100_runtime_not_captured=True,original_2048_requested_upper_and_own1536_position_cap_preserved=True,original_decoder_start_57565_and_prompt_prefix_slicing_not_silently_changed=True,original_stop_EOS2_and_forced_EOS2_preserved=True,full_five_reports_required_for_next_real_stage=True,ready_for_real_NPU=False,missing_native_adapter_YAML_independent_full_numeric_verifier=True,no_learned_CPU_or_NPU_forward=True,no_new_score_G_or_qualification=True,elapsed_seconds=time.monotonic()-start)
 put('review.json',review);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,own_inputs200_completed=count,source_META_exact_names_closed=review['passed'],no_learned_CPU_or_NPU_forward=True));print(json.dumps(review,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_learned_CPU_or_NPU_forward=True))
  raise
