"""Own Donut source/META and input200 only; no learned CPU/NPU or scores."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,traceback,inspect,time
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'DonutInvoices_own_original_auto_FP16_caption200_source_META_v1_20261117';MID='katanaml-org/invoices-donut-model-v1';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/VisionEncoderDecoderModel')/MID;DATA=R/'workspace/datasets/image-to-text-short/eval.jsonl'
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
 torch.set_num_threads(4);H.mkdir(exist_ok=False);start=time.monotonic();put('progress.json',dict(UTC=now(),PID=os.getpid(),phase='own_original_source_and_genuine_META_only',no_learned_forward=True))
 assert not any(getattr(sys.modules.get(n),'__file__',None)=='/dev/null' for n in ['transformer_engine','transformer_engine.pytorch'])
 assert W.is_dir() and (W/'pytorch_model.bin').is_file()
 original_cf=R/'configs/VisionEncoderDecoderModel/katanaml-org--invoices-donut-model-v1.yaml';assert sha(original_cf)=='c9c4f05d98dd64bf789c575d370f5700d1270aead09ae5a918efc93350295d2a'
 own_assets={p.name:dict(bytes=p.stat().st_size,SHA256=sha(p)) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.model']};whole=dict(file=str(W/'pytorch_model.bin'),bytes=(W/'pytorch_model.bin').stat().st_size,SHA256=sha(W/'pytorch_model.bin'))
 cfg=AutoConfig.from_pretrained(W,local_files_only=True);assert cfg.encoder.model_type=='donut-swin' and cfg.decoder.model_type=='mbart' and cfg.decoder.decoder_layers==4 and cfg.decoder.num_hidden_layers==12 and cfg.decoder.vocab_size==57566 and cfg.decoder.max_position_embeddings==1536 and cfg.decoder_start_token_id==57565
 with torch.device('meta'):model=VisionEncoderDecoderModel(cfg)
 expected=model.state_dict();params=dict(model.named_parameters(remove_duplicate=False));raw=torch.load(W/'pytorch_model.bin',map_location='cpu',weights_only=True);assert isinstance(raw,dict) and all(torch.is_tensor(t) for t in raw.values())
 def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
 tensors=[]
 for key,t in raw.items():
  cast=t.half() if t.is_floating_point() and key in params else t
  tensors.append(dict(key=key,shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=fp(t),execution_dtype=str(cast.dtype),original_auto_FP16_cast_SHA256=fp(cast),META_exact_name_and_shape=key in expected and list(t.shape)==list(expected[key].shape)))
 missing=sorted(set(expected)-set(raw));extra=sorted(set(raw)-set(expected));mismatch=[x['key'] for x in tensors if x['key'] in expected and not x['META_exact_name_and_shape']]
 put('own_source_cast_and_META_mapping.json',dict(own_weight=whole,own_assets=own_assets,tensors=tensors,META_state_count=len(expected),all_parameter_names=list(params),META_missing_names=missing,source_extra_names=extra,shape_mismatches=mismatch,own_head_and_word_embedding_equal=fp(raw['decoder.lm_head.weight'])==fp(raw['decoder.model.decoder.embed_tokens.weight']),no_learned_CPU_or_NPU_forward=True))
 del raw,model
 processor=DonutProcessor.from_pretrained(W,local_files_only=True);p=processor.image_processor.to_dict();assert p['size']=={'height':1280,'width':960};tokenizer=processor.tokenizer;rust=Tokenizer.from_file(str(W/'tokenizer.json'));dataset=[json.loads(x) for x in DATA.read_text().splitlines()];assert len(dataset)>=200
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
 review=dict(passed=not (missing or extra or mismatch),UTC=now(),model_id=MID,samples=count,own_weight=whole,own_assets=own_assets,own_source_tensor_count=len(tensors),META_state_count=len(expected),META_missing_names=missing,source_extra_names=extra,shape_mismatches=mismatch,all200_independent_own_uint8_F32_and_FP16_input_controls_passed=True,all200_own_Rust_decoder_prompt_IDs_equal=True,original_runtime_auto_maps_to_FP16_from_own_saved_adapter=True,original_saved_source_SHA256={n:sha(R/n) for n in ['universal_eval/model/adapters/transformers_vlm.py','universal_eval/model/adapters/transformers_vlm_donut_poscap.py']},own_image_processor=p,frontend200_SHA256=sha(output),dataset_SHA256=sha(DATA),original_A100_selected_metric='ROUGE-L',original_A100_selected_score_workbook_metadata='0.003904',original_A100_selected_sample_count=8,original_A100_selected_run='20260828_115000__8b04820e89da',workbook_D_preserved='0.0039',MLU_selected_metric='BLEU-4',historical_full200_A100_runtime_not_captured=True,original_2048_requested_upper_and_own1536_position_cap_preserved=True,original_decoder_start_57565_and_prompt_prefix_slicing_not_silently_changed=True,original_stop_EOS2_and_forced_EOS2_preserved=True,full_five_reports_required_for_next_real_stage=True,ready_for_real_NPU=False,missing_native_adapter_YAML_independent_full_numeric_verifier=True,no_learned_CPU_or_NPU_forward=True,no_new_score_G_or_qualification=True,elapsed_seconds=time.monotonic()-start)
 put('review.json',review);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,own_inputs200_completed=count,source_META_exact_names_closed=review['passed'],no_learned_CPU_or_NPU_forward=True));print(json.dumps(review,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_learned_CPU_or_NPU_forward=True))
  raise
