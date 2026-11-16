"""Independent Falcon own-source, full head/cache, original input and CER audit.
CPU operator references are diagnostics, never a model grade or A100 result.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import json,hashlib,re,csv,contextlib,math
import numpy as np
import torch
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
MID='tiiuae/Falcon-OCR';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None):
 p=Path(p);v=sha(p);assert h is None or v==h,(str(p),'SHA');assert str(p) not in EXTERNAL or EXTERNAL[str(p)]==v;EXTERNAL[str(p)]=v;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def bs(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def ft(x):return torch.from_numpy(x.copy()).view(torch.bfloat16) if x.dtype==np.uint16 else torch.from_numpy(x.copy())
def rel(actual,wanted):
 actual=actual.float();wanted=wanted.float();return float(torch.sqrt(torch.mean((actual-wanted)**2))/torch.sqrt(torch.mean(wanted**2)).clamp(min=1e-6))
def independent_pixels(path):
 from PIL import Image
 image=Image.open(path).convert('RGB');w,h=image.size;ratio=w/h
 if not (64<=w<=1024 and 64<=h<=1024):
  if w<64 or h<64:
   nw,nh=(64,int(64/ratio)) if w<h else (int(64*ratio),64)
  else:nw,nh=(1024,int(1024/ratio)) if w<h else (int(1024*ratio),1024)
  if nw>1024:nw=1024;nh=int(nw/ratio)
  if nh>1024:nh=1024;nw=int(nh*ratio)
  image=image.resize((nw,nh))
 w,h=image.size;hb=round(h/16)*16;wb=round(w/16)*16
 if hb*wb>28*28*1280:
  beta=np.sqrt(h*w/(28*28*1280));hb=math.floor(h/beta/16)*16;wb=math.floor(w/beta/16)*16
 elif hb*wb<56*56:
  beta=np.sqrt(56*56/(h*w));hb=math.ceil(h*beta/16)*16;wb=math.ceil(w*beta/16)*16
 image=image.resize((wb,hb),Image.Resampling.BICUBIC);p=(np.asarray(image).astype(np.float64)*(1/255)).astype(np.float32);p=(p-np.float32(.5))/np.float32(.5)
 padded=np.zeros((1,1,1024,1024,3),dtype=np.float32);mask=np.zeros((1,1,1024,1024),dtype=np.int64);padded[0,0,:hb,:wb]=p;mask[0,0,:hb,:wb]=1
 return padded,mask
def editdistance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]
def verify(folder,count):
 torch.set_num_threads(4);assert count in [3,200] and {x.name for x in folder.iterdir()}==FILES and folder.parent==R/'workspace/reports/tiiuae_Falcon-OCR/image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert report['summary']['primary_metric']=='CER' and cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==8 and not cfg['data']['filter_empty']
 assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['original_Triton_BF16_intermediate_FFN_direct_kernel_restored'] and p['model_id']==MID and p['workbook_D_reference_authorized'] and p['own_original_plain_BF16_512_outer8_inner1_protocol_preserved']
 for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
 for n,h in p['static_evidence_SHA256'].items():check(n,h)
 W=Path(p['weight_path'])
 for n,h in p['own_assets_SHA256'].items():check(W/n,h)
 for n,v in p['own_weight_files'].items():assert (W/n).stat().st_size==v['bytes'];check(W/n,v['SHA256'])
 check(p['reference_workbook'],p['reference_workbook_SHA256']);check(p['dataset_path'],p['dataset_SHA256']);front=jl(check(p['own_frontend200'],p['own_frontend200_SHA256']));assert len(front)==200
 probe=js(check(p['own_source_dtype_probe'],p['own_source_dtype_probe_SHA256']));assert probe['passed'] and len(probe['source'])==115 and probe['persistent_buffer_dtypes']=={'freqs_cis_golden':'torch.float32'}
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['devices']==[4] and line['head_size']==65536 and line['configured_max_new_tokens']==512 and line['configured_outer_batch_size']==8
 for n,h in line['runtime_library_source_SHA256'].items():check(n,h)
 final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed'] and final['actual_inner_serial1']
 assert [v['indices'] for v in final['actual_outer_batches']]==[list(range(i,min(i+8,count))) for i in range(0,count,8)]
 weight=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert len(weight['source'])==weight['actual_NPU_source_tensor_count']==115
 from safetensors import safe_open
 source={}
 with safe_open(str(W/'model.safetensors'),framework='pt',device='cpu') as sf:
  assert set(sf.keys())==set(weight['source'])==set(probe['source'])
  for n in sf.keys():
   t=sf.get_tensor(n);v=weight['source'][n];assert list(t.shape)==v['source']['shape'] and str(t.dtype)==v['source']['dtype'] and bs(t.contiguous().view(torch.uint8).numpy())==v['source']['SHA256']
   dtype=getattr(torch,probe['source'][n]['original_saved_BF16_HF_loaded_CPU_dtype_diagnostic']['dtype'].split('.')[-1]);cast=t.to(dtype);assert v['actual_NPU']==v['original_HF_loaded_cast']==dict(shape=list(cast.shape),dtype=str(dtype),SHA256=bs(cast.contiguous().view(torch.uint8).numpy())) and v['device']=='npu:4';source[n]=cast
 from universal_eval.model.falconocr_own_original_20261116.rope import precompute_freqs_cis,apply_golden_freqs_cis_to_visual_pos,apply_3d_rotary_emb
 from universal_eval.model.falconocr_own_original_20261116.configuration_falcon_ocr import FalconOCRConfig
 from universal_eval.model.falconocr_own_original_20261116.processing_falcon_ocr import process_batch
 from transformers import AutoTokenizer
 from universal_eval.algorithm.inference import GreedyInference
 from universal_eval.metric.ocr import ocr_extract_plate,_prepare_ocr_pair
 owncfg=FalconOCRConfig(**js(W/'config.json'));tokenizer=AutoTokenizer.from_pretrained(str(W),trust_remote_code=True,local_files_only=True);freq=precompute_freqs_cis(32,8192,10000);assert weight['analytic_freqs']==dict(shape=list(freq.shape),dtype=str(freq.dtype),SHA256=bs(freq.view(torch.uint8).numpy()))
 traces=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
 for rows in [traces,ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
 edits=denominator=total=0;reviews=[];empty=[];truncated=[]
 for i,small in enumerate(traces):
  e=js(check(small['actual_execution_file'],small['actual_execution_SHA256']));ref=front[i];N=len(ref['input_ids']);tokens=e['generated_tokens'];n=e['actual_native_forwards'];S=((N+512+127)//128)*128;total+=n
  assert e['error'] is None and small['error'] is None and n==len(tokens)+1==len(e['frames']) and 1<=len(tokens)<=S-N and e['original_allocation_S']==S and e['original_seed']==42 and e['configured_max_new_tokens']==512 and e['original_stop_ids']==ref['original_stop_ids']
  assert not any(t in ref['original_stop_ids'] for t in tokens[:-1]);budget=len(tokens)==S-N and tokens[-1] not in ref['original_stop_ids'];assert budget or tokens[-1] in ref['original_stop_ids'];assert e['original_budget_hit']==budget
  assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and GreedyInference._add_answer_instruction(ref['prompt'])==e['actual_model_prompt_ignored_by_original_Falcon'] and e['original_image_prompt']==ref['actual_image_prompt']
  assert ins[i]['reference']==outs[i]['reference']==details[i]['reference']==ref['reference'];check(ref['media_path'],ref['image_SHA256'])
  raw=tokenizer.decode(tokens,skip_special_tokens=False).replace('<|end_of_query|>','').replace('<|end_of_text|>','').strip();choice=ocr_extract_plate(raw);choice=raw if choice is None else choice;assert raw==e['raw_output']==outs[i]['raw_output'] and choice==outs[i]['prediction']==details[i]['prediction']
  from PIL import Image
  original=process_batch(tokenizer,owncfg,[(Image.open(ref['media_path']).convert('RGB'),ref['actual_image_prompt'])],max_length=4096,min_dimension=64,max_dimension=1024)
  pix,pmask=independent_pixels(ref['media_path']);assert np.array_equal(pix,original['pixel_values'].numpy()) and np.array_equal(pmask,original['pixel_mask'].numpy())
  with ReadOnlyNPZ(check(e['archive'],e['archive_SHA256'])) as a:
   assert a['tokens_0'].tolist()==[ref['input_ids']] and np.array_equal(a['input_pixel_values'],pix) and np.array_equal(a['input_pixel_mask'],pmask)
   for key in ['rope_pos_t','rope_pos_hw']:
    got=a['input_'+key];wanted=original['pos_t' if key.endswith('pos_t') else 'pos_hw'].numpy();assert np.array_equal(got,wanted,equal_nan=True)
   heads=a['all_full_heads'];hidden=a['head_inputs'];assert heads.shape==(n,65536) and hidden.shape==(n,768) and heads.dtype==hidden.dtype==np.uint16 and torch.isfinite(ft(heads).float()).all()
   assert ft(heads[:-1]).float().argmax(-1).tolist()==tokens
   htarget=torch.nn.functional.linear(ft(hidden),source['output.weight']);hr=rel(ft(heads),htarget);assert hr<.02
   cache=a['kv_prefill'];delta=a['kv_decode_appends'];assert cache.shape==(22,2,1,16,N,64) and delta.shape==(len(tokens),22,2,1,16,1,64) and cache.dtype==delta.dtype==np.uint16
   combined=np.concatenate([cache,np.concatenate(list(delta),axis=-2)],axis=-2);assert list(combined.shape)==e['native_final_valid_cache']['shape'] and bs(combined)==e['native_final_valid_cache']['SHA256']
   assert e['native_KV_storage_shape']==[22,2,1,16,S,64] and e['all_native_cache_prefixes_verified_unchanged']
   padded=np.full((S,),original['pad_token_id'],dtype=np.int64);padded[:N]=ref['input_ids'];eos=padded==11;eos[-1]=True;seq=np.concatenate([[0],np.cumsum(eos)[:-1]]);nonpad=np.cumsum(padded!=original['pad_token_id'])>0;acc=np.cumsum(padded==244);image_mask=(acc-np.cumsum(padded==230))>0;img_ids=acc*image_mask
   for step,frame in enumerate(e['frames']):
    assert frame['step']==step and frame['cache_before']==(0 if step==0 else N+step-1) and frame['cache_after']==N+step and frame['LM_layers']==frame['QKV_layers']==frame['original_Triton_FFN_gate_layers']==22 and frame['native_cache_prefix_unchanged']
    want_ids=[ref['input_ids']] if step==0 else [[tokens[step-1]]];assert frame['input_tokens']==a['tokens_'+str(step)].tolist()==want_ids
    if frame['full_cache_SHA256'] is not None:assert frame['full_cache_SHA256']==bs(combined[...,:N+step,:])
    qpos=np.arange(N) if step==0 else np.array([N+step-1]);kpos=np.arange(N+step)
    allowed=((qpos[:,None]>=kpos[None,:])&(seq[qpos,None]==seq[None,kpos])&nonpad[None,kpos])|(image_mask[qpos,None]&image_mask[None,kpos]&(img_ids[qpos,None]==img_ids[None,kpos]));assert np.array_equal(a['allowed_attention_mask_'+str(step)][0,0],allowed)
    positions=original['pos_t'].long() if step==0 else torch.tensor(frame['pos_t'],dtype=torch.long);wanted=freq[positions];assert np.array_equal(a['freqs_cis_'+str(step)],wanted.numpy())
   expected_golden=apply_golden_freqs_cis_to_visual_pos(source['freqs_cis_golden'],original['pos_hw']);assert float((ft(a['freqs_cis_golden_prefill'])-expected_golden).abs().max())<.00001
   import einops as Ein
   pixels=ft(a['input_pixel_values']).to(torch.bfloat16);patches=Ein.rearrange(pixels,'n (t pt) (h ph) (w pw) c -> (n t h w) (pt ph pw c)',pt=1,ph=16,pw=16);mask=Ein.reduce(ft(a['input_pixel_mask']).bool(),'n (t pt) (h ph) (w pw) -> (n t h w)','any',pt=1,ph=16,pw=16);patches=patches[mask];assert torch.equal(patches,ft(a['valid_image_patch_projector_input']))
   projected=ft(a['image_projector_output']);pr=rel(projected,torch.nn.functional.linear(patches,source['img_projector.weight']));assert pr<.02
   merged=ft(a['merged_embeddings']);im=torch.tensor([ref['input_ids']])==227;text=source['tok_embeddings.weight'][torch.tensor([ref['input_ids']])];assert torch.equal(merged[im],projected) and torch.equal(merged[~im],text[~im]) and torch.equal(ft(a['text_embeddings']),text)
   qkv_reviews=[]
   if i<3:
    for layer in range(22):
     packed=ft(a[f'FFN_packed_{layer}_step_0']);gate=ft(a[f'FFN_original_Triton_gate_{layer}_step_0']);expected_gate=(torch.relu(packed[...,0::2])**2)*packed[...,1::2];assert packed.dtype==gate.dtype==torch.bfloat16 and torch.equal(gate,expected_gate),'Original Triton BF16 intermediate gate semantics'
     qkv=ft(a[f'QKV_{layer}_step_0']);qkvin=ft(a[f'QKV_input_{layer}_step_0']);qr=rel(qkv,torch.nn.functional.linear(qkvin,source[f'layers.{layer}.attention.wqkv.weight']));assert qr<.02
     q,k,v=qkv.split([1024,512,512],dim=-1);q=q.reshape(1,N,16,64);k=k.reshape(1,N,8,64);v=v.reshape(1,N,8,64);q=torch.nn.functional.rms_norm(q,(64,));k=torch.nn.functional.rms_norm(k,(64,));k=k.repeat_interleave(2,dim=2);v=v.repeat_interleave(2,dim=2);q,k=apply_3d_rotary_emb(q,k,freq[original['pos_t'].long()],expected_golden,original['pos_hw']);kr=rel(ft(cache[layer,0]),k.permute(0,2,1,3));assert kr<.02 and torch.equal(ft(cache[layer,1]),v.permute(0,2,1,3));qkv_reviews.append(dict(layer=layer,QKV_CPU_same_operand_relative_RMSE=qr,RoPE_K_CPU_same_operand_relative_RMSE=kr,V_exact=True))
   for name,v in e['activations'].items():
    if name in a.files:assert list(a[name].shape)==v['shape'] and bs(a[name])==v['SHA256']
   assert sum(k.startswith('layer_') for k in e['activations'])==22*n and sum(k.startswith('QKV_') and not k.startswith('QKV_input_') for k in e['activations'])==22*n
  pred,gt=_prepare_ocr_pair(choice,ref['reference']);edits+=editdistance(pred,gt);denominator+=len(gt);empty.extend([i] if not raw else []);truncated.extend([i] if budget else [])
  reviews.append(dict(index=i,full_head_rows=n,full_head_all_rows_CPU_same_operand_relative_RMSE=hr,projector_CPU_same_operand_relative_RMSE=pr,all22_KV_values_losslessly_reconstructed=True,all_masks_positions_F32_rotary_and_own_input_pixels_checked=True,QKV_prefill_checks=qkv_reviews,CPU_reference_only_diagnostic_not_grade=True))
 events=0
 for name,h in final['first_real_mixed_image_prefill_profiler_SHA256'].items():
  f=check(name,h)
  if f.name=='task_time.csv' or re.search(r'/task_time_[^/]+\.csv$',name):
   rows=list(csv.DictReader(f.open()));assert all(v['Device_id']=='4' for v in rows);events+=sum(v['kernel_type'].startswith('KERNEL') for v in rows)
 assert events>100
 score=edits/max(1,denominator);assert report['metrics']=={'CER':score} and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
 errors={k:str(abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v))) for k,v in p['current_A100_D_values'].items()};qualified=count==200 and all(Decimal(v)<=Decimal('.05') for v in errors.values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'CER':score},edit_distance=edits,total_reference_characters=denominator,error_indices=[],empty_indices=empty,invalid_indices=[],truncated_indices=truncated,actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=total,source_tensor_count=115,all_own_source_actual_NPU_SHA256_verified=True,own_original_BF16_parameters_and_F32_golden_buffer_preserved=True,all_native_KV_values_losslessly_reconstructed=True,all_full_inputs_tokens_raw_references_errors_preserved=True,original_rounded_generation_budget_and_extra_terminal_forward_preserved=True,original_Triton_BF16_intermediate_FFN_exact_in_all_first3_all22_prefill_layers=True,numeric_sample_checks=reviews,NPU_profiler_kernel_events=events,relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=qualified,original_protocol_final_qualification_complete=qualified,qualification_basis='User workbook D; own saved Falcon plain HF BF16 512 rounded allocation/outer8 inner1/real NPU22cache/F32 golden buffer and original OCR CER extraction',historical_source_boundary=p['historical_source_boundary'],A100_historical_engine_or_tensor_capture_claimed=False,independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):result=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 home=Path(js(Path(sys.argv[1])/'operator_lineage.json')['actual_execution']['weight_evidence']).parent;out=home/'independent_report_verification.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
