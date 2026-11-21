"""Independent own-source visual reconstruction; CPU values are diagnostics only."""
from pathlib import Path
import contextlib,gc,hashlib,json,os
import numpy as np,torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.mllama.modeling_mllama import MllamaVisionModel
from scripts.mllama11b_original_independent_cpu_910b import sha,fp,tr,inputs
from scripts.egm8b_original_independent_cpu_910b import decode_arrays
def verify_own_visual_values(p,line,by,count):
 w=Path(p['weight_path']);home=Path(line['weight_loading_evidence']).parent/'own_independent_visual_values';reuse=p.get('own_completed_true3_visual_reference')
 if reuse:
  ref=Path(reuse['review']);assert sha(ref)==reuse['SHA256'];review=json.loads(ref.read_bytes())
  assert review['passed'] and review['model_id']==p['model_id'] and review['reference_samples']==3 and review['source_checkpoint_files']==p['all_source_file_sha256']
  for n,h in review['external_evidence_sha256'].items():assert sha(n)==h
  return dict(review,completed_true3_visual_controls_reused_without_learned_CPU_or_NPU_replay=True,actual_control_forwards_this_acceptance=0)
 assert count==3 and not home.exists();home.mkdir()
 expected={x['source_key']:x for x in p['all906_source_to_native_semantic_name_shape_mapping_verified']};selected={k:v for k,v in expected.items() if v['target_key'].startswith(('model.vision_model.','model.multi_modal_projector.'))};assert len(selected)==511
 arrays={};records={};source_rows={};CPU_differences={};external={};NPU=int(line['devices'][0])
 def build(device,tag):
  c=AutoConfig.from_pretrained(w,local_files_only=True).vision_config;assert c._attn_implementation is None
  with torch.device('meta'):v=MllamaVisionModel(c);pr=torch.nn.Linear(7680,4096,bias=True)
  assert v.config._attn_implementation=='sdpa';vp=dict(v.named_parameters());pp=dict(pr.named_parameters());rows=[]
  for shard in p['all_source_file_sha256']:
   with safe_open(str(w/shard),framework='pt',device='cpu') as sf:
    for k in sf.keys():
     if k not in selected:continue
     row=selected[k];name=row['target_key'];vis=name.startswith('model.vision_model.');short=name.removeprefix('model.vision_model.' if vis else 'model.multi_modal_projector.');parent=v if vis else pr;params=vp if vis else pp;x=sf.get_tensor(k)
     assert x.dtype==torch.bfloat16 and fp(x)==row['full_sha256'] and list(x.shape)==list(params[short].shape)
     y=x.to(device);assert fp(y)==row['full_sha256'];root,leaf=short.rsplit('.',1) if '.' in short else ('',short);setattr(parent.get_submodule(root) if root else parent,leaf,torch.nn.Parameter(y,requires_grad=False));rows.append(dict(source_key=k,target_key=name,source_SHA256=fp(x),actual_SHA256=fp(y),device=str(device),dtype=str(y.dtype)))
  assert len(rows)==511;v.eval();pr.eval();source_rows[tag]=rows;return v,pr
 for tag,device in [('CPU_original_BF16',torch.device('cpu')),('NPU_independent_original_BF16',torch.device('npu:'+str(NPU)))]:
  if device.type=='npu':
   import torch_npu
   torch.npu.set_device(device)
   for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
  v,pr=build(device,tag)
  for i in range(3):
   entry=by[i];q=Path(entry['group_execution']);g=json.loads(q.read_bytes());assert g['indices']==[i] and sha(g['archive'])==g['archive_sha256']
   original=decode_arrays(g['archive'],g['arrays']);inp,_,_=inputs(p,[i])
   assert torch.equal(inp['pixel_values'].reshape(4,3,560,560).bfloat16(),original['actual_patch_input_BF16'])
   pix=inp['pixel_values'].bfloat16().to(device);ar=inp['aspect_ratio_ids'].to(device);mask=inp['aspect_ratio_mask'].to(device)
   with torch.inference_mode():a=v(pixel_values=pix,aspect_ratio_ids=ar,aspect_ratio_mask=mask).last_hidden_state;b=pr(a)
   if device.type=='npu':torch.npu.synchronize(device)
   for field,tensor,key in [('encoder',a,'actual_vision_output'),('projected',b,'actual_projected_vision')]:
    actual=tensor.detach().cpu().contiguous();assert actual.dtype==torch.bfloat16 and torch.isfinite(actual).all();arrays[tag+'_'+str(i)+'_'+field]=actual;records[tag+'_'+str(i)+'_'+field]=tr(actual)
    if device.type=='npu':assert torch.equal(actual,original[key]),(i,field,'Independent same-own-source full BF16 NPU reconstruction differs')
    else:
     delta=(actual.float()-original[key].float()).abs();CPU_differences[str(i)+'_'+field]=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),RMSE=float(delta.square().mean().sqrt()),diagnostic_not_A100_or_official_CPU_score=True)
   del original,pix,ar,mask,a,b
   (home/'progress.json').write_text(json.dumps(dict(tag=tag,completed_visual_samples=i+1,total=3,no_language_generation_or_score=True))+'\n')
  del v,pr;gc.collect()
  if device.type=='npu':torch.npu.empty_cache()
 archive=home/'own_complete_independent_visual_controls.npz';np.savez_compressed(archive,**{n:v.reshape(-1).view(torch.uint8).numpy() for n,v in arrays.items()});weights=home/'all511_independent_parameter_SHA.json';weights.write_text(json.dumps(source_rows,indent=2)+'\n')
 external.update({str(archive):sha(archive),str(weights):sha(weights)})
 review=dict(passed=True,model_id=p['model_id'],reference_samples=3,actual_control_forwards_this_acceptance=6,only_visual_encoder_and_projector_no_language_score=True,original_SDPA_BF16_math_not_modified=True,all511_own_source_CPU_and_actual_NPU_full_SHA_verified=True,all3_original_NPU_encoder_and_projected_full_values_bitwise_reconstructed=True,CPU_original_BF16_measured_differences=CPU_differences,CPU_not_A100_and_no_historical_engine_equivalence_claim=True,source_checkpoint_files=p['all_source_file_sha256'],array_records=records,archive=str(archive),archive_SHA256=sha(archive),external_evidence_sha256=external)
 result=home/'review.json';result.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');review['external_evidence_sha256'][str(result)]=sha(result);return review
