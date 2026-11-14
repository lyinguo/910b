"""Independent original-group CPU inputs, all logits, positions, masks and source audit."""
import os,pathlib,sys,json,re,collections
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from safetensors import safe_open
from PIL import Image
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from egm8b_original_independent_cpu_910b import sha,fp,tr,clean,input_tensors,positions,decode_arrays
from strict_mc_independent_reference_910b import independent_choice
from cosmos_reason2_stable_rope_reference_910b import independent_text_rope
from idefics2_original_accuracy_reference_910b import original_accuracy_sample
torch.set_num_threads(4)
reuse_schema_recheck=False;assert len(sys.argv)==3
assert not reuse_schema_recheck or sys.argv[3]=='--reuse-existing-real3-for-empty-raw-schema-repair'
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in (3,200) and (not reuse_schema_recheck or count==3);assert {p.name for p in folder.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];assert sha(kw['protocol_file'])==kw['protocol_sha256'];p=json.loads(pathlib.Path(kw['protocol_file']).read_text());w=pathlib.Path(p['weight_path']);line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];device='npu:'+str(kw['npu_devices'][0]);assert cfg['model']['name']==p['model_id']=='nvidia/EGM-8B' and cfg['model']['model_path']==str(w) and folder.parent.parent.name=='nvidia_EGM-8B' and folder.parent.name==cfg['task_type']=='image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['algorithm']['batch_size']==10 and cfg['algorithm']['adapter']=='universal_eval/algorithm/native_explicit_mc_v2.py:NativeExplicitMCV2' and cfg['algorithm']['kwargs']=={'extract_answer':True}
assert cfg['metrics']['override']==['Strict_MC_Accuracy'] and cfg['metrics']['adapters']==['universal_eval/algorithm/tinyllava_strict_mc.py:StrictMCAccuracy'] and cfg['metrics']['include_failed_predictions'] and cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert not cfg['data']['download'] and not cfg['model']['download'] and cfg['data']['preprocess'] and not cfg['data']['filter_empty'] and not cfg['data']['apply_template']
assert line['protocol']=='EGM8B_original_BF16_eager_true_rightpad_size_grouped_free_generation' and line['protocol_sha256']==kw['protocol_sha256'] and line['weight_tensor_count']==750 and line['all_expected_parameters_loaded'] and line['runtime_dtype']=='BF16' and line['original_max_new_tokens']==4096 and line['original_outer_batch_size']==10 and line['original_untied_full151936_head_preserved'] and line['all267_original_padded_head_IDs_preserved'] and line['original_eager_attention_all_text_and_vision'] and set(line['module_placements'].values())=={device} and line['HF32_enabled']=={'conv':False,'matmul':False}
for src,digest in dict(p['native_runtime_sha256'],**p['historical_sources']).items():assert sha(src)==digest
for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(w/name)==digest
from transformers import AutoConfig,Qwen3VLForConditionalGeneration,GenerationConfig
mc=AutoConfig.from_pretrained(w,local_files_only=True);mc._attn_implementation='eager';mc.text_config._attn_implementation='eager';mc.vision_config._attn_implementation='eager'
with torch.device('meta'):meta=Qwen3VLForConditionalGeneration(mc)
meta.generation_config=GenerationConfig.from_pretrained(w,local_files_only=True);gen,_=meta._prepare_generation_config(None,max_new_tokens=4096,do_sample=False,temperature=0.0);assert gen.to_dict()==p['original_resolved_generation_config']==line['generation_config'] and gen.use_cache and gen.num_beams==1 and not gen.do_sample and gen.repetition_penalty==1 and gen.no_repeat_ngram_size==0 and gen.forced_bos_token_id is gen.forced_eos_token_id is None and gen.eos_token_id==[151645,151645,151643]
params=dict(meta.named_parameters());assert len(params)==750 and meta.lm_head.weight is not meta.model.language_model.embed_tokens.weight;acts={n:type(m).__name__ for n,m in meta.named_modules() if n.endswith('act_fn')};assert acts==line['activation_modules']==p['native_activation_modules'] and len(acts)==67
loadpath=pathlib.Path(line['weight_loading_evidence']);loaded=json.loads(loadpath.read_text());lookup={x['source_key']:x for x in loaded};expected={x['source_key']:x for x in p['source_tensor_records']};assert len(loaded)==len(lookup)==len(expected)==750;position_weight=None;embed_weight=None;control32_expected={}
for shard,head in p['source_files'].items():
 assert sha(w/shard)==head['sha256'] and (w/shard).stat().st_size==head['bytes']
 with safe_open(str(w/shard),framework='pt',device='cpu') as f:
  for k in f.keys():
   t=f.get_tensor(k);r=lookup[k];e=expected[k];assert t.dtype==torch.bfloat16 and list(t.shape)==e['shape']==list(params[k].shape)==r['shape'] and fp(t)==e['full_sha256']==r['source_sha256']==r['original_BF16_cast_sha256']==r['actual_NPU_sha256'] and r['device']==device and r['runtime_dtype']=='torch.bfloat16' and r['target_key']==k and r['original_cast_changes']==0
   if k.startswith('model.visual.'):control32_expected[k[len('model.visual.'):]]=fp(t.float())
   if k=='model.visual.pos_embed.weight':position_weight=t.clone()
   if k=='model.language_model.embed_tokens.weight':embed_weight=t.clone()
assert position_weight is not None and embed_weight.shape==(151936,4096) and expected['lm_head.weight']['full_sha256']!=expected['model.language_model.embed_tokens.weight']['full_sha256'];assert mc.text_config.num_hidden_layers==36 and mc.vision_config.depth==27 and mc.vision_config.hidden_size==1152 and mc.vision_config.deepstack_visual_indexes==[8,16,24] and p['padding_side']=='right'
text_inv=1/(5000000**(torch.arange(0,128,2,dtype=torch.int64).float()/128));vision_inv=1/(10000**(torch.arange(0,36,2,dtype=torch.float32)/36))
assert line['analytic_F32_buffers']=={'model.visual.rotary_pos_emb.inv_freq':tr(vision_inv),'model.language_model.rotary_emb.inv_freq':tr(text_inv),'model.language_model.rotary_emb.original_inv_freq':tr(text_inv)}
data=[json.loads(l) for l in pathlib.Path(p['dataset_path']).read_text().splitlines()];assert sha(p['dataset_path'])==p['dataset_sha256'] and len(data)==4329;old=json.loads(pathlib.Path(p['reference_report']).read_text());assert len(old['detailed_results'])==200 and sum(original_accuracy_sample(x['prediction'],x['reference']) for x in old['detailed_results'])/200==old['metrics']['Accuracy']==.815
history_output_path=pathlib.Path(p['reference_report']).parent/'sample_outputs.jsonl';assert str(history_output_path)==p['raw_report_empty_field_join_receipt']['own_sample_output_file'] and sha(history_output_path)==p['raw_report_empty_field_join_receipt']['SHA'];history_outputs=[json.loads(l) for l in history_output_path.read_text().splitlines()];assert len(history_outputs)==200 and [x['index'] for x in history_outputs]==list(range(200))
for i,h in enumerate(old['detailed_results']):assert history_outputs[i]['raw_output']==h.get('raw_output',h.get('extra',{}).get('raw_output')) and history_outputs[i]['prediction']==h['prediction']
assert sum(independent_choice(x['raw_output'])==str(x['reference']).strip().upper() for x in history_outputs)/200==.63 and sum(not x['raw_output'] for x in history_outputs)==15
for i,r in enumerate(old['detailed_results']):assert r['prompt']==p['input_records'][i]['prompt']==clean(data[i]['prompt']) and r['reference']==data[i]['reference'] and sha(r['media_path'])==p['input_records'][i]['media_sha256']
ins=[json.loads(l) for l in (folder/'sample_inputs.jsonl').read_text().splitlines()];outs=[json.loads(l) for l in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];tracepath=pathlib.Path(line['per_sample_evidence']);trace=[json.loads(l) for l in tracepath.read_text().splitlines()];actpath=pathlib.Path(line['activation_execution_trace']);acttrace=[json.loads(l) for l in actpath.read_text().splitlines()]
assert len(ins)==len(outs)==len(details)==len(trace)==count
for rows in (ins,outs,details):assert [r['index'] for r in rows]==list(range(count))
assert sorted(r['index'] for r in trace)==list(range(count));by={r['index']:r for r in trace};groups=[]
for start in range(0,count,10):
 g=collections.OrderedDict()
 for i in range(start,min(start+10,count)):
  with Image.open(p['input_records'][i]['media_path']) as im:key=(im.width//64,im.height//64)
  g.setdefault(key,[]).append(i)
 groups.extend(g.values())
assert len(acttrace)==len(groups);external={str(q):sha(q) for q in [tracepath,actpath,loadpath,pathlib.Path(line['profiler_trace']),pathlib.Path(line['native_GELU_probe']['path']),loadpath.parent/'generation_progress.jsonl']};allraw={};steps_total=0;max_rope=0.;max_visual_position_CPU_NPU=0.
for group_no,(indices,act) in enumerate(zip(groups,acttrace),1):
 entries=[by[i] for i in indices];q=pathlib.Path(entries[0]['group_execution']);g=json.loads(q.read_text());assert all(x['group_execution']==str(q) and x['group_execution_sha256']==sha(q) for x in entries) and g['group']==group_no and g['indices']==indices and g['batch_size']==len(indices) and g['error'] is None and g['vision_calls']==1;external[str(q)]=sha(q);assert sha(g['archive'])==g['archive_sha256'];external[g['archive']]=sha(g['archive']);t=decode_arrays(g['archive'],g['arrays']);inp,texts,rust=input_tensors(p,indices)
 assert texts==g['rendered'] and {k:tr(v) for k,v in inp.items()}==g['CPU_inputs'] and inp['input_ids'].tolist()==g['input_ids'] and inp['attention_mask'].tolist()==g['attention_mask'] and inp['mm_token_type_ids'].tolist()==g['mm_token_type_ids'] and inp['image_grid_thw'].tolist()==g['image_grid_thw'];size,length=inp['input_ids'].shape;assert torch.equal(t['actual_vision_pixels'],inp['pixel_values'].bfloat16());N=inp['pixel_values'].shape[0];M=N//4
 coords=[]
 for temporal,h,wid in inp['image_grid_thw'].tolist():
  assert temporal==1
  coords.extend([[y+dy,x+dx] for y in range(0,h,2) for x in range(0,wid,2) for dy in range(2) for dx in range(2)])
 coords=torch.tensor(coords);assert torch.equal(coords,t['actual_vision_position_ids2']) and torch.equal((coords[:,:,None]*vision_inv).flatten(1),t['actual_vision_rotary_frequencies'])
 bidx=t['actual_visual_bilinear_indices'];weights=t['actual_visual_bilinear_weights'];assert bidx.shape==weights.shape==(4,N) and bidx.min()>=0 and bidx.max()<2304 and weights.dtype==torch.float32 and (weights>=0).all() and torch.allclose(weights.sum(0),torch.ones(N),rtol=0,atol=torch.finfo(torch.float32).eps);assert torch.equal(position_weight[bidx],t['actual_visual_position_gather'])
 interpolated=(t['actual_visual_position_gather']*weights[:,:,None]).sum(0).bfloat16();position_control=t['actual_visual_patch_embeddings']+interpolated;max_visual_position_CPU_NPU=max(max_visual_position_CPU_NPU,float((position_control.float()-t['actual_visual_block0_input'].float()).abs().max()))
 assert t['actual_encoder'].shape==(N,1152) and t['actual_final_visual'].shape==(M,4096) and t['actual_prefill_embeds'].shape==(size,length,4096);image=inp['input_ids']==151655;assert image.sum()==M and torch.equal(t['actual_prefill_embeds'][image],t['actual_final_visual']) and torch.equal(t['actual_prefill_embeds'][~image],embed_weight[inp['input_ids'][~image]])
 for i in range(3):assert t['actual_deepstack_%d'%i].shape==(M,4096) and torch.equal(t['actual_deepstack_%d'%i],t['actual_deepstack_received_%d'%i])
 steps=g['steps'];tokens=g['generated_padded_token_ids'];assert 1<=len(steps)<=4096 and len(tokens)==size and all(len(v)==len(steps) for v in tokens);assert act['group']==group_no and act['indices']==indices and act['batch_size']==size and act['decoder_forwards']==len(steps) and act['module_call_counts']==g['activation_counts'] and act['device']==device and act['error'] is None
 for n,calls in act['module_call_counts'].items():assert n in acts and calls==(1 if '.visual.' in n else len(steps))
 assert set(act['module_call_counts'])==set(acts);prefix=inp['input_ids'];mask2=inp['attention_mask'];p4,delta=positions(inp);finished=[False]*size
 for n,step in enumerate(steps,1):
  total=length+n-1;qlen=length if n==1 else 1;pos=p4 if n==1 else p4[:,:,-1:]+n-1;assert step['step']==n and step['full_prefix']==tr(prefix) and step['input_ids']==(prefix if n==1 else prefix[:,-1:]).tolist() and step['attention_mask2']==mask2.tolist() and step['position_ids4']==pos.tolist() and step['pixels_present']==(n==1)
  if n==1:assert step['original_final_visual_injection_bitwise'] and step['deepstack_actual_additions']==[{'index':i,'NPU_unmodified_original_result_verified':True} for i in range(3)]
  logits=t['%04d_full_vocab_logits'%n];assert logits.dtype==torch.bfloat16 and logits.shape==(size,151936) and torch.isfinite(logits).all();argmax=logits.float().argmax(-1).tolist();chosen=[tokens[i][n-1] for i in range(size)];assert chosen==[151643 if finished[i] else argmax[i] for i in range(size)]
  for i,value in enumerate(chosen):finished[i]|=value in [151645,151643]
  cc,ss=independent_text_rope(pos,text_inv);diff=max(float((cc-t['%04d_text_cos'%n]).abs().max()),float((ss-t['%04d_text_sin'%n]).abs().max()));max_rope=max(max_rope,diff);assert diff<=torch.finfo(torch.bfloat16).eps
  mr=step['causal_mask']
  if mr is None:assert n>1 and mask2.all()
  else:
   mask=t['%04d_causal_mask'%n];assert tr(mask)==mr and mask.shape==(size,1,qlen,total);qidx=torch.arange(total-qlen,total)[:,None];kidx=torch.arange(total)[None,:];allowed=(qidx>=kidx)[None].expand(size,-1,-1)&mask2[:,None,:].bool()
   if mask.dtype==torch.bool:assert torch.equal(mask[:,0],allowed)
   else:assert torch.equal(mask[:,0],torch.where(allowed,torch.tensor(0,dtype=mask.dtype),torch.tensor(torch.finfo(mask.dtype).min,dtype=mask.dtype)))
  assert len(step['all36_attention'])==len(step['cache36'])==36
  for i,(a,c) in enumerate(zip(step['all36_attention'],step['cache36'])):assert a['layer']==c['layer']==i and a['mask']==mr and a['device']==c['device']==device and c['dtype']=='torch.bfloat16' and c['shape']==[size,8,total,128] and c['original_prefix_NPU_bitwise_unchanged']
  prefix=torch.cat([prefix,torch.tensor(chosen)[:,None]],1);mask2=torch.cat([mask2,torch.ones((size,1),dtype=torch.int64)],1)
 assert all(finished) or len(steps)==4096
 for slot,(i,e) in enumerate(zip(indices,entries)):
  raw=rust.decode(tokens[slot],skip_special_tokens=True).strip();first=next((j for j,v in enumerate(tokens[slot]) if v in [151645,151643]),None);stop=len(steps) if first is None else first+1;assert raw==g['raw_outputs'][slot]==e['raw_output'] and e['generation_token_ids']==tokens[slot] and e['first_EOS_index']==first and e['generation_count']==stop and e['truncated']==(stop>=4096) and e['group_slot']==slot and e['group_batch_size']==size and e['error'] is None and e['prompt']==p['input_records'][i]['prompt'] and e['actual_model_prompt']==p['input_records'][i]['model_prompt'] and e['empty']==(not raw) and e['invalid']==(independent_choice(raw) is None) and e['unsupported_original_head_token_IDs']==[v for v in tokens[slot] if v>=151669];allraw[i]=raw
 steps_total+=len(steps)
assert sorted(allraw)==list(range(count));correct=0;empty=[];invalid=[];truncated=[];per=[]
for i,(a,b,r) in enumerate(zip(ins,outs,details)):
 e=by[i];raw=allraw[i];choice=independent_choice(raw);prediction=raw if choice is None else choice;assert a['prompt']==r['prompt']==clean(data[i]['prompt']) and a['reference']==b['reference']==r['reference']==data[i]['reference'] and b['raw_output']==r.get('raw_output','')==raw and ('raw_output' in r or raw=='') and b['prediction']==r['prediction']==prediction and b['error'] is None and r['extra']['execution']==e;hit=int(choice is not None and choice==str(r['reference']).strip().upper());correct+=hit;per.append(hit)
 if e['empty']:empty.append(i)
 if e['invalid']:invalid.append(i)
 if e['truncated']:truncated.append(i)
score=correct/count;assert report['metrics']['Strict_MC_Accuracy']==report['summary']['current_score']==score and report['summary']['primary_metric']=='Strict_MC_Accuracy' and report['statistics']['error_count']==0 and report['statistics']['total_samples']==count and report['statistics']['batch_size']==10 and len(external)==2*len(groups)+6
probe=line['native_GELU_probe'];assert sha(probe['path'])==probe['sha256'] and probe['CPU_threads']==4
with np.load(probe['path'],allow_pickle=False) as z:
 x=torch.linspace(-8,8,20001).bfloat16();assert np.array_equal(x.view(torch.uint8).numpy(),z['input'])
 for mode in ['none','tanh']:assert np.array_equal(torch.nn.functional.gelu(x,approximate=mode).view(torch.uint8).numpy(),z['CPU_'+mode]) and torch.isfinite(torch.from_numpy(z['NPU_'+mode].copy()).view(torch.bfloat16)).all()
profile=json.loads(pathlib.Path(line['profiler_trace']).read_text());events=profile if isinstance(profile,list) else profile['traceEvents'];hw={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};kernels=[e for e in events if e.get('pid') in hw and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},'sample_Accuracy':per,'correct':correct,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'historical_record_reuse':reuse_schema_recheck,'inference_reused_for_empty_raw_schema_verifier_repair':reuse_schema_recheck,'report_top_level_empty_raw_omission_preserved_indices':[i for i,r in enumerate(details) if 'raw_output' not in r],'original200_legacy_and_strict_recomputed':{'legacy_Accuracy':.815,'same_old_raw_Strict_MC_Accuracy':.63,'original_explicit_empty_count':15,'different_metric_extraction_definition_not_hardware_loss':True},'all750_source_cast_actual_NPU_SHA_verified':True,'all_visual_and_3DeepStack_original_additions_NPU_verified':True,'original_true_size_groups_verified':groups,'group_histogram':dict(collections.Counter(map(len,groups))),'all36_cache_positions_masks_steps_verified':steps_total,'all_full151936_native_greedy_argmax_replayed':steps_total,'all267_original_untokenized_head_IDs_preserved':True,'actual_activation_module_count':67,'all67_activation_counts_verified':True,'native_RoPE_independent_FP64_libm_reference_BF16_max_abs':max_rope,'independent_RoPE_reference_disclosure':'Original CPU-parsed FP32 frequencies and integer positions; exact FP64 product rounded once to original FP32 angles, scalar FP64 libm only as CPU diagnostic, explicit BF16 nearest-even bits. Native NPU forward untouched, same original one-BF16-epsilon comparison bound.','native_visual_position_accumulation_CPU_NPU_max_abs':max_visual_position_CPU_NPU,'NPU_profiler_kernel_events':len(kernels),'external_evidence_sha256':external,'report_sha256':{q.name:sha(q) for q in folder.iterdir()},'protocol_file':kw['protocol_file'],'protocol_sha256':kw['protocol_sha256'],'precision_disclosure':line['precision_disclosure'],'pre_numerical_review_real3_not_approved':True}
assert 'native_numerical_review' in kw
if 'native_numerical_review' in kw:
 nrp=pathlib.Path(kw['native_numerical_review']);assert sha(nrp)==kw['native_numerical_review_sha256'];nr=json.loads(nrp.read_text());assert nr==line['native_numerical_review'] and nr['official_inference_remains_original_BF16_grouped_eager'] and nr['CPU_FP32_controls_diagnostic_only'];ev=dict(nr['evidence_sha256']);ev[str(nrp)]=sha(nrp)
 resource_path=pathlib.Path(nr['own_resource_integrity_receipt']);assert sha(resource_path)==nr['own_resource_integrity_receipt_SHA'];resource=json.loads(resource_path.read_text());assert resource['model_id']==p['model_id'] and resource['weight_path']==str(w)
 for n,h in resource['own_nonweight_resource_sha256'].items():assert pathlib.Path(n).name==n and sha(w/n)==h
 for path,digest in ev.items():assert pathlib.Path(path).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(path)==digest
 control=json.loads(pathlib.Path(nr['full_visual_numerical_audit']).read_text());unit=json.loads(pathlib.Path(nr['BF16_operator_audit']).read_text());assert control['official_inference_remains_original_BF16_grouped_eager'] and control['all351_source_BF16_values_and_FP32_control_CPU_NPU_SHA_equal'] and control['CPU_only_original_nn_GELU_modules_changed_for_tanh_diagnostic']==['merger.act_fn','deepstack_merger_list.0.act_fn','deepstack_merger_list.1.act_fn','deepstack_merger_list.2.act_fn']
 controls=json.loads(pathlib.Path(control['parameter_SHA_evidence']).read_text());assert set(controls)=={'CPU_BF16','CPU_FP32','NPU4_FP32'} and len(control32_expected)==351
 for kind,records in controls.items():
  assert len(records)==351 and len({r['key'] for r in records})==351
  for record in records:
   key=record['key'];expected16=expected['model.visual.'+key]['full_sha256'];target=expected16 if kind=='CPU_BF16' else control32_expected[key];assert record['original_source_BF16_sha256']==expected16 and record['actual_control_sha256']==record['same_BF16_rounded_source_control_sha256']==target and record['device']==('npu:4' if kind=='NPU4_FP32' else 'cpu')
 ca=decode_arrays(control['archive'],control['arrays']);ua=decode_arrays(unit['archive'],unit['array_records']);assert set(unit['parameter_SHA'])==set(control32_expected)
 for key,value in unit['parameter_SHA'].items():assert value==expected['model.visual.'+key]['full_sha256']
 assert unit['all351_CPU_NPU4_BF16_parameter_SHA_equal'] and unit['selected_layers']==[0,8,16,24,26] and unit['original_NPU3_and_diagnostic_NPU4_BF16_encoder_bitwise_equal'] and unit['original_NPU3_and_diagnostic_NPU4_BF16_encoder_max_difference']==0 and torch.equal(ua['actual_diagnostic_NPU4_full_encoder'],ua['original_NPU3_full_encoder']) and torch.equal(ua['original_NPU3_full_encoder'],ca['actual_NPU_original_BF16_encoder'])
 cases={'original_NPU_BF16_vs_CPU_original_erf':('actual_NPU_original_BF16_','CPU_original_BF16_erf_'),'original_NPU_BF16_vs_CPU_module_tanh_diagnostic':('actual_NPU_original_BF16_','CPU_diagnostic_BF16_tanh_'),'same_BF16_rounded_weights_pixels_FP32_NPU4_vs_CPU_original_erf':('NPU4_diagnostic_FP32_','CPU_diagnostic_FP32_erf_'),'same_BF16_rounded_weights_pixels_FP32_NPU4_vs_CPU_module_tanh_diagnostic':('NPU4_diagnostic_FP32_','CPU_diagnostic_FP32_tanh_')}
 for case,(left,right) in cases.items():
  for field,value in control['summaries'][case].items():
   a=ca[left+field].float();b=ca[right+field].float();diff=(a-b).abs();assert float(diff.max())==value['all3_max_abs'] and float(diff.mean())==value['all3_mean_abs'];off=0;grid=torch.tensor(json.loads(pathlib.Path(control['original_group_json']).read_text())['image_grid_thw']);sizes=grid.prod(-1).tolist() if field=='encoder' else (grid.prod(-1)//4).tolist()
   for length,part in zip(sizes,value['per_original_image']):
    cut=diff[off:off+length];assert {'max_abs':float(cut.max()),'mean_abs':float(cut.mean()),'RMSE':float(cut.square().mean().sqrt())}==part;off+=length
   assert off==diff.shape[0]
 for key,m in unit['same_input_operator_measurements'].items():
  a=ua[key+'_CPU'].float();b=ua[key+'_NPU'].float();diff=(a-b).abs();idx=int(diff.reshape(-1).argmax());assert fp(ua[key+'_same_input'])==m['input_SHA'] and float(diff.max())==m['max_abs'] and float(diff.mean())==m['mean_abs'] and float(diff.square().mean().sqrt())/max(float(a.square().mean().sqrt()),1e-30)==m['relative_RMSE'] and float(a.reshape(-1)[idx])==m['largest_difference_CPU_value'] and float(b.reshape(-1)[idx])==m['largest_difference_NPU_value']
 for source in [control['NPU4_FP32_profiler'],unit['profiler']]:
  prof=json.loads(pathlib.Path(source).read_text());events=prof if isinstance(prof,list) else prof['traceEvents'];hardware={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};assert sum(e.get('pid') in hardware and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL') for e in events)>100
 proof.update(all351_control_parameter_SHA_verified=True,all_static_control_array_SHA_and_measured_differences_independently_recomputed=True,original_NPU3_NPU4_BF16_full_encoder_bitwise_replayed=True,static_NPU4_controls_only_not_official_FP32_inference=True)
 proof.update(pre_numerical_review_real3_not_approved=False,native_numerical_review=nr,cosmos_reason2_numerical_evidence_sha256=ev,static_CPU_diagnostics_from_prior_real3_not_recomputed_per_report=True)
proof=json.loads(json.dumps(proof,ensure_ascii=False))
q=loadpath.parent/'independent_report_verification.json';assert not q.exists() or json.loads(q.read_text())==proof
if not q.exists():q.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
print(json.dumps(proof,ensure_ascii=False,indent=2))
