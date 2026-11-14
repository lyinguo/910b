"""Independent complete own-weight, actual-A100 HighNine serial1 and every-greedy-step audit."""
import pathlib,os,sys,json,collections,re,math
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig,Gemma3ForConditionalGeneration,GenerationConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from zelk12_A100_HighNine_independent_CPU_0913_20261106 import sha,fp,tr,clean,inputs,arrays
from idefics2_original_accuracy_reference_910b import original_accuracy_sample
torch.set_num_threads(4)
def check(x,r):assert tr(x)==r
def mapped(k):
 for old,new in [('language_model.model.','model.language_model.'),('language_model.lm_head.','lm_head.'),('vision_tower.vision_model.','model.vision_tower.'),('multi_modal_projector.','model.multi_modal_projector.')]:
  if k.startswith(old):return new+k[len(old):]
 raise AssertionError(k)
def independent_choice(raw):
 s=str(raw).strip()
 for rule in [r'<([A-E])>',r'\[([A-E])\]',r'\(([A-E])\)',r'\*\*([A-E])\*\*',r'([A-E])']:
  m=re.fullmatch(rule,s,re.I)
  if m:return m[1].upper()
 m=re.fullmatch(r'<([A-E])>\s*([A-E])\s*</([A-E])>',s,re.I)
 if m and len({c.upper() for c in m.groups()})==1:return m[1].upper()
 for rule in [r'<answer>\s*([A-E])\s*</answer>',r'\\boxed\{\s*([A-E])\s*\}']:
  a=re.findall(rule,s,re.I)
  if a:return a[-1].upper()
 prefixes=[r'(?i:(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*:?|:)\s*)',r'(?i:(?:correct\s+)?option\s+(?:is\s+)?\s*)',r'(?i:(?:choose|select|pick)\s+)']
 for i,prefix in enumerate(prefixes):
  a=re.findall(prefix+r'[\'"(\[]\s*([A-Ea-e])\s*\.?\s*[\'"\])]',s)
  if a:return a[-1].upper()
  a=re.findall(prefix+(r'([A-Ea-e])' if i==1 else r'([A-E])')+r'(?=[\s.,:;!?]|$)',s)
  if a:return a[-1].upper()
 m=re.match(r'^([A-Ea-e])\s*[.)](?=\s|$)',s)
 if m:return m[1].upper()
 a=re.findall(r'(?m)^\s*([A-Ea-e])\s*$',s);return a[-1].upper() if a else None
assert independent_choice('a sample') is None and independent_choice('A sample of an object.') is None
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in [3,200]
assert {q.name for q in folder.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];assert sha(kw['protocol_file'])==kw['protocol_sha256'];p=json.loads(pathlib.Path(kw['protocol_file']).read_text());wp=pathlib.Path(p['weight_path']);line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];device='npu:'+str(kw['npu_devices'][0])
assert cfg['model']['name']==p['model_id']=='zelk12/MT-Gen1-gemma-3-12B' and cfg['model']['model_path']==str(wp) and folder.parent.parent.name=='zelk12_MT-Gen1-gemma-3-12B' and folder.parent.name==cfg['task_type']=='image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['algorithm']['batch_size']==1 and cfg['algorithm']['adapter']=='universal_eval/algorithm/native_explicit_mc_v2.py:NativeExplicitMCV2' and cfg['algorithm']['kwargs']=={'extract_answer':True} and cfg['metrics']['override']==['Strict_MC_Accuracy'] and cfg['metrics']['adapters']==['universal_eval/algorithm/tinyllava_strict_mc.py:StrictMCAccuracy'] and cfg['metrics']['include_failed_predictions']
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True} and cfg['data']['preprocess'] and not cfg['data']['filter_empty'] and not cfg['data']['apply_template'] and not cfg['data']['download'] and not cfg['model']['download']
assert line['protocol']=='zelk12_actual_A100_HighNine_HF_BF16_serial1_generation' and line['protocol_sha256']==kw['protocol_sha256'] and line['runtime_dtype']=='BF16' and line['weight_tensor_count']==1065 and line['all_expected_parameters_loaded'] and line['original_tied_output_head_preserved'] and line['max_new_tokens']==2048 and line['original_algorithm_batch_size']==1 and line['original_two_BOS_preserved'] and line['padding_side']=='right' and line['own_template_explicit_injection']==p['own_tokenizer_template'] and len(line['own_template_explicit_injection'])==1532 and set(line['module_placements'].values())=={device} and line['HF32_enabled']=={'conv':False,'matmul':False}
for src,digest in dict(p['runtime_sha256'],**p['historical_sources'],**p['evidence_sha256']).items():assert sha(src)==digest
for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(wp/name)==digest
assert not (wp/'generation_config.json').exists();mc=AutoConfig.from_pretrained(wp,local_files_only=True)
mc._attn_implementation='sdpa';mc.text_config._attn_implementation='sdpa';mc.vision_config._attn_implementation='eager'
with torch.device('meta'):meta=Gemma3ForConditionalGeneration(mc)
assert line['actual_A100_attention']=={'text':'sdpa','vision':'eager'}
meta.generation_config=GenerationConfig.from_dict(p['original_resolved_generation_config'])
gen,_=meta._prepare_generation_config(None,max_new_tokens=2048,do_sample=False);assert gen.to_dict()==line['original_resolved_generation_config']==p['original_resolved_generation_config'] and gen.num_beams==1 and gen.use_cache and not gen.do_sample and gen.max_new_tokens==2048 and gen.min_new_tokens is None and gen.repetition_penalty==1 and gen.forced_bos_token_id is gen.forced_eos_token_id is None and gen.bad_words_ids is None and not gen.no_repeat_ngram_size
params=dict(meta.named_parameters(remove_duplicate=False));assert len(dict(meta.named_parameters()))==1065;expected_act={n:type(m).__name__ for n,m in meta.named_modules() if type(m).__name__=='GELUTanh'};assert len(expected_act)==75 and line['activation_modules']==expected_act
loading_path=pathlib.Path(line['weight_loading_evidence']);loading=json.loads(loading_path.read_text());loaded={r['source_key']:r for r in loading};expected={r['source_key']:r for r in p['source_tensor_records']};assert len(loading)==len(expected)==1065;hashes={}
for shard,header in p['weight_headers'].items():
 assert (wp/shard).stat().st_size==header['bytes'] and sha(wp/shard)==header['file_sha256']
 with safe_open(str(wp/shard),framework='pt',device='cpu') as f:
  for key in f.keys():
   x=f.get_tensor(key);r=loaded[key];e=expected[key];name=mapped(key);assert x.dtype==torch.bfloat16 and str(x.dtype)==r['source_dtype']==e['source_dtype'] and name==r['target_key']==e['target_key'] and list(x.shape)==r['shape']==list(params[name].shape) and fp(x)==r['source_sha256']==e['source_sha256']==r['original_BF16_cast_sha256']==e['original_BF16_cast_sha256']==r['actual_NPU_sha256'] and r['device']==device and r['runtime_dtype']=='torch.bfloat16' and not r['nonzero_underflow'] and not r['original_BF16_roundtrip_changed_elements'];hashes[name]=fp(x)
for names in p['native_alias_groups']:assert len({hashes[n] for n in names if n in hashes})==1
rope=Gemma3RotaryEmbedding(mc.text_config)
for n,v in rope.named_buffers():check(v,line['text_rotary_buffers'][n]);assert v.dtype==torch.float32
check(torch.tensor(math.sqrt(mc.text_config.hidden_size),dtype=torch.bfloat16),line['original_embedding_scale'])
data=[json.loads(x) for x in pathlib.Path(p['dataset_path']).read_text().splitlines()];old=json.loads(pathlib.Path(p['reference_report']).read_text());assert len(data)==4329 and len(old['detailed_results'])==200 and sum(original_accuracy_sample(r['prediction'],r['reference']) for r in old['detailed_results'])/200==p['original_report_Accuracy']==old['metrics']['Accuracy']
for i,r in enumerate(old['detailed_results']):assert r['prompt']==clean(data[i]['prompt']) and r['reference']==data[i]['reference'] and sha(p['input_identity_records'][i]['media_path'])==p['reference_first200_image_sha256'][p['input_identity_records'][i]['media_path']]
ins=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];trace_path=pathlib.Path(line['per_sample_evidence']);trace=[json.loads(x) for x in trace_path.read_text().splitlines()];acts_path=pathlib.Path(line['activation_execution_trace']);acts=[json.loads(x) for x in acts_path.read_text().splitlines()]
assert len(ins)==len(outs)==len(details)==len(trace)==count
for rows in [ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
assert sorted(e['index'] for e in trace)==list(range(count));trace_by={e['index']:e for e in trace};external={str(q):sha(q) for q in [trace_path,acts_path,loading_path,pathlib.Path(line['profiler_trace']),pathlib.Path(line['native_GELU_probe']['path'])]}
# Original actual A100 HighNine pipeline batch_size=1, independent serial request sequence.
expected_groups=[[i] for i in range(count)]
assert len(acts)==len(expected_groups);totalsteps=0;max_rope=0.;sample_raw={};all_indices=[]
for group_id,(indices,act) in enumerate(zip(expected_groups,acts),1):
 entries=[trace_by[i] for i in indices];path=pathlib.Path(entries[0]['group_execution']);g=json.loads(path.read_text());assert sha(path)==entries[0]['group_execution_sha256'] and all(e['group_execution']==str(path) and e['group_execution_sha256']==sha(path) for e in entries) and g['group']==group_id and g['indices']==indices and g['batch_size']==len(indices) and g['error'] is None and g['actual_visual_injection_verified'] and g['actual_vision_calls']==g['actual_connector_calls']==1
 external[str(path)]=sha(path);assert sha(g['archive'])==g['archive_sha256'];external[g['archive']]=sha(g['archive']);t=arrays(g['archive'],g['tensor_records']);own,texts,rust=inputs(p,indices);assert g['rendered_texts']==texts and g['input_ids']==own['input_ids'].tolist() and g['attention_mask']==own['attention_mask'].tolist() and g['token_type_ids']==own['token_type_ids'].tolist()
 for name,value in own.items():check(value,g['CPU_inputs'][name])
 size,length=own['input_ids'].shape;assert t['actual_encoder'].shape==(size,4096,1152) and t['actual_connector'].shape==(size,256,3840) and torch.equal(t['actual_vision_pixels'],own['pixel_values'].bfloat16()) and torch.equal(t['actual_vision_positions'],torch.arange(4096)[None]);image=own['input_ids']==262144;assert image.sum()==256*size and torch.equal(t['actual_injected_visual'],t['actual_connector'].reshape(-1,3840)) and torch.equal(t['actual_prefill_inputs_embeds'][image],t['actual_injected_visual'])
 steps=g['steps'];tokens=g['generated_padded_token_ids'];assert 1<=len(steps)<=2048 and len(tokens)==size and all(len(v)==len(steps) for v in tokens);assert act['group']==group_id and act['batch_size']==size and act['decoder_forwards']==len(steps) and act['error'] is None and act['device']==device and act['dtype']=='torch.bfloat16' and act['module_call_counts']==g['native_activation_call_counts'] and set(act['module_call_counts'])==set(expected_act)
 for name,calls in act['module_call_counts'].items():assert calls==(1 if '.vision_tower.' in name else len(steps))
 finished=[False]*size;prefix=own['input_ids'];active_mask=own['attention_mask'];vision_ids=own['input_ids']==262144
 for n,step in enumerate(steps,1):
  assert step['step']==n and step['full_generation_prefix']==tr(prefix) and step['attention_mask_2D']==active_mask.tolist() and step['pixel_values_present']==(n==1)
  actual_ids=prefix if n==1 else prefix[:,-1:];assert step['input_ids']==actual_ids.tolist();pos=active_mask.cumsum(-1)-1;pos.masked_fill_(active_mask==0,0);pos=pos if n==1 else pos[:,-1:];assert step['position_ids']==pos.tolist()
  logits=t['%04d_logits'%n];check(logits,step['full_vocab_logits']);assert logits.dtype==torch.bfloat16 and logits.shape==(size,262208) and torch.isfinite(logits).all();chosen=logits.float().argmax(-1).tolist();actual=[tokens[i][n-1] for i in range(size)];assert actual==[0 if finished[i] else chosen[i] for i in range(size)]
  for i,v in enumerate(actual):finished[i]=finished[i] or v in gen.eos_token_id
  total=length+n-1;qlen=length if n==1 else 1
  for kind in ['full_attention','sliding_attention']:
   c,s=rope(torch.zeros((size,qlen,3840),dtype=torch.bfloat16),pos,kind);dc=float((c-t['%04d_%s_cos'%(n,kind)]).abs().max());ds=float((s-t['%04d_%s_sin'%(n,kind)]).abs().max());max_rope=max(max_rope,dc,ds);assert max(dc,ds)<=torch.finfo(torch.bfloat16).eps
   mr=step['masks'][kind]
   if mr is None:
    assert n>1;K=min(total,1024) if kind=='sliding_attention' else total;assert active_mask[:,-K:].all()
   else:
    mask=t['%04d_%s_mask'%(n,kind)];check(mask,mr);assert mask.shape[:3]==(size,1,qlen);K=mask.shape[-1];offset=total-K;q=torch.arange(total-qlen,total)[:,None];k=torch.arange(offset,total)[None,:];allowed=(q>=k)[None].expand(size,-1,-1).clone()
    if n==1:allowed|=vision_ids[:,:,None]&vision_ids[:,None,:]
    if kind=='sliding_attention':allowed&=(q-k<1024)
    allowed&=active_mask[:,None,offset:total].bool();assert allowed.shape==mask[:,0].shape
    if mask.dtype==torch.bool:assert torch.equal(mask[:,0],allowed)
    else:assert torch.equal(mask[:,0],torch.where(allowed,torch.tensor(0.,dtype=mask.dtype),torch.tensor(torch.finfo(mask.dtype).min,dtype=mask.dtype)))
  assert len(step['all48_attention'])==len(step['cache'])==48
  for i,(a,c) in enumerate(zip(step['all48_attention'],step['cache'])):
   kind=mc.text_config.layer_types[i];N=min(total,1023) if kind=='sliding_attention' else total;assert a['layer']==c['layer']==i and a['layer_type']==kind and a['mask']==step['masks'][kind] and a['scaling']==mc.text_config.query_pre_attn_scalar**-.5 and a['sliding_window']==(1024 if kind=='sliding_attention' else None) and c['shape']==c['value_shape']==[size,8,N,256] and c['cumulative_length']==total and c['dtype']=='torch.bfloat16' and c['device']==device
  prefix=torch.cat([prefix,torch.tensor(actual)[:,None]],1);active_mask=torch.cat([active_mask,torch.ones((size,1),dtype=torch.int64)],1)
 assert all(finished) or len(steps)==2048
 for slot,(i,e) in enumerate(zip(indices,entries)):
  raw=rust.decode(tokens[slot],skip_special_tokens=True).strip();assert raw==g['raw_outputs'][slot]==e['raw_output'] and e['generation_token_ids']==tokens[slot] and e['group_slot']==slot and e['group_batch_size']==size and e['error'] is None and e['index']==i and e['prompt']==p['input_identity_records'][i]['original_prompt'] and e['actual_original_model_prompt']==p['input_identity_records'][i]['actual_original_model_prompt_after_instruction']
  first=next((j for j,v in enumerate(tokens[slot]) if v in gen.eos_token_id),None);stop=first+1 if first is not None else len(steps);assert e['first_EOS_index']==first and e['active_generation_token_ids']==tokens[slot][:stop] and e['generation_count']==stop and e['truncated']==(stop>=2048 and first is None) and e['empty']==(not raw) and e['invalid']==(independent_choice(raw) is None);sample_raw[i]=raw
 totalsteps+=len(steps);all_indices+=indices
assert sorted(all_indices)==list(range(count)) and len(set(all_indices))==count
correct=legacy=0;empty=[];invalid=[];truncated=[];sample_scores=[]
for i,(a,b,r) in enumerate(zip(ins,outs,details)):
 e=trace_by[i];raw=sample_raw[i];choice=independent_choice(raw);prediction=choice if choice is not None else raw;assert a['prompt']==r['prompt']==clean(data[i]['prompt']) and a['reference']==b['reference']==r['reference']==data[i]['reference'] and r['raw_output']==b['raw_output']==raw and r['prediction']==b['prediction']==prediction and b['error'] is None and r['extra']['execution']==e
 hit=int(choice is not None and choice==str(r['reference']).strip().upper());correct+=hit;sample_scores.append(hit);legacy+=original_accuracy_sample(prediction,r['reference'])
 if e['empty']:empty.append(i)
 if e['invalid']:invalid.append(i)
 if e['truncated']:truncated.append(i)
score=correct/count;assert report['metrics']['Strict_MC_Accuracy']==report['summary']['current_score']==score and report['summary']['primary_metric']=='Strict_MC_Accuracy' and report['statistics']['error_count']==0 and report['statistics']['total_samples']==count and report['statistics']['batch_size']==1 and len(external)==2*len(expected_groups)+5
probe=line['native_GELU_probe'];assert sha(probe['path'])==probe['sha256'] and probe['device']==device;torch.set_num_threads(probe['CPU_threads'])
with np.load(probe['path'],allow_pickle=False) as z:
 x=torch.linspace(-8,8,20001).bfloat16();assert np.array_equal(z['input_BF16_bytes'],x.view(torch.uint8).numpy())
 for mode in ['none','tanh']:
  cpu=torch.nn.functional.gelu(x,approximate=mode);native=torch.from_numpy(z['NPU_'+mode+'_BF16_bytes'].copy()).view(torch.bfloat16);assert torch.isfinite(native).all() and np.array_equal(z['CPU_'+mode+'_BF16_bytes'],cpu.view(torch.uint8).numpy()) and float((native-cpu).abs().max())==probe[mode]['max_abs_CPU_same_declared_GELU']
profile=json.loads(pathlib.Path(line['profiler_trace']).read_text());events=profile if isinstance(profile,list) else profile['traceEvents'];hw={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};kernels=[e for e in events if e.get('pid') in hw and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},'sample_Accuracy':sample_scores,'correct':correct,'original_legacy_Accuracy_diagnostic_only':legacy/count,'original200_legacy_Accuracy_independently_recomputed':p['original_report_Accuracy'],'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'historical_record_reuse':False,'all1065_source_cast_actual_NPU_SHA_verified':True,'all_expected_visual_features_actually_injected':True,'original_true_size_groups_verified':expected_groups,'group_size_histogram':dict(collections.Counter(map(len,expected_groups))),'all200_CPU_Jinja_Rust_pixels_original_serial_right_padding_twoBOS_replayed':count,'all48_native_positions_masks_cache_all_steps_verified':totalsteps,'all_full_vocab_native_greedy_argmax_replayed':totalsteps,'actual_activation_module_count':75,'all75_activation_module_counts_verified':True,'native_RoPE_CPU_NPU_max_abs':max_rope,'full_vocab_size':262208,'NPU_profiler_kernel_events':len(kernels),'external_evidence_sha256':external,'report_sha256':{q.name:sha(q) for q in folder.iterdir()},'protocol_file':kw['protocol_file'],'protocol_sha256':kw['protocol_sha256'],'precision_disclosure':line['precision_disclosure'],'pre_numerical_review_real3_not_approved':True,'native_padding_position0_exact_official_generation_formula_verified':True,'all_active_token_positions_exact_cumsum_minus1':True,'inference_reused_for_verifier_repair':'--reuse-for-verifier-repair' in sys.argv}
if 'native_numerical_review' in kw:
 q=pathlib.Path(kw['native_numerical_review']);assert sha(q)==kw['native_numerical_review_sha256'];nr=json.loads(q.read_text());assert nr==line['native_numerical_review'] and nr['official_inference_remains_original_A100_HF_BF16_serial1_generation'] and nr['CPU_FP32_controls_diagnostic_only'] and nr['all_control_CPU_NPU_parameter_SHA_equal']
 ev=dict(nr['evidence_sha256']);ev[str(q)]=sha(q)
 for path,digest in ev.items():assert pathlib.Path(path).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(path)==digest
 proof.update(pre_numerical_review_real3_not_approved=False,native_numerical_difference_retained_and_disclosed=True,native_numerical_review=nr,zelk12_numerical_diagnostic_evidence_sha256=ev,static_CPU_diagnostics_from_prior_real3_not_recomputed_per_report=True)
from audit_Zelk12_A100_HighNine_raw_and_snapshots_0913_20261106 import audit_raw_and_snapshots
configured_rel=cfg['model']['adapter'].rsplit(':',1)[0];assert not pathlib.PurePosixPath(configured_rel).is_absolute() and '..' not in pathlib.PurePosixPath(configured_rel).parts
configured_source=pathlib.Path('/root/projects/comperror')/configured_rel;configured_snapshot=loading_path.parent/'source_snapshot'/configured_source.relative_to('/')
assert sha(configured_snapshot)==sha(configured_source),'Configured own adapter source snapshot must match actual source'
closure=audit_raw_and_snapshots(folder,external)
proof['external_evidence_sha256'].update(closure['external_SHA256']);proof['external_evidence_sha256'][closure['manifest']]=closure['manifest_SHA256']
proof.update(full_new_raw_CANN_and_source_snapshot_closure=closure['manifest'],full_raw_source_manifest_SHA256=closure['manifest_SHA256'],full_raw_source_file_count=closure['file_count'],new_uncovered_raw_and_source_files_hashed=closure['newly_hashed'],primary_new_run_evidence_hashes_reused_without_reinference=closure['primary_hashes_reused'])
relative=abs(score-p['original_report_Accuracy'])/abs(p['original_report_Accuracy']);proof.update(relative_error_to_A100_table_D=relative,formal_alignment_within_5pct=relative<=.05,actual_A100_original_serial1_verified=True,actual_A100_text_SDPA_vision_eager_verified=True)
dest=loading_path.parent/'independent_report_verification.json';assert not dest.exists() or json.loads(dest.read_text())==proof
if not dest.exists():dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
print(json.dumps(proof,ensure_ascii=False,indent=2))
