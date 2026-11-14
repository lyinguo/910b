"""Independent own Mllama pixels/Jinja/Rust, source SHA, masks/cache and logits."""
import os,pathlib,sys,json,re,collections,math
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
from PIL import Image
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from alpindale_mllama_A100_independent_cpu_0910_20261104 import sha,fp,tr,inputs,clean
from egm8b_original_independent_cpu_910b import decode_arrays
from strict_mc_independent_reference_910b import independent_choice
from idefics2_original_accuracy_reference_910b import original_accuracy_sample
from cosmos_reason2_stable_rope_reference_910b import _bf16_rne
from mllama11b_masks_rope_reference_910b import original_vision_mask
torch.set_num_threads(4)
R=pathlib.Path(__file__).resolve().parent.parent
assert len(sys.argv)==3;reuse_placement_schema=False;folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in (3,200)
assert {q.name for q in folder.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
report=json.loads((folder/'eval_report.json').read_text());assert report['statistics']['total_samples']==count and report['statistics']['error_count']==0,'Actual sample execution errors cannot be accepted';cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];assert sha(kw['protocol_file'])==kw['protocol_sha256']
p=json.loads(pathlib.Path(kw['protocol_file']).read_text());w=pathlib.Path(p['weight_path']);line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];assert len(kw['npu_devices'])==1;vd=ld='npu:'+str(kw['npu_devices'][0])
assert cfg['model']['name']==p['model_id']=='alpindale/Llama-3.2-11B-Vision-Instruct' and cfg['model']['model_path']==str(w) and folder.parent.parent.name=='alpindale_Llama-3.2-11B-Vision-Instruct' and folder.parent.name==cfg['task_type']=='image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['algorithm']=={'name':'native_explicit_mc_v2','adapter':'universal_eval/algorithm/native_explicit_mc_v2.py:NativeExplicitMCV2','batch_size':10,'kwargs':{'extract_answer':True}}
assert cfg['metrics']=={'override':['Strict_MC_Accuracy'],'adapters':['universal_eval/algorithm/tinyllava_strict_mc.py:StrictMCAccuracy'],'include_failed_predictions':True}
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert not cfg['data']['download'] and not cfg['model']['download'] and cfg['data']['preprocess'] and not cfg['data']['filter_empty'] and not cfg['data']['apply_template']
assert line['protocol']=='Alpindale_actual_A100_FP16load_then_BF16_SDPA_serial1_2048_single_BOS' and line['weight_tensor_count']==906 and line['all_expected_parameters_loaded'] and line['runtime_dtype']=='BF16' and line['original_max_new_tokens']==2048 and line['original_min_new_tokens'] is None and line['original_outer_batch_metadata']==10 and line['original_untied_head128256_embedding128264_tokenizer128257_preserved'] and line['original_single_BOS128000_preserved'] and line['actual_attention']=={'root':'sdpa','text':'sdpa','vision':'sdpa'} and line['HF32_enabled']=={'conv':False,'matmul':False} and line['original_FP16_parameter_load_then_BF16_preserved'] and line['original_roundtrip_changed_elements']==4186034 and line['original_FP16_nonzero_underflows']==1107716
assert set(line['module_placements'].values())=={vd,ld}
for name,dev in line['module_placements'].items():assert dev==(vd if name=='model.vision_model' or name.startswith('model.vision_model.') or name=='model.multi_modal_projector' else ld)
assert sha(p['CPU200_protocol'])==p['CPU200_protocol_sha256'] and p['actual_A100_CPU200_and_independent_R_CPU200_passed'] and p['actual_batch_size']==1 and p['original_outer_batch_metadata']==10 and len(p['input_records'])==200
for n,h in p['runtime_and_original_source_sha256'].items():assert sha(n)==h
for n,h in p['own_resource_sha256'].items():assert pathlib.Path(n).name==n and sha(w/n)==h
from transformers import MllamaConfig,MllamaForConditionalGeneration,GenerationConfig
import copy
mc=MllamaConfig.from_dict(copy.deepcopy(p['original_A100_config']));mc.dtype=torch.float16;mc.text_config.dtype=torch.float16;mc.vision_config.dtype=torch.float16;mc._attn_implementation='sdpa';mc.text_config._attn_implementation='sdpa';mc.vision_config._attn_implementation='sdpa'
with torch.device('meta'):meta=MllamaForConditionalGeneration._from_config(mc,torch_dtype=torch.float16)
meta.generation_config=GenerationConfig.from_dict(copy.deepcopy(p['original_A100_generation_config_file']));meta.generation_config.cache_implementation=None;gen,_=meta._prepare_generation_config(None,max_new_tokens=2048,do_sample=False,temperature=0.0)
assert gen.to_dict()==p['original_resolved_generation']==line['generation_config'] and gen.use_cache and gen.num_beams==1 and not gen.do_sample and gen.repetition_penalty==1 and gen.no_repeat_ngram_size==0 and gen.forced_bos_token_id is gen.forced_eos_token_id is None and gen.eos_token_id==[128001,128008,128009] and gen.pad_token_id==128004 and gen.min_new_tokens is None
params=dict(meta.named_parameters());acts={n:type(m).__name__ for n,m in meta.named_modules() if n.endswith(('act_fn','activation_fn'))}
assert len(params)==906 and acts==line['activation_modules']==p['native_activation_modules'] and len(acts)==80 and meta.lm_head.weight is not meta.model.language_model.embed_tokens.weight
loadpath=pathlib.Path(line['weight_loading_evidence']);loaded=json.loads(loadpath.read_text());lookup={x['source_key']:x for x in loaded};expected={x['source_key']:x for x in p['all906_source_to_native_semantic_name_shape_mapping_verified']};assert len(lookup)==len(loaded)==len(expected)==906
embed=None;source_vision={}
for shard,head in p['all_source_file_sha256'].items():
    assert sha(w/shard)==head['sha256'] and (w/shard).stat().st_size==head['bytes']
    with safe_open(str(w/shard),framework='pt',device='cpu') as f:
        for key in f.keys():
            t=f.get_tensor(key);r=lookup[key];e=expected[key];name=e['target_key']
            assert t.dtype==torch.bfloat16 and list(t.shape)==e['shape']==list(params[name].shape)==r['shape']
            half=t.half();final=half.bfloat16();assert fp(t)==e['full_sha256']==r['source_sha256'] and fp(half)==e['original_CPU_FP16_load_sha256']==r['original_FP16_load_sha256'] and fp(final)==e['original_CPU_BF16_final_sha256']==r['original_BF16_final_sha256']==r['actual_NPU_sha256'] and r['device']==ld and r['runtime_dtype']=='torch.bfloat16' and r['target_key']==name;assert int((t!=final).sum())==e['original_roundtrip_changed_elements']==r['original_roundtrip_changed_elements'] and int(((t!=0)&(half==0)).sum())==e['original_FP16_nonzero_underflows']==r['original_FP16_nonzero_underflows']
            if name=='model.language_model.embed_tokens.weight':embed=final.clone()
            if name.startswith(('model.vision_model.','model.multi_modal_projector.')):source_vision[name]={'source_BF16':fp(t),'FP16_load':fp(half),'BF16':fp(final),'FP32_widen_diagnostic_only':fp(final.float())}
assert embed.shape==(128264,4096) and params['lm_head.weight'].shape==(128256,4096) and len(source_vision)==511
inv=1/(500000.**(torch.arange(0,128,2,dtype=torch.int64).float()/128));wavelen=2*math.pi/inv;smooth=(8192/wavelen-1)/3
scaled=torch.where(wavelen>8192,inv/8,inv);inv=torch.where((wavelen>=2048)&(wavelen<=8192),(1-smooth)*scaled/8+smooth*scaled,scaled)
assert fp(inv)=='4ae228093081627907bb1abceaa696354a9d187a0202a20492a1105c1ade70e3'
assert line['analytic_F32_buffers']=={'model.language_model.rotary_emb.inv_freq':tr(inv),'model.language_model.rotary_emb.original_inv_freq':tr(inv)}
data=[json.loads(l) for l in pathlib.Path(p['dataset_path']).read_text().splitlines()];assert sha(p['dataset_path'])==p['dataset_SHA'] and len(data)==4329
old=json.loads(pathlib.Path(p['reference_report']).read_text());assert sha(p['reference_report'])==p['reference_report_SHA'] and len(old['detailed_results'])==200
assert sum(original_accuracy_sample(r['prediction'],r['reference']) for r in old['detailed_results'])/200==old['metrics']['Accuracy']==.785
old_outputs=old['detailed_results'];assert len(old_outputs)==200 and [x['index'] for x in old_outputs]==list(range(200))
assert sum(independent_choice(r['raw_output'])==str(r['reference']).strip().upper() for r in old_outputs)/200==.770
for i,r in enumerate(old['detailed_results']):
    assert r['prompt']==p['input_records'][i]['prompt']==clean(data[i]['prompt']) and r['reference']==data[i]['reference'] and sha(p['input_records'][i]['media_path'])==p['input_records'][i]['media_sha256']
    assert old_outputs[i]['prediction']==r['prediction'] and old_outputs[i]['raw_output']==r.get('raw_output',r.get('extra',{}).get('raw_output'))
ins=[json.loads(l) for l in (folder/'sample_inputs.jsonl').read_text().splitlines()];outs=[json.loads(l) for l in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results']
tracepath=pathlib.Path(line['per_sample_evidence']);trace=[json.loads(l) for l in tracepath.read_text().splitlines()];actpath=pathlib.Path(line['activation_execution_trace']);acttrace=[json.loads(l) for l in actpath.read_text().splitlines()]
assert len(ins)==len(outs)==len(details)==len(trace)==count
for rows in [ins,outs,details]:assert [x['index'] for x in rows]==list(range(count))
assert sorted(r['index'] for r in trace)==list(range(count));by={r['index']:r for r in trace};groups=[]
for start in range(0,count,1):
    group=collections.OrderedDict()
    for i in range(start,min(start+1,count)):
        with Image.open(p['input_records'][i]['media_path']) as im:key=(im.width//64,im.height//64)
        group.setdefault(key,[]).append(i)
    groups.extend(group.values())
assert len(acttrace)==len(groups)
external={str(q):sha(q) for q in [tracepath,actpath,loadpath,*[pathlib.Path(x) for x in line['profiler_traces'].values()],pathlib.Path(line['native_GELU_probe']['path']),loadpath.parent/'generation_progress.jsonl']}
raws={};total_steps=0;max_rope=0.;rust=Tokenizer.from_file(str(w/'tokenizer.json'))
for group_no,(indices,act) in enumerate(zip(groups,acttrace),1):
    entries=[by[i] for i in indices];q=pathlib.Path(entries[0]['group_execution']);g=json.loads(q.read_text())
    assert g['group']==group_no and g['indices']==indices and g['batch_size']==len(indices) and g['error'] is None and g['vision_calls']==1
    assert all(e['group_execution']==str(q) and e['group_execution_sha256']==sha(q) for e in entries)
    assert sha(g['archive'])==g['archive_sha256'];external[str(q)]=sha(q);external[g['archive']]=sha(g['archive']);t=decode_arrays(g['archive'],g['arrays'])
    inp,texts,geometry=inputs(p,indices);size,length=inp['input_ids'].shape
    assert texts==g['rendered'] and {k:tr(v) for k,v in inp.items()}==g['CPU_inputs'] and inp['input_ids'].tolist()==g['input_ids'] and inp['attention_mask'].tolist()==g['attention_mask']
    assert torch.equal(t['actual_vision_original_F32_pixels'],inp['pixel_values']) and torch.equal(t['actual_aspect_ratio_ids'],inp['aspect_ratio_ids']) and torch.equal(t['actual_aspect_ratio_mask'],inp['aspect_ratio_mask'])
    assert torch.equal(t['actual_patch_input_BF16'],inp['pixel_values'].reshape(-1,3,560,560).bfloat16())
    assert t['actual_vision_output'].shape==(size,1,4,1601,7680) and t['actual_projected_vision'].shape==(size,1,4,1601,4096) and torch.isfinite(t['actual_vision_output']).all() and torch.isfinite(t['actual_projected_vision']).all()
    assert torch.equal(t['actual_prefill_text_embeds'],embed[inp['input_ids']])
    assert torch.equal(t['actual_vision_attention_mask'],original_vision_mask(inp['aspect_ratio_mask']))
    step_rows=g['steps'];tokens=g['generated_padded_token_ids'];assert len(tokens)==size and all(len(z)==len(step_rows) for z in tokens) and 1<=len(step_rows)<=2048
    assert act['group']==group_no and act['indices']==indices and act['decoder_forwards']==len(step_rows) and act['module_call_counts']==g['activation_counts'] and act['vision_device']==vd and act['language_device']==ld and act['error'] is None
    for n,c in g['activation_counts'].items():assert c==(1 if '.vision_model.' in n else len(step_rows))
    assert set(g['activation_counts'])==set(acts)
    prefix=inp['input_ids'].clone();pos0=inp['attention_mask'].cumsum(-1)-1;pos0.masked_fill_(inp['attention_mask']==0,0)
    for n,s in enumerate(step_rows,1):
        assert s['step']==n and s['full_prefix']==tr(prefix) and s['input_ids']==(inp['input_ids'] if n==1 else torch.tensor(tokens)[:,n-2:n-1]).tolist()
        pos=pos0 if n==1 else pos0[:,-1:]+n-1;assert s['actual_text_position_ids']==pos.tolist() and s['position_ids']==pos.tolist()
        attmask=inp['attention_mask'] if n==1 else torch.cat([inp['attention_mask'],torch.ones((size,n-1),dtype=torch.int64)],dim=1);assert s['attention_mask2']==attmask.tolist()
        angles=(pos.numpy().astype(np.float64)[...,None]*inv.numpy().astype(np.float64)).astype(np.float32);angles=np.concatenate([angles,angles],axis=-1)
        c=_bf16_rne(np.fromiter((math.cos(float(x)) for x in angles.ravel()),dtype=np.float64,count=angles.size).reshape(angles.shape));si=_bf16_rne(np.fromiter((math.sin(float(x)) for x in angles.ravel()),dtype=np.float64,count=angles.size).reshape(angles.shape))
        for name,ref in [('cos',c),('sin',si)]:
            a=t['%04d_text_%s'%(n,name)];delta=float((a.float()-ref.float()).abs().max());assert a.shape==ref.shape and a.dtype==torch.bfloat16 and delta<=torch.finfo(torch.bfloat16).eps;max_rope=max(max_rope,delta)
        cross4=inp['cross_attention_mask'] if n==1 else torch.cat([inp['cross_attention_mask'],inp['cross_attention_mask'][:,-1:].expand(-1,n-1,-1,-1)],dim=1)
        assert torch.equal(t['%04d_original_cross_attention_mask4'%n],cross4) and s['cross_attention_mask4']==tr(cross4)
        binary=cross4.repeat_interleave(1601,dim=3).reshape(size,-1,6404)[:,None];rows=binary.any(-1,keepdim=True).bfloat16();cross=torch.where(binary.bool(),torch.tensor(0,dtype=torch.bfloat16),torch.tensor(torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16))*rows
        if n>1:cross=cross[:,:,-1:];rows=rows[:,:,-1:]
        assert torch.equal(t['%04d_expanded_cross_attention_mask'%n],cross) and torch.equal(t['%04d_full_text_row_mask'%n],rows) and torch.equal(t['%04d_cross_mask'%n],cross)
        qlen=length if n==1 else 1;klen=length+n-1;query=torch.arange(klen-qlen,klen)[:,None];keys=torch.arange(klen)[None,:];valid=(keys<=query)[None,None]&attmask.bool()[:,None,None,:]
        causal=torch.where(valid,torch.tensor(0,dtype=torch.bfloat16),torch.tensor(torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16))
        if s['self_mask'] is None:assert n==1 or torch.all(causal==0)
        else:assert torch.equal(t['%04d_self_mask'%n],causal)
        assert len(s['all40_attention'])==40 and [z['layer'] for z in s['all40_attention']]==list(range(40))
        for i,rec in enumerate(s['all40_attention']):assert rec['type']==('cross' if i in [3,8,13,18,23,28,33,38] else 'self') and rec['device']==ld and rec['mask']==s[rec['type']+'_mask']
        for i,rec in enumerate(s['cache40']):
            cross_layer=i in [3,8,13,18,23,28,33,38];assert rec['layer']==i and rec['type']==('cross' if cross_layer else 'self') and rec['shape']==[size,8,6404 if cross_layer else klen,128] and rec['dtype']=='torch.bfloat16' and rec['device']==ld and rec['original_prefix_NPU_bitwise_unchanged']
        assert s['pixels_present']==(n==1)
        if n==1:
            assert s['vision_input_device_transfer_bitwise'] and s['projector_device_transfer_bitwise'] and s['actual_projected_visual_received_bitwise'] and len(s['cross_prefill_projection_cache_bitwise'])==8
            assert len(s['all40_vision_attention'])==40 and len({z['name'] for z in s['all40_vision_attention']})==40 and all(z['device']==vd and z['mask']==tr(t['actual_vision_attention_mask']) for z in s['all40_vision_attention'])
            for r in s['cross_prefill_projection_cache_bitwise']:assert r['original_projection_received_in_native_cache_bitwise'] and r['key']['shape']==r['value']['shape']==[size,8,6404,128] and r['key']['dtype']==r['value']['dtype']=='torch.bfloat16'
        # Actual native SDPA calls, including implicit causal masks and the
        # installed HF NPU binary-mask representation; no logical-mask change.
        calls=s['actual_SDPA_calls'];assert len(calls)==(80 if n==1 else 40);vision_calls=[];text_calls=[]
        for call in calls:
            assert call['dtype']=='torch.bfloat16' and call['device']==ld and call['dropout_p']==0.0
            isvision=call['module'].startswith('model.vision_model.')
            if isvision:
                vision_calls.append(call);assert n==1 and call['q_shape']==call['k_shape']==call['v_shape']==[size,16,6432,80] and not call['enable_gqa'] and not call['is_causal']
                expected_allowed=t['actual_vision_attention_mask']==0;assert call['mask_shape']==list(expected_allowed.shape) and call['mask_dtype']=='torch.bool' and call['logical_allowed_mask_SHA256']==fp(expected_allowed) and call['scale'] in (None,80**-.5)
            else:
                text_calls.append(call);layer=int(call['module'].split('.layers.')[1].split('.')[0]);is_cross=layer in [3,8,13,18,23,28,33,38];ql=length if n==1 else 1;kl=6404 if is_cross else length+n-1
                assert call['q_shape']==[size,32,ql,128] and call['scale'] in (None,128**-.5)
                assert call['k_shape']==call['v_shape']==[size,8 if call['enable_gqa'] else 32,kl,128]
                if is_cross:
                    allow=cross==0;assert call['module'].endswith('.cross_attn') and call['mask_shape']==list(allow.shape) and call['mask_dtype']=='torch.bool' and call['logical_allowed_mask_SHA256']==fp(allow) and not call['enable_gqa'] and not call['is_causal']
                elif call['mask_dtype'] is None:
                    assert call['module'].endswith('.self_attn') and s['self_mask'] is None and call['mask_shape'] is None and call['is_causal']==(ql>1) and call['enable_gqa']
                else:
                    assert call['module'].endswith('.self_attn') and call['mask_dtype']=='torch.bool' and call['mask_shape']==list(valid.shape) and call['logical_allowed_mask_SHA256']==fp(valid) and not call['is_causal'] and not call['enable_gqa']
        assert len(text_calls)==40 and len({z['module'] for z in text_calls})==40
        assert len(vision_calls)==(40 if n==1 else 0) and len({z['module'] for z in vision_calls})==len(vision_calls)
        logits=t['%04d_full_vocab_logits'%n];assert logits.shape==(size,128256) and logits.dtype==torch.bfloat16 and torch.isfinite(logits).all();best=logits.argmax(-1).tolist()
        for j in range(size):
            ended=any(v in [128001,128008,128009] for v in tokens[j][:n-1]);assert tokens[j][n-1]==(128004 if ended else best[j])
        prefix=torch.cat([prefix,torch.tensor(tokens)[:,n-1:n]],dim=1)
    for j,i in enumerate(indices):
        e=entries[j];raw=rust.decode(tokens[j],skip_special_tokens=True).strip();first=next((k for k,v in enumerate(tokens[j]) if v in [128001,128008,128009]),None);stop=len(step_rows) if first is None else first+1
        assert raw==g['raw_outputs'][j]==e['raw_output'] and e['generation_token_ids']==tokens[j] and e['first_EOS_index']==first and e['generation_count']==stop and e['truncated']==(stop>=2048) and e['group_slot']==j and e['group_batch_size']==size and e['error'] is None and e['prompt']==p['input_records'][i]['prompt'] and e['actual_model_prompt']==p['input_records'][i]['model_prompt'] and e['empty']==(not raw) and e['invalid']==(independent_choice(raw) is None)
        raws[i]=raw
    total_steps+=len(step_rows)
correct=0;empty=[];invalid=[];truncated=[];per=[]
for i,(a,b,r) in enumerate(zip(ins,outs,details)):
    e=by[i];raw=raws[i];choice=independent_choice(raw);prediction=raw if choice is None else choice
    assert a['prompt']==r['prompt']==clean(data[i]['prompt']) and a['reference']==b['reference']==r['reference']==data[i]['reference'] and b['raw_output']==r.get('raw_output','')==raw and ('raw_output' in r or raw=='') and b['prediction']==r['prediction']==prediction and b['error'] is None and r['extra']['execution']==e
    hit=int(choice is not None and choice==str(r['reference']).strip().upper());correct+=hit;per.append(hit)
    if e['empty']:empty.append(i)
    if e['invalid']:invalid.append(i)
    if e['truncated']:truncated.append(i)
score=correct/count;assert report['summary']['primary_metric']=='Strict_MC_Accuracy' and report['summary']['current_score']==score and report['statistics']['total_samples']==count and report['statistics']['error_count']==0 and report['statistics']['batch_size']==10 and report['metrics']=={'Strict_MC_Accuracy':score} and len(external)==2*len(groups)+7
with np.load(line['native_GELU_probe']['path'],allow_pickle=False) as z:
    vals=torch.linspace(-8,8,20001).bfloat16();assert np.array_equal(z['input'],vals.view(torch.uint8).numpy())
    for mode in ['none','tanh']:
        assert np.array_equal(z['CPU_'+mode],torch.nn.functional.gelu(vals,approximate=mode).view(torch.uint8).numpy())
        for device in [vd,ld]:assert torch.isfinite(torch.from_numpy(z[device+'_'+mode].copy()).view(torch.bfloat16)).all()
kernels={}
for kind,path in line['profiler_traces'].items():
    prof=json.loads(pathlib.Path(path).read_text());events=prof if isinstance(prof,list) else prof['traceEvents'];hw={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};ks=[e for e in events if e.get('pid') in hw and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(ks)>100;kernels[kind]=len(ks)
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},'sample_Accuracy':per,'correct':correct,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'historical_record_reuse':reuse_placement_schema,'inference_reused_for_module_placement_schema_verifier_repair':reuse_placement_schema,'all906_source_cast_actual_NPU_SHA_verified':True,'all511_original_visual_control_parameter_SHA_expected':source_vision,'original_untied128256_head128264_embedding128257_tokenizer_preserved':True,'own_single_BOS_and_original_serial1_all200_preserved':True,'actual_activation_module_count':80,'all80_actual_activation_counts_verified':True,'all40_native_cache_layers_including8_cross_and32_self_verified_steps':total_steps,'all_full128256_native_greedy_argmax_replayed':total_steps,'original_true_size_groups_verified':groups,'group_histogram':dict(collections.Counter(map(len,groups))),'independent_FP64_libm_reference_original_F32_angles_BF16_max_abs':max_rope,'original200_legacy_and_same_raw_strict_recomputed':{'legacy_Accuracy':.785,'same_old_raw_Strict_MC_Accuracy':.770,'metric_definition_difference_not_hardware_loss':True},'NPU_profiler_kernel_events':sum(kernels.values()),'first_actual_prefill_kernel_events_by_vision_and_language':kernels,'external_evidence_sha256':external,'report_sha256':{q.name:sha(q) for q in folder.iterdir()},'protocol_file':kw['protocol_file'],'protocol_sha256':kw['protocol_sha256'],'precision_disclosure':line['precision_disclosure'],'pre_numerical_review_real3_not_approved':True}


from decimal import Decimal
score_decimal=Decimal(str(score));a100=Decimal(str(p['A100_D']));assert a100!=0
relative=abs(score_decimal-a100)/abs(a100)
assert p['original_report_metric_qualification']=={'legacy_Accuracy':0.785,'same_old_raw_Strict_MC_Accuracy':0.77,'strict_invalid_count':5,'metric_definition_difference_not_hardware_loss':True}
proof.update(actual_attention={'root':'sdpa','text':'sdpa','vision':'sdpa'},all_native_SDPA_mask_causality_scale_GQA_verified=True,original_FP16_parameter_load_then_BF16_SHA_verified=True,original_roundtrip_changed_elements=4186034,original_FP16_nonzero_underflows=1107716,original_FP32_RoPE_buffers_preserved=True,original_actual_serial1_outer_metadata10_verified=True,original2048_budget_and_single_BOS_rightpad_original_AA_template_verified=True,A100_D=str(a100),relative_error_to_A100_table_D=float(relative),formal_alignment_within_5pct=(count==200 and relative<=Decimal('.05')),pre_numerical_review_real3_not_approved=not bool(kw.get('numerical_review')))
def reviewed_numerical_evidence_path(path,source_report):
    source=pathlib.Path(source_report).resolve()
    expected=R/'workspace/reports/alpindale_Llama-3.2-11B-Vision-Instruct/image-text-to-text'
    if source.parent!=expected or re.fullmatch(r'\d{8}_\d{6}',source.name) is None:return False
    q=pathlib.Path(path).resolve()
    return q.is_relative_to(R/'workspace/diagnostics') or (q==source/'operator_lineage.json' and q.parent==source)

if kw.get('numerical_review'):
    nrpath=pathlib.Path(kw['numerical_review']);assert sha(nrpath)==kw['numerical_review_sha256'];nr=json.loads(nrpath.read_text());assert nr==line['numerical_review'] and nr['passed'] and nr['diagnostic_only'] and nr['original_parameter_only_FP16load_then_BF16_and_FP32_RoPE_SDPA_preserved']
    for path,digest in nr['evidence_sha256'].items():assert reviewed_numerical_evidence_path(path,nr['NPU_source_report']) and sha(path)==digest;external[path]=digest
    assert nr['all906_A100_GPU_and_actual_NPU_final_SHA_equal'] and nr['same_full_vocab_no_resource_substitution'] and nr['all_measured_differences_retained_no_forced_tolerance']
    from alpindale_A100_numerical_reference_0910_20261104 import recompute
    independent_numerics=recompute(nr['control_file'],nr['NPU_source_report'])
    assert independent_numerics['results']==nr['results'] and all(nr[k]==independent_numerics[k] for k in independent_numerics if k!='evidence_sha256')
    for path,digest in independent_numerics['evidence_sha256'].items():assert nr['evidence_sha256'][path]==digest
    external[str(nrpath)]=sha(nrpath)
    proof.update(numerical_review=nr,pre_numerical_review_real3_not_approved=False)
proof['external_evidence_sha256']=external
proof=json.loads(json.dumps(proof));q=loadpath.parent/'independent_report_verification.json';assert not q.exists() or json.loads(q.read_text())==proof
if not q.exists():q.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
print(json.dumps(proof,ensure_ascii=False,indent=2))
