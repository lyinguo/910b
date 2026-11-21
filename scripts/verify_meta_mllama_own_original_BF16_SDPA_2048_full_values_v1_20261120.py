"""Independent own Mllama pixels/Jinja/Rust, source SHA, masks/cache and logits."""
import os,pathlib,sys,json,re,collections,math
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
import hashlib
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
from safetensors import safe_open
from tokenizers import Tokenizer
from PIL import Image
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from mllama11b_original_independent_cpu_910b import sha,fp,tr,inputs,clean
from egm8b_original_independent_cpu_910b import decode_arrays
from strict_mc_independent_reference_910b import independent_choice
from idefics2_original_accuracy_reference_910b import original_accuracy_sample
from cosmos_reason2_stable_rope_reference_910b import _bf16_rne
from mllama11b_masks_rope_reference_910b import original_vision_mask
torch.set_num_threads(4)
assert len(sys.argv)==3;folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in (3,200)
assert {q.name for q in folder.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];assert sha(kw['protocol_file'])==kw['protocol_sha256']
p=json.loads(pathlib.Path(kw['protocol_file']).read_text());w=pathlib.Path(p['weight_path']);line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];vd='npu:'+str(kw['npu_devices'][0]);ld='npu:'+str(kw['npu_devices'][1])
assert cfg['model']['name']==p['model_id']=='meta-llama/Llama-3.2-11B-Vision-Instruct' and cfg['model']['model_path']==str(w) and folder.parent.parent.name=='meta-llama_Llama-3.2-11B-Vision-Instruct' and folder.parent.name==cfg['task_type']=='image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1)
assert cfg['algorithm']['batch_size']==10 and cfg['algorithm']['kwargs']=={'extract_answer':True}
assert cfg['metrics']['override']==['Accuracy'] and cfg['metrics']['include_failed_predictions']
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert not cfg['data']['download'] and not cfg['model']['download'] and cfg['data']['preprocess'] and not cfg['data']['filter_empty'] and not cfg['data']['apply_template']
assert line['protocol']=='MetaMllama_own_original_BF16_defaultSDPA_outer10_inner1_2048' and line['weight_tensor_count']==906 and line['all_expected_parameters_loaded'] and line['runtime_dtype']=='BF16' and line['original_max_new_tokens']==2048 and line['original_outer_batch_size']==10 and line['original_untied_head128256_embedding128264_tokenizer128257_preserved'] and line['original_two_BOS128000_preserved'] and line['original_default_SDPA_all_text_and_vision'] and line['HF32_enabled']=={'conv':False,'matmul':False}
assert set(line['module_placements'].values())=={vd,ld}
for name,dev in line['module_placements'].items():assert dev==(vd if name=='model.vision_model' or name.startswith('model.vision_model.') or name=='model.multi_modal_projector' else ld)
assert sha(p['CPU200_protocol'])==p['CPU200_protocol_sha256'] and p['original_outer10_inner_serial1'] and p['group_count']==200
for n,h in p['runtime_and_original_source_sha256'].items():assert sha(n)==h
for n,h in p['own_resource_sha256'].items():assert pathlib.Path(n).name==n and sha(w/n)==h
from transformers import AutoConfig,MllamaForConditionalGeneration,GenerationConfig
mc=AutoConfig.from_pretrained(w,local_files_only=True);assert mc._attn_implementation is mc.text_config._attn_implementation is mc.vision_config._attn_implementation is None
with torch.device('meta'):meta=MllamaForConditionalGeneration(mc)
meta.generation_config=GenerationConfig.from_pretrained(w,local_files_only=True);gen,_=meta._prepare_generation_config(None,max_new_tokens=2048,do_sample=False,temperature=0.0)
assert gen.to_dict()==p['original_resolved_generation']==line['generation_config'] and gen.use_cache and gen.num_beams==1 and not gen.do_sample and gen.repetition_penalty==1 and gen.no_repeat_ngram_size==0 and gen.forced_bos_token_id is gen.forced_eos_token_id is None and gen.eos_token_id==[128001,128008,128009] and gen.pad_token_id==128004
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
            assert fp(t)==e['full_sha256']==r['source_sha256']==r['original_BF16_cast_sha256']==r['actual_NPU_sha256'] and r['device']==(vd if name.startswith(('model.vision_model.','model.multi_modal_projector.')) else ld) and r['runtime_dtype']=='torch.bfloat16' and r['target_key']==name and r['original_cast_changes']==0
            if name=='model.language_model.embed_tokens.weight':embed=t.clone()
            if name.startswith(('model.vision_model.','model.multi_modal_projector.')):source_vision[name]={'BF16':fp(t),'FP32':fp(t.float())}
assert embed.shape==(128264,4096) and params['lm_head.weight'].shape==(128256,4096) and len(source_vision)==511
inv=1/(500000.**(torch.arange(0,128,2,dtype=torch.int64).float()/128));wavelen=2*math.pi/inv;smooth=(8192/wavelen-1)/3
scaled=torch.where(wavelen>8192,inv/8,inv);inv=torch.where((wavelen>=2048)&(wavelen<=8192),(1-smooth)*scaled/8+smooth*scaled,scaled)
assert line['analytic_F32_buffers']=={'model.language_model.rotary_emb.inv_freq':tr(inv),'model.language_model.rotary_emb.original_inv_freq':tr(inv)}
data=[json.loads(l) for l in pathlib.Path(p['dataset_path']).read_text().splitlines()];assert sha(p['dataset_path'])==p['dataset_SHA'] and len(data)==4329
ins=[json.loads(l) for l in (folder/'sample_inputs.jsonl').read_text().splitlines()];outs=[json.loads(l) for l in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results']
tracepath=pathlib.Path(line['per_sample_evidence']);trace=[json.loads(l) for l in tracepath.read_text().splitlines()];actpath=pathlib.Path(line['activation_execution_trace']);acttrace=[json.loads(l) for l in actpath.read_text().splitlines()]
assert len(ins)==len(outs)==len(details)==len(trace)==count
for rows in [ins,outs,details]:assert [x['index'] for x in rows]==list(range(count))
assert sorted(r['index'] for r in trace)==list(range(count));by={r['index']:r for r in trace};groups=[]
groups=[[i] for i in range(count)]
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
    reconstructed_KV={};prefix=inp['input_ids'].clone();pos0=inp['attention_mask'].cumsum(-1)-1;pos0.masked_fill_(inp['attention_mask']==0,0)
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
        if s['self_mask'] is None:assert p['original_default_attention']=='sdpa' and torch.all(inp['attention_mask']==1)
        else:assert torch.equal(t['%04d_self_mask'%n],causal)
        assert len(s['all40_attention'])==40 and [z['layer'] for z in s['all40_attention']]==list(range(40))
        for i,rec in enumerate(s['all40_attention']):assert rec['type']==('cross' if i in [3,8,13,18,23,28,33,38] else 'self') and rec['device']==ld and rec['mask']==s[rec['type']+'_mask']
        for i,rec in enumerate(s['cache40']):
            cross_layer=i in [3,8,13,18,23,28,33,38];assert rec['layer']==i and rec['type']==('cross' if cross_layer else 'self') and rec['shape']==[size,8,6404 if cross_layer else klen,128] and rec['dtype']=='torch.bfloat16' and rec['device']==ld and rec['original_prefix_NPU_bitwise_unchanged']
        for i,rec in enumerate(s['cache40']):
            cross_layer=i in [3,8,13,18,23,28,33,38]
            for kind in ['key','value']:
                name='%04d_cache_%02d_%s'%(n,i,kind)
                if n==1:reconstructed_KV[(i,kind)]=t[name]
                elif not cross_layer:reconstructed_KV[(i,kind)]=torch.cat([reconstructed_KV[(i,kind)],t[name]],dim=-2)
                assert tr(reconstructed_KV[(i,kind)])==rec['complete_'+kind]
        calls=s['actual_native_SDPA_calls'];assert len(calls)==(80 if n==1 else 40)
        assert len([r for r in calls if r['section']=='text'])==40
        for r in calls:
            assert r['dropout']==0. and r['dtype']=='torch.bfloat16' and r['device']==(vd if r['section']=='vision' else ld)
            assert r['query_shape'][1]==(16 if r['section']=='vision' else 32) and r['query_shape'][-1]==(80 if r['section']=='vision' else 128)
            assert r['scale']==1./math.sqrt(r['query_shape'][-1])
            if r['mask'] is None:
                assert r['kind']=='self' and r['is_causal']==(r['query_shape'][-2]>1) and torch.all(inp['attention_mask']==1)
            else:
                actualmask=t[r['mask_array']];assert tr(actualmask)==r['mask'] and not r['is_causal']
                expected_mask=t['actual_vision_attention_mask'] if r['section']=='vision' else (causal if r['kind']=='self' else cross)
                assert actualmask.dtype==torch.bool and torch.equal(actualmask,torch.logical_not(expected_mask.bool()))
        assert s['pixels_present']==(n==1)
        if n==1:
            assert s['vision_input_device_transfer_bitwise'] and s['projector_device_transfer_bitwise'] and s['actual_projected_visual_received_bitwise'] and len(s['cross_prefill_projection_cache_bitwise'])==8
            assert len(s['all40_vision_attention'])==40 and len({z['name'] for z in s['all40_vision_attention']})==40 and all(z['device']==vd and z['mask']==tr(t['actual_vision_attention_mask']) for z in s['all40_vision_attention'])
            for r in s['cross_prefill_projection_cache_bitwise']:assert r['original_projection_received_in_native_cache_bitwise'] and r['key']['shape']==r['value']['shape']==[size,8,6404,128] and r['key']['dtype']==r['value']['dtype']=='torch.bfloat16'
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
score=correct/count;assert report['summary']['primary_metric']=='Accuracy' and report['summary']['current_score']==score and report['statistics']['total_samples']==count and report['statistics']['error_count']==0 and report['statistics']['batch_size']==10 and report['metrics']=={'Accuracy':score} and len(external)==2*len(groups)+7
with np.load(line['native_GELU_probe']['path'],allow_pickle=False) as z:
    vals=torch.linspace(-8,8,20001).bfloat16();assert np.array_equal(z['input'],vals.view(torch.uint8).numpy())
    for mode in ['none','tanh']:
        assert np.array_equal(z['CPU_'+mode],torch.nn.functional.gelu(vals,approximate=mode).view(torch.uint8).numpy())
        for device in [vd,ld]:assert torch.isfinite(torch.from_numpy(z[device+'_'+mode].copy()).view(torch.bfloat16)).all()
kernels={}
for kind,path in line['profiler_traces'].items():
    prof=json.loads(pathlib.Path(path).read_text());events=prof if isinstance(prof,list) else prof['traceEvents'];hw={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'}
    ks=[e for e in events if e.get('pid') in hw and e.get('ph')=='X' and (str(e.get('args',{}).get('Task Type','')).startswith('KERNEL') or str(e.get('args',{}).get('Task Type','')) in {'AI_CORE','AI_CPU','AICPU','AIV','AIC'})]
    assert len(ks)>100;kernels[kind]=len(ks)
from meta_mllama_own_visual_value_reference_20261120 import verify_own_visual_values
visual=verify_own_visual_values(p,line,by,count)
external.update(visual['external_evidence_sha256'])
for path,digest in p['runtime_and_original_source_sha256'].items():external[path]=digest
for name,head in p['all_source_file_sha256'].items():external[str(w/name)]=head['sha256']
for name,digest in p['own_resource_sha256'].items():external[str(w/name)]=digest
external[p['CPU200_protocol']]=p['CPU200_protocol_sha256'];external[kw['protocol_file']]=kw['protocol_sha256']
proof=dict(passed=True,report_dir=str(folder),sample_count=count,Accuracy=score,metrics={'Accuracy':score},sample_Accuracy=per,correct=correct,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,actual_native_forwards=total_steps,all906_own_source_cast_actual_NPU_SHA_verified=True,all511_own_visual_source_SHA_verified=True,all40_physical_KV_full_values_reconstructed_and_complete_SHA_verified_each_forward=True,all_native_SDPA_call_masks_causal_scale_head_arguments_independently_verified=True,original_outer10_inner_serial1_preserved=True,original_max_new_tokens=2048,original_two_BOS_and_untied_128256_head_128264_embedding_preserved=True,all_full128256_native_greedy_argmax_replayed=total_steps,independent_original_F32_angles_BF16_max_abs=max_rope,NPU_profiler_kernel_events=sum(kernels.values()),first_actual_vision_and_language_prefill_kernels_only=kernels,external_evidence_sha256=external,report_sha256={q.name:sha(q) for q in folder.iterdir()},protocol_file=kw['protocol_file'],protocol_sha256=kw['protocol_sha256'],precision_disclosure=line['precision_disclosure'],own_original_visual_numerical_controls=visual,historical_A100_engine_or_tensor_capture_claimed=False,all_original_input_output_raw_ref_token_error_denominators_preserved=True)
q=loadpath.parent/'independent_report_verification.json';proof=json.loads(json.dumps(proof));assert not q.exists() or json.loads(q.read_text())==proof
if not q.exists():q.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
print(json.dumps(proof,ensure_ascii=False,indent=2))
