"""Independent Ferret own-source/input/visual/CE/full-vocab/report audit (CPU only)."""
import os,sys,pathlib,json,hashlib,re,math,collections
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,'/root/projects/comperror')
import numpy as np,torch
from PIL import Image
import sentencepiece as spm
from safetensors import safe_open
torch.set_num_threads(4)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(v):return {'shape':list(v.shape),'dtype':str(v.dtype),'sha256':fp(v)}
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in [3,200]
required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {q.name for q in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];pfile=pathlib.Path(kw['protocol_file']);assert sha(pfile)==kw['protocol_sha256'];p=json.loads(pfile.read_text());wp=pathlib.Path(p['weight_path']);assert wp==pathlib.Path(cfg['model']['model_path']) and cfg['model']['name']==p['model_id']=='jadechoghari/Ferret-UI-Gemma2b'
assert cfg['task_type']=='image-text-to-text' and folder.parent.name==cfg['task_type'] and folder.parent.parent.name==p['model_id'].replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_range']=='0:'+str(count-1) and cfg['sample_count']==count and cfg['algorithm']['batch_size']==1 and cfg['data']['filter_empty'] is False and cfg['data']['download'] is cfg['model']['download'] is False
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert p['candidate_count']==525 and p['original_runtime_dtype']=='BF16' and p['original_text_attention']=='eager' and p['use_cache'] is False and p['native_parameter_count']==575
for source,digest in dict(p['historical_sources'],**p['native_source_sha256'],**p['preprocessor_source_sha256']).items():assert sha(source)==digest
for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(wp/name)==digest
assert sha(p['declared_clip_config'])==p['declared_clip_config_sha256'];assert sha(p['records_path'])==p['records_sha256'];cpu=json.loads(pathlib.Path(p['records_path']).read_text());assert len(cpu)==200
line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];dev='npu:'+str(kw['npu_devices'][0]);assert line['devices']==kw['npu_devices'] and line['protocol_file']==str(pfile) and line['protocol_sha256']==kw['protocol_sha256'] and line['own_source_tensor_count']==576 and line['independent_learned_parameter_count']==575 and line['all_expected_parameters_loaded']
assert line['native_attention_implementation']=={'text':'eager','vision':p['native_vision_attention']} and line['original_use_cache'] is False and line['external_CLIP_learned_values_loaded'] is False and line['actual_LM_batch_size']==1 and line['full_vocab_size']==256001 and not any(line['HF32_enabled'].values())
external={}
def check_external(path,digest=None):
 q=pathlib.Path(path).resolve();assert q.is_relative_to('/root/projects/comperror/workspace/diagnostics');h=sha(q)
 if digest is not None:assert h==digest
 if str(q) in external:assert external[str(q)]==h
 external[str(q)]=h;return q
loading=check_external(line['weight_loading_evidence']);loaded=json.loads(loading.read_text());expected={r['source_key']:r for r in p['source_tensor_records']};assert len(loaded)==len(expected)==576 and len({r['source_key'] for r in loaded})==576
for shard,header in p['weight_headers'].items():
 assert sha(wp/shard)==header['file_sha256'] and (wp/shard).stat().st_size==header['size']
 with safe_open(wp/shard,framework='pt',device='cpu') as f:
  for r in loaded:
   if r['source_key'] not in f.keys():continue
   e=expected[r['source_key']];v=f.get_tensor(r['source_key']);assert list(v.shape)==e['shape']==r['shape'] and str(v.dtype)==r['source_dtype']=='torch.bfloat16' and fp(v)==r['source_sha256']==r['original_BF16_cast_sha256']==r['actual_NPU_sha256'] and r['device']==dev and r['target_key']==e['target_key']
assert p['original_tied_source_alias_checks'][0]['all_stored_values_identical'] and line['original_own_tied_embedding_head_preserved']
traces=[json.loads(x) for x in check_external(line['per_sample_evidence']).read_text().splitlines()];candidates=[json.loads(x) for x in check_external(line['per_candidate_evidence']).read_text().splitlines()];activation=[json.loads(x) for x in check_external(line['activation_execution_trace']).read_text().splitlines()]
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];assert len(inputs)==len(outputs)==len(details)==len(traces)==count
for rows in [inputs,outputs,details]:assert [r['index'] for r in rows]==list(range(count))
assert [e['index'] for e in traces]==list(range(count)) and all(e['attempt']==e['index']+1 and e['error'] is None for e in traces)
sp=spm.SentencePieceProcessor(model_file=str(wp/'tokenizer.model'));assert sp.vocab_size()==256000
def ids(text):
 result=[2]
 for i,part in enumerate(text.split('<image>')):
  if i:result.append(-200)
  result.extend(sp.encode(part))
 return result
def independent_pixels(image):
 w,h=image.size;options=[]
 for i,(gw,gh) in enumerate(p['own_model_config']['image_grid_pinpoints']):
  scale=min(gw/w,gh/h);effective=min(int(w*scale)*int(h*scale),w*h);options.append((-effective,gw*gh-effective,i,gw,gh))
 _,_,_,gw,gh=min(options);stretched=image.resize((gw,gh));pieces=[image.resize((336,336))]+[stretched.crop((x,y,x+336,y+336)) for y in range(0,gh,336) for x in range(0,gw,336)];config=json.loads((wp/'preprocessor_config.json').read_text());arrays=[]
 for piece in pieces:
  a=(np.asarray(piece.resize((336,336),Image.Resampling.BICUBIC)).astype(np.float64)*config['rescale_factor']).astype(np.float32);a=(a-np.array(config['image_mean'],np.float32))/np.array(config['image_std'],np.float32);arrays.append(a.transpose(2,0,1).copy())
 return torch.from_numpy(np.stack(arrays)),(gw//336,gh//336)
def decode_bytes(z,name,meta):
 dtype={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32,'torch.int64':torch.int64}[meta['dtype']];t=torch.from_numpy(z[name].copy()).view(dtype).reshape(meta['shape']);assert tr(t)==meta;return t
score=0;sample_scores=[];total=0;max_ce_delta=0.;tilehist=collections.Counter();seen=[];expected_candidates=[]
for i,(e,a,b,r) in enumerate(zip(traces,inputs,outputs,details)):
 original=cpu[i];assert e['index']==i and e['error'] is None and r.get('error') is None and b['error'] is None and e['source_image_sha256']==sha(original['media_path'])==original['media_sha256']
 assert a['prompt']==r['prompt']==original['input'] and a['reference']==b['reference']==r['reference']==original['reference'] and r['extra']['execution']==e
 pixel,grid=independent_pixels(Image.open(original['media_path']).convert('RGB'));assert tr(pixel)==original['pixels']==e['pixel_values'];tilehist[len(pixel)]+=1
 system='A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0, 0], bottom-right [width-1, height-1]. Increasing x moves right; y moves down. Bounding box: [x1, y1, x2, y2]. Image size: 1000x1000. Follow instructions.'
 base=system+'<start_of_turn>user\n<image>\n'+original['input']+'<end_of_turn>\n<start_of_turn>model\n';assert base==original['base_prompt'] and ids(base)==original['base_ids']
 assert len(e['candidates'])==len(original['candidates']);scores={};cpu_scores={};visual_ref=None
 for candidate,expected_candidate in zip(e['candidates'],original['candidates']):
  c=candidate;expected_candidates.append(c);total+=1;assert c['index']==i and c['error'] is None and c['attempt']==i+1 and c['input_ids']==expected_candidate['input_ids']==ids(base+c['original_continuation']) and c['original_continuation']==expected_candidate['continuation'] and c['labels_before_original_multimodal_packing']==expected_candidate['labels'] and c['actual_visual_recomputed'] and c['actual_visual_injection_bitwise_verified'] and c['actual_vision_calls']==1 and c['cache'] is None
  assert c['original_continuation'].endswith('<end_of_turn>') and c['letter']==expected_candidate['letter'];length=min(expected_candidate['merged_length_before_original_4096_truncation'],4096);assert c['actual_full_logits_shape']==[1,length,256001]
  index=c['input_ids'].index(-200);labels=torch.tensor(c['labels_before_original_multimodal_packing']);merged=torch.cat([labels[:index],torch.full((576*len(pixel),),-100,dtype=torch.int64),labels[index+1:]])[:4096];valid=merged[1:]!=-100;npz=check_external(c['scored_logits_path'],c['scored_logits_sha256'])
  with np.load(npz,allow_pickle=False) as z:
   logits=np.array(z['full_vocab_scored_logits_FP32'],copy=True);target=np.array(z['continuation_labels'],copy=True);assert logits.dtype==np.float32 and logits.shape==(int(valid.sum()),256001) and np.isfinite(logits).all() and np.array_equal(target,merged[1:][valid].numpy()) and np.array_equal(z['merged_labels'],merged.numpy())
   # Independent FP64 whole-vocabulary stable logsumexp, no NPU model/answers.
   x=logits.astype(np.float64);m=x.max(axis=-1);nll=m+np.log(np.exp(x-m[:,None]).sum(axis=-1))-x[np.arange(len(target)),target];reference_score=-float(nll.mean());cpu_scores[c['letter']]=reference_score;delta=abs(reference_score-c['score']);max_ce_delta=max(max_ce_delta,delta);assert delta<2e-5,('Native FP32 CE differs from independent FP64 beyond diagnostic bound',i,c['letter'],delta)
   assert c['score']==-c['actual_native_loss'] and c['native_loss_dtype']=='torch.float32';pos=decode_bytes(z,'actual_positions',c['tensor_records']['actual_positions']);assert torch.equal(pos,torch.arange(length).reshape(1,-1))
   cos=decode_bytes(z,'actual_cos',c['tensor_records']['actual_cos']);sin=decode_bytes(z,'actual_sin',c['tensor_records']['actual_sin']);assert cos.shape==sin.shape==(1,length,256) and cos.dtype==sin.dtype==torch.bfloat16
   freq=(1/(10000.**(np.arange(0,256,2,dtype=np.float32)/256))).astype(np.float32);angle=(np.arange(length,dtype=np.float32)[:,None]*freq[None,:]).astype(np.float32);angle=np.concatenate([angle,angle],axis=-1).astype(np.float64);assert np.max(np.abs(cos.float().numpy()[0]-np.cos(angle)))<=.003907 and np.max(np.abs(sin.float().numpy()[0]-np.sin(angle)))<=.003907
  vpath=check_external(c['visual_path'],c['visual_sha256'])
  with np.load(vpath,allow_pickle=False) as z:
   tensors={k:decode_bytes(z,k,c['tensor_records'][k]) for k in z.files};selected=tensors['selected_original_CLIP_penultimate_patches'];projected=tensors['actual_projected_patches'];actual=tensors['actual_injected_visual'];assert selected.shape==(len(pixel),576,1024) and projected.shape==(len(pixel),576,2048)
   gw,gh=grid;joined=torch.cat([projected[0],projected[1:].reshape(gh,gw,24,24,2048).permute(0,2,1,3,4).reshape(-1,2048)],dim=0);assert torch.equal(actual,joined[:min(len(joined),4096-index)])
  assert len(c['actual_attention_layers'])==18
  for j,layer in enumerate(c['actual_attention_layers']):
   mask=layer['mask'];dtype={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32}[mask['dtype']];expected_mask=torch.full((length,length),torch.finfo(dtype).min,dtype=dtype).triu(1).reshape(1,1,length,length);assert layer['layer']==j and layer['cache'] is None and layer['scaling']==256**-.5 and tr(expected_mask)==mask
  counts=c['activation_call_counts'];assert set(counts)==set(line['activation_modules']) and all(v==(0 if '.region_geo_sampler.' in n else 1) for n,v in counts.items());scores[c['letter']]=c['score'];seen.append((i,c['letter']))
 best=max(scores,key=scores.get);assert max(cpu_scores,key=cpu_scores.get)==best and scores==e['candidate_scores'];raw=f'<answer>{best}</answer>';assert raw==e['raw_output']==r['raw_output']==b['raw_output'] and r['prediction']==b['prediction']==best
 hit=int(best==r['reference']);score+=hit;sample_scores.append(hit);assert e['empty'] is e['invalid'] is e['truncated'] is False and e['generation_count']==0 and e['generation_token_ids']==[] and e['original_2048_generation_fallback_not_executed']
assert candidates==expected_candidates and len(activation)==total and [(r['index'],r['letter']) for r in activation]==seen
for a,c in zip(activation,candidates):assert a['module_call_counts']==c['activation_call_counts'] and a['device']==dev and a['error'] is None
assert report['statistics']['error_count']==0 and report['statistics']['total_samples']==count and report['statistics']['batch_size']==1 and score/count==report['metrics']['Strict_MC_Accuracy']==report['summary']['current_score'] and report['summary']['primary_metric']=='Strict_MC_Accuracy'
if count==200:assert total==525 and dict(tilehist)=={3:168,5:32}
probe=line['native_GELU_probe'];check_external(probe['path'],probe['sha256'])
with np.load(probe['path'],allow_pickle=False) as z:
 x=torch.linspace(-8,8,20001).bfloat16();assert np.array_equal(z['input_BF16'],x.view(torch.uint8).numpy())
 for mode in ['none','tanh']:
  cpu_value=torch.nn.functional.gelu(x,approximate=mode);assert np.array_equal(z['CPU_'+mode],cpu_value.view(torch.uint8).numpy());native=torch.from_numpy(z['NPU_'+mode].copy()).view(torch.bfloat16);assert torch.isfinite(native).all() and float((native-cpu_value).abs().max())==probe['CPU_vs_NPU_max_abs'][mode]
profile=json.loads(check_external(line['profiler_trace']).read_text());events=profile if isinstance(profile,list) else profile['traceEvents'];hardware={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};kernels=[e for e in events if e.get('pid') in hardware and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'Accuracy':score/count,'metrics':{'Strict_MC_Accuracy':score/count},'correct':score,'sample_Accuracy':sample_scores,'error_indices':[],'empty_indices':[],'invalid_indices':[],'truncated_indices':[],'index_duplicates':False,'historical_record_reuse':False,'all576_source_cast_actual_NPU_SHA_verified':True,'original_equal_tied_head_preserved':True,'all_original_candidate_visual_forwards_really_recomputed':total,'full_256001_vocab_FP64_CE_recomputed':True,'native_FP32_CE_vs_independent_FP64_max_abs':max_ce_delta,'all18_no_cache_positions_masks_and_RoPE_verified':True,'all_visual_spatial_prefix_injection_verified':True,'own_SentencePiece_full_input_and_candidate_tokens_verified':True,'original_anyres_patch_histogram':dict(tilehist),'NPU_profiler_kernel_events':len(kernels),'external_evidence_sha256':external,'report_sha256':{f.name:sha(f) for f in folder.iterdir()},'protocol_file':str(pfile),'protocol_sha256':kw['protocol_sha256'],'precision_disclosure':line['precision_disclosure'],'pre_numerical_review_real3_not_approved':True}
assert 'native_numerical_review' in kw and 'native_numerical_review_sha256' in kw
from audit_ferret_static_numerics_0106 import audit as audit_static
numerical=audit_static(kw['native_numerical_review'],kw['native_numerical_review_sha256'],p,line)
assert numerical['review']==line['native_numerical_review'] and line['static_CPU_NPU_controls_from_prior_true3_not_recomputed_per_report']
external.update(numerical['evidence_sha256']);proof.update(pre_numerical_review_real3_not_approved=False,native_numerical_difference_retained_and_disclosed=True,static_controls_from_prior_true3_not_recomputed_per_report=True,native_numerical_review=numerical['review'],native_numerical_diagnostic_records=numerical['diagnostic_records'],external_evidence_sha256=external)
reuse='--reuse-for-verifier-JSON-schema-repair' in sys.argv
assert not reuse or count==3
proof['inference_reused_for_verifier_JSON_schema_repair']=reuse
proof=json.loads(json.dumps(proof,ensure_ascii=False))
dest=loading.parent/('independent_report_verification_JSON_v3.json' if reuse else 'independent_report_verification.json');assert not dest.exists() or json.loads(dest.read_text())==proof
if not dest.exists():dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
print(json.dumps(proof,ensure_ascii=False,indent=2))
