"""Independent own-source/cast/pixel/token/visual-prefix/beam-cache and ROUGE verification for original Florence."""
import pathlib,sys,os,json,hashlib,re,math,bisect
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from PIL import Image
from tokenizers import Tokenizer
from safetensors import safe_open
from transformers import AutoConfig,AutoImageProcessor,AutoTokenizer,Florence2ForConditionalGeneration,GenerationConfig
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent));sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from florence_original_cpu_reference_910b import independent_pixels,semantic_target
from universal_eval.metric.rouge_910b_porter import PorterStemmer
torch.set_num_threads(4)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def check(t,r):assert list(t.shape)==r['shape'] and str(t.dtype)==r['dtype'] and fp(t)==r['sha256']
stem=PorterStemmer()
def words(s):return [stem.stem(x) if len(x)>3 else x for x in re.findall('[a-z0-9]+',str(s).lower())]
def rouge(a,b):
 a=words(a);b=words(' '.join(b) if isinstance(b,list) else b)
 if not a or not b:return 0.
 positions={}
 for i,w in enumerate(b):positions.setdefault(w,[]).append(i)
 tail=[]
 for w in a:
  for i in reversed(positions.get(w,[])):
   k=bisect.bisect_left(tail,i)
   if k==len(tail):tail.append(i)
   else:tail[k]=i
 return 2.*len(tail)/(len(a)+len(b))
tags=re.compile(r'</s>|<s>|<pad>|<unk>|<mask>|<fake_token_around_image>|<image>|<[A-Z][A-Z0-9_]*>|<loc_\d+>|<seg>\d+</seg>')
def cleanup(s):return ' '.join(tags.sub(' ',s or '').split()).strip()
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {x.name for x in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];p=cfg['model']['kwargs']['protocol'];wp=pathlib.Path(cfg['model']['model_path']);assert cfg['model']['name']==p['model_id']=='MiaoshouAI/Florence-2-base-PromptGen' and str(wp)==p['weight_path'] and wp.resolve().is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels')
assert cfg['task_type']==p['workbook_task']==folder.parent.name and folder.parent.parent.name==p['model_id'].replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['algorithm']['batch_size']==1 and cfg['algorithm']['kwargs']=={'extract_answer':False} and cfg['data']['download'] is False and cfg['model']['download'] is False and cfg['metrics']['override']==['ROUGE-L'] and cfg['metrics']['include_failed_predictions'] is True and report['summary']['primary_metric']=='ROUGE-L'
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];assert len(inputs)==len(outputs)==len(details)==count
for rows in [inputs,outputs,details]:assert [x['index'] for x in rows]==list(range(count))
line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];assert line['protocol']=='original_Florence_caption_FP16_legacy_visual_prefix_beam3' and line['protocol_review']==p and line['source_dtype']=='F32' and line['runtime_dtype']=='F16' and line['all_expected_parameters_loaded'] and line['original_head_untied_preserved'] and line['original_three_embedding_aliases_preserved'] and line['original_final_logits_bias_executed_on_NPU'] and line['HF32_enabled']=={'conv':False,'matmul':False}
device='npu:'+str(line['devices'][0]);assert line['devices']==cfg['model']['kwargs']['npu_devices'] and set(line['module_placements'].values())=={device}
for rel,digest in p['resource_sha256'].items():assert pathlib.Path(rel).name==rel and sha(wp/rel)==digest
for source,digest in dict(p['runtime_sha256'],**p['historical_sources']).items():assert sha(source)==digest
for name in ['reference_report','reference_log','dataset_path','original_dataset','mapping_pixel_review']:assert sha(p[name])==p[name+'_sha256']
old=json.loads(pathlib.Path(p['reference_report']).read_text());assert len(old['detailed_results'])==200 and old['pipeline_config']['model']['name']==p['model_id'] and old['pipeline_config']['model']['kwargs']['torch_dtype']=='float16' and old['pipeline_config']['algorithm']['batch_size']==1 and old['pipeline_config']['algorithm']['kwargs']['extract_answer'] is False
assert re.search(r'开始推理循环: 共 200 条样本[^\n]*提取答案=False',pathlib.Path(p['reference_log']).read_text())
graph=json.loads(pathlib.Path(p['CPU_original_vision_graph_audit']).read_text());assert sha(p['CPU_original_vision_graph_audit'])==p['CPU_original_vision_graph_audit_sha256'] and graph['passed'] and graph['CPU_only_no_approval'] and graph['equations']=={'channel_modules':12,'window_modules':12}
mc=AutoConfig.from_pretrained(wp,local_files_only=True,trust_remote_code=False);assert mc.text_config.encoder_layers==mc.text_config.decoder_layers==6 and mc.text_config.scale_embedding is False;mc.tie_word_embeddings=False;mc.text_config.tie_word_embeddings=False
with torch.device('meta'):model=Florence2ForConditionalGeneration(mc)
lm=model.model.language_model;lm.encoder.embed_tokens.weight=lm.shared.weight;lm.decoder.embed_tokens.weight=lm.shared.weight;model.register_buffer('original_final_logits_bias',torch.empty((1,51289),device='meta'));expected=model.state_dict();params=set(dict(model.named_parameters(remove_duplicate=False)));assert len(list(model.parameters()))==665 and len(list(model.buffers()))==2
loading=json.loads(pathlib.Path(line['weight_loading_evidence']).read_text());assert len(loading)==667==line['weight_tensor_count'] and line['learned_parameter_count']==665 and line['original_buffer_count']==2;mapping={r['source']:r for r in p['source_target_mapping']};seen=set();positions={};source_text=None;source_head=None;underflows=0
with safe_open(str(wp/'model.safetensors'),framework='pt',device='cpu') as f:
 assert set(f.keys())==set(mapping)=={r['source'] for r in loading}
 for r in loading:
  key=r['source'];name,op=semantic_target(key);v=f.get_tensor(key);cast=v.half();target=cast.T.contiguous() if op=='transpose_2d' else cast;m=mapping[key];assert name==r['target']==m['target'] and op==r['operation']==m['operation'] and name not in seen and target.shape==expected[name].shape;seen.add(name)
  assert r['source_dtype']=='torch.float32' and r['runtime_dtype']=='torch.float16' and r['device']==device and r['source_shape']==list(v.shape) and r['target_shape']==list(target.shape) and r['is_parameter']==(name in params)
  assert fp(v)==r['source_sha256']==m['source_sha256'] and fp(cast)==r['original_FP16_cast_sha256']==m['original_FP16_cast_sha256'] and fp(target)==r['target_FP16_sha256']==r['actual_NPU_sha256']==m['target_FP16_sha256'] and torch.isfinite(cast).all();lost=int(((v!=0)&(cast==0)).sum());assert lost==r['nonzero_FP16_underflows']==m['nonzero_FP16_underflows'];underflows+=lost
  if name=='model.language_model.shared.weight':source_text=cast.clone()
  if name=='lm_head.weight':source_head=cast.clone()
  if name.endswith('.embed_positions.weight'):positions[name]=cast.clone()
  if name=='original_final_logits_bias':assert not torch.count_nonzero(cast)
assert set(expected)-seen=={'model.language_model.encoder.embed_tokens.weight','model.language_model.decoder.embed_tokens.weight'} and underflows==463==line['source_FP16_underflow_total'] and source_text is not None and source_head is not None and not torch.equal(source_text,source_head)
ip=AutoImageProcessor.from_pretrained(wp,local_files_only=True,trust_remote_code=False,backend='pil');tok=AutoTokenizer.from_pretrained(wp,local_files_only=True,trust_remote_code=False);rust=Tokenizer.from_file(str(wp/'tokenizer.json'));assert json.loads(json.dumps(ip.to_dict()))==p['image_processor']==line['effective_image_processor'] and len(tok)==51289 and all(rust.id_to_token(i) is not None for i in range(51289))
ids=tok([p['expanded_prompt']],return_tensors='pt',padding=True,return_token_type_ids=False)['input_ids'];assert ids.tolist()==p['original_task_input_ids'] and p['original_task_prompt']=='<DETAILED_CAPTION>' and p['expanded_prompt']=='Describe in detail what is shown in the image.'
gen,_=model._prepare_generation_config(GenerationConfig.from_pretrained(wp,local_files_only=True),max_new_tokens=256,do_sample=False,num_beams=3,repetition_penalty=1.1,forced_bos_token_id=None,forced_eos_token_id=None);assert gen.to_dict()==p['generation_config']==line['effective_generation_config'] and gen.use_cache is True and gen.num_beams==3 and gen.no_repeat_ngram_size==3 and gen.early_stopping is True and gen.decoder_start_token_id==2 and gen.eos_token_id==2 and gen.forced_bos_token_id is None and gen.forced_eos_token_id is None
dataset=pathlib.Path(p['dataset_path']);assert sha(dataset)=='ceb1670f2848955290915dbe375dca8b34b62d6980e9f6cd1abe38bf0e68963b';data=[json.loads(x) for x in dataset.read_text().splitlines()];fixture=[json.loads(x) for x in pathlib.Path(p['original_dataset']).read_text().splitlines()];assert len(data)==1459
oldvalues=[]
for i,(r,d,f) in enumerate(zip(old['detailed_results'],data,fixture)):
 assert r['index']==i and r['prompt']==d['prompt']==f['prompt'] and r['reference']==d['reference']==f['reference'] and r['raw_output']==r['prediction'];path=(dataset.parent/d['media_path']).resolve();oldpath=(pathlib.Path(p['original_dataset']).parent/f['media_path']).resolve();assert sha(path)==sha(oldpath)==p['reference_first200_image_sha256'][str(path)];oldvalues.append(rouge(r['prediction'],r['reference']))
assert abs(sum(oldvalues)/200-old['metrics']['ROUGE-L'])<1e-12
assert report['statistics']['error_count']==0,report['statistics']
tracepath=pathlib.Path(line['per_sample_evidence']);trace=[json.loads(x) for x in tracepath.read_text().splitlines()];gpath=pathlib.Path(line['native_GELU_execution_trace']);gelus=[json.loads(x) for x in gpath.read_text().splitlines()];assert len(trace)==len(gelus)==count and [r['attempt'] for r in trace]==[r['attempt'] for r in gelus]==list(range(1,count+1))
probe=line['native_GELU_probe'];assert sha(probe['path'])==probe['sha256'] and probe['device']==device and probe['dtype']=='torch.float16';torch.set_num_threads(probe['CPU_threads'])
with np.load(probe['path'],allow_pickle=False) as z:
 v=torch.from_numpy(z['input_FP16'].copy());erf=torch.nn.functional.gelu(v,approximate='none');tanh=torch.nn.functional.gelu(v,approximate='tanh');npu=torch.from_numpy(z['actual_NPU_FP16'].copy());assert torch.equal(erf,torch.from_numpy(z['CPU_original_erf'])) and torch.equal(tanh,torch.from_numpy(z['CPU_tanh_diagnostic'])) and torch.isfinite(npu).all() and float((npu-erf).abs().max())==probe['max_abs_CPU_original_erf'] and float((npu-tanh).abs().max())==probe['max_abs_CPU_tanh_diagnostic']
torch.set_num_threads(4);external={str(tracepath):sha(tracepath),str(gpath):sha(gpath),line['weight_loading_evidence']:sha(line['weight_loading_evidence']),line['profiler_trace']:sha(line['profiler_trace']),probe['path']:sha(probe['path'])};empty=[];invalid=[];truncated=[];search_hits=[];values=[];length=577+ids.shape[1]
for i,(a,o,r,g,d) in enumerate(zip(inputs,outputs,trace,gelus,details)):
 source=data[i];assert a['prompt']==r['prompt']==source['prompt'] and a['reference']==o['reference']==d['reference']==source['reference'] and o['extra']['execution']==r and o['error'] is None and r['error'] is None and r['no_reference_passed_to_model'] and r['original_input_prompt_ignored_in_favour_of_fixed_task'];path=pathlib.Path(a['media_path']).resolve();assert path==(dataset.parent/source['media_path']).resolve() and sha(path)==r['source_image_sha256']==p['reference_first200_image_sha256'][str(path)]
 with Image.open(path) as im:manual=independent_pixels(im,ip.to_dict());prepared=ip(images=im.convert('RGB').resize((768,768),Image.Resampling.LANCZOS),return_tensors='pt')['pixel_values']
 assert torch.equal(manual,prepared);check(manual,r['CPU_original_PIL_pixels']);check(manual.half(),r['NPU_original_FP16_pixels']);assert ids.tolist()==r['original_task_input_ids']
 archive=pathlib.Path(r['visual_feature_archive']);assert sha(archive)==r['visual_feature_archive_sha256'];external[str(archive)]=sha(archive)
 with np.load(archive,allow_pickle=False) as z:arrays={k:torch.from_numpy(z[k].copy()).view(torch.float16) for k in z.files}
 assert set(arrays)=={'vision_spatial','projected_visual','original_unscaled_text_embedding','actual_BART_encoder_inputs','actual_BART_encoder_output','actual_decoder_cross_attention_input'}
 for name,t in arrays.items():assert torch.isfinite(t).all();check(t,r['visual_tensor_records'][name])
 visual=arrays['projected_visual'];text=arrays['original_unscaled_text_embedding'];merged=arrays['actual_BART_encoder_inputs'];encoded=arrays['actual_BART_encoder_output'];cross=arrays['actual_decoder_cross_attention_input'];assert visual.shape==(1,577,768) and text.shape==(1,ids.shape[1],768) and merged.shape==encoded.shape==(1,length,768) and cross.shape==(3,length,768) and torch.equal(text,source_text[ids]) and torch.equal(merged,torch.cat([visual,text],dim=1)) and torch.equal(cross,encoded.repeat_interleave(3,dim=0));assert r['actual_vision_forward_calls']==r['actual_projector_forward_calls']==1 and r['actual_visual_encoder_prefix_verified'] and r['actual_decoder_cross_attention_equal_to_beam_expanded_encoder']
 check(positions['model.language_model.encoder.embed_positions.weight'][torch.arange(length)+2].unsqueeze(0),r['encoder_positions'])
 if r['encoder_bidirectional_mask'] is not None:check(torch.zeros((1,1,length,length),dtype=torch.float16),r['encoder_bidirectional_mask'])
 seq=r['generation_sequence_ids'];assert seq[0]==2;selected=seq.index(2,1) if 2 in seq[1:] else len(seq)-1;steps=r['actual_beam_search_steps'];assert 1<=selected<=256 and selected==r['selected_generation_token_count'] and 1<=steps<=256 and len(r['actual_beam_prefixes'])==len(r['decoder_positions'])==len(r['decoder_self_attention_masks'])==r['finite_actual_next_token_logit_vectors']==r['original_zero_final_logits_bias_addition_calls']==r['final_cache_length']==steps
 for j,(beam,pos,mask) in enumerate(zip(r['actual_beam_prefixes'],r['decoder_positions'],r['decoder_self_attention_masks'])):
  assert len(beam)==3 and all(len(b)==j+1 and b[0]==2 for b in beam) and pos['decoder_input_ids']==[[b[-1]] for b in beam] and pos['past_length']==j and pos['position_ids']==[j] and pos['device']==device;check(positions['model.language_model.decoder.embed_positions.weight'][j+2].reshape(1,1,768),pos['actual_learned_positions']);check(positions['model.language_model.decoder.embed_positions.weight'][j+2].reshape(1,1,768).expand(3,-1,-1),pos['original_beam_broadcast_positions'])
  if mask is not None:check(torch.zeros((3,1,1,j+1),dtype=torch.float16),mask)
  if j<selected:assert seq[:j+1] in beam
 raw_before=rust.decode(seq,skip_special_tokens=False).strip();assert raw_before==r['full_decoded_output_before_original_TAGS_cleanup'];raw=cleanup(raw_before);assert raw==r['raw_output']==o['raw_output']==o['prediction']==d['raw_output']==d['prediction']
 assert r['selected_budget_reached']==r['budget_reached']==r['truncated']==(selected==256) and r['beam_search_budget_reached']==(steps==256) and r['forced_eos_at_budget'] is False and r['allocator_before']==r['allocator_after']==0
 assert r['native_GELU_call_counts']==g['module_call_counts'] and g['attempt']==i+1 and g['device']==device and g['dtype']=='torch.float16' and g['error'] is None and g['beam_search_steps']==steps and len(g['module_call_counts'])==36 and all(v==(steps if name.startswith('model.language_model.decoder.') else 1) for name,v in g['module_call_counts'].items())
 for name,slots in r['cache_device_and_shapes'].items():
  assert name in ['self_attention_cache','cross_attention_cache'] and len(slots)==6
  for j,slot in enumerate(slots):assert slot['slot']==j and slot['dtype']=='torch.float16' and slot['key_device']==slot['value_device']==device and slot['key_shape']==slot['value_shape']==[3,12,steps if name=='self_attention_cache' else length,64]
 if not raw:empty.append(i)
 if raw and not any(c.isalnum() for c in raw):invalid.append(i)
 if selected==256:truncated.append(i)
 if steps==256:search_hits.append(i)
 values.append(rouge(raw,o['reference']))
score=sum(values)/count;assert math.isfinite(score) and abs(score-report['metrics']['ROUGE-L'])<1e-12 and report['statistics']['total_samples']==count and report['statistics']['error_count']==0 and len(external)==count+5
prof=json.loads(pathlib.Path(line['profiler_trace']).read_text());events=prof if isinstance(prof,list) else prof['traceEvents'];hardware={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};kernels=[e for e in events if e.get('pid') in hardware and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'ROUGE-L':score,'metrics':{'ROUGE-L':score},'sample_ROUGE_L_F1':values,'historical_record_reuse':False,'source_dtype':'F32','runtime_dtype':'F16','full_original_source_and_FP16_cast_NPU_SHAs_verified':667,'learned_parameter_count':665,'original_buffer_count':2,'source_to_original_FP16_underflow_total':463,'original_num_beams':3,'original_use_cache':True,'original_max_new_tokens':256,'florence_original_visual_prefix_and_cross_attention_replays':count,'independent_CPU_pixel_replays':count,'own_original_token_TAGS_cleanup_replays':count,'original_position_mask_beam_cache_replays':count,'original200_input_prediction_ROUGE_replayed':200,'original_head_untied_preserved':True,'original_three_embedding_aliases_preserved':True,'original_final_logits_bias_NPU_execution_verified':True,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'beam_search_budget_indices':search_hits,'index_duplicates':False,'NPU_profiler_kernel_events':len(kernels),'external_evidence_sha256':external,'CPU_original_vision_graph_audit':p['CPU_original_vision_graph_audit'],'CPU_original_vision_graph_audit_sha256':p['CPU_original_vision_graph_audit_sha256'],'original_mapping_pixel_review':p['mapping_pixel_review'],'original_mapping_pixel_review_sha256':p['mapping_pixel_review_sha256'],'native_GELU_probe':probe,'report_sha256':{q.name:sha(q) for q in folder.iterdir()},'protocol':'Matched original caption1459 image-only fixed detailed task, old PIL/NumPy processing, original explicit FP16 cast, complete own output head and original3 embedding aliases/final_logits_bias, own legacy577 visual prefix with original unscaled shared text embedding. Instance-only own exact original vision q-scaling/layout/softmax/projector equations CPU-checked, all native NPU learned layers. Original beam3/cacheTrue/256/noRepeat3/earlyStopping and explicit forcedBOS/forcedEOS=None retained. Full original tokens/uncleaned special-tag raw/cleaned raw, beam prefixes/positions/cache, true actual visual encoder and decoder cross attention, independent full denominator ROUGE. Native arithmetic differences retained, not lowbit acceleration or CPU official inference.'}
nr=p['native_numerical_review'];assert nr==line['native_numerical_review']
for source,digest in nr['evidence_sha256'].items():assert pathlib.Path(source).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(source)==digest
cpu=json.loads(pathlib.Path(nr['CPU_original_reference']).read_text());control=json.loads(pathlib.Path(nr['FP32_operator_control']).read_text());assert cpu['passed_diagnostic_integrity'] and cpu['CPU_only_not_official_scores'] and cpu['source_runtime_F16_cast_preserved'] and cpu['native_NPU_activation_attention_cache_math_unchanged_by_diagnostic'] and control['passed_diagnostic_control'] and control['CPU_and_FP32_NPU4_diagnostic_only'] and control['official_inference_remains_original_FP16'] and control['same_original_FP16_rounded_parameters_and_pixels'] and control['all_control_CPU_NPU_tensor_SHA_equal'] and control['HF32_enabled']=={'conv':False,'matmul':False}
assert cpu['CPU_original_vision_module_erf_vs_NPU_and_CPU_only_tanh']==nr['CPU_original_and_tanh_diagnostic_comparisons'] and control['comparisons']==nr['same_original_FP16_rounded_FP32_control_comparisons']
for diagnostic in [cpu,control]:
 for name,digest in diagnostic['original_real3_file_sha256'].items():assert sha(pathlib.Path(diagnostic.get('original_real3_report',diagnostic.get('original_real3_report')))/name)==digest
assert sha(p['native_GELU_method_note'])==p['native_GELU_method_note_sha256']=='0842aa75ef490056d6ccf8035fbb63684a98d709b193f59dcb2dba10b21d41d6'
proof['native_numerical_review']=nr;proof['florence_numerical_diagnostic_evidence_sha256']=nr['evidence_sha256'];proof['native_FP16_numerical_difference_retained_and_disclosed']=True;proof['prior_real3_CPU_diagnostic_reused_not_official_inference']=True
dest=pathlib.Path(line['weight_loading_evidence']).parent/'independent_report_verification.json';assert not dest.exists();dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2));print(json.dumps(proof,ensure_ascii=False,indent=2))
