"""Independent complete A100-aligned FP16 TrOCR input/cast/raw/CER/kernel audit; no inference or workbook writes."""
import pathlib,json,sys,hashlib,re,struct,math,os,importlib.util
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
torch.set_num_threads(4)
from PIL import Image
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from transformers import AutoConfig,ViTImageProcessor,PreTrainedTokenizerFast,TrOCRProcessor,VisionEncoderDecoderModel,GenerationConfig
from safetensors import safe_open

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def check(t,r):assert list(t.shape)==r['shape'] and str(t.dtype)==r['dtype'] and fp(t)==r['sha256']
def distance(a,b):
 # Independent O(len(reference)) memory algorithm, rather than production's 2D DP.
 costs=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  previous=costs[0];costs[0]=i
  for j,y in enumerate(b,1):
   old=costs[j];costs[j]=min(costs[j]+1,costs[j-1]+1,previous+(x!=y));previous=old
 return costs[-1]
def normalize(s):return re.sub(r'[\s\-\.\,\;\:\!\?\'\"\(\)\[\]\{\}\/\\\@\#\$\%\^\&\*\+\=\_\~\`\|]','',str(s or '').strip().upper())

folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];p=kw['protocol'];mid=cfg['model']['name'];assert p['model_id']==mid
assert folder.parent.name==cfg['task_type']=='image-to-text' and folder.parent.parent.name==mid.replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_range']=='0:'+str(count-1) and cfg['sample_count']==count and cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['model']['download'] is False
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert cfg['algorithm']['name']=='greedy_inference' and cfg['algorithm']['kwargs']=={'extract_answer':True,'ocr_extract':True} and p['extract_answer'] is True and p['ocr_extract'] is True and cfg['algorithm']['batch_size']==1
assert cfg['metrics']['override']==['CER'] and cfg['metrics']['include_failed_predictions'] is True
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];assert len(inputs)==len(outputs)==len(details)==count
for rows in (inputs,outputs,details):assert [r['index'] for r in rows]==list(range(count))
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];device='npu:'+str(kw['npu_devices'][0]);assert lineage['devices']==kw['npu_devices'] and len(kw['npu_devices'])==1 and set(lineage['module_placements'].values())=={device}
assert lineage['all_expected_parameters_loaded'] and lineage['source_dtype']=='F32' and lineage['execution_dtype']=='FP16' and lineage['actual_attention']==p['A100_resolved_attention'] and lineage['HF32_enabled']=={'conv':False,'matmul':False} and lineage['protocol_review']==p
assert lineage['weight_files']==p['weights'] and lineage['model_resource_sha256']==p['resource_sha256'] and lineage['runtime_source_sha256']==p['runtime_sha256']
for path,digest in dict(p['runtime_sha256'],**p['original_source_sha256']).items():assert sha(path)==digest
directory=pathlib.Path(cfg['model']['model_path']).resolve();assert directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels') and directory.parent.name+'/'+directory.name==mid
for name,digest in p['resource_sha256'].items():assert sha(directory/name)==digest
mc=AutoConfig.from_pretrained(str(directory),local_files_only=True);assert mc.encoder.model_type in ('vit','deit') and mc.decoder.model_type=='trocr'
with torch.device('meta'):blueprint=VisionEncoderDecoderModel(mc)
assert {'root':blueprint.config._attn_implementation,'encoder':blueprint.encoder.config._attn_implementation,'decoder':blueprint.decoder.config._attn_implementation}==p['A100_resolved_attention']
loading=json.loads(pathlib.Path(lineage['weight_loading_evidence']).read_text());weight=directory/next(iter(p['weights']));review=p['weights'][weight.name];assert weight.stat().st_size==review['size']
if weight.suffix=='.safetensors':
 with weight.open('rb') as f:size=f.read(8);header=f.read(struct.unpack('<Q',size)[0])
 assert hashlib.sha256(size+header).hexdigest()==review['header_sha256'];handle=safe_open(str(weight),framework='pt',device='cpu');source_keys=set(handle.keys());get_tensor=handle.get_tensor
else:
 assert sha(weight)==review['sha256']
 try:source=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
 except RuntimeError as error:
  if 'mmap' not in str(error):raise
  source=torch.load(weight,map_location='cpu',weights_only=True)
 source_keys=set(source);get_tensor=source.__getitem__
disabled=not any(k.startswith('encoder.pooler.') for k in source_keys)
assert lineage['disabled_absent_unused_encoder_pooler']==disabled
if disabled:blueprint.encoder.pooler=None
expected={n:list(v.shape) for n,v in blueprint.named_parameters(remove_duplicate=False)}
def name_map(key):
 parts=key.split('.')
 if parts[:3]==['encoder','encoder','layer']:
  layer=parts[3];tail='.'.join(parts[4:]);patterns={'attention.attention.query':'attention.q_proj','attention.attention.key':'attention.k_proj','attention.attention.value':'attention.v_proj','attention.output.dense':'attention.o_proj','intermediate.dense':'mlp.fc1','output.dense':'mlp.fc2'}
  for old,new in patterns.items():
   if tail.startswith(old+'.'):tail=new+tail[len(old):];break
  return 'encoder.layers.'+layer+'.'+tail
 return key
legacy={x['source_key'] for x in lineage['legacy_unlearned_sentinels']};assert legacy<=({'decoder.model.decoder.embed_positions._float_tensor'} if not mc.decoder.use_learned_position_embeddings else set())
assert len(loading)==lineage['weight_tensor_count'] and {x['source_key'] for x in loading}==source_keys-legacy and len({x['target_key'] for x in loading})==len(loading)
for r in loading:
 key=r['source_key'];target=name_map(key);tensor=get_tensor(key);assert r['source_file']==weight.name and r['target_key']==target and list(tensor.shape)==r['shape']==expected[target] and tensor.dtype==torch.float32
 assert r['source_dtype']=='torch.float32' and r['execution_dtype']=='torch.float16' and r['device']==device and fp(tensor)==r['source_tensor_sha256'] and fp(tensor.to(torch.float16))==r['original_A100_FP16_cast_sha256']==r['NPU_tensor_sha256']
 assert r['cast_roundtrip_changed_elements']==int((tensor.to(torch.float16).float()!=tensor).sum()) and r['nonzero_underflow_elements']==int(((tensor!=0)&(tensor.to(torch.float16)==0)).sum())
head='decoder.output_projection.weight';embed='decoder.model.decoder.embed_tokens.weight';tied=bool(mc.decoder.tie_word_embeddings);mapped={x['target_key'] for x in loading};assert set(expected)-mapped in (set(),{head}) and (head in mapped or tied)
assert lineage['declared_decoder_tie_word_embeddings']==lineage['actual_head_embedding_shared']==tied and fp(get_tensor(embed).to(torch.float16))==lineage['embedding_sha256']
assert fp((get_tensor(head) if head in source_keys else get_tensor(embed)).to(torch.float16))==lineage['head_sha256']
if tied:assert lineage['head_sha256']==lineage['embedding_sha256']
else:assert head in source_keys
for r in lineage['legacy_unlearned_sentinels']:
 t=get_tensor(r['source_key']);assert t.numel()==1 and list(t.shape)==r['shape'] and str(t.dtype)==r['dtype'] and fp(t)==r['sha256']
assert lineage['decoder_embedding_scale']==(math.sqrt(mc.decoder.hidden_size) if mc.decoder.scale_embedding else 1.0)
from tokenizers import Tokenizer,models,decoders,pre_tokenizers
backend=Tokenizer(models.BPE.from_file(str(directory/'vocab.json'),str(directory/'merges.txt'),unk_token='<unk>'));backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);backend.decoder=decoders.ByteLevel()
backend.add_special_tokens(['<s>','</s>','<pad>','<unk>','<mask>'])
tok=PreTrainedTokenizerFast(tokenizer_object=backend,bos_token='<s>',eos_token='</s>',pad_token='<pad>',unk_token='<unk>',mask_token='<mask>')
processor=TrOCRProcessor(image_processor=ViTImageProcessor.from_pretrained(str(directory),local_files_only=True),tokenizer=tok)
assert processor.tokenizer.get_vocab()==json.loads((directory/'vocab.json').read_text()) and json.loads(json.dumps(processor.image_processor.to_dict()))==lineage['effective_image_processor']
gen=GenerationConfig.from_pretrained(str(directory),local_files_only=True) if (directory/'generation_config.json').is_file() else GenerationConfig.from_model_config(mc);blueprint.generation_config=gen
resolved,_=blueprint._prepare_generation_config(None,max_new_tokens=p['max_new_tokens'],do_sample=False,temperature=0.0)
assert resolved.to_dict()==p['effective_generation_config']==lineage['effective_generation_config'] and resolved.do_sample is False and resolved.use_cache is False
eos=resolved.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos);assert eos==lineage['eos_token_ids']
position_name='decoder.model.decoder.embed_positions'
def analytic_positions(length):
 dim=mc.decoder.hidden_size;half=dim//2;rate=np.log(10000.0)/(half-1)
 # Exact reference math uses Torch CPU exp/sin/cos and explicit original dtype.
 omega=torch.exp(torch.arange(half,dtype=torch.int64).float()*-float(rate));phase=torch.arange(length,dtype=torch.int64).float()[:,None]*omega[None,:]
 matrix=torch.cat([phase.sin(),phase.cos()],dim=1)
 if dim%2:matrix=torch.cat([matrix,torch.zeros(length,1)],dim=1)
 matrix[mc.decoder.pad_token_id,:]=0
 return matrix
if not mc.decoder.use_learned_position_embeddings:
 initial=analytic_positions(p['NPU_host_sinusoidal_rebuilt']['need']);assert set(lineage['analytic_positions'])=={position_name}
 check(initial,lineage['analytic_positions'][position_name]['formula_cpu_F32'])
 check(initial.half(),lineage['analytic_positions'][position_name]['native_CPU_FP16']);check(initial.half(),lineage['analytic_positions'][position_name]['actual_NPU_FP16'])
 assert lineage['analytic_positions'][position_name]['native_CPU_FP16']==p['NPU_host_sinusoidal_rebuilt']['original_FP16']
else:assert not lineage['analytic_positions']
def position_replay(r):
 ids=torch.tensor(r['input_ids'],dtype=torch.int64);past=r['past_key_values_length'];assert past==0
 if mc.decoder.use_learned_position_embeddings:position=torch.arange(past,past+ids.shape[-1])[None,:].expand(ids.shape[0],-1)+2;values=get_tensor(position_name+'.weight').to(torch.float16)[position]
 else:
  mask=(ids!=mc.decoder.pad_token_id).int();position=(torch.cumsum(mask,dim=1).type_as(mask)+past)*mask+mc.decoder.pad_token_id;values=analytic_positions(p['NPU_host_sinusoidal_rebuilt']['need']).half()[position]
 check(values,r['output'])
dataset=pathlib.Path(cfg['data']['dataset_path']).resolve();assert dataset==pathlib.Path(p['dataset_path']).resolve() and dataset.is_relative_to('/root/projects/comperror/workspace/datasets') and sha(dataset)==p['dataset_sha256']
data=[json.loads(x) for x in dataset.read_text().splitlines()];assert len(data)==3000
reference=pathlib.Path(p['reference_report']);assert reference.is_relative_to(pathlib.Path(p['A100_reference_root'])) and sha(reference)==p['reference_report_sha256'];old=json.loads(reference.read_text());pc=old['pipeline_config'];assert len(old['detailed_results'])==200 and pc['model']['name']==mid and pc['metrics']['override']==['CER'] and pc['model']['kwargs']['max_new_tokens']==p['max_new_tokens']
assert pc['algorithm']['kwargs']['extract_answer']==p['extract_answer'] and pc['algorithm']['kwargs'].get('ocr_extract',False)==p['ocr_extract'] and not pc['model']['kwargs'].get('ocr_allowed_charset') and not pc['model']['kwargs'].get('ocr_plate_grammar')
assert sha(p['reference_log'])==p['reference_log_sha256'];flags=re.findall(r'开始推理循环: 共 200 条样本[^\n]*提取答案=(True|False)',pathlib.Path(p['reference_log']).read_text());assert flags and len(set(flags))==1 and (flags[0]=='True')==p['extract_answer']
# Use the frozen original OCR normalization/extraction recipe, not the new adapter or production CER implementation.
recipe_path=pathlib.Path('universal_eval/metric/A100_CER_alignment_20261103.py').resolve();assert str(recipe_path) in p['original_source_sha256'] and sha(recipe_path)==p['original_source_sha256'][str(recipe_path)]
spec=importlib.util.spec_from_file_location('frozen_original_ocr_recipe',recipe_path);recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
def prediction(raw):
 assert p['extract_answer'] and p['ocr_extract']
 selected=recipe.ocr_extract_plate(raw)
 return selected if selected is not None else raw
def pair(a,b):
 normalized=normalize(a)
 if re.search(r'\*\*|:\s|\n',a) or len(normalized)>16:
  extracted=recipe.ocr_extract_plate(a)
  if extracted is not None:normalized=normalize(extracted)
 return normalized,normalize(b)
old_errors=old_chars=0
for i,(r,s) in enumerate(zip(old['detailed_results'],data)):
 assert r['index']==i and r['prompt']==s['prompt'] and r['reference']==s['reference']
 raw=r.get('raw_output',r.get('extra',{}).get('raw_output'));assert raw is not None and prediction(raw)==r['prediction']
 a,b=pair(r['prediction'],r['reference']);old_errors+=distance(a,b);old_chars+=len(b)
assert abs(old_errors/old_chars-old['metrics']['CER'])<1e-12
trace=[json.loads(x) for x in pathlib.Path(lineage['per_sample_evidence']).read_text().splitlines()];assert len(trace)==count
errors=characters=0;empty=[];invalid=[];truncated=[];per_sample=[];artifacts={}
for i,(a,b,d,s,e) in enumerate(zip(inputs,outputs,details,data,trace)):
 assert a['prompt']==s['prompt'] and a['reference']==b['reference']==d['reference']==s['reference'] and b['extra']['execution']==e and e['error'] is None and b['error'] is None
 media=pathlib.Path(a['media_path']).resolve();source=pathlib.Path(s['media_path']);source=source if source.is_absolute() else dataset.parent/source
 assert media==source.resolve() and media.is_relative_to(dataset.parent) and sha(media)==e['source_image_sha256']
 assert e['prompt']==a['prompt']+('\n\nProvide only the final answer.' if p['extract_answer'] else '') and e['text_prompt_used_by_model'] is False
 with Image.open(media) as image:prepared=processor(images=image.convert('RGB'),return_tensors='pt')
 assert set(prepared)=={'pixel_values'};check(prepared['pixel_values'],e['CPU_pixel_values'])
 # Independent actual Torchvision uint8 resize primitive, explicit fused FP32 normalization and original FP16 rounding.
 with Image.open(media) as im: u8=torch.from_numpy(np.ascontiguousarray(np.asarray(im.convert('RGB')).transpose(2,0,1)))[None]
 resized=torch.nn.functional.interpolate(u8,size=(384,384),mode='bilinear',align_corners=False,antialias=True);assert resized.dtype==torch.uint8
 manual=(resized.float()-127.5)/127.5
 check(manual,e['CPU_pixel_values']);check(manual.to(torch.float16),e['original_A100_FP16_CPU_pixels']);check(manual.to(torch.float16),e['actual_NPU_pixels'])
 A100_pixel=p['A100_CPU200_pixels'][i];assert A100_pixel['index']==i and A100_pixel['image_sha256']==sha(media) and A100_pixel['F32_pixel_sha256']==fp(manual) and A100_pixel['FP16_pixel_sha256']==fp(manual.to(torch.float16))
 archive=pathlib.Path(e['visual_feature_archive']);assert archive.parent.parent==pathlib.Path(lineage['weight_loading_evidence']).parent and archive.parent.name=='visual_features' and sha(archive)==e['visual_feature_archive_sha256'];artifacts[str(archive)]=e['visual_feature_archive_sha256']
 with np.load(archive,allow_pickle=False) as z:
  assert set(z.files)=={'encoder_output_bits','decoder_cross_attention_bits'}
  vision=torch.from_numpy(z['encoder_output_bits'].copy()).view(torch.float16).reshape(e['encoder_output']['shape']);features=torch.from_numpy(z['decoder_cross_attention_bits'].copy()).view(torch.float16).reshape(e['decoder_cross_attention_features']['shape']);check(vision,e['encoder_output']);check(features,e['decoder_cross_attention_features'])
  assert vision.dtype==features.dtype==torch.float16 and bool(torch.isfinite(vision).all()) and bool(torch.isfinite(features).all()) and bool(features.abs().sum()>0)
  assert vision.shape[0]==1 and vision.shape[-1]==mc.encoder.hidden_size
  if not hasattr(blueprint,'enc_to_dec_proj'):assert torch.equal(features,vision)
  assert e['actual_decoder_beam_batch']==resolved.num_beams and e['visual_cross_attention_verified'] is True
  expected=features.repeat_interleave(resolved.num_beams//features.shape[0],dim=0);check(expected,e['actual_cross_attention_input'])
 position_replay(e['position_first']);position_replay(e['position_last'])
 assert e['position_first']['input_ids']==[[resolved.decoder_start_token_id]]*resolved.num_beams
 tokens=e['generation_sequence_ids'];assert tokens[0]==resolved.decoder_start_token_id and e['generation_count']==len(tokens)-1<=p['max_new_tokens']
 raw=backend.decode(tokens,skip_special_tokens=True).strip();assert raw==e['raw_output']==b['raw_output'];pred=prediction(raw);assert pred==b['prediction']==d['prediction']
 assert e['actual_full_vocabulary_size']==mc.decoder.vocab_size and e['actual_greedy_full_vocab_argmax_ids']==tokens[1:]
 clip=len(tokens)-1>=p['max_new_tokens'];assert clip==e['truncated']
 if clip:truncated.append(i)
 if not raw:empty.append(i)
 x,y=pair(pred,b['reference']);assert y
 if not x:invalid.append(i)
 edit=distance(x,y);errors+=edit;characters+=len(y);per_sample.append({'index':i,'normalized_prediction':x,'normalized_reference':y,'edit_distance':edit,'reference_characters':len(y)})
score=errors/characters
assert abs(report['metrics']['CER']-score)<1e-12 and report['summary']['primary_metric']=='CER' and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
npu=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=npu if isinstance(npu,list) else npu['traceEvents'];hardware={x['pid'] for x in events if x.get('name')=='process_name' and 'Ascend Hardware' in x.get('args',{}).get('name','')};kernels=[x for x in events if x.get('pid') in hardware and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'report_dir':str(folder),'passed':True,'sample_count':count,'metrics':{'CER':score},'CER':score,'historical_record_reuse':False,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'CPU_image_preprocessing_replayed':count,'original_vocabulary_and_decoded_sequences_replayed':count,'CPU_position_embeddings_replayed':count,'real_visual_cross_attention_replayed':count,'historical_inputs_references_extraction_and_CER_replayed':200,'historical_reference_CER':old_errors/old_chars,'reference_character_denominator':characters,'total_edit_distance':errors,'per_sample_edit_distances':per_sample,'NPU_profiler_kernel_events':len(kernels),'source_dtype':'F32','execution_dtype':'FP16','full_vocabulary_argmax_greedy_steps_replayed':sum(len(e['generation_sequence_ids'])-1 for e in trace),'actual_attention':p['A100_resolved_attention'],'original_max_new_tokens':p['max_new_tokens'],'original_num_beams':resolved.num_beams,'original_use_cache':resolved.use_cache,'external_evidence_sha256':artifacts,'protocol':'A100 image-only OCR/CER with exact own F32-to-FP16 conversion, original default SDPA vision and eager cross-attention TrOCR decoder, own complete vocabulary, original beam/cache/decoding budget and runtime-log grounded OCR extraction. Independent CPU pixel, complete parameter SHA, actual cross-attention arrays, position, token decoding and one-row edit-distance/full-reference denominator replay. No model/data downloads, substituted weights, new character constraints or low-bit acceleration claims.','report_sha256':{q.name:sha(q) for q in folder.iterdir()}}
proof['A100_full_reference_CER']=old_errors/old_chars;proof['A100_table_D']=p['A100_table_D'];proof['relative_error_to_A100_table_D']=abs(score-p['A100_table_D'])/abs(p['A100_table_D']);proof['formal_alignment_within_5pct']=count==200 and abs(score-p['A100_table_D'])/abs(p['A100_table_D'])<=.05;proof['new_A100_aligned_inference_evidence_only_no_G_overwrite']=True
for path in [lineage['per_sample_evidence'],lineage['weight_loading_evidence'],lineage['profiler_trace']]:proof['external_evidence_sha256'][path]=sha(path)
nr=pathlib.Path(p['A100_numerical_review']);assert sha(nr)==p['A100_numerical_review_sha256'];numerics=json.loads(nr.read_text());assert numerics['passed'] and numerics['official_A100_aligned_execution_remains_FP16_SDPA_vision_eager_decoder'] and numerics['CPU_visual_only_no_LM_forward_no_answer_or_score_substitution'] and numerics['native_NPU_differences_retained_not_claimed_eliminated'] and numerics['control_parameters']==162
for path,h in numerics['evidence_sha256'].items():assert sha(path)==h;proof['external_evidence_sha256'][path]=h
proof['external_evidence_sha256'][str(nr)]=sha(nr)
pr=pathlib.Path(p['CPU_position_numerical_review']);assert sha(pr)==p['CPU_position_numerical_review_sha256'];pos=json.loads(pr.read_text())
assert pos['same_exact_native_CPU_formula_source'] and pos['same_original_FP32_formula_then_FP16_cast'] and pos['A100_FP16_sha256']==p['A100_sinusoidal_original_rebuilt']['original_FP16']['sha256'] and pos['910B_host_FP16_sha256']==p['NPU_host_sinusoidal_rebuilt']['original_FP16']['sha256']
proof['external_evidence_sha256'][str(pr)]=sha(pr)
assert lineage['static_numerical_disclosure']['sha256']==sha(nr) and lineage['original_CPU_position_rounding_disclosure']['sha256']==sha(pr)
proof['A100_FP16_numerical_difference_disclosure']={'static_review':str(nr),'sha256':sha(nr),'CPU_only_diagnostics_from_prior_real3_not_repeated_for_each_report':True,'per_image':numerics['per_image'],'official_FP16_unchanged_no_CPU_answer_or_error_elimination_claim':True}
proof['cross_CPU_analytic_position_rounding_disclosure']={'review':str(pr),'sha256':sha(pr),'same_exact_source_formula_and_original_FP16_precision':True,'actual_full_A100_vs_native_CPU_difference':pos,'NPU_position_outputs_independently_replayed_against_unchanged_native_host_formula':True,'not_claimed_bitwise_cross_CPU_equality':True}

dest=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json';assert not dest.exists();dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2));print(json.dumps(proof,ensure_ascii=False,indent=2))
