"""Independent full-weight/CPU-pixel/original-protocol/raw-token/prefix/cache/kernel/ROUGE verification."""
import os,pathlib,json,hashlib,sys,struct,re,math,bisect,ast,typing
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from PIL import Image
from tokenizers import Tokenizer
from safetensors import safe_open
from transformers import AutoConfig,AutoTokenizer,AutoImageProcessor,PaliGemmaProcessor,PaliGemmaForConditionalGeneration,GenerationConfig
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from universal_eval.metric.rouge_910b_porter import PorterStemmer
torch.set_num_threads(4)
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def check(t,r):assert list(t.shape)==r['shape'] and str(t.dtype)==r['dtype'] and fp(t)==r['sha256']
def pixels(image,ip):
 size=(ip.size['width'],ip.size['height']);values=np.asarray(image.resize(size,resample=Image.Resampling(ip.resample))).astype(np.float64)
 values=(values*ip.rescale_factor).astype(np.float32);values=(values-np.asarray(ip.image_mean,dtype=np.float32))/np.asarray(ip.image_std,dtype=np.float32)
 return torch.from_numpy(values.transpose(2,0,1).copy())[None]
stem=PorterStemmer()
def words(x):return [stem.stem(w) if len(w)>3 else w for w in re.findall('[a-z0-9]+',str(x).lower())]
def rouge(pred,ref):
 a=words(' '.join(ref) if isinstance(ref,list) else ref);b=words(pred)
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
def clean(text):
 text=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',text);text=re.sub(r'[ \t]+',' ',text);text=re.sub(r'\n{3,}','\n\n',text).strip();return re.sub(r'<image\d+>\s*','',text)
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {x.name for x in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];p=cfg['model']['kwargs']['protocol'];wp=pathlib.Path(cfg['model']['model_path']);dtype={'F32':torch.float32,'BF16':torch.bfloat16}[p['source_dtype']]
assert cfg['model']['name']==p['model_id']==wp.parent.name+'/'+wp.name and wp.resolve().is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels') and cfg['model']['download'] is False and not p.get('LoRA_backup')
assert cfg['task_type']==p['workbook_task']==folder.parent.name and folder.parent.parent.name==p['model_id'].replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['data']['filter_empty'] is False and cfg['data']['download'] is False
assert cfg['algorithm']['name']=='paligemma_reference_caption_inference' and cfg['algorithm']['batch_size']==p['original_batch_size']==1 and cfg['algorithm']['kwargs']=={'extract_answer':p['extract_answer']}
assert cfg['metrics']['override']==['ROUGE-L'] and cfg['metrics']['include_failed_predictions'] is True and report['summary']['primary_metric']=='ROUGE-L'
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];assert len(inputs)==len(outputs)==len(details)==count
for rows in [inputs,outputs,details]:assert [x['index'] for x in rows]==list(range(count))
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];assert lineage['protocol']=='paired_paligemma_original_1459_caption' and lineage['protocol_review']==p and lineage['all_expected_parameters_loaded'] and lineage['tied_output_head_restored'] and lineage['source_dtype']==p['source_dtype'] and lineage['HF32_enabled']=={'conv':False,'matmul':False} and lineage['native_allocator_setting']=='expandable_segments:True'
assert lineage['devices']==cfg['model']['kwargs']['npu_devices'] and len(lineage['devices'])==1;device='npu:'+str(lineage['devices'][0]);assert set(lineage['module_placements'].values())=={device}
for rel,digest in p['resource_sha256'].items():assert pathlib.Path(rel).name==rel and sha(wp/rel)==digest
for source,digest in dict(p['runtime_sha256'],**p['historical_sources']).items():assert sha(source)==digest
for name in ['reference_report','reference_log','original_dataset']:assert sha(p[name])==p[name+'_sha256']
old=json.loads(pathlib.Path(p['reference_report']).read_text());oldrows=old['detailed_results'];assert len(oldrows)==200 and old['pipeline_config']['model']['name']==p['model_id']
flags=re.findall(r'开始推理循环: 共 200 条样本[^\n]*提取答案=(True|False)',pathlib.Path(p['reference_log']).read_text());assert flags and len(set(flags))==1 and (flags[0]=='True')==p['extract_answer']
algorithm_source=pathlib.Path('/mnt/afs/comp_error/universal_eval/algorithm/inference.py');assert str(algorithm_source) in p['historical_sources'];tree=ast.parse(algorithm_source.read_text());original_cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GreedyInference');members=[n for n in original_cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ANSWER_PATTERNS' for t in n.targets) or isinstance(n,ast.FunctionDef) and n.name in ['_extract_answer','_add_answer_instruction','_extract_committed_option']];node=ast.ClassDef(name='OriginalAlgorithm',bases=[],keywords=[],body=members,decorator_list=[]);space={'re':re,'Optional':typing.Optional};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<independent original protocol>','exec'),space);algorithm=space['OriginalAlgorithm']()
sent_source=pathlib.Path('/mnt/afs/comp_error/workspace/diagnostics/paired_ten_20261018/runner/universal_eval/model/adapters/paligemma_batch4_c500.py');tree=ast.parse(sent_source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ABBREV_RE' for t in n.targets) or isinstance(n,ast.FunctionDef) and n.name=='_first_sentence'];space={'re':re,'List':typing.List};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<original first sentence>','exec'),space);first_sentence=space['_first_sentence']
mc=AutoConfig.from_pretrained(wp,local_files_only=True);assert mc.tie_word_embeddings and mc.architectures==['PaliGemmaForConditionalGeneration'] and mc.vision_config.image_size==p['expected_image_size']
with torch.device('meta'):model=PaliGemmaForConditionalGeneration(mc)
expected=dict(model.named_parameters());assert len(expected)==p['learned_tensor_count']==lineage['weight_tensor_count'];loading=json.loads(pathlib.Path(lineage['weight_loading_evidence']).read_text());assert len(loading)==len(expected) and {r['target_key'] for r in loading}==set(expected)
def target_key(k):
 parts=k.split('.')
 if parts[0]=='vision_tower':return '.'.join(['model','vision_tower']+parts[2 if parts[1]=='vision_model' else 1:])
 if parts[0]=='multi_modal_projector':return 'model.'+k
 if parts[:2]==['language_model','model']:return '.'.join(['model','language_model']+parts[2:])
 if parts[:2]==['language_model','lm_head']:return '.'.join(parts[1:])
 return k
index=json.loads((wp/'model.safetensors.index.json').read_text())['weight_map'];assert set(index)=={r['source_key'] for r in loading}
for shard,review in p['weight_headers'].items():
 path=wp/shard
 with path.open('rb') as f:n=f.read(8);head=f.read(struct.unpack('<Q',n)[0])
 assert path.stat().st_size==review['size'] and hashlib.sha256(n+head).hexdigest()==review['header_sha256']
 with safe_open(str(path),framework='pt',device='cpu') as f:
  assert set(f.keys())=={k for k,v in index.items() if v==shard}
  for row in [r for r in loading if r['source_file']==shard]:
   key=row['source_key'];v=f.get_tensor(key);assert row['target_key']==target_key(key) and row['dtype']==str(v.dtype)==str(dtype) and row['device']==device and row['shape']==list(v.shape)==list(expected[row['target_key']].shape) and fp(v)==row['source_sha256']==row['NPU_sha256']
tok=AutoTokenizer.from_pretrained(wp,local_files_only=True);rust=Tokenizer.from_file(str(wp/'tokenizer.json'));ip=AutoImageProcessor.from_pretrained(wp,local_files_only=True,backend='pil');ip.image_seq_length=p['image_seq_length'];proc=PaliGemmaProcessor(image_processor=ip,tokenizer=tok);assert json.loads(json.dumps(ip.to_dict()))==lineage['effective_image_processor']
assert p['generation_config']==lineage['effective_generation_config'];gen=GenerationConfig.from_dict(p['generation_config']);assert gen.use_cache is True and gen.num_beams==1 and gen.do_sample is False and gen.max_new_tokens==p['max_new_tokens']
own=GenerationConfig.from_pretrained(wp,local_files_only=True);own.do_sample=False;own.max_new_tokens=p['max_new_tokens'];own.use_cache=True
if p['prompt_mode']=='vllm_prefix_template':own.num_beams=1;own.repetition_penalty=1.;own.min_new_tokens=0;own.min_length=0
resolved,_=model._prepare_generation_config(own);assert resolved.to_dict()==p['generation_config'];eos=gen.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
rp=mc.text_config.rope_parameters;dim=mc.text_config.head_dim or mc.text_config.hidden_size//mc.text_config.num_attention_heads;freq=1/(float(rp['rope_theta'])**(torch.arange(0,dim,2).float()/dim));assert torch.equal(freq,torch.tensor(lineage['text_inv_freq']))
seq=(mc.vision_config.image_size//mc.vision_config.patch_size)**2;assert seq==lineage['image_seq_length']==p['image_seq_length'];native_buffers=dict(model.named_buffers());assert set(native_buffers)==set(lineage['analytic_buffers'])
for name,v in native_buffers.items():
 if name.endswith('position_ids'):tensor=torch.arange(seq,dtype=torch.int64).reshape(1,-1)
 elif name.endswith('embed_scale'):tensor=torch.tensor(mc.text_config.hidden_size**.5,dtype=v.dtype)
 elif name.endswith('inv_freq'):tensor=freq
 else:raise AssertionError('Unknown analytic buffer '+name)
 check(tensor,lineage['analytic_buffers'][name])
dataset=pathlib.Path(p['dataset_path']);assert str(dataset)==cfg['data']['dataset_path'] and sha(dataset)==p['dataset_sha256']=='ceb1670f2848955290915dbe375dca8b34b62d6980e9f6cd1abe38bf0e68963b';source=[json.loads(x) for x in dataset.read_text().splitlines()];original=[json.loads(x) for x in pathlib.Path(p['original_dataset']).read_text().splitlines()];assert len(source)==p['dataset_count']==1459 and len(original)>=200
oldscore=0.
for i,(r,s,o) in enumerate(zip(oldrows,source,original)):
 assert r['index']==i and r['prompt']==s['prompt']==o['prompt'] and r['reference']==s['reference']==o['reference']
 img=(dataset.parent/s['media_path']).resolve();oldimg=(pathlib.Path(p['original_dataset']).parent/o['media_path']).resolve();assert sha(img)==sha(oldimg)==p['reference_first200_image_sha256'][str(img)]
 prediction=(algorithm._extract_answer(r['raw_output'],is_multiple_choice=False) or r['raw_output']) if p['extract_answer'] else r['raw_output'];assert prediction==r['prediction'];oldscore+=rouge(prediction,r['reference'])/200
assert abs(oldscore-old['metrics']['ROUGE-L'])<1e-12
tracepath=pathlib.Path(lineage['per_sample_evidence']);trace=[json.loads(x) for x in tracepath.read_text().splitlines()];assert len(trace)==count and [r['attempt'] for r in trace]==list(range(1,count+1))
external={str(tracepath):sha(tracepath),lineage['weight_loading_evidence']:sha(lineage['weight_loading_evidence']),lineage['profiler_trace']:sha(lineage['profiler_trace'])};empty=[];invalid=[];truncated=[];scores=[]
for i,(a,b,d,s,e) in enumerate(zip(inputs,outputs,details,source,trace)):
 assert a['prompt']==clean(s['prompt']) and a['reference']==b['reference']==d['reference']==s['reference'] and b['extra']['execution']==e and e['error'] is None and b['error'] is None
 incoming=algorithm._add_answer_instruction(a['prompt']) if p['extract_answer'] else a['prompt'];assert e['prompt']==incoming
 text=incoming
 if p['strip_eval_instruction']:
  for pattern in [r'\n\n(?:请仔细分析图像和题目[^\n]*|Answer the question by providing only the option letter[^\n]*)\s*$',r'\n\nProvide only the final answer\.\s*$']:text=re.sub(pattern,'',text)
 text=p['prefix']+text+p['mc_instruction']
 if p['prompt_mode']=='vllm_prefix_template':text='<image>'+text
 assert text==e['processor_text'];media=pathlib.Path(a['media_path']).resolve();assert media==(dataset.parent/s['media_path']).resolve() and sha(media)==e['source_image_sha256']==p['reference_first200_image_sha256'][str(media)]
 with Image.open(media) as im:image=im.convert('RGB')
 prepared=proc(text=text,images=image,return_tensors='pt');manual=pixels(image,ip);assert torch.equal(manual,prepared['pixel_values'])
 for name,v in prepared.items():check(v,e['CPU_inputs'][name])
 assert prepared['input_ids'].tolist()==e['input_ids'] and prepared['attention_mask'].tolist()==e['attention_mask'] and prepared['token_type_ids'].tolist()==e['token_type_ids'] and (prepared['input_ids']==mc.image_token_id).sum()==seq and (prepared['token_type_ids']==0).all()
 check(manual.to(dtype=dtype),e['actual_NPU_pixels']);assert e['excluded_training_labels']==('labels' in prepared) and ('labels' not in prepared or (prepared['labels']==-100).all())
 length=prepared['input_ids'].shape[-1];check(torch.arange(1,length+1).reshape(1,-1),e['text_position_ids']);names={'full_attention','sliding_attention'} if getattr(mc.text_config,'sliding_window',None) is not None else {'full_attention'};assert set(e['prefix_attention_masks'])==names
 for mask in e['prefix_attention_masks'].values():check(torch.zeros([1,1,length,length],dtype=dtype),mask)
 archive=pathlib.Path(e['visual_feature_archive']);assert sha(archive)==e['visual_feature_archive_sha256'];external[str(archive)]=sha(archive)
 with np.load(archive,allow_pickle=False) as z:visual=torch.from_numpy(z['projected_visual_u8'].copy()).view(dtype).reshape(1,seq,mc.text_config.hidden_size);prefix=torch.from_numpy(z['actual_decoder_visual_prefix_u8'].copy()).view(dtype).reshape(visual.shape)
 check(visual,e['projected_visual']);check(prefix,e['actual_decoder_visual_prefix']);assert torch.equal(visual,prefix) and torch.isfinite(visual).all() and e['actual_visual_prefix_verified'] and e['vision_forward_calls']>0
 ids=e['generation_token_ids'];assert len(ids)==e['generation_count']<=p['max_new_tokens'];full=rust.decode(ids,skip_special_tokens=True)
 if tok.clean_up_tokenization_spaces:
  for x,y in [(' .','.'),(' ?','?'),(' !','!'),(' ,',','),(" ' ","'"),(" n't","n't"),(" 'm","'m"),(" 's","'s"),(" 've","'ve"),(" 're","'re")]:full=full.replace(x,y)
 assert full==e['full_decoded_output_before_original_postprocess'];raw=full if p['prompt_mode']=='vllm_prefix_template' else full.strip()
 if p['truncate_first_sentence']:raw=first_sentence(raw)
 assert raw==e['raw_output']==b['raw_output'];prediction=(algorithm._extract_answer(raw,is_multiple_choice=False) or raw) if p['extract_answer'] else raw;assert prediction==b['prediction']==d['prediction']
 if not raw:empty.append(i)
 if raw and not re.search('[a-zA-Z0-9]',raw):invalid.append(i)
 cut=len(ids)>=p['max_new_tokens'] and (not ids or ids[-1] not in eos);assert cut==e['truncated']
 if cut:truncated.append(i)
 assert e['allocator_before']==e['allocator_after']==0 and e['final_cache_length']==length+len(ids)-1 and len(e['cache_device_and_shapes'])==mc.text_config.num_hidden_layers
 for layer in e['cache_device_and_shapes']:assert layer['key_device']==layer['value_device']==device
 scores.append(rouge(prediction,b['reference']))
score=sum(scores)/count;assert math.isfinite(score) and abs(score-report['metrics']['ROUGE-L'])<1e-12 and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
prof=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=prof if isinstance(prof,list) else prof['traceEvents'];hardware={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'};kernels=[x for x in events if x.get('pid') in hardware and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'metrics':{'ROUGE-L':score},'ROUGE-L':score,'historical_record_reuse':False,'source_dtype':p['source_dtype'],'full_original_tensor_SHA_verified':len(loading),'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'paligemma_paired_visual_prefix_replays':count,'independent_CPU_pixel_replays':count,'own_original_input_token_decode_replays':count,'native_prefix_position_mask_cache_replays':count,'original200_prediction_and_ROUGE_replayed':200,'NPU_profiler_kernel_events':len(kernels),'external_evidence_sha256':external,'sample_ROUGE_L_F1':scores,'report_sha256':{f.name:sha(f) for f in folder.iterdir()},'protocol':'Matched original paired/high7 caption1459, full original source F32/BF16 NPU native layers and all source SHA, HF32off, no CPU official inference. Original per-model prefix/template/strip/first-sentence/extract/budget/cache/greedy retained. Full generated token stream before original postprocess, native decoder visual-prefix bytes, CPU pixels/token/1-based positions/bidirectional masks/actual NPU cache, allocator noOOM and kernels independently verified. Fixed full denominator including genuine empty/invalid/truncated; no downloads or replacement resources.'}
dest=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json';assert not dest.exists();dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2));print(json.dumps(proof,ensure_ascii=False,indent=2))
