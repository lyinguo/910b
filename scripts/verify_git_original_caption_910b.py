"""Independent complete source/pixel/prefix/position/cache/token/kernel and BLEU/ROUGE audit."""
import os,pathlib,sys,json,hashlib,struct,re,math,bisect,collections
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
torch.set_num_threads(4)
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoConfig,AutoTokenizer,AutoImageProcessor,GitProcessor,GitForCausalLM,GenerationConfig
from safetensors import safe_open
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from universal_eval.metric.rouge_910b_porter import PorterStemmer
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def check(t,r):assert list(t.shape)==r['shape'] and str(t.dtype)==r['dtype'] and fp(t)==r['sha256']
def independent_pixel(image,ip):
 width,height=image.size;short=ip.size['shortest_edge']
 if width<=height:size=(short,int(height*short/width))
 else:size=(int(width*short/height),short)
 resized=image.resize(size,resample=Image.Resampling.BICUBIC);cw=ip.crop_size['width'];ch=ip.crop_size['height'];left=(size[0]-cw)//2;top=(size[1]-ch)//2;cropped=resized.crop((left,top,left+cw,top+ch))
 values=(np.asarray(cropped).astype(np.float64)*ip.rescale_factor).astype(np.float32);values=(values-np.asarray(ip.image_mean,dtype=np.float32))/np.asarray(ip.image_std,dtype=np.float32)
 return torch.from_numpy(values.transpose(2,0,1).copy())[None]
def bleu(pred,ref):
 words=str(pred or '').strip().split();references=[str(x).strip().split() for x in ([ref] if isinstance(ref,str) else ref)]
 if not words or not references or any(not x for x in references):return 0.
 logs=[]
 for order in range(1,5):
  grams=list(zip(*[words[j:] for j in range(order)]));unique=set(grams);matched=0
  for gram in unique:
   available=max(sum(1 for k in range(len(r)-order+1) if tuple(r[k:k+order])==gram) for r in references);matched+=min(grams.count(gram),available)
  if order==1 and matched==0:return 0.
  logs.append(math.log((matched if matched else 1./10)/max(1,len(grams))))
 closest=sorted((abs(len(r)-len(words)),len(r)) for r in references)[0][1];bp=min(1.,math.exp(1.-closest/len(words)))
 return bp*math.exp(math.fsum(logs)/4.)
stem=PorterStemmer()
def terms(x):return [stem.stem(w) if len(w)>3 else w for w in re.findall('[a-z0-9]+',str(x).lower())]
def rouge(pred,ref):
 a=terms(' '.join(ref) if isinstance(ref,list) else ref);b=terms(pred or '')
 if not a or not b:return 0.
 matches={}
 for i,w in enumerate(b):matches.setdefault(w,[]).append(i)
 end=[]
 for w in a:
  for i in reversed(matches.get(w,[])):
   j=bisect.bisect_left(end,i)
   if j==len(end):end.append(i)
   else:end[j]=i
 return 2.*len(end)/(len(a)+len(b))
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];p=cfg['model']['kwargs']['protocol'];wp=pathlib.Path(cfg['model']['model_path'])
assert cfg['model']['name']==p['model_id']=='microsoft/git-large-coco' and cfg['model']['download'] is False and wp.resolve().is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels')
assert cfg['task_type']==folder.parent.name=='image-to-text' and folder.parent.parent.name==cfg['model']['name'].replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_range']=='0:'+str(count-1) and cfg['sample_count']==count and cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['algorithm']['kwargs']=={'extract_answer':False} and cfg['algorithm']['batch_size']==4
assert cfg['metrics']['override']==['BLEU-4','ROUGE-L'] and cfg['metrics']['include_failed_predictions'] is True and report['summary']['primary_metric']=='ROUGE-L'
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];assert len(inputs)==len(outputs)==len(details)==count
for rows in (inputs,outputs,details):assert [x['index'] for x in rows]==list(range(count))
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];assert lineage['protocol']=='original_GIT_image_only_caption_768' and lineage['protocol_review']==p and lineage['HF32_enabled']=={'conv':False,'matmul':False} and lineage['all_expected_parameters_loaded']
assert lineage['devices']==cfg['model']['kwargs']['npu_devices'] and len(lineage['devices'])==1;device='npu:'+str(lineage['devices'][0]);assert set(lineage['module_placements'].values())=={device} and lineage['source_dtype']=='F32'
for rel,digest in p['resource_sha256'].items():assert sha(wp/rel)==digest
for path,digest in p['runtime_sha256'].items():assert sha(path)==digest
for key in ['reference_report','reference_log','historical_adapter','historical_parent']:assert sha(p[key])==p[key+'_sha256']
assert p['max_new_tokens']==768 and p['extract_answer'] is False and p['skip_text_input'] is True
archival=pathlib.Path(p['historical_adapter']).read_text();assert 'max_new_tokens=self._max_new_tokens' in archival and '_eff_max_new' not in archival and 'min_new_tokens' not in archival
old=json.loads(pathlib.Path(p['reference_report']).read_text());assert old['pipeline_config']['model']['kwargs']['max_new_tokens']==768 and old['pipeline_config']['model']['kwargs']['skip_text_input'] and '提取答案=False' in pathlib.Path(p['reference_log']).read_text() and len(old['detailed_results'])==200
old_scores={'BLEU-4':sum(bleu(x['raw_output'],x['reference']) for x in old['detailed_results'])/200,'ROUGE-L':sum(rouge(x['raw_output'],x['reference']) for x in old['detailed_results'])/200}
for metric,value in old_scores.items():assert abs(value-old['metrics'][metric])<1e-12
mc=AutoConfig.from_pretrained(wp,local_files_only=True);mc._attn_implementation='eager';mc.vision_config._attn_implementation='eager';assert mc.tie_word_embeddings is False
with torch.device('meta'):blueprint=GitForCausalLM(mc)
params=dict(blueprint.named_parameters(remove_duplicate=False));buffers=dict(blueprint.named_buffers());assert len(params)==497 and len(buffers)==2
loading=json.loads(pathlib.Path(lineage['weight_loading_evidence']).read_text());assert len(loading)==499 and {x['source_key'] for x in loading}==set(params)|set(buffers)
for shard,review in p['weight_headers'].items():
 path=wp/shard
 with path.open('rb') as f:n=f.read(8);h=f.read(struct.unpack('<Q',n)[0])
 assert path.stat().st_size==review['size'] and hashlib.sha256(n+h).hexdigest()==review['header_sha256']
 with safe_open(str(path),framework='pt',device='cpu') as f:
  assert set(f.keys())=={r['source_key'] for r in loading if r['source_file']==shard}
  for row in (r for r in loading if r['source_file']==shard):
   key=row['source_key'];v=f.get_tensor(key);assert row['target_key']==key and row['device']==device and row['dtype']==str(v.dtype) and row['shape']==list(v.shape)==list((params|buffers)[key].shape) and fp(v)==row['source_sha256']==row['NPU_sha256']
   assert v.dtype==(torch.float32 if key in params else torch.int64)
   if key in buffers:assert torch.equal(v,torch.arange(v.shape[-1]).reshape(v.shape))
tok=AutoTokenizer.from_pretrained(wp,local_files_only=True);rust=Tokenizer.from_file(str(wp/'tokenizer.json'));ip=AutoImageProcessor.from_pretrained(wp,local_files_only=True,backend='pil');proc=GitProcessor(image_processor=ip,tokenizer=tok);assert json.loads(json.dumps(ip.to_dict()))==lineage['effective_image_processor']
gen=GenerationConfig.from_pretrained(wp,local_files_only=True);blueprint.generation_config=gen;resolved,_=blueprint._prepare_generation_config(None,max_new_tokens=768,do_sample=False);assert resolved.to_dict()==p['effective_generation_config']==lineage['effective_generation_config'] and resolved.use_cache and resolved.num_beams==1 and resolved.repetition_penalty==1. and resolved.min_new_tokens in (0,None)
dataset=pathlib.Path(cfg['data']['dataset_path']);assert str(dataset)==p['dataset_path'] and sha(dataset)==p['dataset_sha256']=='709cf45623d2feb5cb2e5d6306a9c024e47762f8debc718a055884d3def0adbf'
source=[json.loads(x) for x in dataset.read_text().splitlines()];assert len(source)==1500
def clean(s):
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
for i,(a,s) in enumerate(zip(old['detailed_results'],source)):assert a['index']==i and a['prompt']==s['prompt'] and a['reference']==s['reference'] and a['prediction']==a['raw_output']
trace_path=pathlib.Path(lineage['per_sample_evidence']);trace=[json.loads(x) for x in trace_path.read_text().splitlines()];assert len(trace)==count and [r['attempt'] for r in trace]==list(range(1,count+1))
external={str(trace_path):sha(trace_path),lineage['weight_loading_evidence']:sha(lineage['weight_loading_evidence']),lineage['profiler_trace']:sha(lineage['profiler_trace'])};empty=[];invalid=[];truncated=[];bleu_values=[];rouge_values=[]
nvis=lineage['visual_token_count'];assert nvis==257
for i,(a,b,d,s,e) in enumerate(zip(inputs,outputs,details,source,trace)):
 assert a['prompt']==e['prompt']==clean(s['prompt']) and a['reference']==b['reference']==d['reference']==s['reference'] and b['extra']['execution']==e and b['error'] is None and e['error'] is None and e['model_receives_no_prompt_or_reference']
 media=pathlib.Path(a['media_path']).resolve();assert media==(dataset.parent/s['media_path']).resolve() and media.is_relative_to(dataset.parent) and sha(media)==e['source_image_sha256']
 with Image.open(media) as im:image=im.convert('RGB')
 pixels=independent_pixel(image,ip);actual=proc(images=[image],return_tensors='pt')['pixel_values'];assert torch.equal(pixels,actual),('independent pixels',i,float(abs(pixels-actual).max()));check(pixels,e['CPU_pixel_values'])
 archive=pathlib.Path(e['visual_feature_archive']);assert sha(archive)==e['visual_feature_archive_sha256'];external[str(archive)]=sha(archive)
 with np.load(archive,allow_pickle=False) as z:vision=torch.from_numpy(z['vision_output'].copy());projected=torch.from_numpy(z['projected_visual'].copy());prefix=torch.from_numpy(z['actual_decoder_visual_prefix'].copy())
 check(vision,e['vision_output']);check(projected,e['projected_visual']);check(prefix,e['actual_decoder_visual_prefix']);assert torch.equal(projected,prefix) and vision.shape==torch.Size([1,257,mc.vision_config.hidden_size]) and projected.shape==torch.Size([1,257,mc.hidden_size]) and torch.isfinite(vision).all() and torch.isfinite(projected).all() and e['actual_visual_prefix_verified'] and e['vision_forward_calls']>0 and e['decoder_visual_bridge_calls']>0
 seq=e['generation_sequence_ids'];tokens=seq[1:];assert seq[0]==101 and len(tokens)==e['generation_token_count']<=768
 pos=[r for r in e['text_positions'] if r['phase']=='generate'];assert len(pos)==len(tokens)
 for j,r in enumerate(pos):
  assert r['resolved_text_position_ids']==[[j]] and r['past_total_cache_length']==(0 if j==0 else 257+j) and r['input_token_ids']==[[seq[j]]]
  supplied=r['supplied_position_ids']
  if supplied is not None:assert supplied==[[j+(r['past_total_cache_length'] if j else 0)]]
 masks=[r for r in e['attention_masks'] if r['phase']=='generate'];assert len(masks)==len(tokens)
 for j,r in enumerate(masks):
  if j==0:
   allowed=torch.tril(torch.ones(nvis+1,nvis+1,dtype=torch.bool));allowed[:nvis,:nvis]=True;allowed=allowed[None,None]
  else:allowed=torch.ones(1,1,1,nvis+j+1,dtype=torch.bool)
  candidates=[torch.where(allowed,0.,torch.finfo(torch.float32).min),torch.where(allowed,0.,float('-inf'))];assert any(list(x.shape)==r['mask']['shape'] and fp(x)==r['mask']['sha256'] for x in candidates),('native mask',i,j,r['mask'])
 assert e['final_cache_length']==257+len(tokens) and len(e['cache_device_and_shapes'])==mc.num_hidden_layers
 for layer in e['cache_device_and_shapes']:assert layer['key_device']==layer['value_device']==device and layer['key_shape']==layer['value_shape'] and layer['key_shape'][-2]==e['final_cache_length']
 raw=rust.decode(seq,skip_special_tokens=True)
 if tok.clean_up_tokenization_spaces:
  for x,y in [(' .', '.'), (' ?', '?'), (' !', '!'), (' ,', ','), (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")]:raw=raw.replace(x,y)
 raw=raw.strip();assert raw==e['raw_output']==b['raw_output']==b['prediction']==d['prediction'],('own token decode',i,raw,b['raw_output'])
 if not raw:empty.append(i)
 if raw and not re.search('[a-zA-Z0-9]',raw):invalid.append(i)
 cut=len(tokens)>=768 and tokens[-1]!=102;assert cut==e['truncated']
 if cut:truncated.append(i)
 bleu_values.append(bleu(raw,b['reference']));rouge_values.append(rouge(raw,b['reference']))
scores={'BLEU-4':sum(bleu_values)/count,'ROUGE-L':sum(rouge_values)/count}
for metric,score in scores.items():assert math.isfinite(score) and abs(score-report['metrics'][metric])<1e-12
assert report['statistics']['total_samples']==count and report['statistics']['error_count']==0
profile=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=profile if isinstance(profile,list) else profile['traceEvents'];hardware={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'};kernels=[x for x in events if x.get('pid') in hardware and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'metrics':scores,**scores,'historical_record_reuse':False,'source_dtype':'F32','error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'independent_CPU_pixel_replays':count,'independent_visual_prefix_replays':count,'independent_generation_position_and_mask_replays':count,'full_original_tensor_SHA_verified':499,'historical_complete200_inputs_raws_BLEU_ROUGE_replayed':200,'NPU_profiler_kernel_events':len(kernels),'sample_BLEU4_method1':bleu_values,'sample_ROUGE_L_F1':rouge_values,'external_evidence_sha256':external,'report_sha256':{x.name:sha(x) for x in folder.iterdir()},'protocol':'Original archived before matched report GIT image-only/no-text/extractFalse/768/cacheTrue/greedy without later min64 or clamp; own sourceF32 all learned NPU layers and originalI64 buffers, HF32off. Native current double-added decode cache position offset is compensated instance-only to original text-only learned positions0..N. Original257 bidirectional image prefix, text causal masks, full original KV cache and all own tokenizer words preserved. Full denominator original whitespace case-sensitive BLEU method1 independently enumerated clipped ngrams and Porter/Hunt-Szymanski ROUGE; no downloads or zero fallback.'}
destination=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json';assert not destination.exists();destination.write_text(json.dumps(proof,ensure_ascii=False,indent=2));print(json.dumps(proof,ensure_ascii=False,indent=2))
