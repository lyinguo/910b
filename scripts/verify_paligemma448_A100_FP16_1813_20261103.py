"""Independent own-source/cast/CPU200/raw/ROUGE/native-SDPA/cache/full-vocabulary audit."""
import os,pathlib,json,sys,hashlib,re,math,bisect,decimal
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
torch.set_num_threads(4)
from safetensors import safe_open
from PIL import Image
from tokenizers import Tokenizer
from torchvision.transforms.v2 import functional as F2
from torchvision.transforms import InterpolationMode
from transformers import AutoConfig,PaliGemmaForConditionalGeneration,GenerationConfig
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from universal_eval.metric.rouge_910b_porter import PorterStemmer
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def check(t,r):assert list(t.shape)==r['shape'] and str(t.dtype)==r['dtype'] and fp(t)==r['sha256']
def target(k):
 s=k.split('.')
 if s[0]=='vision_tower':return '.'.join(['model','vision_tower']+s[2 if s[1]=='vision_model' else 1:])
 if s[0]=='multi_modal_projector':return 'model.'+k
 if s[:2]==['language_model','model']:return '.'.join(['model','language_model']+s[2:])
 if s[:2]==['language_model','lm_head']:return '.'.join(s[1:])
 return k
stem=PorterStemmer()
def words(s):return [stem.stem(w) if len(w)>3 else w for w in re.findall('[a-z0-9]+',str(s).lower())]
def rouge(a,b):
 a=words(a);b=words(' '.join(b) if isinstance(b,list) else b)
 if not a or not b:return 0.
 positions={}
 for i,w in enumerate(b):positions.setdefault(w,[]).append(i)
 tail=[]
 for w in a:
  for i in reversed(positions.get(w,[])):
   j=bisect.bisect_left(tail,i)
   if j==len(tail):tail.append(i)
   else:tail[j]=i
 return 2*len(tail)/(len(a)+len(b))

folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in (3,200)
required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {p.name for p in folder.iterdir()}==required
r=json.loads((folder/'eval_report.json').read_text());cfg=r['pipeline_config'];p=cfg['model']['kwargs']['protocol'];mid=cfg['model']['name'];device='npu:'+str(cfg['model']['kwargs']['npu_devices'][0])
assert mid==p['model_id']=='google/paligemma-3b-pt-448' and folder.parent.parent.name==mid.replace('/','_') and folder.parent.name==cfg['task_type']=='image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert cfg['sample_range']=='0:'+str(count-1) and cfg['sample_count']==count and cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['model']['download'] is False
assert cfg['algorithm']=={'name':'greedy_inference','batch_size':1,'kwargs':{'extract_answer':False}} and cfg['metrics']['override']==['ROUGE-L'] and cfg['metrics']['include_failed_predictions'] is True
inputs=[json.loads(s) for s in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(s) for s in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=r['detailed_results']
for rows in [inputs,outputs,details]:assert len(rows)==count and [x['index'] for x in rows]==list(range(count))
l=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];assert l['all_expected_parameters_loaded'] and l['source_dtype']=='F32' and l['execution_dtype']=='FP16' and l['actual_tied_head_embedding'] and l['weight_tensor_count']==603
assert l['actual_attention']==p['A100_resolved_attention']=={'root':'sdpa','text':'sdpa','vision':'sdpa'} and l['protocol_review']==p and l['HF32_enabled']=={'conv':False,'matmul':False} and set(l['module_placements'].values())=={device}
for path,h in dict(p['runtime_sha256'],**p['original_source_sha256']).items():assert sha(path)==h,path
for name in ['CPU200_reference','source_cast_manifest','reference_report','reference_log']:assert sha(p[name])==p[name+'_sha256']
cpu=json.loads(pathlib.Path(p['CPU200_reference']).read_text());cast={v['source_key']:v for v in json.loads(pathlib.Path(p['source_cast_manifest']).read_text())};assert len(cpu)==200 and len(cast)==603
directory=pathlib.Path(cfg['model']['model_path']).resolve();assert directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels') and directory.parent.name+'/'+directory.name==mid
for name,h in p['resource_sha256'].items():assert sha(directory/name)==h
mc=AutoConfig.from_pretrained(directory,local_files_only=True)
with torch.device('meta'):blue=PaliGemmaForConditionalGeneration(mc)
expected={k:list(v.shape) for k,v in blue.named_parameters()};assert len(expected)==603 and mc.tie_word_embeddings
loading=json.loads(pathlib.Path(l['weight_loading_evidence']).read_text());assert len(loading)==603 and len({v['source_key'] for v in loading})==603 and {v['target_key'] for v in loading}==set(expected)
assert sha(p['original_A100_real3_diagnostic'])==p['original_A100_real3_diagnostic_sha256']
aa=json.loads(pathlib.Path(p['original_A100_real3_diagnostic']).read_text());assert aa['all3_historical_raw_equal'] and aa['actual_attention']==l['actual_attention'] and aa['reference_report_sha256']==p['actual_A100_original_report_sha256']
assert len(aa['actual_parameter_sha256'])==603 and all(aa['actual_parameter_sha256'][row['target_key']]==row['actual_NPU_sha256'] for row in loading)
index=json.loads((directory/'model.safetensors.index.json').read_text())['weight_map'];assert set(index)==set(cast)
for shard,w in p['weights'].items():
 f=directory/shard;assert f.stat().st_size==w['size'] and sha(f)==w['sha256']
 with safe_open(f,framework='pt',device='cpu') as handle:
  for row in [v for v in loading if v['source_file']==shard]:
   k=row['source_key'];x=handle.get_tensor(k);y=x.to(torch.float16);c=cast[k]
   assert row['target_key']==target(k) and row['shape']==expected[target(k)]==list(x.shape) and x.dtype==torch.float32 and row['device']==device
   assert row['source_sha256']==c['source_sha256']==fp(x) and row['actual_NPU_sha256']==row['FP16_sha256']==c['FP16_sha256']==fp(y)
   assert row['roundtrip_changed_elements']==int((x!=y.to(torch.float32)).sum()) and row['nonzero_underflow']==int(((x!=0)&(y==0)).sum())
assert sum(v['roundtrip_changed_elements'] for v in loading)==l['source_cast_roundtrip_changed_sum']==2923995056 and sum(v['nonzero_underflow'] for v in loading)==l['source_cast_underflow_sum']==477109
for name,rec in l['analytic_buffers'].items():
 if name.endswith('position_ids'):v=torch.arange(1024).reshape(1,-1)
 elif name.endswith('embed_scale'):v=torch.tensor(mc.text_config.hidden_size**.5,dtype=torch.float16)
 elif name.endswith('inv_freq'):v=1/(float(mc.text_config.rope_parameters['rope_theta'])**(torch.arange(0,mc.text_config.head_dim,2).float()/mc.text_config.head_dim))
 else:raise AssertionError(name)
 check(v,rec)
if p.get('actual_A100_analytic_buffers'):assert l['analytic_buffers']==p['actual_A100_analytic_buffers']
gen=GenerationConfig.from_pretrained(directory,local_files_only=True);blue.generation_config=gen;resolved,_=blue._prepare_generation_config(None,max_new_tokens=64,do_sample=False)
assert resolved.to_dict()==l['effective_generation_config']==p['effective_generation_config'] and resolved.use_cache is True and resolved.num_beams==1
dataset=pathlib.Path(cfg['data']['dataset_path']);assert str(dataset)==p['dataset_path'] and sha(dataset)==p['dataset_sha256']=='ceb1670f2848955290915dbe375dca8b34b62d6980e9f6cd1abe38bf0e68963b'
original=json.loads(pathlib.Path(p['reference_report']).read_text());assert len(original['detailed_results'])==200 and original['pipeline_config']['model']['name']==mid
oldscore=sum(rouge(v['raw_output'],v['reference']) for v in original['detailed_results'])/200;assert abs(oldscore-original['metrics']['ROUGE-L'])<1e-12
rust=Tokenizer.from_file(str(directory/'tokenizer.json'));tc=json.loads((directory/'tokenizer_config.json').read_text());steps=[json.loads(s) for s in pathlib.Path(l['generation_step_evidence']).read_text().splitlines()];trace=[json.loads(s) for s in pathlib.Path(l['per_sample_evidence']).read_text().splitlines()]
assert len(steps)==len(trace)==count and [v['attempt'] for v in trace]==list(range(1,count+1))
scores=[];empty=[];invalid=[];truncated=[];fullsteps=0;external={}
for i,(inp,out,detail,step,actual) in enumerate(zip(inputs,outputs,details,steps,trace)):
 ref=cpu[i];assert inp['prompt']==ref['prompt'] and inp['reference']==out['reference']==detail['reference']==ref['reference'] and out['error'] is None and actual['error'] is None
 e=out['extra']['execution'];assert e==actual and e['original_index']==i and e['processor_text']==ref['transformed_prompt']==p['caption_prefix']+ref['prompt'] and e['processor_inputs']==ref['inputs'] and e['native_SDPA_prefix_and_cache_checked']
 media=pathlib.Path(inp['media_path']).resolve();assert str(media)==ref['resolved_media_path'] and sha(media)==ref['source_image_sha256']==e['source_image_sha256']
 with Image.open(media) as im:rgb=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1)
 resized=F2.resize(rgb,[448,448],interpolation=InterpolationMode.BICUBIC,antialias=True);pixels=F2.normalize(resized.float(),mean=[127.5]*3,std=[127.5]*3).unsqueeze(0)
 assert fp(pixels)==ref['pixel_values_F32_sha256'];check(pixels,e['CPU_pixels']);check(pixels.half(),e['original_A100_FP16_pixels']);check(pixels.half(),e['actual_NPU_pixels'])
 token_text='<image>'*1024+'<bos>'+ref['transformed_prompt']+'\n';ids=rust.encode(token_text,add_special_tokens=False).ids
 assert tc['add_bos_token'] is True and p['effective_processor_tokenizer_flags']['add_bos_token'] is False and p['effective_processor_tokenizer_flags']['add_eos_token'] is False
 assert ids==ref['inputs']['input_ids'][0] and ref['inputs']['attention_mask']==[[1]*len(ids)] and ref['inputs']['token_type_ids']==[[0]*len(ids)] and ref['inputs']['labels']==[[-100]*len(ids)]
 tokens=e['generation_token_ids'];assert len(tokens)==e['generation_count']<=64 and all(0<=t<257216 for t in tokens) and tokens==step['actual_full_vocab_argmax_ids']
 raw=rust.decode(tokens,skip_special_tokens=True)
 if tc.get('clean_up_tokenization_spaces',False):
  for a,b in [(' .','.'),(' ?','?'),(' !','!'),(' ,',','),(" ' ","'"),(" n't","n't"),(" 'm","'m"),(" 's","'s"),(" 've","'ve"),(" 're","'re")]:raw=raw.replace(a,b)
 raw=raw.strip();assert raw==e['raw_output']==out['prediction']==detail['prediction'] and out.get('raw_output','')==raw
 if not raw:empty.append(i)
 elif not any(x.isalnum() for x in raw):invalid.append(i)
 assert e['truncated']==(len(tokens)>=64)
 if e['truncated']:truncated.append(i)
 assert len(step['forward'])==len(step['cache'])==len(tokens) and step['original_index']==i
 for j,(f,c) in enumerate(zip(step['forward'],step['cache'])):
  q=len(ids) if j==0 else 1;past=0 if j==0 else len(ids)+j-1
  assert f['past']==past and f['input_ids']==([ids] if j==0 else [[tokens[j-1]]]) and f['position_ids']==[list(range(past+1,past+q+1))]
  assert set(f['masks'])=={'full_attention'}
  for name,m in f['masks'].items():
   if m is None:assert q==1;continue
   shape=m['shape'];assert shape[:3]==[1,1,q] and shape[-1]>=past+q
   allowed=(torch.arange(shape[-1])<past+q)[None,None,None,:].expand(1,1,q,-1)
   if m['dtype']=='torch.bool':mask=allowed
   else:
    dtype={'torch.float16':torch.float16,'torch.float32':torch.float32,'torch.bfloat16':torch.bfloat16}[m['dtype']];mask=torch.zeros(shape,dtype=dtype).masked_fill(~allowed,torch.finfo(dtype).min)
   check(mask,m)
  assert c['seen']==past+q and c['layers']==18 and c['device']==device and c['key_value_dtype']=='torch.float16' and c['key_shape']==[1,1,past+q,256]
 assert len(step['activation_counts'])==45
 for name,v in step['activation_counts'].items():assert v==(1 if 'vision_tower' in name else len(tokens))
 archive=pathlib.Path(e['visual_feature_archive']);assert sha(archive)==e['visual_feature_archive_sha256'];external[str(archive)]=sha(archive)
 with np.load(archive) as arrays:
  for key,rec in [('encoder_bits',e['encoder']),('projector_bits',e['projector']),('actual_LM_image_bits',e['actual_LM_images'])]:
   a=arrays[key];assert a.dtype==np.uint8 and hashlib.sha256(a.tobytes()).hexdigest()==rec['sha256'] and rec['dtype']=='torch.float16'
  assert np.array_equal(arrays['projector_bits'],arrays['actual_LM_image_bits'])
 assert e['encoder']['shape']==[1,1024,1152] and e['projector']['shape']==e['actual_LM_images']['shape']==[1,1024,2048]
 scores.append(rouge(raw,out['reference']));fullsteps+=len(tokens)
score=sum(scores)/count;assert abs(score-r['metrics']['ROUGE-L'])<1e-12 and r['summary']['primary_metric']=='ROUGE-L' and r['statistics']['total_samples']==count and r['statistics']['error_count']==0
traceprof=json.loads(pathlib.Path(l['profiler_trace']).read_text());events=traceprof if isinstance(traceprof,list) else traceprof['traceEvents'];hw={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'}
kernels=[x for x in events if x.get('pid') in hw and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
for key in ['weight_loading_evidence','per_sample_evidence','generation_step_evidence','profiler_trace']:external[l[key]]=sha(l[key])
for key in ['CPU200_reference','source_cast_manifest','reference_report','reference_log','original_A100_real3_diagnostic']:external[p[key]]=sha(p[key])
value=r['metrics']['ROUGE-L'];D=decimal.Decimal(str(p['A100_D']));relative=abs(decimal.Decimal(str(value))-D)/abs(D);assert D!=0
proof={'report_dir':str(folder),'passed':True,'sample_count':count,'metrics':{'ROUGE-L':value},'ROUGE-L':value,'historical_record_reuse':False,'source_dtype':'F32','execution_dtype':'FP16','source_tensor_count':603,'actual_activation_module_count':45,'actual_native_cache_layers':18,'actual_attention':l['actual_attention'],'actual_full_vocabulary_size':257216,'full_vocab_greedy_steps':fullsteps,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'sample_ROUGE_L':scores,'CPU_inputs_replayed':count,'original_A100_ROUGE_L_recomputed':oldscore,'NPU_profiler_kernel_events':len(kernels),'profiler_scope':l['profiler_scope'],'external_evidence_sha256':external,'relative_error_to_A100_table_D':float(relative),'formal_alignment_within_5pct':bool(count==200 and relative<=decimal.Decimal('.05')),'A100_D':p['A100_D'],'report_sha256':{x.name:sha(x) for x in folder.iterdir()},'protocol':'Actual A100 FP16/SDPA/own Torchvision-v2 uint8BICUBIC and fused normalization, original caption1459/caption_en/64/serial1/cacheTrue; independent full source-cast-NPU hashes, all token argmax, prefix masks/positions/native cache/visual injection and fixed-denominator ROUGE. No budget or mathematical-path change to force score.'}
if p.get('numerical_review'):
 assert sha(p['numerical_review'])==p['numerical_review_sha256'];num=json.loads(pathlib.Path(p['numerical_review']).read_text());assert num['passed'] and num['official_precision_unchanged']=='FP16' and num['CPU_diagnostics_only']
 for path,h in num['evidence_sha256'].items():assert sha(path)==h;external[path]=h
 external[p['numerical_review']]=sha(p['numerical_review']);proof['numerical_disclosure_reviewed']=True
dest=pathlib.Path(l['per_sample_evidence']).parent/'independent_report_verification.json';assert not dest.exists();dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(json.dumps(proof,ensure_ascii=False,indent=2))
