"""Independent full native OCR checkpoint, buffer, alias, cache-position, input, cross-attention and CER verification."""
import pathlib,json,sys,hashlib,re,struct,math,os,importlib.util,collections
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from PIL import Image
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from transformers import AutoConfig,AutoImageProcessor,AutoTokenizer,TrOCRProcessor,VisionEncoderDecoderModel,GenerationConfig
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers.activations import GELUActivation
import torch.nn.functional as F
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def check(t,r):assert list(t.shape)==r['shape'] and str(t.dtype)==r['dtype'] and fp(t)==r['sha256']
def distance(a,b):
 costs=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  previous=costs[0];costs[0]=i
  for j,y in enumerate(b,1):
   old=costs[j];costs[j]=min(costs[j]+1,costs[j-1]+1,previous+(x!=y));previous=old
 return costs[-1]
def normalize(s):return re.sub(r'[\s\-\.\,\;\:\!\?\'\"\(\)\[\]\{\}\/\\\@\#\$\%\^\&\*\+\=\_\~\`\|]','',str(s or '').strip().upper())
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {q.name for q in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];p=kw['protocol'];mid=cfg['model']['name'];assert p['model_id']==mid
allowed={'daekeun-ml/ko-trocr-base-nsmc-news-chatbot','ddobokki/ko-trocr','kkatiz/thai-trocr-thaigov-v2','openthaigpt/thai-trocr'};assert mid in allowed
assert folder.parent.name==cfg['task_type']=='image-to-text' and folder.parent.parent.name==mid.replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_range']=='0:'+str(count-1) and cfg['sample_count']==count and cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['model']['download'] is False
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert cfg['algorithm']['name']=='greedy_inference' and cfg['algorithm']['kwargs']=={'extract_answer':True,'ocr_extract':True} and p['extract_answer'] is True and p['ocr_extract'] is True and cfg['algorithm']['batch_size']==1
assert cfg['metrics']['override']==['CER'] and cfg['metrics']['include_failed_predictions'] is True
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];assert len(inputs)==len(outputs)==len(details)==count
for rows in (inputs,outputs,details):assert [r['index'] for r in rows]==list(range(count))
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];device='npu:'+str(kw['npu_devices'][0]);assert lineage['devices']==kw['npu_devices'] and len(kw['npu_devices'])==1 and set(lineage['module_placements'].values())=={device}
execution_dtype=torch.float16 if mid=='ddobokki/ko-trocr' else torch.float32;assert p['source_dtype']==str(execution_dtype)
assert lineage['all_expected_parameters_loaded'] and lineage['source_dtype']==('F16' if execution_dtype==torch.float16 else 'F32') and lineage['HF32_enabled']=={'conv':False,'matmul':False} and lineage['protocol_review']==p
assert lineage['weight_files']==p['weights'] and lineage['model_resource_sha256']==p['resource_sha256'] and lineage['runtime_source_sha256']==p['runtime_sha256']
for path,digest in dict(p['runtime_sha256'],**p['original_source_sha256']).items():assert sha(path)==digest
directory=pathlib.Path(cfg['model']['model_path']).resolve();assert directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels') and directory.parent.name+'/'+directory.name==mid
for name,digest in p['resource_sha256'].items():assert sha(directory/name)==digest
raw_config=json.loads((directory/'config.json').read_text());mc=AutoConfig.from_pretrained(str(directory),local_files_only=True);assert mc.encoder.model_type in ('vit','deit') and mc.decoder.model_type==p['decoder_family'] in ('bert','roberta','camembert','electra')
with torch.device('meta'):blueprint=VisionEncoderDecoderModel(mc)
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
disabled=not any(k.startswith('encoder.pooler.') for k in source_keys);assert lineage['disabled_absent_unused_encoder_pooler']==disabled
if disabled:blueprint.encoder.pooler=None
expected={n:list(t.shape) for n,t in blueprint.named_parameters(remove_duplicate=False)};buffer_names=set(dict(blueprint.named_buffers()));groups=collections.defaultdict(list)
for name,t in blueprint.named_parameters(remove_duplicate=False):groups[id(t)].append(name)
aliases=sorted([sorted(names) for names in groups.values() if len(names)>1]);assert aliases==sorted(p['alias_groups'])==sorted(row['names'] for row in lineage['parameter_alias_groups'])
def name_map(key):
 parts=key.split('.')
 if parts[:3]==['encoder','encoder','layer']:
  layer=parts[3];tail='.'.join(parts[4:]);patterns={'attention.attention.query':'attention.q_proj','attention.attention.key':'attention.k_proj','attention.attention.value':'attention.v_proj','attention.output.dense':'attention.o_proj','intermediate.dense':'mlp.fc1','output.dense':'mlp.fc2'}
  for old,new in patterns.items():
   if tail.startswith(old+'.'):tail=new+tail[len(old):];break
  return 'encoder.layers.'+layer+'.'+tail
 return key
assert len(loading)==lineage['weight_tensor_count'] and len({x['target_key'] for x in loading})==len(loading) and {x['source_key'] for x in loading}==source_keys-buffer_names
for r in loading:
 key=r['source_key'];target=name_map(key);tensor=get_tensor(key);assert r['source_file']==weight.name and r['target_key']==target and list(tensor.shape)==r['shape']==expected[target] and tensor.dtype==execution_dtype
 assert r['source_dtype']==r['execution_dtype']==str(execution_dtype) and r['device']==device and fp(tensor)==r['source_tensor_sha256']==r['NPU_tensor_sha256']
mapped={x['target_key'] for x in loading};alias_names=set()
for evidence in lineage['parameter_alias_groups']:
 names=evidence['names'];present=[n for n in names if n in source_keys];assert evidence['all_shared'] and set(evidence['present_source_names'])==set(present) and present;alias_names.update(names)
 for name in present:assert fp(get_tensor(name))==evidence['sha256']
assert set(expected)-mapped<=alias_names and mapped<=set(expected)
materialized=json.loads(pathlib.Path(lineage['buffer_materialization']).read_text());assert {r['name'] for r in materialized}==buffer_names
for r in materialized:
 shape=dict(blueprint.named_buffers())[r['name']].shape
 if r['name'].endswith('.position_ids'):cpu=torch.arange(shape[-1],dtype=torch.int64).reshape(1,-1).expand(shape)
 elif r['name'].endswith('.token_type_ids'):cpu=torch.zeros(shape,dtype=torch.int64)
 else:raise AssertionError(r['name'])
 assert r['source_checkpoint_buffer_present']==(r['name'] in source_keys) and r['device']==device;check(cpu,r['canonical_CPU']);check(cpu,r['actual_NPU'])
 if r['name'] in source_keys:assert torch.equal(cpu,get_tensor(r['name']))
processor=TrOCRProcessor(image_processor=AutoImageProcessor.from_pretrained(str(directory),local_files_only=True,backend='pil'),tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True))
assert json.loads(json.dumps(processor.image_processor.to_dict()))==lineage['effective_image_processor']
own_backend=Tokenizer.from_file(str(directory/'tokenizer.json'));own_vocab=own_backend.get_vocab();assert own_vocab==processor.tokenizer.get_vocab()
assert len(own_vocab)==p['vocabulary_contract']['tokenizer_size'] and mc.decoder.vocab_size==p['vocabulary_contract']['model_vocab_size'] and all(0<=i<mc.decoder.vocab_size for i in own_vocab.values())
assert hashlib.sha256(json.dumps(sorted([(i,t) for t,i in own_vocab.items()]),ensure_ascii=False).encode()).hexdigest()==p['vocabulary_contract']['core_vocabulary_sha256']
unmapped_ids=sorted(set(range(mc.decoder.vocab_size))-set(own_vocab.values()));assert lineage['vocabulary_contract']==dict(p['vocabulary_contract'],original_unmapped_model_ids=unmapped_ids)
all_model_ids=[[i] for i in range(mc.decoder.vocab_size)]
assert own_backend.decode_batch(all_model_ids,skip_special_tokens=True)==processor.tokenizer.batch_decode(all_model_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)

genfile=directory/'generation_config.json';gen=GenerationConfig.from_pretrained(str(directory),local_files_only=True) if genfile.is_file() else GenerationConfig.from_model_config(json.loads((directory/'config.json').read_text()));blueprint.generation_config=gen
resolved,_=blueprint._prepare_generation_config(None,max_new_tokens=p['max_new_tokens'],do_sample=False,temperature=0.)
assert resolved.to_dict()==p['effective_generation_config']==lineage['effective_generation_config'] and resolved.do_sample is False and resolved.use_cache is True and p['max_new_tokens']==4096
assert resolved.num_beams==(1 if mid=='ddobokki/ko-trocr' else 4) and resolved.length_penalty==(1. if mid=='ddobokki/ko-trocr' else 2.) and resolved.no_repeat_ngram_size==(0 if mid=='ddobokki/ko-trocr' else 3)
eos=resolved.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos);assert eos==lineage['eos_token_ids']
if not genfile.is_file():
 assert 'Raw own config.json' in lineage['generation_config_source'] and raw_config['max_length']==resolved.max_length
 if mid=='daekeun-ml/ko-trocr-base-nsmc-news-chatbot':assert raw_config['num_beams']==resolved.num_beams==4
position_name=p['position_embedding_module'];assert position_name==lineage['position_embedding_module'] and p['cross_attention_module']==lineage['cross_attention_module']
def position_replay(r):
 ids=torch.tensor(r['input_ids'],dtype=torch.int64);past=r['past_key_values_length'];assert isinstance(past,int) and past>=0
 if mc.decoder.model_type in ('roberta','camembert'):
  mask=(ids!=mc.decoder.pad_token_id).int();expected_ids=((torch.cumsum(mask,dim=1).type_as(mask)+past)*mask).long()+mc.decoder.pad_token_id
 else:expected_ids=torch.arange(past,past+ids.shape[-1],dtype=torch.int64).reshape(1,-1)
 observed=torch.tensor(r['position_ids'],dtype=torch.int64);assert torch.equal(observed,expected_ids)
 values=get_tensor(position_name+'.weight')[observed];check(values,r['output'])
disclosure=lineage['native_GELU_disclosure'];assert disclosure['native_graph_unchanged'] is True and disclosure['no_NPU_ERF_replacement'] is True
expected_native_gelu=sorted(name for name,module in blueprint.named_modules() if type(module) is GELUActivation)
assert disclosure['module_names']==expected_native_gelu
execution=pathlib.Path(disclosure['execution_evidence']);gelu_rows=[json.loads(x) for x in execution.read_text().splitlines()]
assert execution.parent==pathlib.Path(lineage['weight_loading_evidence']).parent and len(gelu_rows)==count and [r['attempt'] for r in gelu_rows]==list(range(1,count+1))
totals={n:0 for n in expected_native_gelu}
for row in gelu_rows:
 assert sorted(row['module_forward_counts_delta'])==expected_native_gelu
 for name in expected_native_gelu:
  assert row['module_forward_counts_delta'][name]>=1 and row['actual_devices'][name]==[device] and row['actual_dtypes'][name]==[str(execution_dtype)]
  totals[name]+=row['module_forward_counts_delta'][name];assert totals[name]==row['cumulative_counts'][name]

probe=pathlib.Path(disclosure['probe']);assert probe.parent==pathlib.Path(lineage['weight_loading_evidence']).parent and sha(probe)==disclosure['probe_sha256']
with np.load(probe,allow_pickle=False) as z:
 assert set(z.files)=={'input_F32','actual_native_NPU_GELU'}
 x=torch.from_numpy(z['input_F32']);native=torch.from_numpy(z['actual_native_NPU_GELU'])
 assert torch.equal(x,torch.linspace(-8,8,8193,dtype=torch.float32)) and native.dtype==torch.float32
 exact_error=float((native-F.gelu(x,approximate='none')).abs().max());tanh_error=float((native-F.gelu(x,approximate='tanh')).abs().max())
 assert exact_error>1e-4 and tanh_error<2e-6 and exact_error==disclosure['native_NPU_vs_CPU_erf_max_abs'] and tanh_error==disclosure['native_NPU_vs_CPU_tanh_max_abs']
dataset=pathlib.Path(cfg['data']['dataset_path']).resolve();assert dataset==pathlib.Path(p['dataset_path']).resolve() and dataset.is_relative_to('/root/projects/comperror/workspace/datasets') and sha(dataset)==p['dataset_sha256']
data=[json.loads(x) for x in dataset.read_text().splitlines()];assert len(data)==3000
reference=pathlib.Path(p['reference_report']);assert reference.is_relative_to('/mnt/afs/comp_error/workspace/reports') and sha(reference)==p['reference_report_sha256'];old=json.loads(reference.read_text());pc=old['pipeline_config'];assert len(old['detailed_results'])==200 and pc['model']['name']==mid and pc['metrics']['override']==['CER'] and pc['model']['kwargs']['max_new_tokens']==p['max_new_tokens']
assert pc['algorithm']['kwargs']['extract_answer']==p['extract_answer'] and pc['algorithm']['kwargs'].get('ocr_extract',False)==p['ocr_extract'] and not pc['model']['kwargs'].get('ocr_allowed_charset') and not pc['model']['kwargs'].get('ocr_plate_grammar')
assert sha(p['reference_log'])==p['reference_log_sha256'];flags=re.findall(r'开始推理循环: 共 200 条样本[^\n]*提取答案=(True|False)',pathlib.Path(p['reference_log']).read_text());assert flags and len(set(flags))==1 and (flags[0]=='True')==p['extract_answer']
# Use the frozen original OCR normalization/extraction recipe, not the new adapter or production CER implementation.
recipe_path=pathlib.Path('universal_eval/metric/ocr.py').resolve();assert str(recipe_path) in p['original_source_sha256'] and sha(recipe_path)==p['original_source_sha256'][str(recipe_path)]
spec=importlib.util.spec_from_file_location('frozen_original_ocr_recipe',recipe_path);recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
def prediction(raw):
 selected=recipe.ocr_extract_plate(raw) if p['extract_answer'] and p['ocr_extract'] else None
 assert not p['extract_answer'] or p['ocr_extract'] # These reviewed protocols do not use generic non-OCR answer parsing.
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
errors=characters=0;empty=[];invalid=[];truncated=[];per_sample=[];artifacts={};empty_evidence=[]
for i,(a,b,d,s,e) in enumerate(zip(inputs,outputs,details,data,trace)):
 assert a['prompt']==s['prompt'] and a['reference']==b['reference']==d['reference']==s['reference'] and b['extra']['execution']==e and e['error'] is None and b['error'] is None
 media=pathlib.Path(a['media_path']).resolve();source=pathlib.Path(s['media_path']);source=source if source.is_absolute() else dataset.parent/source
 assert media==source.resolve() and media.is_relative_to(dataset.parent) and sha(media)==e['source_image_sha256']
 assert e['prompt']==a['prompt']+('\n\nProvide only the final answer.' if p['extract_answer'] else '') and e['text_prompt_used_by_model'] is False
 with Image.open(media) as image:prepared=processor(images=image.convert('RGB'),return_tensors='pt')
 assert set(prepared)=={'pixel_values'};check(prepared['pixel_values'],e['CPU_pixel_values'])
 archive=pathlib.Path(e['visual_feature_archive']);assert archive.parent.parent==pathlib.Path(lineage['weight_loading_evidence']).parent and archive.parent.name=='visual_features' and sha(archive)==e['visual_feature_archive_sha256'];artifacts[str(archive)]=e['visual_feature_archive_sha256']
 with np.load(archive,allow_pickle=False) as z:
  assert set(z.files)=={'encoder_output','decoder_cross_attention_features'}
  vision=torch.from_numpy(z['encoder_output']);features=torch.from_numpy(z['decoder_cross_attention_features']);check(vision,e['encoder_output']);check(features,e['decoder_cross_attention_features'])
  assert vision.dtype==features.dtype==execution_dtype and bool(torch.isfinite(vision).all()) and bool(torch.isfinite(features).all()) and bool(features.abs().sum()>0)
  assert vision.shape[0]==1 and vision.shape[-1]==mc.encoder.hidden_size
  if not hasattr(blueprint,'enc_to_dec_proj'):assert torch.equal(features,vision)
  assert e['actual_decoder_beam_batch']==resolved.num_beams and e['visual_cross_attention_verified'] is True
  expected=features.repeat_interleave(resolved.num_beams//features.shape[0],dim=0);check(expected,e['actual_cross_attention_input'])
 position_replay(e['position_first']);position_replay(e['position_last'])
 assert e['position_first']['input_ids']==[[resolved.decoder_start_token_id]]*resolved.num_beams
 tokens=e['generation_sequence_ids'];assert tokens[0]==resolved.decoder_start_token_id and e['generation_count']==len(tokens)-1<=p['max_new_tokens']
 raw_base=own_backend.decode(tokens,skip_special_tokens=True)
 cleanup_enabled=processor.tokenizer.clean_up_tokenization_spaces and (type(processor.tokenizer.backend_tokenizer.model).__name__!='BPE' or processor.tokenizer.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output)
 if cleanup_enabled:
  for old,new in [(" .","."),(" ?","?"),(" !","!"),(" ,",","),(" ' ","'"),(" n't","n't"),(" 'm","'m"),(" 's","'s"),(" 've","'ve"),(" 're","'re")]:raw_base=raw_base.replace(old,new)
 raw=raw_base.strip();assert raw==e['raw_output']==b['raw_output'];pred=prediction(raw);assert pred==b['prediction']==d['prediction']
 clip=len(tokens)-1>=p['max_new_tokens'] and tokens[-1] not in eos;assert clip==e['truncated']
 if clip:truncated.append(i)
 if not raw:
  assert e['stop_reason']=='eos' or e['truncated']
  assert all(isinstance(t,int) and 0<=t<mc.decoder.vocab_size for t in tokens)
  empty.append(i);empty_evidence.append({'index':i,'sequence_ids':tokens,'original_decode_without_special_tokens':own_backend.decode(tokens,skip_special_tokens=True),'ended_with_original_EOS':tokens[-1] in eos,'truncated':e['truncated'],'unmapped_source_model_IDs_in_sequence':sorted(set(tokens)&set(unmapped_ids))})
 x,y=pair(pred,b['reference']);assert y
 if not x:invalid.append(i)
 edit=distance(x,y);errors+=edit;characters+=len(y);per_sample.append({'index':i,'normalized_prediction':x,'normalized_reference':y,'edit_distance':edit,'reference_characters':len(y)})
score=errors/characters
assert abs(report['metrics']['CER']-score)<1e-12 and report['summary']['primary_metric']=='CER' and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
npu=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=npu if isinstance(npu,list) else npu['traceEvents'];hardware={x['pid'] for x in events if x.get('name')=='process_name' and 'Ascend Hardware' in x.get('args',{}).get('name','')};kernels=[x for x in events if x.get('pid') in hardware and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'report_dir':str(folder),'passed':True,'sample_count':count,'metrics':{'CER':score},'CER':score,'historical_record_reuse':'--reuse-existing-inference' in sys.argv[3:],'inference_reused_for_verifier_repair':'--reuse-existing-inference' in sys.argv[3:],'decode_protocol':'Own original tokenizer JSON plus original batch_decode cleanup flag; BPE keeps native skip rule, Unigram/WordPiece use exact standard punctuation cleanup when enabled. Independent raw replay only; no model inference, raw output or metric replacement.','error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'genuine_empty_decoding_evidence':empty_evidence,'original_model_vocab_size':mc.decoder.vocab_size,'own_tokenizer_vocab_size':len(own_vocab),'original_unmapped_model_IDs':unmapped_ids,'all_source_model_IDs_decode_replayed':mc.decoder.vocab_size,'CPU_image_preprocessing_replayed':count,'original_vocabulary_and_decoded_sequences_replayed':count,'CPU_position_embeddings_replayed':count,'real_visual_cross_attention_replayed':count,'historical_inputs_references_extraction_and_CER_replayed':200,'historical_reference_CER':old_errors/old_chars,'reference_character_denominator':characters,'total_edit_distance':errors,'per_sample_edit_distances':per_sample,'NPU_profiler_kernel_events':len(kernels),'source_dtype':'F16' if execution_dtype==torch.float16 else 'F32','original_decoder_family':mc.decoder.model_type,'native_GELU_calculation':'original native API with observed vendor tanh approximation','native_GELU_probe':str(probe),'native_GELU_probe_sha256':sha(probe),'native_GELU_execution':str(execution),'native_GELU_execution_sha256':sha(execution),'native_GELU_erf_difference_disclosed':True,'original_max_new_tokens':p['max_new_tokens'],'original_num_beams':resolved.num_beams,'original_use_cache':resolved.use_cache,'external_evidence_sha256':artifacts,'protocol':'Original image-only OCR/CER with exact own F32 or F16 tensors, native eager vision and cross-attention Bert/Roberta/Camembert/Electra decoder, own complete vocabulary, original beam/cache/decoding budget and runtime-log grounded OCR extraction. Independent CPU pixel, complete parameter SHA, actual cross-attention arrays, position, token decoding and one-row edit-distance/full-reference denominator replay. No model/data downloads, substituted weights, new character constraints or low-bit acceleration claims.','report_sha256':{q.name:sha(q) for q in folder.iterdir()}}
dest=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json';reuse_existing='--reuse-existing-inference' in sys.argv[3:]
if dest.exists() and reuse_existing:dest=dest.with_name('independent_report_verification_unigram_cleanup_v3.json')
assert not dest.exists();dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2));print(json.dumps(proof,ensure_ascii=False,indent=2))
