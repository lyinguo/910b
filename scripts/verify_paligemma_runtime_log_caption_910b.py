"""Independent original-caption ROUGE, image/prefix/token/precision and actual NPU execution audit."""
import os,pathlib,json,sys,hashlib,re,struct,torch,yaml,math,bisect,ast,typing
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from transformers import AutoConfig,AutoTokenizer,AutoImageProcessor,GenerationConfig,PaliGemmaProcessor,PaliGemmaForConditionalGeneration
from safetensors import safe_open
from PIL import Image
from universal_eval.metric.rouge_910b_porter import PorterStemmer
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs']
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results']
assert len(inputs)==len(outputs)==len(details)==count
for rows in (inputs,outputs,details):assert [r['index'] for r in rows]==list(range(count))
weight_dir=pathlib.Path(cfg['model']['model_path']).resolve();assert weight_dir.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels')
assert cfg['model']['name']==weight_dir.parent.name+'/'+weight_dir.name and cfg['model']['download'] is False
mc=AutoConfig.from_pretrained(str(weight_dir),local_files_only=True);assert mc.architectures==['PaliGemmaForConditionalGeneration'] and not getattr(mc,'quantization_config',None)
assert not (weight_dir/'adapter_config.json').is_file() and not (weight_dir/'adapter_model.safetensors').is_file()
assert cfg['task_type']=='image-text-to-text' and folder.parent.name==cfg['task_type'] and folder.parent.parent.name==cfg['model']['name'].replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_range']=='0:'+str(count-1) and cfg['data']['filter_empty'] is False and cfg['data']['download'] is False
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert cfg['algorithm']['name']=='paligemma_reference_caption_inference' and isinstance(cfg['algorithm']['kwargs'].get('extract_answer'),bool)
assert cfg['metrics']['override']==['ROUGE-L'] and cfg['metrics']['include_failed_predictions'] is True
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];assert lineage['all_expected_parameters_loaded'] and lineage['tied_output_head_restored'] and mc.tie_word_embeddings
assert lineage['protocol']=='runtime_log_paligemma_caption_reference' and lineage['HF32_enabled']=={'conv':False,'matmul':False}
assert lineage['devices']==kw['npu_devices'] and len(lineage['devices'])==1;device='npu:'+str(lineage['devices'][0]);assert set(lineage['module_placements'].values())=={device}
dtype={'BF16':torch.bfloat16,'F32':torch.float32}[kw['source_dtype']];assert lineage['source_dtype']==kw['source_dtype']
for source,digest in lineage['runtime_source_sha256'].items():assert hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()==digest
for rel,digest in lineage['model_resource_sha256'].items():assert hashlib.sha256((weight_dir/rel).read_bytes()).hexdigest()==digest
assert lineage['model_resource_sha256']==kw['resource_sha256'] and lineage['runtime_source_sha256']==kw['runtime_sha256'] and lineage['weight_headers']==kw['weight_headers']
loading=json.loads(pathlib.Path(lineage['weight_loading_evidence']).read_text());index=weight_dir/'model.safetensors.index.json'
if index.is_file():weight_map=json.loads(index.read_text())['weight_map']
else:
    with safe_open(str(weight_dir/'model.safetensors'),framework='pt',device='cpu') as h:weight_map={k:'model.safetensors' for k in h.keys()}
assert len(loading)==len(weight_map)==lineage['weight_tensor_count'] and {r['source_key'] for r in loading}==set(weight_map)
assert len({r['target_key'] for r in loading})==len(loading)
# Independently derive native parameter names from the original legacy module hierarchy.
def target_name(key):
    parts=key.split('.')
    if parts[0]=='vision_tower':return '.'.join(['model','vision_tower']+parts[(2 if parts[1]=='vision_model' else 1):])
    if parts[0]=='multi_modal_projector':return 'model.'+key
    if parts[:2]==['language_model','model']:return '.'.join(['model','language_model']+parts[2:])
    if parts[:2]==['language_model','lm_head']:return '.'.join(parts[1:])
    return key
with torch.device('meta'):blueprint=PaliGemmaForConditionalGeneration(mc)
expected=dict(blueprint.named_parameters());assert {r['target_key'] for r in loading}==set(expected)
for shard,review in lineage['weight_headers'].items():
    path=weight_dir/shard
    with path.open('rb') as h:length=h.read(8);header=h.read(struct.unpack('<Q',length)[0])
    assert path.stat().st_size==review['size'] and hashlib.sha256(length+header).hexdigest()==review['header_sha256']
    with safe_open(str(path),framework='pt',device='cpu') as h:
        assert set(h.keys())=={k for k,v in weight_map.items() if v==shard}
        for row in (r for r in loading if r['source_file']==shard):
            key=row['source_key'];assert row['target_key']==target_name(key) and row['device']==device
            value=h.get_tensor(key);assert row['source_dtype']==row['execution_dtype']==str(value.dtype)==str(dtype) and row['shape']==list(value.shape)==list(expected[row['target_key']].shape)
            locations=sorted({0,value.numel()//2,value.numel()-1});assert row['slice_positions']==locations and value.reshape(-1)[locations].float().tolist()==row['source_slice']==row['device_slice']
original=lineage['original_evaluation_protocol'];reference_path=pathlib.Path(original['reference_report']);assert str(reference_path)==kw['reference_report'] and reference_path.is_relative_to('/mnt/afs/comp_error/workspace/reports')
assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==original['reference_report_sha256']==kw['reference_report_sha256'];reference=json.loads(reference_path.read_text());previous=reference['pipeline_config'];adapter=previous['model']['adapter']
assert previous['model']['name']==cfg['model']['name'] and previous['task_type']==original['original_task']=='image-to-text' and original['workbook_task']==cfg['task_type'] and original['primary_metric']=='ROUGE-L'
assert adapter==original['adapter'] and adapter.endswith((':PaliGemmaMLUAdapter',':PaliGemma2MLUAdapter')) and previous['algorithm']['name']=='greedy_inference'
assert len(reference['detailed_results'])==200 and abs(reference['metrics']['ROUGE-L']-kw['workbook_reference_score'])<0.00011
for path,sha in original['reference_source_sha256'].items():assert pathlib.Path(path).is_relative_to('/mnt/afs/comp_error/universal_eval') and hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==sha
assert original['reference_source_sha256']==kw['reference_source_sha256']
root_yaml=pathlib.Path(original['root_default_config']);assert str(root_yaml)==kw['original_config'] and hashlib.sha256(root_yaml.read_bytes()).hexdigest()==original['root_default_config_sha256']==kw['original_config_sha256'];root_previous=yaml.safe_load(root_yaml.read_text());assert root_previous['model']['name']==cfg['model']['name'] and root_previous['task_type']=='image-to-text' and root_previous['model']['adapter'].endswith(':VLLMModel')
assert hashlib.sha256(pathlib.Path(root_previous['model']['adapter'].rsplit(':',1)[0]).read_bytes()).hexdigest()==original['root_default_adapter_sha256']==kw['original_adapter_sha256']
reference_log=pathlib.Path(original['reference_log']);assert reference_log==reference_path.parent/'run.log' and str(reference_log)==kw['reference_log'] and hashlib.sha256(reference_log.read_bytes()).hexdigest()==original['reference_log_sha256']==kw['reference_log_sha256']
flags=re.findall(r'开始推理循环: 共 200 条样本[^\n]*提取答案=(True|False)',reference_log.read_text());assert flags and len(set(flags))==1;actual_extract=flags[0]=='True'
assert actual_extract==original['actual_historical_extract_answer']==cfg['algorithm']['kwargs']['extract_answer']
assert mc.vision_config.image_size==kw['expected_image_size']
old=previous['model']['kwargs'];budget=int(old.get('max_new_tokens',2048));prefix=old.get('vqa_prefix','') or ''
protocol={'vqa_prefix':prefix,'mc_instruction':old.get('mc_instruction','\n\nRespond with only the letter of the correct option.') or '', 'strip_eval_instruction':bool(old.get('strip_eval_instruction',False)), 'prompt_override':old.get('prompt_override','') or '', 'extract_answer':actual_extract}
assert protocol==original['reference_protocol']==kw['reference_protocol'] and protocol['extract_answer']==actual_extract
assert budget==kw['max_new_tokens']==lineage['max_new_tokens']==original['max_new_tokens'] and prefix==kw['vqa_prefix']==lineage['vqa_prefix']
afs=pathlib.Path('/mnt/afs/comp_error');code=(afs/'universal_eval/algorithm/inference.py').read_text();tree=ast.parse(code);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GreedyInference');members=[x for x in cls.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ANSWER_PATTERNS' for t in x.targets) or isinstance(x,ast.FunctionDef) and x.name in ('_extract_answer','_add_answer_instruction','_extract_committed_option')];node=ast.ClassDef(name='ReferenceAlgorithm',bases=[],keywords=[],body=members,decorator_list=[]);namespace={'re':re,'Optional':typing.Optional};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<independent original caption extraction>','exec'),namespace);original_algorithm=namespace['ReferenceAlgorithm']()
model_code=(afs/'universal_eval/model/adapters/paligemma_mlu.py').read_text();tree=ast.parse(model_code);suffix=ast.literal_eval(next(x.value for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EVAL_MC_INSTRUCTION' for t in x.targets)));cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PaliGemmaMLUAdapter');method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_transform_prompt');namespace={'EVAL_MC_INSTRUCTION':suffix};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'<independent original caption prompt>','exec'),namespace);original_transform=namespace['_transform_prompt'];dummy=type('OriginalPromptProtocol',(),{})()
for name,value in protocol.items():setattr(dummy,'_'+name,value)
tok=AutoTokenizer.from_pretrained(str(weight_dir),local_files_only=True);ip=AutoImageProcessor.from_pretrained(str(weight_dir),local_files_only=True,backend='pil')
seq=(mc.vision_config.image_size//mc.vision_config.patch_size)**2;ip.image_seq_length=seq;assert seq==lineage['image_seq_length'] and json.loads(json.dumps(ip.to_dict()))==lineage['effective_image_processor']
processor=PaliGemmaProcessor(image_processor=ip,tokenizer=tok);assert processor.image_token_id==mc.image_token_id
gen=GenerationConfig.from_pretrained(str(weight_dir),local_files_only=True) if (weight_dir/'generation_config.json').is_file() else GenerationConfig.from_model_config(mc)
gen.do_sample=False;eos=gen.eos_token_id if gen.eos_token_id is not None else mc.text_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos);gen.eos_token_id=eos
if gen.pad_token_id is None:gen.pad_token_id=tok.pad_token_id
assert eos==lineage['eos_token_ids'] and gen.to_dict()==lineage['effective_generation_config']
rp=mc.text_config.rope_parameters;assert rp['rope_type']=='default';dim=mc.text_config.head_dim or mc.text_config.hidden_size//mc.text_config.num_attention_heads
freq=1/(float(rp['rope_theta'])**(torch.arange(0,dim,2).float()/dim));assert torch.equal(freq,torch.tensor(lineage['text_inv_freq'])) and rp==lineage['text_rope_parameters']
def fingerprint(value):return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def check_tensor(value,record):assert list(value.shape)==record['shape'] and str(value.dtype)==record['dtype'] and fingerprint(value)==record['sha256']
buffer_info=dict(blueprint.named_buffers());buffers=lineage['analytic_buffers'];assert set(buffers)==set(buffer_info)
for name,native in buffer_info.items():
    if name.endswith('position_ids'):value=torch.arange(seq,dtype=torch.int64).reshape(1,-1)
    elif name.endswith('embed_scale'):value=torch.tensor(mc.text_config.hidden_size**0.5,dtype=native.dtype);assert float(value)==lineage['text_embedding_scale']
    elif name.endswith('inv_freq'):value=freq
    else:raise AssertionError('Unreviewed native buffer '+name)
    check_tensor(value,buffers[name])
dataset=pathlib.Path(cfg['data']['dataset_path']);expected_sha='709cf45623d2feb5cb2e5d6306a9c024e47762f8debc718a055884d3def0adbf'
assert dataset.is_relative_to('/root/projects/comperror/workspace/datasets/protocol_snapshots') and str(dataset)==original['dataset_snapshot']==kw['dataset_path']
assert hashlib.sha256(dataset.read_bytes()).hexdigest()==original['dataset_sha256']==kw['dataset_sha256']==expected_sha
source=[json.loads(x) for x in dataset.read_text().splitlines()];assert len(source)==1500
def clean(s):
    s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
stemmer=PorterStemmer()
def word_tokens(text):return [stemmer.stem(x) if len(x)>3 else x for x in re.findall('[a-z0-9]+',str(text).lower())]
def independent_rouge(pred,ref):
    a=word_tokens(' '.join(ref) if isinstance(ref,list) else ref);b=word_tokens(pred or '')
    if not a or not b:return 0.0
    # Hunt-Szymanski: LCS through strictly increasing matched positions, independent of the production row-DP recurrence.
    positions={}
    for n,word in enumerate(b):positions.setdefault(word,[]).append(n)
    tail=[]
    for word in a:
        for n in reversed(positions.get(word,[])):
            k=bisect.bisect_left(tail,n)
            if k==len(tail):tail.append(n)
            else:tail[k]=n
    return 2.0*len(tail)/(len(a)+len(b))
empty=[];invalid=[];truncated=[];scores=[]
for i,(a,b,d,s) in enumerate(zip(inputs,outputs,details,source)):
    assert a['prompt']==clean(s['prompt']) and a['reference']==b['reference']==d['reference']==s['reference']
    e=b['extra']['execution'];assert b['error'] is None and e['error'] is None and e['visual_embedding_injection_verified'] is True and e['vision_forward_calls']>=1
    model_prompt=original_algorithm._add_answer_instruction(a['prompt']) if actual_extract else a['prompt'];text=original_transform(dummy,model_prompt)
    assert e['protocol']==lineage['protocol'] and e['prompt']==model_prompt and e['processor_text']==text
    media=pathlib.Path(a['media_path']);ref_media=pathlib.Path(s['media_path']);ref_media=ref_media if ref_media.is_absolute() else dataset.parent/ref_media
    assert media.resolve()==ref_media.resolve() and media.resolve().is_relative_to(dataset.parent) and hashlib.sha256(media.read_bytes()).hexdigest()==e['source_image_sha256']
    with Image.open(media) as im:image=im.convert('RGB')
    prepared=processor(text=text,images=image,return_tensors='pt')
    for name,value in prepared.items():
        if name=='pixel_values':check_tensor(value,e['image_tensors'][name])
        else:assert value.tolist()==e[name],('CPU processor input differs',i,name)
    assert e['excluded_inference_fields']==(['labels'] if 'labels' in prepared else []) and ('labels' not in prepared or bool((prepared['labels']==-100).all()))
    assert bool((prepared['token_type_ids']==0).all()) and bool(prepared['attention_mask'].all()) and int((prepared['input_ids']==mc.image_token_id).sum())==seq
    assert e['image_features_shape']==[1,seq,mc.text_config.hidden_size]
    length=prepared['input_ids'].shape[-1];check_tensor(torch.arange(1,length+1).reshape(1,-1),e['text_position_ids'])
    expected_masks={'full_attention','sliding_attention'} if getattr(mc.text_config,'sliding_window',None) is not None else {'full_attention'};assert set(e['prefix_attention_masks'])==expected_masks
    for mask in e['prefix_attention_masks'].values():check_tensor(torch.zeros([1,1,length,length],dtype=dtype),mask)
    tokens=e['generation_token_ids'];assert len(tokens)==e['generation_count'] and len(tokens)<=budget
    raw=tok.decode(tokens,skip_special_tokens=True).strip();assert raw==e['raw_output']==b['raw_output']
    prediction=original_algorithm._extract_answer(raw,is_multiple_choice=False) if actual_extract else None;prediction=prediction if prediction is not None else raw;assert prediction==b['prediction']==d['prediction']
    if not raw:empty.append(i)
    if raw and not re.search('[a-zA-Z0-9]',raw):invalid.append(i)
    clipped=len(tokens)>=budget and (not tokens or tokens[-1] not in eos);assert clipped==e['truncated']
    if clipped:truncated.append(i)
    scores.append(independent_rouge(prediction,b['reference']))
historical_scores=[]
for i,(row,sample) in enumerate(zip(reference['detailed_results'],source)):
    assert row['index']==i and row['reference']==sample['reference'] and row['prompt']==sample['prompt']
    raw=row.get('raw_output',row.get('extra',{}).get('raw_output'));assert raw is not None
    prediction=original_algorithm._extract_answer(raw,is_multiple_choice=False) if actual_extract else None;prediction=prediction if prediction is not None else raw;assert prediction==row['prediction'],('Historical caption extraction recipe differs',i)
    if prediction and row['reference']:historical_scores.append(independent_rouge(prediction,row['reference']))
assert historical_scores and abs(sum(historical_scores)/len(historical_scores)-reference['metrics']['ROUGE-L'])<1e-12
score=sum(scores)/count;assert math.isfinite(score) and abs(report['metrics']['ROUGE-L']-score)<1e-12 and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
assert report['summary']['primary_metric']=='ROUGE-L'
trace=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=trace if isinstance(trace,list) else trace['traceEvents'];hardware={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'}
kernels=[x for x in events if x.get('pid') in hardware and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'report_dir':str(folder),'passed':True,'sample_count':count,'ROUGE-L':score,'metrics':{'ROUGE-L':score},'historical_record_reuse':False,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'CPU_image_preprocessing_replayed':count,'tokenizer_replayed':count,'decoded_generation_replayed':count,'prefix_masks_and_positions_replayed':count,'source_dtype':kw['source_dtype'],'original_weight_slice_checks':True,'NPU_profiler_kernel_events':len(kernels),'sample_ROUGE-L_F1':scores,'dataset_sha256':expected_sha,'protocol':'Exact existing PaliGemma checkpoint on NPU with original F32/BF16 and HF32 disabled. Table-matched original caption200 reference supplies exact own prefix, prompt override, instruction, extraction and budget. Original caption algorithm copied independently; CPU verifier reads original hashed source for recipe replay. Independent Hunt-Szymanski ROUGE-L, full fixed denominator, complete CPU image/token/position/bidirectional prefix masks/raw decode; genuine NPU kernels. Original root default VLLM YAML remains unchanged. Workbook broad task alias and original image-to-text/ROUGE provenance disclosed.','report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
proof['reference_log']=str(reference_log);proof['reference_log_sha256']=original['reference_log_sha256'];proof['actual_historical_extract_answer']=actual_extract;proof['reference_report']=str(reference_path);proof['reference_report_sha256']=original['reference_report_sha256'];proof['historical_input_and_extraction_replayed']=200;proof['historical_ROUGE_L_replayed']=sum(historical_scores)/len(historical_scores);proof['historical_metric_nonempty_denominator']=len(historical_scores)
destination=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json';assert not destination.exists();destination.write_text(json.dumps(proof,indent=2),encoding='utf8');print(json.dumps(proof,indent=2))
