"""Independent Idefics3 source/input/PIL/patch-position/token/output/metric/kernel audit."""
import os,pathlib,json,sys,hashlib,re,struct,torch,yaml
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoConfig,AutoTokenizer,AutoImageProcessor,GenerationConfig,Idefics3Processor
from safetensors import safe_open
from PIL import Image
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config']
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results']
assert len(inputs)==len(outputs)==len(details)==count
for rows in (inputs,outputs,details):assert [r['index'] for r in rows]==list(range(count))
weight_dir=pathlib.Path(cfg['model']['model_path']).resolve();assert weight_dir.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels')
assert cfg['model']['name']==weight_dir.parent.name+'/'+weight_dir.name and cfg['model']['download'] is False
mc=AutoConfig.from_pretrained(str(weight_dir),local_files_only=True);assert mc.architectures==['Idefics3ForConditionalGeneration'] and not getattr(mc,'quantization_config',None)
assert cfg['task_type']=='image-text-to-text' and folder.parent.name==cfg['task_type'] and folder.parent.parent.name==cfg['model']['name'].replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['sample_range']=='0:'+str(count-1)
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];assert lineage['all_expected_parameters_loaded'] and lineage['tied_output_head_restored']==bool(mc.tie_word_embeddings)
assert lineage['devices']==cfg['model']['kwargs']['npu_devices'] and len(lineage['devices'])==1;device='npu:'+str(lineage['devices'][0]);assert set(lineage['module_placements'].values())=={device}
for source,digest in lineage['runtime_source_sha256'].items():assert hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()==digest
for rel,digest in lineage['model_resource_sha256'].items():assert hashlib.sha256((weight_dir/rel).read_bytes()).hexdigest()==digest
assert lineage['model_resource_sha256']==cfg['model']['kwargs']['resource_sha256']
loading=json.loads(pathlib.Path(lineage['weight_loading_evidence']).read_text());index=weight_dir/'model.safetensors.index.json'
if index.is_file():weight_map=json.loads(index.read_text())['weight_map']
else:
    with safe_open(str(weight_dir/'model.safetensors'),framework='pt',device='cpu') as h:weight_map={k:'model.safetensors' for k in h.keys()}
assert len(loading)==len(weight_map)==lineage['weight_tensor_count'] and {r['source_key'] for r in loading}==set(weight_map)
assert len({r['target_key'] for r in loading})==len(loading)
for shard,review in lineage['weight_headers'].items():
    path=weight_dir/shard
    with path.open('rb') as h:length=h.read(8);header=h.read(struct.unpack('<Q',length)[0])
    assert path.stat().st_size==review['size'] and hashlib.sha256(length+header).hexdigest()==review['header_sha256']
    with safe_open(str(path),framework='pt',device='cpu') as h:
        assert set(h.keys())=={k for k,v in weight_map.items() if v==shard}
        for row in (r for r in loading if r['source_file']==shard):
            key=row['source_key'];assert row['target_key']==key and row['device']==device
            value=h.get_tensor(key);assert row['source_dtype']==row['execution_dtype']==str(value.dtype)=='torch.bfloat16' and row['shape']==list(value.shape)
            locations=sorted({0,value.numel()//2,value.numel()-1});assert row['slice_positions']==locations and value.reshape(-1)[locations].float().tolist()==row['source_slice']==row['device_slice']
protocol=lineage['protocol'];original=lineage['original_evaluation_protocol'];path=pathlib.Path(original['config'])
assert hashlib.sha256(path.read_bytes()).hexdigest()==original['config_sha256']==cfg['model']['kwargs']['original_config_sha256']
previous=yaml.safe_load(path.read_text());adapter=previous['model']['adapter'];assert adapter==original['adapter']
assert previous['model']['name']==cfg['model']['name'] and previous['task_type']==cfg['task_type'] and 'image-text-to-text-mmbench/eval.jsonl' in previous['data']['dataset_path']
assert hashlib.sha256(pathlib.Path(adapter.rsplit(':',1)[0]).read_bytes()).hexdigest()==original['adapter_source_sha256']==cfg['model']['kwargs']['original_adapter_sha256']
kw=previous['model'].get('kwargs',{});budget=int(kw.get('max_new_tokens',kw.get('max_tokens',2048)))
expected_protocol='vllm_greedy';effective_budget=budget
if adapter.endswith(':HighTenReference'):
    if kw.get('kind')=='smol256':expected_protocol='high_ten_smol256'
    elif kw.get('kind')=='smol2' and kw.get('protocol')=='c500':expected_protocol='high_ten_smol2_c500';effective_budget=min(budget,64)
    else:raise AssertionError('Unreviewed original reference protocol')
else:assert adapter.endswith(':VLLMModel')
assert protocol==expected_protocol==cfg['model']['kwargs']['processor_protocol'] and budget==original['original_max_tokens']
assert effective_budget==original['effective_max_new_tokens']==cfg['model']['kwargs']['max_new_tokens']==lineage['max_new_tokens']
tok=AutoTokenizer.from_pretrained(str(weight_dir),local_files_only=True);ip=AutoImageProcessor.from_pretrained(str(weight_dir),local_files_only=True,backend='pil')
if protocol=='high_ten_smol2_c500':ip.max_image_size={'longest_edge':384};ip.size={'longest_edge':1536}
assert json.loads(json.dumps(ip.to_dict()))==lineage['effective_image_processor']
seq_len=(mc.vision_config.image_size//mc.vision_config.patch_size)**2//mc.scale_factor**2;assert seq_len==lineage['image_seq_len_per_patch']
if (weight_dir/'processor_config.json').is_file():assert json.loads((weight_dir/'processor_config.json').read_text()).get('image_seq_len',seq_len)==seq_len
template=(weight_dir/'chat_template.jinja').read_text() if (weight_dir/'chat_template.jinja').is_file() else (json.loads((weight_dir/'chat_template.json').read_text()).get('chat_template') if (weight_dir/'chat_template.json').is_file() else tok.chat_template);assert template
processor=Idefics3Processor(image_processor=ip,tokenizer=tok,image_seq_len=seq_len,chat_template=template)
gen=GenerationConfig.from_pretrained(str(weight_dir),local_files_only=True) if (weight_dir/'generation_config.json').is_file() else GenerationConfig.from_model_config(mc)
eos=gen.eos_token_id if gen.eos_token_id is not None else mc.text_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
assert eos==lineage['eos_token_ids'] and lineage['effective_generation_config']['do_sample'] is False
assert lineage['effective_generation_config']['repetition_penalty']==(1.0 if protocol=='vllm_greedy' else gen.repetition_penalty)
if protocol=='high_ten_smol2_c500':assert lineage['effective_generation_config']['pad_token_id']==tok.eos_token_id
rp=mc.text_config.rope_parameters;assert rp['rope_type'] in ('default','llama3');dim=mc.text_config.head_dim or mc.text_config.hidden_size//mc.text_config.num_attention_heads
freq=1/(float(rp['rope_theta'])**(torch.arange(0,dim,2).float()/dim))
if rp['rope_type']=='llama3':
    wavelength=2*torch.pi/freq;lo=rp['original_max_position_embeddings']/rp['low_freq_factor'];hi=rp['original_max_position_embeddings']/rp['high_freq_factor'];scaled=torch.where(wavelength>lo,freq/rp['factor'],freq)
    smooth=(rp['original_max_position_embeddings']/wavelength-rp['low_freq_factor'])/(rp['high_freq_factor']-rp['low_freq_factor']);blend=(1-smooth)*scaled/rp['factor']+smooth*scaled;freq=torch.where((wavelength<=lo)&(wavelength>=hi),blend,scaled)
assert torch.equal(freq,torch.tensor(lineage['text_inv_freq'])) and rp==lineage['text_rope_parameters']
dataset=pathlib.Path(cfg['data']['dataset_path']);assert dataset==pathlib.Path('/root/projects/comperror/workspace/datasets/image-text-to-text-mmbench/eval.jsonl');source=[json.loads(x) for x in dataset.read_text().splitlines()][:count]
def fingerprint(value):return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def check_tensor(value,record):assert list(value.shape)==record['shape'] and str(value.dtype)==record['dtype'] and fingerprint(value)==record['sha256']
def clean(s):
    s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
def extract(raw):
    value=raw.strip()
    # Independent recognition of complete, unambiguous wrapper forms.
    pairs=[('<','>'),('[',']'),('(',')'),('**','**')]
    for left,right in pairs:
        if value.startswith(left) and value.endswith(right):
            body=value[len(left):len(value)-len(right)].strip()
            if len(body)==1 and body.upper() in 'ABCDE':return body.upper()
    closed=re.fullmatch(r'<([A-Ea-e])>\s*([A-Ea-e])\s*</([A-Ea-e])>',value)
    if closed and closed[1].upper()==closed[2].upper()==closed[3].upper():return closed[1].upper()
    enclosed=[r'<answer>\s*([A-E])\s*</answer>',r'\\boxed\{\s*([A-E])\s*\}']
    for pattern in enclosed:
        values=re.findall(pattern,value,flags=re.I)
        if values:return values[-1].upper()
    starts=[r'(?i:(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*:?|:)\s*)',r'(?i:(?:correct\s+)?option\s+(?:is\s+)?\s*)',r'(?i:(?:choose|select|pick)\s+)']
    for index,start in enumerate(starts):
        quoted=re.findall(start+r"['\"(\[]\s*([A-Ea-e])\s*\.?\s*['\"\])]",value)
        if quoted:return quoted[-1].upper()
        label=r'([A-Ea-e])' if index==1 else r'([A-E])'
        tokens=re.findall(start+label+r'(?=[\s.,:;!?]|$)',value)
        if tokens:return tokens[-1].upper()
    first=re.match(r'^([A-Ea-e])\s*[.)](?=\s|$)',value)
    if first:return first[1].upper()
    lines=re.findall(r'(?m)^\s*([A-Ea-e])\s*$',value)
    return lines[-1].upper() if lines else raw

instruction='\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) in the format <answer>X</answer>.'
correct=0;empty=[];invalid=[];truncated=[]
for i,(a,b,d,s) in enumerate(zip(inputs,outputs,details,source)):
    assert a['prompt']==clean(s['prompt']) and a['reference']==b['reference']==d['reference']==s['reference']
    e=b['extra']['execution'];assert b['error'] is None and e['error'] is None and e['visual_embedding_injection_verified'] is True and e['vision_forward_calls']>=1
    assert e['processor_protocol']==protocol and e['prompt']==a['prompt']+instruction
    media=pathlib.Path(a['media_path']);original_media=pathlib.Path(s['media_path']);original_media=original_media if original_media.is_absolute() else dataset.parent/original_media
    assert media.is_relative_to(dataset.parent) and media.resolve()==original_media.resolve() and hashlib.sha256(media.read_bytes()).hexdigest()==e['source_image_sha256']
    text=e['prompt']
    if protocol=='high_ten_smol256':text+='\n\nRespond with ONLY the letter of the correct option inside <answer> tags, for example <answer>C</answer>.'
    chat=('<|im_start|>user\n<image>\n'+text+'<|im_end|>\n<|im_start|>assistant\n') if protocol=='high_ten_smol2_c500' else processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}],tokenize=False,add_generation_prompt=True)
    assert chat==e['chat_text']
    with Image.open(media) as image:image=image.convert('RGB')
    kwargs={'padding':True} if protocol.startswith('high_ten') else {'add_special_tokens':False};prepared=processor(text=chat,images=[image],return_tensors='pt',**kwargs)
    for name,value in prepared.items():
        if name in ('pixel_values','pixel_attention_mask'):check_tensor(value,e['image_tensors'][name])
        else:assert value.tolist()==e[name],('CPU text input differs',i,name)
    pixels=prepared['pixel_values'].reshape(-1,*prepared['pixel_values'].shape[-3:]);valid=(pixels!=0).any(dim=(-1,-2,-3));patch_count=int(valid.sum())
    assert patch_count==e['image_patch_count'] and int((prepared['input_ids']==mc.image_token_id).sum())==patch_count*seq_len
    assert e['image_features_shape']==[patch_count,seq_len,mc.text_config.hidden_size]
    # Rebuild image positional lookup indices from the CPU replayed padding mask, independently of the learned model.
    pm=prepared['pixel_attention_mask'].reshape(-1,*prepared['pixel_attention_mask'].shape[-2:])[valid];patch_size=mc.vision_config.patch_size
    mask=(pm.unfold(1,patch_size,patch_size).unfold(2,patch_size,patch_size).sum((-1,-2))>0)
    sides=mc.vision_config.image_size//patch_size;boundaries=torch.arange(1/sides,1.0,1/sides);positions=torch.zeros((patch_count,mask.shape[1]*mask.shape[2]),dtype=torch.int64)
    for n,m in enumerate(mask):
        h=int(m[:,0].sum());w=int(m[0,:].sum());ch=(torch.arange(mask.shape[1]).float()*(torch.tensor(1.0)/h)).clamp(max=1-1e-6).to(torch.bfloat16)
        cw=(torch.arange(mask.shape[2]).float()*(torch.tensor(1.0)/w)).clamp(max=1-1e-6).to(torch.bfloat16)
        row=torch.bucketize(ch,boundaries,right=True);col=torch.bucketize(cw,boundaries,right=True);lookup=(row[:,None]*sides+col[None,:]).flatten();positions[n,m.flatten()]=lookup[m.flatten()]
    check_tensor(positions,e['vision_position_ids'])
    assert bool(prepared['attention_mask'].all());check_tensor(torch.arange(prepared['input_ids'].shape[-1]).reshape(1,-1),e['text_position_ids'])
    tokens=e['generation_token_ids'];assert len(tokens)==e['generation_count'] and len(tokens)<=effective_budget
    raw=tok.decode(tokens,skip_special_tokens=True).strip();assert raw==b['raw_output']==e['raw_output']
    prediction=extract(raw);assert prediction==b['prediction']==d['prediction'],('Independent extraction differs',i,repr(raw),repr(prediction),repr(b['prediction']))
    if not raw:empty.append(i)
    if prediction not in re.findall(r'(?m)^\s*([A-E])\s*[.)]\s+',a['prompt']):invalid.append(i)
    clipped=len(tokens)>=effective_budget and (not tokens or tokens[-1] not in eos);assert clipped==e['truncated']
    if clipped:truncated.append(i)
    correct+=str(prediction).strip().lower()==str(b['reference']).strip().lower()
score=correct/count;assert abs(report['metrics']['Strict_MC_Accuracy']-score)<1e-12 and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
trace=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=trace if isinstance(trace,list) else trace['traceEvents'];hardware={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'}
kernels=[x for x in events if x.get('pid') in hardware and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert kernels
proof={'report_dir':str(folder),'passed':True,'sample_count':count,'correct':correct,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},'extraction_protocol':'native_explicit_mc_v2','historical_record_reuse':False,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'CPU_image_preprocessing_replayed':count,'tokenizer_replayed':count,'decoded_generation_replayed':count,'vision_position_lookup_replayed':count,'text_positions_replayed':count,'original_weight_slice_checks':True,'NPU_profiler_kernel_events':len(kernels),'processor_protocol':protocol,'protocol':'Exact existing BF16 Idefics3; original free generation protocol, own image/tokenizer/template assets and PIL LANCZOS; full CPU input/mask/image-position/text-position/token/output replay and full denominator strict MC score; all actual image patches injected into NPU language model. Invalid/truncated outputs retained.','report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
destination=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json';assert not destination.exists();destination.write_text(json.dumps(proof,indent=2),encoding='utf8');print(json.dumps(proof,indent=2))
