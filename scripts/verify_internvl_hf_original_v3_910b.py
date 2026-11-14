"""Independent original-HF InternVL CPU preprocessing/token/position/weights and full-denominator MC audit."""
import pathlib,json,sys,hashlib,re,struct,torch,yaml,os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from transformers import AutoConfig,AutoProcessor,InternVLForConditionalGeneration,GenerationConfig
from safetensors import safe_open
from PIL import Image
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];mid=cfg['model']['name']
allowed={'internlm/CapRL-InternVL3.5-8B','OpenGVLab/InternVL3-1B-hf','OpenGVLab/InternVL3-8B-hf','OpenGVLab/InternVL3_5-1B-HF','OpenGVLab/InternVL3_5-8B-HF'};assert mid in allowed
assert folder.parent.name==cfg['task_type']=='image-text-to-text' and folder.parent.parent.name==mid.replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_range']=='0:'+str(count-1) and cfg['sample_count']==count and cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['model']['download'] is False
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert cfg['algorithm']['adapter']=='universal_eval/algorithm/native_explicit_mc_v2.py:NativeExplicitMCV2' and cfg['algorithm']['kwargs']['extract_answer'] is True
assert cfg['metrics']['override']==['Strict_MC_Accuracy'] and cfg['metrics']['include_failed_predictions'] is True
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];assert len(inputs)==len(outputs)==len(details)==count
for records in (inputs,outputs,details):assert [x['index'] for x in records]==list(range(count))
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];device='npu:'+str(kw['npu_devices'][0]);assert lineage['devices']==kw['npu_devices'] and len(lineage['devices'])==1 and set(lineage['module_placements'].values())=={device}
assert lineage['all_expected_parameters_loaded'] and lineage['source_dtype']=='BF16' and lineage['HF32_enabled']=={'conv':False,'matmul':False} and lineage['untied_source_head_retained'] is True
assert lineage['model_resource_sha256']==kw['resource_sha256'] and lineage['runtime_source_sha256']==kw['runtime_sha256'] and lineage['weight_headers']==kw['weight_headers']
for path,digest in lineage['runtime_source_sha256'].items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==digest
directory=pathlib.Path(cfg['model']['model_path']).resolve();assert directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels') and directory.parent.name+'/'+directory.name==mid
for name,digest in lineage['model_resource_sha256'].items():assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
raw_cfg=json.loads((directory/'config.json').read_text());mc=AutoConfig.from_pretrained(str(directory),local_files_only=True);assert raw_cfg.get('tie_word_embeddings') is not True and raw_cfg['text_config'].get('tie_word_embeddings') is not True
mc.tie_word_embeddings=False;mc.text_config.tie_word_embeddings=False
with torch.device('meta'):blueprint=InternVLForConditionalGeneration(mc)
expected={n:list(t.shape) for n,t in blueprint.named_parameters()};index=directory/'model.safetensors.index.json'
if index.exists():weight_map=json.loads(index.read_text())['weight_map']
else:
 with safe_open(str(directory/'model.safetensors'),framework='pt',device='cpu') as h:weight_map={k:'model.safetensors' for k in h.keys()}
loading=json.loads(pathlib.Path(lineage['weight_loading_evidence']).read_text());assert len(loading)==len(weight_map)==lineage['weight_tensor_count']==len(expected) and {x['source_key'] for x in loading}==set(weight_map)
def target_name(k):
 parts=k.split('.')
 if parts[:2]==['language_model','model']:return '.'.join(['model','language_model']+parts[2:])
 if parts[:2]==['language_model','lm_head']:return '.'.join(parts[1:])
 if parts[0] in ('vision_tower','multi_modal_projector'):return 'model.'+k
 raise AssertionError(('Unreviewed original key',k))
assert {target_name(x['source_key']) for x in loading}==set(expected)
for shard,review in lineage['weight_headers'].items():
 p=directory/shard
 with p.open('rb') as f:length=f.read(8);header=f.read(struct.unpack('<Q',length)[0])
 assert p.stat().st_size==review['size'] and hashlib.sha256(length+header).hexdigest()==review['header_sha256']
 with safe_open(str(p),framework='pt',device='cpu') as handle:
  assert set(handle.keys())=={k for k,v in weight_map.items() if v==shard}
  for r in (x for x in loading if x['source_file']==shard):
   key=r['source_key'];target=target_name(key);tensor=handle.get_tensor(key);assert r['target_key']==target and list(tensor.shape)==expected[target]==r['shape'] and tensor.dtype==torch.bfloat16
   locs=sorted({0,tensor.numel()//2,tensor.numel()-1});assert r['slice_positions']==locs and tensor.reshape(-1)[locs].float().tolist()==r['source_slice']==r['device_slice']
   assert r['device']==device and r['source_dtype']==r['execution_dtype']=='torch.bfloat16'
for source_key,sha_name in [('language_model.lm_head.weight','source_head_sha256'),('language_model.model.embed_tokens.weight','source_embedding_sha256')]:
 with safe_open(str(directory/weight_map[source_key]),framework='pt',device='cpu') as h:
  t=h.get_tensor(source_key);digest=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();assert digest==lineage[sha_name]==kw[sha_name]
assert lineage['source_head_sha256']!=lineage['source_embedding_sha256']
original=pathlib.Path(kw['original_config']);assert hashlib.sha256(original.read_bytes()).hexdigest()==kw['original_config_sha256']==lineage['original_config_sha256'];old=yaml.safe_load(original.read_text());assert old['model']['name']==mid and old['model']['kwargs']['max_new_tokens']==2048 and old['algorithm']['kwargs']['extract_answer'] is True
assert hashlib.sha256(pathlib.Path('universal_eval/model/adapters/transformers_vlm.py').read_bytes()).hexdigest()==kw['original_adapter_sha256']==lineage['original_adapter_sha256']
processor=AutoProcessor.from_pretrained(str(directory),local_files_only=True);assert json.loads(json.dumps(processor.image_processor.to_dict()))==lineage['effective_image_processor'] and processor.image_token_id==mc.image_token_id==lineage['image_token_id'] and processor.image_seq_length==mc.image_seq_length==256
gen=GenerationConfig.from_pretrained(str(directory),local_files_only=True) if (directory/'generation_config.json').exists() else GenerationConfig.from_model_config(mc);blueprint.generation_config=gen;resolved,_=blueprint._prepare_generation_config(None,max_new_tokens=2048,do_sample=False,temperature=0.0)
assert resolved.to_dict()==lineage['effective_generation_config'] and resolved.do_sample is False and resolved.num_beams==1 and kw['max_new_tokens']==lineage['max_new_tokens']==2048
eos=resolved.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos);assert eos==lineage['eos_token_ids']
def fingerprint(t):return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def check_tensor(t,r):assert list(t.shape)==r['shape'] and str(t.dtype)==r['dtype'] and fingerprint(t)==r['sha256']
dim=getattr(mc.text_config,'head_dim',None) or mc.text_config.hidden_size//mc.text_config.num_attention_heads;rp=mc.text_config.rope_parameters;assert rp['rope_type'] in ('default','dynamic')
freq=1.0/(float(rp['rope_theta'])**(torch.arange(0,dim,2,dtype=torch.float32)/dim));assert set(lineage['analytic_buffers'])==set(dict(blueprint.named_buffers()))
for name,record in lineage['analytic_buffers'].items():assert name.endswith(('inv_freq','original_inv_freq'));check_tensor(freq,record)
dataset=pathlib.Path(cfg['data']['dataset_path']);assert dataset.is_relative_to('/root/projects/comperror/workspace/datasets') and hashlib.sha256(dataset.read_bytes()).hexdigest()==kw['dataset_sha256']==lineage['dataset_sha256'];source=[json.loads(x) for x in dataset.read_text().splitlines()];assert len(source)>=200
reference=pathlib.Path(kw['reference_report']);assert hashlib.sha256(reference.read_bytes()).hexdigest()==kw['reference_report_sha256']==lineage['historical_reference_sha256'];historical=json.loads(reference.read_text());assert len(historical['detailed_results'])==200 and historical['pipeline_config']['model']['name']==mid
def clean(s):
 s=''.join(ch for ch in s if ord(ch) not in set(list(range(0,9))+[11,12]+list(range(14,32))+[127]));s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip();return re.sub(r'<image\d+>\s*','',s)
for i,(r,s) in enumerate(zip(historical['detailed_results'],source)):
 assert r['index']==i and r['prompt']==clean(s['prompt']) and r['reference']==s['reference']
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
correct=0;invalid=[];empty=[];truncated=[];scores=[];trace_rows=[json.loads(x) for x in pathlib.Path(lineage['per_sample_evidence']).read_text().splitlines()];assert len(trace_rows)==count
for i,(a,b,d,s,observed) in enumerate(zip(inputs,outputs,details,source,trace_rows)):
 assert a['prompt']==clean(s['prompt']) and a['reference']==b['reference']==d['reference']==s['reference']
 e=b['extra']['execution'];assert e==observed and e['error'] is None and b['error'] is None and e['visual_embedding_injection_verified'] is True and e['vision_forward_calls']>=1
 media=pathlib.Path(a['media_path']).resolve();p=pathlib.Path(s['media_path']);p=p if p.is_absolute() else dataset.parent/p;assert media==p.resolve() and media.is_relative_to(dataset.parent) and hashlib.sha256(media.read_bytes()).hexdigest()==e['source_image_sha256']
 actual_prompt=a['prompt']+instruction;assert e['prompt']==actual_prompt and e['messages']==[{'role':'user','content':'<image>\n'+actual_prompt}]
 # Independently assemble the exact checkpoint template, rather than calling the new adapter's template code.
 chat='<|im_start|>user\n<IMG_CONTEXT>\n'+actual_prompt+'<|im_end|>\n<|im_start|>assistant\n';assert chat==e['chat']
 with Image.open(media) as im:prepared=processor(text=chat,images=[im.convert('RGB')],return_tensors='pt',padding=True)
 for k,t in prepared.items():check_tensor(t,e['CPU_processor_tensors'][k])
 assert prepared['input_ids'].tolist()==e['input_ids'] and prepared['attention_mask'].tolist()==e['attention_mask'] and bool(prepared['attention_mask'].all())
 pixels=prepared['pixel_values'];assert int((prepared['input_ids']==mc.image_token_id).sum())==len(pixels)*256 and e['image_features_shape']==[len(pixels),256,mc.text_config.hidden_size]
 length=prepared['input_ids'].shape[-1];assert length+2048<mc.text_config.max_position_embeddings;check_tensor(torch.arange(length,dtype=torch.int64).reshape(1,-1),e['position_ids'])
 dtype=getattr(torch,e['causal_mask']['dtype'].split('.')[-1]);mask=torch.triu(torch.full((length,length),torch.finfo(dtype).min,dtype=dtype),diagonal=1).reshape(1,1,length,length);check_tensor(mask,e['causal_mask'])
 tokens=e['generation_token_ids'];assert len(tokens)==e['generation_count'] and len(tokens)<=2048;raw=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip();assert raw==e['raw_output']==b['raw_output'];pred=extract(raw);assert pred==b['prediction']==d['prediction']
 if not raw:empty.append(i)
 if pred not in re.findall(r'(?m)^\s*([A-E])\s*[.)]\s+',a['prompt']):invalid.append(i)
 clip=len(tokens)>=2048 and (not tokens or tokens[-1] not in eos);assert clip==e['truncated']
 if clip:truncated.append(i)
 hit=str(pred).strip().lower()==str(b['reference']).strip().lower();correct+=hit;scores.append(int(hit))
score=correct/count;assert abs(report['metrics']['Strict_MC_Accuracy']-score)<1e-12 and report['summary']['primary_metric']=='Strict_MC_Accuracy' and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
trace=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=trace if isinstance(trace,list) else trace['traceEvents'];hardware={x['pid'] for x in events if x.get('name')=='process_name' and 'Ascend Hardware' in x.get('args',{}).get('name','')};kernels=[x for x in events if x.get('pid') in hardware and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'report_dir':str(folder),'passed':True,'sample_count':count,'correct':correct,'metrics':{'Strict_MC_Accuracy':score},'Accuracy':score,'historical_record_reuse':False,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'CPU_image_preprocessing_replayed':count,'tokenizer_replayed':count,'decoded_generation_replayed':count,'causal_masks_and_positions_replayed':count,'historical_inputs_and_references_replayed':200,'NPU_profiler_kernel_events':len(kernels),'source_dtype':'BF16','source_head_and_embedding_preserved_separately':True,'sample_correct':scores,'protocol':'Original own native HF InternVL weights and dynamic image processor/chat; complete unshared original checkpoint head, original2048 native generation. Independent CPU image/token/raw decode/causal positions and strict explicit MC v2, complete denominator and all invalid/empty/clipped rows retained. Genuine NPU kernels, no CPU learned layers or downloads.','report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
destination=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json';assert not destination.exists();destination.write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
