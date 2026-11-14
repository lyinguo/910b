"""Independent CPU input/decode replay and exact Accuracy for native Gemma3."""
import hashlib,json,pathlib,re,sys
import torch
from PIL import Image
from transformers import AutoTokenizer,Gemma3ImageProcessor,Gemma3Processor

folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2])
required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config']
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()]
outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()]
details=report['detailed_results']
assert len(inputs)==len(outputs)==len(details)==count
for records in (inputs,outputs,details):assert [x['index'] for x in records]==list(range(count))
weight_dir=pathlib.Path(cfg['model']['model_path']).resolve()
assert cfg['model']['name']==weight_dir.parent.name+'/'+weight_dir.name and cfg['model']['download'] is False
assert json.loads((weight_dir/'config.json').read_text())['architectures']==['Gemma3ForConditionalGeneration']
assert folder.parent.parent.name==cfg['model']['name'].replace('/','_') and folder.parent.name=='image-text-to-text'
assert re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['data']['filter_empty'] is False
assert cfg['data']['download'] is False and cfg['sample_range']=='0:'+str(count-1)
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution']
assert lineage['all_expected_parameters_loaded'] and lineage['image_token_count_per_image']==256
from transformers import AutoConfig
expected_tied=bool(AutoConfig.from_pretrained(str(weight_dir),local_files_only=True).tie_word_embeddings)
assert lineage['tied_output_head_restored']==expected_tied
loading=json.loads(pathlib.Path(lineage['weight_loading_evidence']).read_text())
from safetensors import safe_open
index=weight_dir/'model.safetensors.index.json'
if index.is_file():keys=set(json.loads(index.read_text())['weight_map'])
else:
    with safe_open(str(weight_dir/'model.safetensors'),framework='pt',device='cpu') as handle:keys=set(handle.keys())
assert len(loading)==len(keys)==lineage['weight_tensor_count'] and {x['source_key'] for x in loading}==keys
assert len({x['target_key'] for x in loading})==len(loading)
for row in loading:
    assert row['source_slice']==row['device_slice'] and row['source_dtype']==row['execution_dtype']=='torch.bfloat16'
    assert row['device'].startswith('npu:')
    old=row['source_key'];mapped=old
    for left,right in [('language_model.model.','model.language_model.'),('language_model.lm_head.','lm_head.'),('vision_tower.vision_model.','model.vision_tower.'),('multi_modal_projector.','model.multi_modal_projector.')]:
        if old.startswith(left):mapped=right+old[len(left):];break
    assert mapped==row['target_key']
for source,digest in lineage['runtime_source_sha256'].items():assert hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()==digest
for rel,digest in lineage['model_resource_sha256'].items():assert hashlib.sha256((weight_dir/rel).read_bytes()).hexdigest()==digest
metadata=pathlib.Path(lineage['processor_metadata_source'])
assert hashlib.sha256(metadata.read_bytes()).hexdigest()==lineage['processor_metadata_sha256']

tok=AutoTokenizer.from_pretrained(cfg['model']['model_path'],trust_remote_code=True,local_files_only=True)
image_processor=Gemma3ImageProcessor(**json.loads(metadata.read_text()))
template=(weight_dir/'chat_template.jinja').read_text() if (weight_dir/'chat_template.jinja').is_file() else tok.chat_template
assert template
processor=Gemma3Processor(image_processor=image_processor,tokenizer=tok,image_seq_length=256,chat_template=template)
dataset=pathlib.Path(cfg['data']['dataset_path'])
source=[json.loads(x) for x in dataset.read_text().splitlines()][:count]
def clean(s):
    s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s)
    s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip()
    return re.sub(r'<image\d+>\s*','',s)
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
model_eos=json.loads((weight_dir/'config.json').read_text()).get('eos_token_id') or [1,106]
instruction='\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) in the format <answer>X</answer>.'
correct=0;invalid=[];empty=[];truncated=[];errors=[];predictions=[]
for i,(a,b,d,s) in enumerate(zip(inputs,outputs,details,source)):
    assert a['prompt']==clean(s['prompt'])
    assert a['reference']==b['reference']==d['reference']==s['reference']
    execution=b['extra']['execution'];assert execution['error'] is None
    assert b['error'] is None and execution['visual_embedding_injection_verified'] is True
    media=pathlib.Path(a['media_path']);assert media.is_relative_to(dataset.parent)
    assert hashlib.sha256(media.read_bytes()).hexdigest()==execution['source_image_sha256']
    actual_prompt=a['prompt']+instruction
    assert execution['prompt']==actual_prompt
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':actual_prompt}]}]
    chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    assert chat==execution['chat_text']
    prepared=processor(text=chat,images=Image.open(media).convert('RGB'),return_tensors='pt',add_special_tokens=False,return_mm_token_type_ids=True)
    pixels=prepared['pixel_values'].contiguous()
    assert hashlib.sha256(pixels.view(torch.uint8).numpy().tobytes()).hexdigest()==execution['pixel_sha256']
    assert list(pixels.shape)==execution['pixel_shape']
    for name,value in prepared.items():
        if name!='pixel_values':assert value.tolist()==execution[name],('CPU input differs',i,name)
    assert int((prepared['input_ids']==tok.image_token_id).sum())==256
    assert execution['vision_forward_calls']>=1 and execution['image_features_shape']==[1,256,json.loads((weight_dir/'config.json').read_text())['text_config']['hidden_size']]

    generated=execution['generation_token_ids']
    assert len(generated)==execution['generation_count']
    raw=tok.decode(generated,skip_special_tokens=True).strip()
    assert raw==b['raw_output']==execution['raw_output']
    prediction=extract(raw)
    assert prediction==b['prediction']==d['prediction'],('Answer extraction disagreement',i,repr(raw),repr(prediction),repr(b['prediction']))
    if not raw:empty.append(i)
    if prediction not in re.findall(r'(?m)^\s*([A-E])\s*[.)]\s+',a['prompt']):invalid.append(i)
    is_truncated=len(generated)>=cfg['model']['kwargs']['max_new_tokens'] and (not generated or generated[-1] not in ([model_eos] if isinstance(model_eos,int) else model_eos))
    assert is_truncated==execution['truncated']
    if is_truncated:truncated.append(i)
    correct+=str(prediction).strip().lower()==str(b['reference']).strip().lower()
    predictions.append(prediction)
score=correct/count
assert abs(report['metrics']['Strict_MC_Accuracy']-score)<1e-12
assert report['statistics']['total_samples']==count and report['statistics']['error_count']==0
trace=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=trace if isinstance(trace,list) else trace['traceEvents']
hardware={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'}
kernels=[e for e in events if e.get('pid') in hardware and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')]
assert kernels
proof={'report_dir':str(folder),'passed':True,'extraction_protocol':'native_explicit_mc_v2','historical_record_reuse':bool(report.get('historical_regrade')),'sample_count':count,'correct':correct,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},'error_indices':errors,'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'CPU_image_preprocessing_replayed':count,'tokenizer_replayed':count,'decoded_generation_replayed':count,'NPU_profiler_kernel_events':len(kernels),'protocol':'Native Gemma3 BF16 generation; exact source checkpoint and own tokenizer/template, existing local processor metadata. Full CPU image/token/token-type/decode replay; strict explicit MC scores over all samples; invalid/truncated outputs retained','report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
destination=pathlib.Path(lineage.get('postprocessing_evidence_dir',pathlib.Path(lineage['weight_loading_evidence']).parent))/'independent_report_verification.json'
destination.write_text(json.dumps(proof,indent=2),encoding='utf8')
print(json.dumps(proof,indent=2))
