"""Independent CPU input/decode replay and exact Accuracy for native InternVL/Qwen2."""
import hashlib,json,pathlib,re,sys
import torch
from PIL import Image
from transformers import AutoTokenizer
import numpy as np

folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2])
required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config']
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()]
outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()]
details=report['detailed_results']
assert len(inputs)==len(outputs)==len(details)==count
for records in (inputs,outputs,details):assert [x['index'] for x in records]==list(range(count))
assert cfg['model']['name'] in {'OpenGVLab/InternVL2-1B','OpenGVLab/InternVL2_5-4B-MPO','OpenGVLab/InternVL3-2B-Instruct'} and cfg['model']['download'] is False
assert cfg['data']['filter_empty'] is False
assert cfg['data']['download'] is False and cfg['sample_range']=='0:'+str(count-1)
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution']
assert lineage['all_expected_parameters_loaded'] and lineage['image_token_count_per_tile']==256
from safetensors import safe_open
weight_dir=pathlib.Path(cfg['model']['model_path'])
index_path=weight_dir/'model.safetensors.index.json'
if index_path.is_file():
    expected_keys=set(json.loads(index_path.read_text())['weight_map'])
else:
    with safe_open(str(weight_dir/'model.safetensors'),framework='pt',device='cpu') as handle:expected_keys=set(handle.keys())
assert lineage['weight_tensor_count']==len(expected_keys)
loading=json.loads(pathlib.Path(lineage['weight_loading_evidence']).read_text())
assert len(loading)==len(expected_keys) and {x['source_key'] for x in loading}==expected_keys
assert all(x['source_slice']==x['device_slice'] for x in loading)
assert all(x['execution_dtype']=='torch.bfloat16' and x['device'].startswith('npu:') for x in loading)
for path,digest in lineage['architecture_source_sha256'].items():
    assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==digest
assert folder.parent.name=='image-text-to-text' and folder.parent.parent.name==cfg['model']['name'].replace('/','_')
assert re.fullmatch(r'\d{8}_\d{6}',folder.name)
raw_cfg=json.loads((pathlib.Path(cfg['model']['model_path'])/'config.json').read_text())
tok=AutoTokenizer.from_pretrained(cfg['model']['model_path'],trust_remote_code=True,local_files_only=True)
def cpu_pixels(image):
    image=image.convert('RGB');w,h=image.size;size=raw_cfg['force_image_size'];low=raw_cfg['min_dynamic_patch'];high=raw_cfg['max_dynamic_patch']
    candidates={(x,y) for n in range(low,high+1) for x in range(1,n+1) for y in range(1,n+1) if low<=x*y<=high}
    best=(1,1);distance=float('inf')
    for x,y in sorted(candidates,key=lambda pair:pair[0]*pair[1]):
        difference=abs(w/h-x/y)
        if difference<distance or (difference==distance and w*h>.5*size*size*x*y):best=(x,y);distance=difference
    resized=image.resize((best[0]*size,best[1]*size))
    crops=[resized.crop((x*size,y*size,(x+1)*size,(y+1)*size)) for y in range(best[1]) for x in range(best[0])]
    if raw_cfg['use_thumbnail'] and len(crops)>1:crops.append(image.resize((size,size)))
    result=[]
    mean=torch.tensor([.485,.456,.406]).reshape(3,1,1);std=torch.tensor([.229,.224,.225]).reshape(3,1,1)
    for crop in crops:
        arr=np.array(crop,dtype=np.uint8,copy=True).transpose(2,0,1)
        value=torch.from_numpy(arr).contiguous().to(torch.float32).div(255)
        value=value.sub(mean).div(std);result.append(value)
    return torch.stack(result)

dataset=pathlib.Path(cfg['data']['dataset_path'])
source=[json.loads(x) for x in dataset.read_text().splitlines()][:count]
def clean(s):
    s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s)
    s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip()
    return re.sub(r'<image\d+>\s*','',s)
def extract(raw):
    import re
    patterns=[r'<answer>\s*([A-E])\s*</answer>',r'\\boxed\{\s*([A-E])\s*\}',r'(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*:?|:)\s*[\'"(\[]*\s*([A-E])(?=[\s.\'"\])]|$)',r'(?:correct\s+)?option\s+(?:is\s+)?[\'"(\[]*([A-E])\b',r'(?:choose|select|pick)\s+[\'"(\[]*([A-E])\b',r'^\s*([A-E])\s*[.)](?=\s|$)',r'^\s*([A-E])\s*$']
    for pattern in patterns:
        matches=re.findall(pattern,raw,flags=re.I)
        if matches:return matches[-1].upper()
    lines=re.findall(r'(?m)^\s*([A-E])\s*$',raw)
    return lines[-1].upper() if lines else raw
instruction='\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) in the format <answer>X</answer>.'
correct=0;invalid=[];empty=[];truncated=[];errors=[];predictions=[]
for i,(a,b,d,s) in enumerate(zip(inputs,outputs,details,source)):
    assert a['prompt']==clean(s['prompt'])
    assert a['reference']==b['reference']==d['reference']==s['reference']
    execution=b['extra']['execution'];assert execution['error'] is None
    assert b['error'] is None and execution['visual_embedding_injection_verified'] is True
    media=pathlib.Path(a['media_path']);assert media.is_relative_to(dataset.parent)
    assert hashlib.sha256(media.read_bytes()).hexdigest()==execution['source_image_sha256']
    pixels=cpu_pixels(Image.open(media)).contiguous()
    assert hashlib.sha256(pixels.view(torch.uint8).numpy().tobytes()).hexdigest()==execution['pixel_sha256']
    actual_prompt=a['prompt']+instruction
    assert execution['prompt']==actual_prompt
    assert list(pixels.shape)==execution['pixel_shape']
    assert execution['image_features_shape']==[len(pixels),256,raw_cfg['llm_config']['hidden_size']]
    # Replay the checkpoint Hermes-2 format independently of the adapter.
    system=pathlib.Path(lineage['architecture_source_snapshot'])/'conversation.py'
    import importlib.util
    if i==0:
        spec=importlib.util.spec_from_file_location('verify_native_conversation',system);conv=importlib.util.module_from_spec(spec);sys.modules[spec.name]=conv;spec.loader.exec_module(conv)
        template=conv.get_conv_template(raw_cfg['template']);system_message=template.system_message
        assert raw_cfg['template'] in ('Hermes-2','internvl2_5') and template.sep_style==conv.SeparatorStyle.MPT
    image_tokens='<img>'+'<IMG_CONTEXT>'*(256*len(pixels))+'</img>'
    query=template.system_template.format(system_message=system_message)+template.sep+template.roles[0]+image_tokens+'\n'+actual_prompt+template.sep+template.roles[1]
    assert query==execution['query']
    ids=tok(query)['input_ids']
    assert ids==execution['input_ids']
    generated=execution['generation_token_ids']
    assert len(generated)==execution['generation_count']
    assert len(generated)<=cfg['model']['kwargs']['max_new_tokens']
    raw=tok.decode(generated,skip_special_tokens=True).strip()
    assert raw==b['raw_output']==execution['raw_output']
    prediction=extract(raw)
    assert prediction==b['prediction']==d['prediction'],('Answer extraction disagreement',i,repr(raw),repr(prediction),repr(b['prediction']))
    if not raw:empty.append(i)
    if prediction not in re.findall(r'(?m)^\s*([A-E])\s*[.)]\s+',a['prompt']):invalid.append(i)
    is_truncated=len(generated)>=cfg['model']['kwargs']['max_new_tokens'] and (not generated or generated[-1]!=tok.convert_tokens_to_ids('<|im_end|>'))
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
proof={'report_dir':str(folder),'passed':True,'sample_count':count,'correct':correct,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},'error_indices':errors,'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'CPU_image_preprocessing_replayed':count,'tokenizer_replayed':count,'decoded_generation_replayed':count,'NPU_profiler_kernel_events':len(kernels),'protocol':'Native BF16 InternVL/Qwen2 generation; full original image tiles and Hermes-2 format independently replayed; explicit MC extraction and full denominator; all invalid/truncated/failed samples retained','report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
destination=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json'
destination.write_text(json.dumps(proof,indent=2),encoding='utf8')
print(json.dumps(proof,indent=2))
