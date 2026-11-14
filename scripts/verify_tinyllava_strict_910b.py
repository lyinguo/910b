"""Independent CPU input/decode replay and exact Accuracy for native TinyLLaVA."""
import hashlib,json,pathlib,re,sys
import torch
from PIL import Image
from transformers import AutoTokenizer,AutoImageProcessor

folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2])
required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config']
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()]
outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()]
details=report['detailed_results']
assert len(inputs)==len(outputs)==len(details)==count
for records in (inputs,outputs,details):assert [x['index'] for x in records]==list(range(count))
assert cfg['model']['name']=='0xAmey/tinyllava-1.1b-v0.1' and cfg['model']['download'] is False
assert cfg['data']['download'] is False and cfg['sample_range']=='0:'+str(count-1)
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution']
assert lineage['all_expected_parameters_loaded'] and lineage['image_token_count']==256
assert lineage['image_placeholder_token']=='[PAD]'
tok=AutoTokenizer.from_pretrained(cfg['model']['model_path'],trust_remote_code=True,local_files_only=True)
processor=AutoImageProcessor.from_pretrained(lineage['vision_metadata_dir'],local_files_only=True)
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
prefix="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: "
instruction='\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) in the format <answer>X</answer>.'
correct=0;invalid=[];empty=[];truncated=[];errors=[];predictions=[]
for i,(a,b,d,s) in enumerate(zip(inputs,outputs,details,source)):
    assert a['prompt']==clean(s['prompt'])
    assert a['reference']==b['reference']==d['reference']==s['reference']
    execution=b['extra']['execution'];assert execution['error'] is None
    assert b['error'] is None and execution['visual_embedding_injection_verified'] is True
    media=pathlib.Path(a['media_path']);assert media.is_relative_to(dataset.parent)
    assert hashlib.sha256(media.read_bytes()).hexdigest()==execution['source_image_sha256']
    pixels=processor(images=Image.open(media).convert('RGB'),return_tensors='pt')['pixel_values'].contiguous()
    assert hashlib.sha256(pixels.view(torch.uint8).numpy().tobytes()).hexdigest()==execution['pixel_sha256']
    actual_prompt=a['prompt']+instruction
    assert execution['prompt']==actual_prompt
    ids=tok(prefix,add_special_tokens=True)['input_ids']+[tok.convert_tokens_to_ids('[PAD]')]*256+tok('\n'+actual_prompt+' ASSISTANT:',add_special_tokens=False)['input_ids']
    assert ids==execution['input_ids']
    generated=execution['generation_token_ids']
    assert len(generated)==execution['generation_count']
    raw=tok.decode(generated,skip_special_tokens=True).strip()
    assert raw==b['raw_output']==execution['raw_output']
    prediction=extract(raw)
    assert prediction==b['prediction']==d['prediction'],('Answer extraction disagreement',i,repr(raw),repr(prediction),repr(b['prediction']))
    if not raw:empty.append(i)
    if prediction not in re.findall(r'(?m)^\s*([A-E])\s*[.)]\s+',a['prompt']):invalid.append(i)
    is_truncated=len(generated)>=cfg['model']['kwargs']['max_new_tokens'] and (not generated or generated[-1]!=2)
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
proof={'report_dir':str(folder),'passed':True,'sample_count':count,'correct':correct,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},'error_indices':errors,'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'CPU_image_preprocessing_replayed':count,'tokenizer_replayed':count,'decoded_generation_replayed':count,'NPU_profiler_kernel_events':len(kernels),'protocol':'Native greedy generation; only explicit option labels count. Ordinary a/e words in explanation are not options. All invalid/truncated/failed samples remain in the full denominator','report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
destination=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json'
destination.write_text(json.dumps(proof,indent=2),encoding='utf8')
print(json.dumps(proof,indent=2))
