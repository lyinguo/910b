"""Independent file, metric and CPU preprocessing verification; no model inference."""
import collections,hashlib,json,pathlib,sys,re
import torch
from PIL import Image
from transformers import AutoTokenizer,AutoImageProcessor

report_dir=pathlib.Path(sys.argv[1]).resolve()
expected_count=int(sys.argv[2])
required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
assert {p.name for p in report_dir.iterdir()}==required
report=json.loads((report_dir/'eval_report.json').read_text())
lineage=json.loads((report_dir/'operator_lineage.json').read_text())
inputs=[json.loads(line) for line in (report_dir/'sample_inputs.jsonl').read_text().splitlines()]
outputs=[json.loads(line) for line in (report_dir/'sample_outputs.jsonl').read_text().splitlines()]
details=report['detailed_results']
assert len(inputs)==len(outputs)==len(details)==expected_count
assert [r['index'] for r in inputs]==list(range(expected_count))
assert [r['index'] for r in outputs]==list(range(expected_count))
config=report['pipeline_config']
assert config['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert config['model']['download'] is False
assert config['sample_range']=='0:'+str(expected_count-1)
root=pathlib.Path(config['model']['model_path'])
vit=next(root.glob('**/preprocessor_config.json')).parent
processor=AutoImageProcessor.from_pretrained(vit,trust_remote_code=True,local_files_only=True)
tokenizer=AutoTokenizer.from_pretrained(root,trust_remote_code=True,local_files_only=True)
dataset=pathlib.Path(config['data']['dataset_path'])
original=[json.loads(line) for line in dataset.read_text().splitlines()][:expected_count]
errors=[];empty=[];invalid=[];truncated=[];correct=0
preprocessed_indices=[]
def canonical_prompt(value):
    value=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',str(value))
    value=re.sub(r'[ \t]+',' ',value)
    value=re.sub(r'\n{3,}','\n\n',value).strip()
    return re.sub(r'<image\d+>\s*','',value)
for i,(inp,out,detail,source) in enumerate(zip(inputs,outputs,details,original)):
    expected_prompt=canonical_prompt(source['prompt'])
    assert inp['prompt']==expected_prompt
    if expected_prompt!=source['prompt']:preprocessed_indices.append(i)
    assert inp['reference']==source['reference']==out['reference']==detail['reference']
    assert out['prediction']==detail['prediction']
    execution=out['extra']['execution']
    if out.get('error'):errors.append(i)
    if not out['prediction']:empty.append(i)
    if execution.get('truncated'):truncated.append(i)
    scores=execution.get('option_scores',{})
    if out['prediction'] not in scores:invalid.append(i)
    if not errors and scores:
        assert out['prediction']==max(scores,key=scores.get)
    correct+=out['prediction']==out['reference']
    assert execution['visual_embedding_injection_verified'] is True
    image_path=pathlib.Path(inp['media_path'])
    assert image_path.is_relative_to(dataset.parent)
    assert hashlib.sha256(image_path.read_bytes()).hexdigest()==execution['source_image_sha256']
    image=Image.open(image_path).convert('RGB')
    w,h=image.size;side=max(w,h)
    background=tuple(int(x*255) for x in processor.image_mean)
    padded=Image.new('RGB',(side,side),background)
    padded.paste(image,((side-w)//2,(side-h)//2))
    pixels=processor(images=padded,return_tensors='pt')['pixel_values'].contiguous()
    digest=hashlib.sha256(pixels.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert digest==execution['pixel_sha256'],('pixel replay mismatch',i)
    ids=execution['input_ids']
    assert sum(v==tokenizer.convert_tokens_to_ids('<|im_sep|>') for v in ids)==1024
    clean=re.sub(r'\n+Answer the question by providing only the option letter.*$','',expected_prompt,flags=re.S)
    clean=re.sub(r'\n+Provide only the final answer\.?\s*$','',clean).strip()
    prefix=('<|im_start|>user\n'+'<|im_sep|>'*1024+'\n'+clean+'<|im_end|>\n<|im_start|>assistant\n').rstrip('\n')+'The correct answer is'
    replay=tokenizer(prefix,add_special_tokens=False)['input_ids']
    assert replay==ids,('tokenizer replay mismatch',i)
    assert detail['extra']['execution']==execution
score=correct/expected_count
assert abs(report['metrics']['Accuracy']-score)<1e-12
assert report['statistics']['total_samples']==expected_count
assert report['statistics']['error_count']==len(errors)
assert not errors and not empty and not invalid
actual=lineage['actual_execution']
assert actual['all_expected_parameters_loaded'] is True
assert actual['hardware']=='Ascend 910B'
trace_path=pathlib.Path(actual['profiler_trace'])
assert trace_path.is_file()
trace=json.loads(trace_path.read_text())
events=trace if isinstance(trace,list) else trace.get('traceEvents',[])
hardware_pids={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'}
npu_events=[e for e in events if e.get('pid') in hardware_pids and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')]
assert npu_events,'No NPU kernel events found in actual profiler trace'
result={'report_dir':str(report_dir),'passed':True,'sample_count':expected_count,'correct':correct,'Accuracy':score,'error_indices':errors,'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'source_text_normalization_indices':preprocessed_indices,'CPU_image_preprocessing_replayed':expected_count,'tokenizer_replayed':expected_count,'NPU_profiler_kernel_events':len(npu_events),'report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in report_dir.iterdir()}}
destination=pathlib.Path(actual['weight_loading_evidence']).parent/'independent_report_verification.json'
destination.write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
