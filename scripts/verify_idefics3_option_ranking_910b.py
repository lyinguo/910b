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

import numpy as np
assert lineage['protocol']==cfg['model']['kwargs']['processor_protocol']=='smoldocling_option_ranking'
original=lineage['original_evaluation_protocol'];original_path=pathlib.Path(original['config']);assert hashlib.sha256(original_path.read_bytes()).hexdigest()==original['config_sha256']==cfg['model']['kwargs']['original_config_sha256']
previous=yaml.safe_load(original_path.read_text());assert previous['model']['name']==cfg['model']['name'] and previous['task_type']==cfg['task_type'] and previous['model']['adapter'].endswith(':SmolDoclingDiscriminativeVLM')
assert previous['model']['kwargs']['scoring']=='option' and 'image-text-to-text-mmbench/eval.jsonl' in previous['data']['dataset_path']
adapter=pathlib.Path(previous['model']['adapter'].rsplit(':',1)[0]);assert hashlib.sha256(adapter.read_bytes()).hexdigest()==original['adapter_source_sha256']==cfg['model']['kwargs']['original_adapter_sha256']
assert previous['model']['kwargs']['max_new_tokens']==cfg['model']['kwargs']['max_new_tokens']==original['original_max_tokens']
tok=AutoTokenizer.from_pretrained(str(weight_dir),local_files_only=True);ip=AutoImageProcessor.from_pretrained(str(weight_dir),local_files_only=True,backend='pil');assert json.loads(json.dumps(ip.to_dict()))==lineage['effective_image_processor']
seq_len=(mc.vision_config.image_size//mc.vision_config.patch_size)**2//mc.scale_factor**2;assert seq_len==lineage['image_seq_len_per_patch']
processor=Idefics3Processor(image_processor=ip,tokenizer=tok,image_seq_len=seq_len,chat_template=json.loads((weight_dir/'chat_template.json').read_text())['chat_template'])
assert mc.text_config.rope_parameters['rope_type']=='default';dim=mc.text_config.head_dim or mc.text_config.hidden_size//mc.text_config.num_attention_heads
freq=1/(float(mc.text_config.rope_parameters['rope_theta'])**(torch.arange(0,dim,2).float()/dim));assert torch.equal(freq,torch.tensor(lineage['text_inv_freq']))
dataset=pathlib.Path(cfg['data']['dataset_path']);assert dataset==pathlib.Path('/root/projects/comperror/workspace/datasets/image-text-to-text-mmbench/eval.jsonl');source=[json.loads(x) for x in dataset.read_text().splitlines()][:count]
def clean(text):
    text=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',text);text=re.sub(r'[ \t]+',' ',text);text=re.sub(r'\n{3,}','\n\n',text).strip();return re.sub(r'<image\d+>\s*','',text)
def independent_options(text):
    boundaries=list(re.finditer(r'^\s*([A-E])\s*[.)]\s+',text,re.M));options=[];expected=ord('A')
    for i,match in enumerate(boundaries):
        end=boundaries[i+1].start() if i+1<len(boundaries) else len(text);choice=text[match.end():end].strip()
        if ord(match.group(1))==expected and choice:options.append([match.group(1),choice]);expected+=1
    return options
def fingerprint(value):return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def check_tensor(value,record):assert list(value.shape)==record['shape'] and str(value.dtype)==record['dtype'] and fingerprint(value)==record['sha256']
def check_inputs(prepared,record):
    assert set(prepared)==set(record)
    for name,value in prepared.items():
        if name in ('pixel_values','pixel_attention_mask'):check_tensor(value,record[name]['tensor'])
        else:assert value.tolist()==record[name]['values'] and str(value.dtype)==record[name]['dtype']
def image_position_ids(prepared):
    pixels=prepared['pixel_values'].reshape(-1,*prepared['pixel_values'].shape[-3:]);valid=(pixels!=0).any(dim=(-1,-2,-3));patch_count=int(valid.sum());pm=prepared['pixel_attention_mask'].reshape(-1,*prepared['pixel_attention_mask'].shape[-2:])[valid];ps=mc.vision_config.patch_size
    mask=pm.unfold(1,ps,ps).unfold(2,ps,ps).sum((-1,-2))>0;sides=mc.vision_config.image_size//ps;boundaries=torch.arange(1/sides,1.0,1/sides);position=torch.zeros(patch_count,mask.shape[1]*mask.shape[2],dtype=torch.int64)
    for n,m in enumerate(mask):
        h=int(m[:,0].sum());w=int(m[0,:].sum());ch=(torch.arange(mask.shape[1]).float()*(torch.tensor(1.0)/h)).clamp(max=1-1e-6).to(torch.bfloat16);cw=(torch.arange(mask.shape[2]).float()*(torch.tensor(1.0)/w)).clamp(max=1-1e-6).to(torch.bfloat16)
        row=torch.bucketize(ch,boundaries,right=True);col=torch.bucketize(cw,boundaries,right=True);lookup=(row[:,None]*sides+col[None,:]).flatten();position[n,m.flatten()]=lookup[m.flatten()]
    return position,patch_count
instruction='\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) in the format <answer>X</answer>.'
correct=0;candidate_count=0;logit_rows=0;max_difference=0.0;all_paths=set()
for index,(a,b,d,s) in enumerate(zip(inputs,outputs,details,source)):
    assert a['prompt']==clean(s['prompt']) and a['reference']==b['reference']==d['reference']==s['reference']
    e=b['extra']['execution'];assert b['error'] is None and e['error'] is None and e['autoregressive_generation_performed'] is False
    assert e['decision_protocol']=='teacher_forced_option_mean_logprob' and e['prompt']==a['prompt']+instruction
    assert e['stripped_prompt']==a['prompt'];options=independent_options(a['prompt']);assert len(options)>=2 and e['options']==options and len(e['candidates'])==len(options)
    prefix='<image>'+a['prompt'].rstrip()+'\n\nThe correct answer is';assert e['prefix_text']==prefix
    media=pathlib.Path(a['media_path']);original_media=pathlib.Path(s['media_path']);original_media=original_media if original_media.is_absolute() else dataset.parent/original_media
    assert media.is_relative_to(dataset.parent) and media.resolve()==original_media.resolve() and hashlib.sha256(media.read_bytes()).hexdigest()==e['source_image_sha256']
    with Image.open(media) as image:image=image.convert('RGB')
    base_inputs=processor(text=[prefix],images=[[image]],return_tensors='pt');check_inputs(base_inputs,e['base_inputs']);assert base_inputs['input_ids'].shape[1]==e['base_length']
    cpu_means=[];npu_means=[]
    for (letter,text),candidate in zip(options,e['candidates']):
        assert candidate['error'] is None and candidate['letter']==letter and candidate['option_text']==text
        continuation=' '+text;assert candidate['continuation']==continuation and candidate['full_text']==prefix+continuation
        prepared=processor(text=[prefix+continuation],images=[[image]],return_tensors='pt');check_inputs(prepared,candidate['inputs'])
        n=len(tok(continuation,add_special_tokens=False)['input_ids']);full_len=prepared['input_ids'].shape[1];assert n>0 and full_len>e['base_length']
        start=max(full_len-n,1);positions=list(range(start,full_len));targets=prepared['input_ids'][0,start:full_len]
        assert candidate['target_positions']==positions and candidate['target_token_ids']==targets.tolist() and candidate['continuation_token_count']==n
        evidence=pathlib.Path(candidate['full_vocabulary_selected_logits']);assert evidence.is_relative_to(pathlib.Path(lineage['candidate_logits_dir'])) and evidence not in all_paths;all_paths.add(evidence)
        assert hashlib.sha256(evidence.read_bytes()).hexdigest()==candidate['logits_npz_sha256']
        with np.load(evidence,allow_pickle=False) as stored:assert stored.files==['logits'];array=stored['logits']
        assert array.dtype==np.float32 and list(array.shape)==candidate['logits_shape']==[len(targets),mc.text_config.vocab_size] and candidate['logits_dtype']=='float32'
        assert np.isfinite(array).all()
        replay=torch.log_softmax(torch.from_numpy(array),dim=-1)[torch.arange(len(targets)),targets];actual=torch.tensor(candidate['token_log_probabilities'],dtype=torch.float32)
        difference=float((replay-actual).abs().max());max_difference=max(max_difference,difference);assert torch.allclose(replay,actual,atol=5e-5,rtol=1e-6),('Full-vocabulary logsoftmax differs',index,letter,difference)
        npu_mean=sum(candidate['token_log_probabilities'])/len(targets);assert npu_mean==candidate['mean_log_probability']
        npu_means.append(npu_mean);cpu_means.append(sum(replay.tolist())/len(targets))
        geometry,patch_count=image_position_ids(prepared);check_tensor(geometry,candidate['vision_position_ids'])
        assert int((prepared['input_ids']==mc.image_token_id).sum())==patch_count*seq_len and candidate['image_features_shape']==[patch_count,seq_len,mc.text_config.hidden_size]
        assert candidate['visual_embedding_injection_verified'] is True and candidate['vision_forward_calls']>=1
        assert bool(prepared['attention_mask'].all());check_tensor(torch.arange(full_len).reshape(1,-1),candidate['text_position_ids'])
        candidate_count+=1;logit_rows+=len(targets)
    assert e['option_mean_log_probabilities']==npu_means
    cpu_choice=max(range(len(options)),key=lambda k:cpu_means[k]);actual_choice=max(range(len(options)),key=lambda k:npu_means[k]);assert cpu_choice==actual_choice,('Numerically ambiguous rank needs review',index)
    answer=options[cpu_choice][0];assert answer==e['selected_letter']==e['raw_output']==b['raw_output']==b['prediction']==d['prediction']
    assert e['truncated'] is False and e['stop_reason']=='all_original_options_scored' and 'generation_token_ids' not in e
    correct+=answer.strip().lower()==str(b['reference']).strip().lower()
score=correct/count;assert abs(report['metrics']['Strict_MC_Accuracy']-score)<1e-12 and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
trace=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=trace if isinstance(trace,list) else trace['traceEvents'];hardware={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'}
kernels=[x for x in events if x.get('pid') in hardware and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert kernels
proof={'report_dir':str(folder),'passed':True,'sample_count':count,'correct':correct,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},'historical_record_reuse':False,'error_indices':[],'empty_indices':[],'invalid_indices':[],'truncated_indices':[],'index_duplicates':False,'CPU_image_preprocessing_replayed':count,'tokenizer_replayed':count,'original_weight_slice_checks':True,'NPU_profiler_kernel_events':len(kernels),'decision_protocol':'original_teacher_forced_option_mean_logprob','candidate_count':candidate_count,'full_vocabulary_logsoftmax_rows_independently_recomputed':logit_rows,'max_NPU_CPU_token_logprob_absolute_difference':max_difference,'all_CPU_NPU_rankings_agree':True,'autoregressive_generation_claimed':False,'protocol':'Original SmolDoclingDiscriminativeVLM scoring=option, exact BF16 checkpoint and original prefix/continuation token accounting. All candidate NPU full-vocabulary logits independently replayed with CPU logsoftmax, exact image/mask/position/input checks, independent option ranking and full denominator MC metric. No alternate generation or single-letter prior protocol.','report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
destination=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json';assert not destination.exists();destination.write_text(json.dumps(proof,indent=2),encoding='utf8');print(json.dumps(proof,indent=2))
