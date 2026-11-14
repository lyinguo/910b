"""Independent original-input, token, M-RoPE, kernel and full-denominator score audit."""
import os,hashlib,json,pathlib,re,sys,itertools,struct
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from PIL import Image
from transformers import AutoConfig,AutoTokenizer,GenerationConfig,Qwen2_5_VLProcessor,Qwen2VLImageProcessor,Qwen2VLVideoProcessor
from safetensors import safe_open

folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2])
required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config']
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()]
outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results']
assert len(inputs)==len(outputs)==len(details)==count
for records in (inputs,outputs,details):assert [x['index'] for x in records]==list(range(count))
weight_dir=pathlib.Path(cfg['model']['model_path']).resolve()
assert weight_dir.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels')
assert cfg['model']['name']==weight_dir.parent.name+'/'+weight_dir.name and cfg['model']['download'] is False
model_config=AutoConfig.from_pretrained(str(weight_dir),local_files_only=True)
assert model_config.architectures==['Qwen2_5_VLForConditionalGeneration'] and not getattr(model_config,'quantization_config',None)
assert cfg['task_type']=='image-text-to-text'
assert folder.parent.parent.name==cfg['model']['name'].replace('/','_') and folder.parent.name==cfg['task_type']
assert re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['sample_range']=='0:'+str(count-1)
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution']
assert lineage['all_expected_parameters_loaded'] and lineage['tied_output_head_restored']==bool(model_config.tie_word_embeddings)
assert lineage['devices']==cfg['model']['kwargs']['npu_devices'] and len(lineage['devices'])==1
device='npu:'+str(lineage['devices'][0]);assert set(lineage['module_placements'].values())=={device}
assert lineage['generation_mode']==cfg['model']['kwargs']['generation_mode']
assert lineage['max_new_tokens']==cfg['model']['kwargs']['max_new_tokens']
assert lineage['spatial_merge_size']==model_config.vision_config.spatial_merge_size
loading=json.loads(pathlib.Path(lineage['weight_loading_evidence']).read_text())
index=weight_dir/'model.safetensors.index.json'
if index.is_file():index_map=json.loads(index.read_text())['weight_map']
else:
    with safe_open(str(weight_dir/'model.safetensors'),framework='pt',device='cpu') as h:index_map={k:'model.safetensors' for k in h.keys()}
assert len(loading)==len(index_map)==lineage['weight_tensor_count']
assert {x['source_key'] for x in loading}==set(index_map) and len({x['target_key'] for x in loading})==len(loading)
for shard,review in lineage['weight_headers'].items():
    p=weight_dir/shard
    with p.open('rb') as h:
        length=h.read(8);header=h.read(struct.unpack('<Q',length)[0])
    assert p.stat().st_size==review['size'] and hashlib.sha256(length+header).hexdigest()==review['header_sha256']
for shard in sorted(set(index_map.values())):
    with safe_open(str(weight_dir/shard),framework='pt',device='cpu') as h:
        assert set(h.keys())=={k for k,v in index_map.items() if v==shard}
        for row in (x for x in loading if x['source_file']==shard):
            key=row['source_key'];assert index_map[key]==shard
            value=h.get_tensor(key);assert str(value.dtype)==row['source_dtype']==row['execution_dtype']=='torch.bfloat16'
            assert row['shape']==list(value.shape) and row['device']==device
            positions=sorted({0,value.numel()//2,value.numel()-1})
            assert positions==row['slice_positions'] and value.reshape(-1)[positions].float().tolist()==row['source_slice']==row['device_slice']
            mapped=('model.'+key if key.startswith('visual.') else 'model.language_model.'+key[6:] if key.startswith('model.') and not key.startswith(('model.visual.','model.language_model.')) else key)
            assert mapped==row['target_key']
for source,digest in lineage['runtime_source_sha256'].items():assert hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()==digest
for rel,digest in lineage['model_resource_sha256'].items():assert hashlib.sha256((weight_dir/rel).read_bytes()).hexdigest()==digest
assert lineage['model_resource_sha256']==cfg['model']['kwargs']['resource_sha256']
rope=lineage['rotary_frequencies'];dim=model_config.text_config.hidden_size//model_config.text_config.num_attention_heads
assert model_config.text_config.rope_parameters['rope_type']=='default'
text_freq=1/(float(model_config.text_config.rope_parameters['rope_theta'])**(torch.arange(0,dim,2).float()/dim))
vision_dim=model_config.vision_config.hidden_size//model_config.vision_config.num_heads//2
vision_freq=1/(10000.0**(torch.arange(0,vision_dim,2).float()/vision_dim))
assert torch.equal(torch.tensor(rope['text_inv_freq']),text_freq) and torch.equal(torch.tensor(rope['vision_inv_freq']),vision_freq)
assert rope['frequencies_dtype']=='torch.float32' and rope['text_parameters']==model_config.text_config.rope_parameters
tok=AutoTokenizer.from_pretrained(str(weight_dir),local_files_only=True)
template=(weight_dir/'chat_template.jinja').read_text() if (weight_dir/'chat_template.jinja').is_file() else (json.loads((weight_dir/'chat_template.json').read_text()).get('chat_template') if (weight_dir/'chat_template.json').is_file() else tok.chat_template)
assert template
processor=Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor(**json.loads((weight_dir/'preprocessor_config.json').read_text())),tokenizer=tok,video_processor=Qwen2VLVideoProcessor(),chat_template=template)
gen=GenerationConfig.from_pretrained(str(weight_dir),local_files_only=True) if (weight_dir/'generation_config.json').is_file() else GenerationConfig.from_model_config(model_config)
eos=gen.eos_token_id if gen.eos_token_id is not None else model_config.text_config.eos_token_id
eos=[eos] if isinstance(eos,int) else list(eos);assert lineage['eos_token_ids']==eos

import yaml
original=lineage['original_evaluation_protocol'];original_path=pathlib.Path(original['config'])
assert hashlib.sha256(original_path.read_bytes()).hexdigest()==original['config_sha256']==cfg['model']['kwargs']['original_config_sha256']
original_cfg=yaml.safe_load(original_path.read_text());assert original_cfg['model']['name']==cfg['model']['name'] and original_cfg['task_type']==cfg['task_type']
original_adapter=original_cfg['model']['adapter'];assert original_adapter==original['adapter']
assert hashlib.sha256(pathlib.Path(original_adapter.rsplit(':',1)[0]).read_bytes()).hexdigest()==original['adapter_source_sha256']==cfg['model']['kwargs']['original_adapter_sha256']
old_kwargs=original_cfg['model'].get('kwargs',{});original_budget=int(old_kwargs.get('max_new_tokens',old_kwargs.get('max_tokens',2048)))
assert original_budget==original['original_max_tokens']==lineage['max_new_tokens']
original_mode='constrained_choice' if original_adapter.endswith(':VLLMMCChoiceModel') else 'greedy'
assert original_mode==lineage['generation_mode']
effective=lineage['effective_generation_config']
assert cfg['model']['name']=='nvidia/Cosmos-Reason1-7B' and original_adapter.endswith(':TransformersVLM')
# Preserve the checkpoint's unset (None) beam metadata. Native generate resolves missing values against its actual model/global defaults.
gen.do_sample=False;assert effective==gen.to_dict(), 'Original generation metadata was changed'
import inspect
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.generation.utils import GenerationMixin
resolver=pathlib.Path(inspect.getfile(GenerationMixin));resolver_sha=hashlib.sha256(resolver.read_bytes()).hexdigest()
assert resolver_sha=='ec9e5bce8b654d5ef8169ddf595be18a06a95c87568fcd909d37d88255ffb8f7'
with torch.device('meta'):resolution_model=Qwen2_5_VLForConditionalGeneration(model_config)
resolved,_=resolution_model._prepare_generation_config(gen,max_new_tokens=original_budget,do_sample=False,eos_token_id=lineage['eos_token_ids'],pad_token_id=(gen.pad_token_id if gen.pad_token_id is not None else tok.pad_token_id),use_cache=True,return_dict_in_generate=True,output_scores=False)
assert resolved.do_sample is False and resolved.num_beams==1 and str(resolved.get_generation_mode())=='GenerationMode.GREEDY_SEARCH'
assert resolved.repetition_penalty==gen.repetition_penalty==1.05
generation_resolution={'preserved_num_beams_metadata':effective['num_beams'],'resolved_num_beams':resolved.num_beams,'resolved_do_sample':resolved.do_sample,'resolved_repetition_penalty':resolved.repetition_penalty,'resolved_mode':str(resolved.get_generation_mode()),'native_resolver_source':str(resolver),'native_resolver_source_sha256':resolver_sha,'note':'Own unset beam metadata is resolved by the same native GenerationMixin defaults as actual inference. No model/config/output/score change.'}
expected_repetition=1.0 if original_adapter.endswith((':VLLMModel',':VLLMMCChoiceModel')) else float(gen.repetition_penalty)
assert effective['repetition_penalty']==original['effective_repetition_penalty']==expected_repetition
assert original['effective_token_limit']==(1 if original_mode=='constrained_choice' else original_budget)
dataset=pathlib.Path(cfg['data']['dataset_path']);assert dataset==pathlib.Path('/root/projects/comperror/workspace/datasets/image-text-to-text-mmbench/eval.jsonl')
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

instruction='\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) in the format <answer>X</answer>.'
correct=0;invalid=[];empty=[];truncated=[];errors=[];predictions=[]
for i,(a,b,d,s) in enumerate(zip(inputs,outputs,details,source)):
    assert a['prompt']==clean(s['prompt']) and a['reference']==b['reference']==d['reference']==s['reference']
    execution=b['extra']['execution'];assert b['error'] is None and execution['error'] is None
    assert execution['visual_embedding_injection_verified'] is True and execution['vision_forward_calls']>=1
    assert execution['generation_mode']==lineage['generation_mode']
    media=pathlib.Path(a['media_path']);assert media.is_relative_to(dataset.parent)
    original_media=pathlib.Path(s['media_path']);original_media=original_media if original_media.is_absolute() else dataset.parent/original_media
    assert media.resolve()==original_media.resolve()
    assert hashlib.sha256(media.read_bytes()).hexdigest()==execution['source_image_sha256']
    prompt=a['prompt']+instruction;assert execution['prompt']==prompt
    chat=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}],tokenize=False,add_generation_prompt=True)
    assert chat==execution['chat_text']
    with Image.open(media) as image:image=image.convert('RGB')
    prepared=processor(text=chat,images=image,return_tensors='pt',add_special_tokens=False,return_mm_token_type_ids=True)
    pixels=prepared['pixel_values'].contiguous()
    assert hashlib.sha256(pixels.view(torch.uint8).numpy().tobytes()).hexdigest()==execution['pixel_sha256'] and list(pixels.shape)==execution['pixel_shape']
    for name,value in prepared.items():
        if name!='pixel_values':assert value.tolist()==execution[name],('CPU input differs',i,name)
    grid=prepared['image_grid_thw'];merge=model_config.vision_config.spatial_merge_size
    feature_count=int(grid.prod(-1).sum())//merge**2
    assert grid.shape==torch.Size([1,3]) and int((prepared['input_ids']==model_config.image_token_id).sum())==feature_count
    assert execution['image_features_shape']==[feature_count,model_config.text_config.hidden_size]
    # Independent M-RoPE indexing: text uses consecutive positions; the image grid uses its three axes.
    assert bool(prepared['attention_mask'].all())
    position=0;parts=[];grid_iter=iter(grid.tolist())
    for kind,group in itertools.groupby(prepared['mm_token_type_ids'][0].tolist()):
        size=len(list(group));assert kind in (0,1)
        if kind==0:parts.append(torch.arange(position,position+size).expand(3,-1));position+=size
        else:
            t,h,w=next(grid_iter);h//=merge;w//=merge
            axes=torch.meshgrid(torch.arange(t),torch.arange(h),torch.arange(w),indexing='ij')
            indices=torch.stack(axes).reshape(3,-1)+position
            assert indices.shape[-1]==size;parts.append(indices);position+=max(h,w)
    expected_positions=torch.cat(parts,dim=-1)[:,None,:]
    actual_positions=torch.tensor(execution['prefill_position_ids'])
    if actual_positions.shape[0]==4:
        # Native generation prepends a text-position row for packed causal-mask construction.
        assert torch.equal(actual_positions[0],torch.arange(prepared['input_ids'].shape[-1]).reshape(1,-1))
        actual_positions=actual_positions[1:]
    assert torch.equal(expected_positions,actual_positions)
    tokens=execution['generation_token_ids'];assert len(tokens)==execution['generation_count']
    raw=tok.decode(tokens,skip_special_tokens=True).strip();assert raw==b['raw_output']==execution['raw_output']
    if lineage['generation_mode']=='constrained_choice':
        labels=sorted(set(re.findall(r'(?:^|\n)\s*([A-E])[.):]\s',prompt))) or list('ABCDE')
        assert execution['choice_labels']==labels
        ids=[tok.encode(x,add_special_tokens=False) for x in labels];assert all(len(x)==1 for x in ids)
        assert execution['choice_token_ids']==[x[0] for x in ids]
        scores=execution['choice_logits'];assert len(scores)==len(labels) and all(torch.isfinite(torch.tensor(scores)))
        best=max(range(len(scores)),key=lambda k:scores[k]);assert raw==labels[best] and tokens==ids[best]
        is_truncated=False;assert execution['stop_reason']=='choice_grammar_complete'
    else:
        is_truncated=len(tokens)>=cfg['model']['kwargs']['max_new_tokens'] and (not tokens or tokens[-1] not in eos)
        assert len(tokens)<=cfg['model']['kwargs']['max_new_tokens']
    prediction=extract(raw);assert prediction==b['prediction']==d['prediction'],('Independent answer extraction differs',i,repr(raw),repr(prediction),repr(b['prediction']))
    if not raw:empty.append(i)
    if prediction not in re.findall(r'(?m)^\s*([A-E])\s*[.)]\s+',a['prompt']):invalid.append(i)
    assert is_truncated==execution['truncated']
    if is_truncated:truncated.append(i)
    correct+=str(prediction).strip().lower()==str(b['reference']).strip().lower();predictions.append(prediction)
score=correct/count
assert abs(report['metrics']['Strict_MC_Accuracy']-score)<1e-12
assert report['statistics']['total_samples']==count and report['statistics']['error_count']==0
trace=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=trace if isinstance(trace,list) else trace['traceEvents']
hardware={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'}
kernels=[e for e in events if e.get('pid') in hardware and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')]
assert kernels
proof={'report_dir':str(folder),'passed':True,'sample_count':count,'correct':correct,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},
    'extraction_protocol':'native_explicit_mc_v2','generation_mode':lineage['generation_mode'],'historical_record_reuse':False,
    'error_indices':errors,'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,
    'CPU_image_preprocessing_replayed':count,'tokenizer_replayed':count,'decoded_generation_replayed':count,'independent_MRoPE_indices_replayed':count,
    'NPU_profiler_kernel_events':len(kernels),'weight_tensor_count':len(loading),'original_weight_slice_checks':True,
    'protocol':'Exact existing BF16 Qwen2.5-VL, own tokenizer/template/processor and original generation protocol; full CPU image/grid/token/M-RoPE/decode replay and strict MC score over all samples. Image-to-language injection observed on actual NPU execution. Invalid/empty/truncated outputs retained.',
    'report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
proof['generation_config_resolution']=generation_resolution
destination=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json'
assert not destination.exists(),'Preserve an existing independent verification'
destination.write_text(json.dumps(proof,indent=2),encoding='utf8');print(json.dumps(proof,indent=2))
