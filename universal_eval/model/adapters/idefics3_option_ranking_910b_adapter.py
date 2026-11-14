"""Existing exact Idefics3 BF16 checkpoints with their reviewed original generation protocol."""
import contextlib,datetime,hashlib,json,logging,os,pathlib,re,struct,time,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
import torch_npu
import yaml
from PIL import Image
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.idefics3_native_910b')

def fingerprint(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def tensor_record(value):return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':fingerprint(value)}

class Idefics3OptionRanking910BAdapter(IModel):
    def __init__(self,npu_devices=None,max_new_tokens=4096,processor_protocol='smoldocling_option_ranking',resource_sha256=None,
                 runtime_sha256=None,weight_headers=None,original_config=None,original_config_sha256=None,
                 original_adapter_sha256=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
        self.devices=list(npu_devices if npu_devices is not None else [3])
        if len(self.devices)!=1:raise ValueError('Exactly one reviewed NPU is required')
        self.device=torch.device('npu:'+str(self.devices[0]));self.max_new_tokens=int(max_new_tokens);self.protocol=processor_protocol
        self.resource_sha256=dict(resource_sha256 or {});self.runtime_sha256=dict(runtime_sha256 or {});self.weight_headers=dict(weight_headers or {})
        self.original_config=original_config;self.original_config_sha256=original_config_sha256;self.original_adapter_sha256=original_adapter_sha256
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=pathlib.Path(evidence_dir).resolve()/('Idefics3_option_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False)
        self.trace_path=self.evidence/'sample_execution.jsonl';self.calls=0;self.profile_done=False;self.details={};self.lineage={}

    @property
    def model_name(self):return getattr(self,'_model_name','unloaded')
    @property
    def model_type(self):return 'Idefics3ForConditionalGeneration'
    @property
    def inference_framework(self):return 'PyTorch + torch_npu / original teacher-forced option mean log probability, native Idefics3 BF16'
    def download(self,*args,**kwargs):raise RuntimeError('No model or dataset downloads are permitted')

    def load(self,model_path,**kwargs):
        from safetensors import safe_open
        from transformers import AutoConfig,AutoTokenizer,AutoImageProcessor,GenerationConfig,Idefics3ForConditionalGeneration,Idefics3Processor
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        directory=pathlib.Path(model_path).resolve()
        if not directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels'):raise ValueError('Unreviewed existing weight root')
        if not torch.npu.is_available():raise RuntimeError('NPU unavailable')
        torch.npu.set_device(self.device);self._model_name=directory.parent.name+'/'+directory.name
        required={'config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json'}
        if not required.issubset(self.resource_sha256) or not self.runtime_sha256 or not self.weight_headers:raise RuntimeError('Complete local resource, native source and checkpoint review are required')
        for rel,digest in self.resource_sha256.items():
            if pathlib.Path(rel).name!=rel or hashlib.sha256((directory/rel).read_bytes()).hexdigest()!=digest:raise RuntimeError('Reviewed own model resource changed: '+rel)
        for source,digest in self.runtime_sha256.items():
            if hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()!=digest:raise RuntimeError('Reviewed native implementation changed: '+source)
        original_path=pathlib.Path(self.original_config)
        if hashlib.sha256(original_path.read_bytes()).hexdigest()!=self.original_config_sha256:raise RuntimeError('Original evaluation YAML changed')
        original=yaml.safe_load(original_path.read_text());adapter=original['model']['adapter'];source=pathlib.Path(adapter.rsplit(':',1)[0]);old=original['model'].get('kwargs',{})
        if hashlib.sha256(source.read_bytes()).hexdigest()!=self.original_adapter_sha256:raise RuntimeError('Original adapter protocol changed')
        if original['model']['name']!=self.model_name or original['task_type']!='image-text-to-text' or 'image-text-to-text-mmbench/eval.jsonl' not in original['data']['dataset_path']:raise RuntimeError('Original model/task/data protocol differs')
        original_budget=int(old.get('max_new_tokens',old.get('max_tokens',2048)))
        if not adapter.endswith(':SmolDoclingDiscriminativeVLM') or old.get('scoring','option')!='option':raise RuntimeError('This implementation preserves only the original option-text mean-logprob protocol')
        if self.protocol!='smoldocling_option_ranking' or self.max_new_tokens!=original_budget:raise RuntimeError('Original protocol or declared fallback generation budget differs')
        cfg=AutoConfig.from_pretrained(str(directory),local_files_only=True)
        if cfg.architectures!=['Idefics3ForConditionalGeneration'] or getattr(cfg,'quantization_config',None) or cfg.text_config.model_type!='llama':raise ValueError('Unreviewed Idefics3 architecture')
        cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
        with torch.device('meta'):self.model=Idefics3ForConditionalGeneration(cfg).to(dtype=torch.bfloat16)
        parameters=dict(self.model.named_parameters());filled=set();records=[];index=directory/'model.safetensors.index.json';indexed=json.loads(index.read_text())['weight_map'] if index.is_file() else None
        shards=sorted(set(indexed.values())) if indexed else ['model.safetensors']
        if set(shards)!=set(self.weight_headers):raise RuntimeError('Source shard inventory changed')
        for shard in shards:
            if pathlib.Path(shard).name!=shard:raise ValueError('Unexpected source shard path')
            p=directory/shard;review=self.weight_headers[shard]
            with p.open('rb') as h:length=h.read(8);header=h.read(struct.unpack('<Q',length)[0])
            if p.stat().st_size!=review['size'] or hashlib.sha256(length+header).hexdigest()!=review['header_sha256']:raise RuntimeError('Reviewed source header changed')
            with safe_open(str(p),framework='pt',device='cpu') as h:
                if indexed is not None and set(h.keys())!={k for k,v in indexed.items() if v==shard}:raise RuntimeError('Source index/header mismatch')
                for key in h.keys():
                    if key not in parameters or key in filled:raise RuntimeError('Unexpected/duplicate source parameter: '+key)
                    value=h.get_tensor(key)
                    if value.dtype!=torch.bfloat16 or list(value.shape)!=list(parameters[key].shape):raise RuntimeError('Unreviewed source precision/shape: '+key)
                    target=value.to(self.device).contiguous();positions=sorted({0,value.numel()//2,value.numel()-1});expected_slice=value.reshape(-1)[positions];device_slice=target.reshape(-1)[torch.tensor(positions,device=self.device)].cpu()
                    if not torch.equal(expected_slice,device_slice):raise RuntimeError('NPU source transfer mismatch: '+key)
                    parent,leaf=key.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));filled.add(key)
                    records.append({'source_file':shard,'source_key':key,'target_key':key,'shape':list(value.shape),'source_dtype':str(value.dtype),'execution_dtype':str(target.dtype),'device':str(target.device),'slice_positions':positions,'source_slice':expected_slice.float().tolist(),'device_slice':device_slice.float().tolist()})
            log.info('Idefics3 NPU loaded %s: %s/%s tensors',shard,len(filled),len(parameters))
        if filled!=set(parameters):raise RuntimeError('Unfilled learned parameters')
        self.model.tie_weights();self.model.model.text_model.rotary_emb=LlamaRotaryEmbedding(cfg.text_config).to(self.device)
        if any(p.device!=self.device for p in self.model.parameters()):raise RuntimeError('A learned parameter is not on the selected NPU')
        for n,t in self.model.named_buffers():
            if t.is_meta or t.device!=self.device:raise RuntimeError('Unmaterialized/misplaced native buffer: '+n)
        tied=self.model.lm_head.weight is self.model.model.text_model.embed_tokens.weight
        if tied!=bool(cfg.tie_word_embeddings):raise RuntimeError('Own output head relation changed')
        self.model.eval();self.tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True)
        image_processor=AutoImageProcessor.from_pretrained(str(directory),local_files_only=True,backend='pil')
        if self.protocol=='high_ten_smol2_c500':image_processor.max_image_size={'longest_edge':384};image_processor.size={'longest_edge':1536}
        self.image_seq_len=(cfg.vision_config.image_size//cfg.vision_config.patch_size)**2//cfg.scale_factor**2
        metadata=directory/'processor_config.json'
        if metadata.is_file() and json.loads(metadata.read_text()).get('image_seq_len',self.image_seq_len)!=self.image_seq_len:raise RuntimeError('Own image sequence metadata differs from native connector')
        if image_processor.max_image_size['longest_edge']!=cfg.vision_config.image_size:raise RuntimeError('Effective original processor size differs from vision architecture')
        jinja=directory/'chat_template.jinja';template_json=directory/'chat_template.json'
        template=jinja.read_text() if jinja.is_file() else (json.loads(template_json.read_text()).get('chat_template') if template_json.is_file() else self.tokenizer.chat_template)
        if not template:raise RuntimeError('Own existing chat template required')
        self.processor=Idefics3Processor(image_processor=image_processor,tokenizer=self.tokenizer,image_seq_len=self.image_seq_len,chat_template=template)
        self.image_id=cfg.image_token_id
        if self.processor.image_token_id!=self.image_id:raise RuntimeError('Own image token disagrees with checkpoint')
        self.generation_config=GenerationConfig.from_pretrained(str(directory),local_files_only=True) if (directory/'generation_config.json').is_file() else GenerationConfig.from_model_config(cfg)
        self.generation_config.do_sample=False
        if self.protocol=='vllm_greedy':self.generation_config.repetition_penalty=1.0;self.generation_config.num_beams=1
        if self.protocol=='high_ten_smol2_c500':self.generation_config.pad_token_id=self.tokenizer.eos_token_id
        eos=self.generation_config.eos_token_id
        if eos is None:eos=cfg.text_config.eos_token_id
        self.eos_ids=[eos] if isinstance(eos,int) else list(eos)
        self.generation_config.eos_token_id=self.eos_ids
        if self.generation_config.pad_token_id is None:self.generation_config.pad_token_id=self.tokenizer.pad_token_id
        if self.generation_config.pad_token_id is None:raise RuntimeError('Own PAD token configuration is required')
        self._active_ids=None;self._features=None;self._bridge_verified=False;self._vision_calls=0;self._vision_positions=None;self._text_positions=None
        def observe_features(module,args,value):
            if value.device!=self.device or not bool(torch.isfinite(value).all()):raise RuntimeError('Invalid projected NPU image features')
            self._features=value;self._bridge_verified=False;self._vision_calls+=1
        def observe_text(module,args,values):
            embeds=values.get('inputs_embeds')
            if self._bridge_verified or self._features is None or embeds is None or embeds.shape[:2]!=self._active_ids.shape:return
            actual=embeds[self._active_ids==self.image_id];expected=self._features.reshape(-1,embeds.shape[-1]).to(embeds.dtype)
            if actual.shape!=expected.shape or not torch.equal(actual,expected) or not bool(expected.abs().sum()>0):raise RuntimeError('All actual image patches did not reach the language model')
            self._bridge_verified=True
        def observe_vision_positions(module,args):
            if args[0].device!=self.device:raise RuntimeError('Vision position lookup is not on NPU')
            self._vision_positions=tensor_record(args[0])
        def observe_rope(module,args,values):
            positions=values.get('position_ids',args[1] if len(args)>1 else None)
            if positions is not None and positions.shape[-1]==self._active_ids.shape[-1]:self._text_positions=tensor_record(positions)
        self.model.model.connector.register_forward_hook(observe_features)
        self.model.model.text_model.register_forward_pre_hook(observe_text,with_kwargs=True)
        self.model.model.vision_model.embeddings.position_embedding.register_forward_pre_hook(observe_vision_positions)
        self.model.model.text_model.rotary_emb.register_forward_pre_hook(observe_rope,with_kwargs=True)
        loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for rel in self.resource_sha256:
            if rel not in ('tokenizer.json','vocab.json','merges.txt'):(snapshot/rel).write_bytes((directory/rel).read_bytes())
        for src in self.runtime_sha256:(snapshot/(pathlib.Path(src).parent.name+'_'+pathlib.Path(src).name)).write_bytes(pathlib.Path(src).read_bytes())
        (snapshot/'original_evaluation.yaml').write_bytes(original_path.read_bytes());(snapshot/'original_adapter.py').write_bytes(source.read_bytes())
        self.lineage={'hardware':'Ascend 910B','devices':self.devices,'torch':torch.__version__,'torch_npu':torch_npu.__version__,
            'precision':'All exact existing learned BF16 tensors execute unchanged on NPU in BF16. Original analytic text rotary frequencies are constructed on CPU in FP32 and transferred exactly to NPU. No low-bit kernel acceleration, downloads or substituted model/data.',
            'protocol':self.protocol,'max_new_tokens':self.max_new_tokens,'eos_token_ids':self.eos_ids,'all_expected_parameters_loaded':True,'weight_tensor_count':len(records),'weight_loading_evidence':str(loading),'weight_headers':self.weight_headers,'checkpoint_key_conversion':'Identity: all source tensors retained one-to-one.',
            'tied_output_head_restored':tied,'image_seq_len_per_patch':self.image_seq_len,'processor_backend':'PIL; preserves own declared LANCZOS without torchvision BICUBIC fallback','effective_image_processor':image_processor.to_dict(),
            'processor_config_disclosure':'Own processor_config.json' if metadata.is_file() else 'Image sequence length derived solely from this exact checkpoint vision image_size, patch_size and scale_factor; no foreign resource.',
            'original_evaluation_protocol':{'config':str(original_path),'config_sha256':self.original_config_sha256,'adapter':adapter,'adapter_source_sha256':self.original_adapter_sha256,'original_max_tokens':original_budget,'effective_max_new_tokens':self.max_new_tokens,'kind':old.get('kind'),'protocol':old.get('protocol')},
            'effective_generation_config':self.generation_config.to_dict(),'text_rope_parameters':cfg.text_config.rope_parameters,'text_inv_freq':self.model.model.text_model.rotary_emb.inv_freq.detach().cpu().tolist(),
            'model_resource_root':str(directory),'model_resource_sha256':self.resource_sha256,'runtime_source_sha256':self.runtime_sha256,'architecture_source_snapshot':str(snapshot),'per_sample_evidence':str(self.trace_path),'profiler_trace':str(self.evidence/'npu_trace.json'),
            'module_placements':{n:str(next(m.parameters()).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}

        import types
        from universal_eval.model.adapters.idefics3_searchsorted_910b_adapter import npu_compatible_vision_embeddings
        embeddings=self.model.model.vision_model.embeddings
        embeddings.forward=types.MethodType(npu_compatible_vision_embeddings,embeddings)
        self.lineage['decision_protocol']='Original SmolDoclingDiscriminativeVLM scoring=option: all parsed option texts are teacher-forced after the unchanged <image> prompt and trigger; select the largest mean NPU token log probability. No autoregressive generation is claimed.'
        self.lineage['NPU_compatibility']='Only SearchSorted coordinates are losslessly represented as FP32 after original BF16 rounding; all learned operations remain NPU BF16. No shared source changes.'
        self.lineage['candidate_logits_dir']=str(self.evidence/'candidate_logits');(self.evidence/'candidate_logits').mkdir()

    def predict(self,prompt,media_path=None):
        import numpy as np
        start_time=time.monotonic();self.calls+=1
        entry={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None,'decision_protocol':'teacher_forced_option_mean_logprob','autoregressive_generation_performed':False,'candidates':[]}
        try:
            media=pathlib.Path(media_path)
            with Image.open(media) as image:image=image.convert('RGB')
            base=strip_eval_instruction(prompt or '');options=parse_options(base)
            if len(options)<2:raise RuntimeError('Fixed reviewed MC dataset requires at least two original option texts; no silent alternate protocol')
            prefix='<image>'+base.rstrip()+'\n\nThe correct answer is'
            base_inputs=self.processor(text=[prefix],images=[[image]],return_tensors='pt')
            entry.update(source_image_sha256=hashlib.sha256(media.read_bytes()).hexdigest(),stripped_prompt=base,prefix_text=prefix,base_length=base_inputs['input_ids'].shape[1],options=[[a,b] for a,b in options],base_inputs=record_inputs(base_inputs))
            means=[]
            for letter,text in options:
                continuation=' '+text
                prepared=self.processor(text=[prefix+continuation],images=[[image]],return_tensors='pt')
                n_cont=len(self.tokenizer(continuation,add_special_tokens=False)['input_ids']);full_len=prepared['input_ids'].shape[1]
                if n_cont<=0 or full_len<=entry['base_length']:raise RuntimeError('Original continuation scorer would have no valid likelihood tokens')
                begin=max(full_len-n_cont,1);positions=list(range(begin,full_len))
                inputs={k:v.to(device=self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in prepared.items()}
                self._active_ids=inputs['input_ids'];self._features=None;self._bridge_verified=False;self._vision_calls=0;self._vision_positions=None;self._text_positions=None
                with torch.inference_mode():
                    if not self.profile_done:
                        with contextlib.chdir(self.evidence):
                            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as profile:
                                result=self.model(**inputs,use_cache=False,return_dict=True)
                                torch.npu.synchronize(self.device)
                            profile.export_chrome_trace(str(self.evidence/'npu_trace.json'))
                        self.profile_done=True
                    else:result=self.model(**inputs,use_cache=False,return_dict=True)
                    logits=result.logits[0];logp=torch.log_softmax(logits.float(),dim=-1)
                    selected=logits[begin-1:full_len-1].float()
                    targets=inputs['input_ids'][0,begin:full_len]
                    actual=logp[torch.arange(begin-1,full_len-1,device=self.device),targets].cpu().tolist()
                    if not bool(torch.isfinite(selected).all()) or not bool(torch.isfinite(torch.tensor(actual)).all()):raise RuntimeError('Nonfinite actual candidate likelihood')
                    arrays=selected.cpu().contiguous().numpy()
                if not self._bridge_verified or self._vision_positions is None or self._text_positions is None:raise RuntimeError('Actual candidate image-to-language NPU evidence missing')
                path=self.evidence/'candidate_logits'/('sample_'+str(self.calls).zfill(4)+'_'+letter+'.npz')
                if path.exists():raise RuntimeError('Preserve existing candidate evidence')
                np.savez_compressed(path,logits=arrays)
                total=0.0
                for value in actual:total+=value
                mean=total/len(actual);means.append(mean)
                record={'letter':letter,'option_text':text,'continuation':continuation,'full_text':prefix+continuation,'inputs':record_inputs(prepared),
                    'target_positions':positions,'target_token_ids':targets.cpu().tolist(),'continuation_token_count':n_cont,'token_log_probabilities':actual,'mean_log_probability':mean,
                    'full_vocabulary_selected_logits':str(path),'logits_npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'logits_shape':list(arrays.shape),'logits_dtype':str(arrays.dtype),
                    'visual_embedding_injection_verified':True,'vision_forward_calls':self._vision_calls,'image_features_shape':list(self._features.shape),'image_features_sha256':fingerprint(self._features),
                    'vision_position_ids':self._vision_positions,'text_position_ids':self._text_positions,'error':None}
                entry['candidates'].append(record)
                del result,logits,logp,selected,arrays,inputs,prepared
                self._features=None
            best=max(range(len(options)),key=lambda k:means[k]);raw=options[best][0]
            entry.update(raw_output=raw,selected_letter=raw,option_mean_log_probabilities=means,truncated=False,stop_reason='all_original_options_scored',visual_embedding_injection_verified=True)
            log.info('Idefics3 option NPU sample %s: decision=%s means=%s',self.calls,raw,means)
            return raw
        except Exception:
            entry['error']=traceback.format_exc();log.error(entry['error']);raise
        finally:
            entry['elapsed_seconds']=time.monotonic()-start_time;clean=re.sub(r'\n+Answer the question by providing only the option letter.*$','',prompt or '',flags=re.S).strip();self.details[(clean,media_path)]=entry
            with self.trace_path.open('a',encoding='utf8') as h:h.write(json.dumps(entry,ensure_ascii=False)+'\n')
            self._features=None

    def batch_predict(self,batch):return [self.predict(p,m) for p,m in batch]
    def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
    def get_execution_lineage(self):return self.lineage
    def unload(self):self.model=None;torch.npu.empty_cache()

def strip_eval_instruction(prompt):
    for pattern in [r'\n\nAnswer the question by providing only the option letter.*$',r'\n\nProvide only the final answer\.\s*$']:
        prompt=re.sub(pattern,'',prompt,flags=re.S)
    return prompt

def parse_options(prompt):
    marks=list(re.finditer(r'(?m)^\s*([A-E])\s*[.)]\s+',prompt));result=[];expect='A'
    for i,m in enumerate(marks):
        text=prompt[m.end():(marks[i+1].start() if i+1<len(marks) else len(prompt))].strip()
        if m.group(1)==expect and text:result.append((m.group(1),text));expect=chr(ord(expect)+1)
    return result

def record_inputs(inputs):
    return {k:({'tensor':tensor_record(v)} if k in ('pixel_values','pixel_attention_mask') else {'values':v.cpu().tolist(),'dtype':str(v.dtype)}) for k,v in inputs.items()}
