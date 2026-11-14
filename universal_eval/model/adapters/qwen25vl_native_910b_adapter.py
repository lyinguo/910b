"""Exact existing Qwen2.5-VL BF16 checkpoints on an explicitly selected Ascend NPU."""
import contextlib,datetime,hashlib,json,logging,os,pathlib,re,struct,time,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
import torch_npu
from PIL import Image
from universal_eval.model.interface import IModel

logger=logging.getLogger(__name__)

def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def checkpoint_name(key):
    if key.startswith('visual.'):return 'model.'+key
    if key.startswith('model.') and not key.startswith(('model.visual.','model.language_model.')):return 'model.language_model.'+key[6:]
    return key

class Qwen25VLNative910BAdapter(IModel):
    def __init__(self,npu_devices=None,max_new_tokens=2048,generation_mode='greedy',resource_sha256=None,
                 runtime_sha256=None,weight_headers=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
        self.devices=list(npu_devices if npu_devices is not None else [1])
        if len(self.devices)!=1:raise ValueError('This adapter requires exactly one explicitly reviewed NPU')
        if generation_mode not in ('greedy','constrained_choice'):raise ValueError('Unreviewed generation protocol')
        self.device=torch.device('npu:'+str(self.devices[0]));self.max_new_tokens=int(max_new_tokens);self.generation_mode=generation_mode
        self.resource_sha256=dict(resource_sha256 or {});self.runtime_sha256=dict(runtime_sha256 or {});self.weight_headers=dict(weight_headers or {})
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=pathlib.Path(evidence_dir).resolve()/('Qwen25VL_native_910b_'+stamp+'_'+str(os.getpid()))
        self.evidence.mkdir(parents=True,exist_ok=False);self.trace_path=self.evidence/'sample_execution.jsonl'
        self.details={};self.lineage={};self.profile_done=False;self.calls=0

    @property
    def model_name(self):return getattr(self,'_model_name','unloaded')
    @property
    def model_type(self):return 'Qwen2_5_VLForConditionalGeneration'
    @property
    def inference_framework(self):return 'PyTorch + torch_npu / native Transformers Qwen2.5-VL BF16'
    def download(self,*args,**kwargs):raise RuntimeError('Model and dataset downloads are forbidden')

    def load(self,model_path,**kwargs):
        from safetensors import safe_open
        from transformers import AutoConfig,AutoTokenizer,GenerationConfig,Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLProcessor,Qwen2VLImageProcessor,Qwen2VLVideoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding,Qwen2_5_VisionRotaryEmbedding
        directory=pathlib.Path(model_path).resolve()
        if not directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels'):raise ValueError('Unreviewed checkpoint root')
        if not torch.npu.is_available():raise RuntimeError('Ascend NPU is unavailable')
        torch.npu.set_device(self.device)
        self._model_name=directory.parent.name+'/'+directory.name
        required={'config.json','tokenizer_config.json','preprocessor_config.json'}
        if not required.issubset(self.resource_sha256):raise RuntimeError('Own reviewed configuration/tokenizer/processor are required')
        if not ('tokenizer.json' in self.resource_sha256 or {'vocab.json','merges.txt'}.issubset(self.resource_sha256)):
            raise RuntimeError('Own tokenizer binary or complete own vocab/merges are required')
        for rel,digest in self.resource_sha256.items():
            if pathlib.Path(rel).name!=rel:raise ValueError('Unexpected resource path')
            if hashlib.sha256((directory/rel).read_bytes()).hexdigest()!=digest:raise RuntimeError('Reviewed local resource changed: '+rel)
        if not self.runtime_sha256 or not self.weight_headers:raise RuntimeError('Reviewed native implementation and complete checkpoint headers are required')
        for source,digest in self.runtime_sha256.items():
            if hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()!=digest:raise RuntimeError('Native implementation changed: '+source)
        for shard,review in self.weight_headers.items():
            if pathlib.Path(shard).name!=shard:raise ValueError('Unexpected shard path')
            p=directory/shard
            with p.open('rb') as h:
                length=h.read(8);header=h.read(struct.unpack('<Q',length)[0])
            if p.stat().st_size!=review['size'] or hashlib.sha256(length+header).hexdigest()!=review['header_sha256']:
                raise RuntimeError('Reviewed checkpoint header/size changed: '+shard)
        config=AutoConfig.from_pretrained(str(directory),local_files_only=True)
        if config.architectures!=['Qwen2_5_VLForConditionalGeneration'] or getattr(config,'quantization_config',None):
            raise ValueError('This adapter requires a reviewed unquantized Qwen2.5-VL checkpoint')
        config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
        with torch.device('meta'):self.model=Qwen2_5_VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
        parameters=dict(self.model.named_parameters());filled=set();records=[]
        index=directory/'model.safetensors.index.json';indexed=json.loads(index.read_text())['weight_map'] if index.is_file() else None
        shards=sorted(set(indexed.values())) if indexed else ['model.safetensors']
        if set(shards)!=set(self.weight_headers):raise RuntimeError('Checkpoint inventory differs from reviewed headers')
        for shard in shards:
            with safe_open(str(directory/shard),framework='pt',device='cpu') as handle:
                if indexed is not None and set(handle.keys())!={k for k,v in indexed.items() if v==shard}:raise RuntimeError('Checkpoint index/header mismatch')
                for key in handle.keys():
                    name=checkpoint_name(key)
                    if name not in parameters or name in filled:raise RuntimeError('Unexpected or duplicate source parameter: '+key)
                    value=handle.get_tensor(key)
                    if value.dtype!=torch.bfloat16 or list(value.shape)!=list(parameters[name].shape):raise RuntimeError('Unreviewed parameter dtype/shape: '+key)
                    target=value.to(self.device).contiguous();positions=sorted({0,value.numel()//2,value.numel()-1})
                    expected=value.reshape(-1)[positions];actual=target.reshape(-1)[torch.tensor(positions,device=self.device)].cpu()
                    if not torch.equal(expected,actual):raise RuntimeError('CPU-to-NPU weight transfer differs: '+key)
                    parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False))
                    records.append({'source_file':shard,'source_key':key,'target_key':name,'shape':list(value.shape),
                                    'source_dtype':str(value.dtype),'execution_dtype':str(target.dtype),'device':str(target.device),
                                    'slice_positions':positions,'source_slice':expected.float().tolist(),'device_slice':actual.float().tolist()})
                    filled.add(name)
            logger.info('Qwen2.5-VL NPU loaded %s: %s/%s parameters',shard,len(filled),len(parameters))
        if filled!=set(parameters):raise RuntimeError('Missing model parameters: '+repr(sorted(set(parameters)-filled)))
        self.model.tie_weights()
        vision_dim=config.vision_config.hidden_size//config.vision_config.num_heads//2
        self.model.model.visual.rotary_pos_emb=Qwen2_5_VisionRotaryEmbedding(vision_dim).to(self.device)
        self.model.model.language_model.rotary_emb=Qwen2_5_VLRotaryEmbedding(config.text_config,device=self.device)
        for name,tensor in self.model.named_buffers():
            if tensor.is_meta or tensor.device!=self.device:raise RuntimeError('Unmaterialized or misplaced buffer: '+name)
        if any(p.device!=self.device for p in self.model.parameters()):raise RuntimeError('Model parameters must all execute on the selected NPU')
        tied=self.model.lm_head.weight is self.model.model.language_model.embed_tokens.weight
        if tied!=bool(config.tie_word_embeddings):raise RuntimeError('Checkpoint output head relation changed')
        self.model.eval();self.tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True)
        jinja=directory/'chat_template.jinja';template_json=directory/'chat_template.json'
        if jinja.is_file():template=jinja.read_text();template_source=str(jinja)
        elif template_json.is_file():template=json.loads(template_json.read_text()).get('chat_template');template_source=str(template_json)
        else:template=self.tokenizer.chat_template;template_source=str(directory/'tokenizer_config.json')+'::chat_template'
        if not template or not isinstance(template,str):raise RuntimeError('This checkpoint own chat template is required')
        self.processor=Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor(**json.loads((directory/'preprocessor_config.json').read_text())),
            tokenizer=self.tokenizer,video_processor=Qwen2VLVideoProcessor(),chat_template=template)
        self.generation_config=GenerationConfig.from_pretrained(str(directory),local_files_only=True) if (directory/'generation_config.json').is_file() else GenerationConfig.from_model_config(config)
        eos=self.generation_config.eos_token_id
        if eos is None:eos=getattr(config.text_config,'eos_token_id',None)
        if eos is None:raise RuntimeError('Own local EOS configuration is required')
        self.eos_ids=[eos] if isinstance(eos,int) else list(eos)
        self.pad_id=self.generation_config.pad_token_id
        if self.pad_id is None:self.pad_id=self.tokenizer.pad_token_id
        if self.pad_id is None:raise RuntimeError('Own PAD configuration is required')
        self.image_id=config.image_token_id;self.merge_size=config.vision_config.spatial_merge_size
        self._active_ids=None;self._features=None;self._bridge_verified=False;self._vision_calls=0;self._position_ids=None
        def observe_visual(module,args,value):
            features=value.pooler_output
            if not torch.is_tensor(features) or features.device!=self.device or not bool(torch.isfinite(features).all()):raise RuntimeError('Invalid NPU visual features')
            self._features=features;self._bridge_verified=False;self._vision_calls+=1
        def observe_language(module,args,values):
            if self._bridge_verified or self._features is None or values.get('inputs_embeds') is None:return
            embeds=values['inputs_embeds']
            if embeds.shape[:2]!=self._active_ids.shape:return
            actual=embeds[self._active_ids==self.image_id];expected=self._features.reshape(-1,embeds.shape[-1]).to(embeds.dtype)
            if actual.shape!=expected.shape or not torch.equal(actual,expected) or not bool(expected.abs().sum()>0):raise RuntimeError('Actual sorted visual features did not reach the language model')
            self._position_ids=values['position_ids'].detach().cpu().tolist();self._bridge_verified=True
        self.model.model.visual.register_forward_hook(observe_visual)
        self.model.model.language_model.register_forward_pre_hook(observe_language,with_kwargs=True)
        loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2))
        snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for rel in self.resource_sha256:
            if rel not in ('tokenizer.json','vocab.json','merges.txt'):(snapshot/rel).write_bytes((directory/rel).read_bytes())
        for source in self.runtime_sha256:(snapshot/(pathlib.Path(source).parent.name+'_'+pathlib.Path(source).name)).write_bytes(pathlib.Path(source).read_bytes())
        rope={'text_inv_freq':self.model.model.language_model.rotary_emb.inv_freq.detach().cpu().tolist(),
              'vision_inv_freq':self.model.model.visual.rotary_pos_emb.inv_freq.detach().cpu().tolist(),
              'text_parameters':config.text_config.rope_parameters,'frequencies_dtype':'torch.float32'}
        self.lineage={'hardware':'Ascend 910B','devices':self.devices,'torch':torch.__version__,'torch_npu':torch_npu.__version__,
            'precision':'All exact existing checkpoint tensors are BF16 and execute on the selected NPU in BF16; analytically derived vision and text rotary frequencies retain FP32. No low-bit kernel acceleration, dequantization, substitute checkpoint or downloads.',
            'protocol':'Own Qwen2.5-VL tokenizer, chat template and dynamic-resolution image preprocessing, original greedy generation budget; constrained one-letter NPU-logit choice is enabled only for original VLLMMCChoiceModel configurations.',
            'generation_mode':self.generation_mode,'max_new_tokens':self.max_new_tokens,'eos_token_ids':self.eos_ids,'pad_token_id':self.pad_id,
            'weight_tensor_count':len(records),'all_expected_parameters_loaded':True,'weight_loading_evidence':str(loading),'weight_headers':self.weight_headers,
            'checkpoint_key_conversion':'Legacy visual.* -> model.visual.*; legacy model.* -> model.language_model.*; lm_head.* unchanged; every source tensor mapped one-to-one.',
            'tied_output_head_restored':tied,'spatial_merge_size':self.merge_size,'rotary_frequencies':rope,
            'processor_metadata_source':str(directory/'preprocessor_config.json'),'chat_template_source':template_source,
            'video_processor_disclosure':'Installed Qwen2VLVideoProcessor is an unused constructor requirement. Only image inputs are evaluated; no video inputs or video evaluation are claimed.',
            'model_resource_root':str(directory),'model_resource_sha256':self.resource_sha256,'runtime_source_sha256':self.runtime_sha256,
            'architecture_source_snapshot':str(snapshot),'per_sample_evidence':str(self.trace_path),'profiler_trace':str(self.evidence/'npu_trace.json'),
            'module_placements':{n:str(next(m.parameters()).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}

    def predict(self,prompt,media_path=None):
        started=time.monotonic();self.calls+=1;self._features=None;self._bridge_verified=False;self._vision_calls=0;self._position_ids=None
        entry={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None,'generation_mode':self.generation_mode}
        try:
            media=pathlib.Path(media_path)
            with Image.open(media) as image:image=image.convert('RGB')
            chat=self.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}],tokenize=False,add_generation_prompt=True)
            prepared=self.processor(text=chat,images=image,return_tensors='pt',add_special_tokens=False,return_mm_token_type_ids=True)
            entry['chat_text']=chat;entry['source_image_sha256']=hashlib.sha256(media.read_bytes()).hexdigest()
            for name,value in prepared.items():
                if not torch.is_tensor(value):raise RuntimeError('Unexpected nontensor input: '+name)
                if name=='pixel_values':entry.update(pixel_shape=list(value.shape),pixel_sha256=fingerprint(value))
                else:entry[name]=value.cpu().tolist()
            expected_features=int(prepared['image_grid_thw'].prod(-1).sum())//self.merge_size**2
            if prepared['image_grid_thw'].shape[0]!=1 or expected_features<=0 or int((prepared['input_ids']==self.image_id).sum())!=expected_features:
                raise RuntimeError('Exactly one complete dynamic-resolution image must reach the model')
            inputs={k:v.to(device=self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in prepared.items()}
            self._active_ids=inputs['input_ids'];length=self._active_ids.shape[1];prefill=None
            with torch.inference_mode():
                if not self.profile_done:
                    with contextlib.chdir(self.evidence):
                        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as profile:
                            prefill=self.model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True)
                            if not bool(torch.isfinite(prefill.logits).all()):raise RuntimeError('Nonfinite NPU prefill logits')
                            torch.npu.synchronize(self.device)
                        profile.export_chrome_trace(str(self.evidence/'npu_trace.json'))
                    self.profile_done=True
                if self.generation_mode=='constrained_choice':
                    labels=sorted(set(re.findall(r'(?:^|\n)\s*([A-E])[.):]\s',prompt))) or list('ABCDE')
                    candidate_ids=[self.tokenizer.encode(x,add_special_tokens=False) for x in labels]
                    if any(len(x)!=1 for x in candidate_ids):raise RuntimeError('Original single-token choice grammar requires single-token labels')
                    if prefill is None:prefill=self.model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True)
                    scores=prefill.logits[0,-1,[x[0] for x in candidate_ids]].float()
                    if not bool(torch.isfinite(scores).all()):raise RuntimeError('Nonfinite constrained candidate logits')
                    selected=int(scores.argmax().cpu());tokens=[candidate_ids[selected][0]]
                    raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip();truncated=False;stop='choice_grammar_complete'
                    entry.update(choice_labels=labels,choice_token_ids=[x[0] for x in candidate_ids],choice_logits=scores.cpu().tolist())
                else:
                    generated=self.model.generate(**inputs,generation_config=self.generation_config,max_new_tokens=self.max_new_tokens,do_sample=False,
                        eos_token_id=self.eos_ids,pad_token_id=self.pad_id,use_cache=True,return_dict_in_generate=True,output_scores=False)
                    tokens=generated.sequences[0,length:].cpu().tolist();raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                    truncated=len(tokens)>=self.max_new_tokens and (not tokens or tokens[-1] not in self.eos_ids)
                    stop='length' if truncated else 'eos' if tokens and tokens[-1] in self.eos_ids else 'other'
            if not self._bridge_verified or not self._vision_calls or self._features.shape[0]!=expected_features:raise RuntimeError('Actual image-to-language execution was not verified')
            entry.update(raw_output=raw,generation_token_ids=tokens,generation_count=len(tokens),truncated=truncated,stop_reason=stop,
                visual_embedding_injection_verified=True,vision_forward_calls=self._vision_calls,image_features_shape=list(self._features.shape),
                image_features_sha256=fingerprint(self._features),prefill_position_ids=self._position_ids)
            logger.info('Qwen2.5-VL NPU sample %s: tokens=%s truncated=%s raw=%r',self.calls,len(tokens),truncated,raw[:160])
            return raw
        except Exception:
            entry['error']=traceback.format_exc();logger.error(entry['error']);raise
        finally:
            entry['elapsed_seconds']=time.monotonic()-started
            clean=re.sub(r'\n+Answer the question by providing only the option letter.*$','',prompt or '',flags=re.S).strip()
            self.details[(clean,media_path)]=entry
            with self.trace_path.open('a',encoding='utf8') as h:h.write(json.dumps(entry,ensure_ascii=False)+'\n')
            self._features=None

    def batch_predict(self,batch):return [self.predict(prompt,media) for prompt,media in batch]
    def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
    def get_execution_lineage(self):return self.lineage
    def unload(self):self.model=None;torch.npu.empty_cache()
