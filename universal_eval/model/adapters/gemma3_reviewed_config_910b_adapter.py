"""Native Transformers Gemma3 BF16 checkpoint inference on one Ascend NPU."""
import contextlib,datetime,hashlib,inspect,json,logging,math,os,pathlib,re,time,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
import torch_npu
from PIL import Image
from universal_eval.model.interface import IModel

logger=logging.getLogger('universal_eval.model.adapters.gemma3_native_910b')

def fingerprint(value):
    value=value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()

def checkpoint_name(name):
    for source,target in [
        ('language_model.model.','model.language_model.'),
        ('language_model.lm_head.','lm_head.'),
        ('vision_tower.vision_model.','model.vision_tower.'),
        ('multi_modal_projector.','model.multi_modal_projector.')]:
        if name.startswith(source):return target+name[len(source):]
    if name.startswith(('model.language_model.','model.vision_tower.','model.multi_modal_projector.','lm_head.')):return name
    raise RuntimeError('Unreviewed checkpoint key: '+name)

def build_reviewed_config(directory, adjustments):
    from transformers import AutoConfig
    raw=json.loads((pathlib.Path(directory)/'config.json').read_text())
    config=AutoConfig.from_pretrained(str(directory),local_files_only=True)
    if set(adjustments)-{'normalize_flat_rope_parameters','preserve_explicit_output_head'}:raise ValueError('Unreviewed configuration adjustment')
    disclosure={'original_config_unchanged':True,'adjustments':dict(adjustments)}
    if adjustments.get('normalize_flat_rope_parameters'):
        text=raw['text_config'];flat=text['rope_parameters']
        if flat.get('rope_type')!='linear' or not isinstance(flat.get('factor'),(int,float)) or flat['factor']<=0:raise ValueError('Expected explicit original linear RoPE')
        if 'full_attention' in flat or 'sliding_attention' in flat:raise ValueError('Expected legacy flat RoPE only')
        config.text_config.rope_parameters={
            'full_attention':{'rope_type':'linear','factor':flat['factor'],'rope_theta':text['rope_theta']},
            'sliding_attention':{'rope_type':'default','rope_theta':text['rope_local_base_freq']}}
        disclosure['normalized_rope_parameters']=config.text_config.rope_parameters
        disclosure['rope_reason']='Preserves original explicit linear factor and global/local bases; the installed config parser otherwise drops the flat-format scaling'
    if adjustments.get('preserve_explicit_output_head'):
        config.tie_word_embeddings=False;config.text_config.tie_word_embeddings=False
        disclosure['output_head_reason']='Loads the separately serialized output-head tensor as its own parameter instead of silently discarding it under an implicit default tie'
    return config,disclosure

class Gemma3ReviewedConfig910BAdapter(IModel):
    def __init__(self,npu_devices=None,max_new_tokens=2048,evidence_dir='workspace/diagnostics/model_runs',
                 processor_metadata=None,processor_metadata_sha256=None,resource_sha256=None,
                 runtime_sha256=None,processor_is_own=True,reviewed_config=None,prompt_mode="native_chat",**kwargs):
        self.devices=list(npu_devices or [0])
        if len(self.devices)!=1:raise ValueError('This implementation is reviewed for one NPU')
        self.device=torch.device('npu:'+str(self.devices[0]));self.max_new_tokens=int(max_new_tokens)
        self.processor_metadata=processor_metadata;self.processor_metadata_sha256=processor_metadata_sha256
        self.resource_sha256=dict(resource_sha256 or {});self.runtime_sha256=dict(runtime_sha256 or {});self.processor_is_own=bool(processor_is_own)
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=pathlib.Path(evidence_dir).resolve()/('Gemma3_native_910b_'+stamp+'_'+str(os.getpid()))
        self.evidence.mkdir(parents=True,exist_ok=False);self.trace_path=self.evidence/'sample_execution.jsonl'
        self.details={};self.lineage={};self.profile_done=False;self.calls=0
        self.reviewed_config=dict(reviewed_config or {});self.prompt_mode=prompt_mode
        if self.prompt_mode not in ("native_chat","legacy_string_images"):raise ValueError("Unreviewed prompt mode")

    @property
    def model_name(self):return getattr(self,'_model_name','unloaded')
    @property
    def model_type(self):return 'Gemma3ForConditionalGeneration'
    @property
    def inference_framework(self):return 'PyTorch + torch_npu / Transformers native Gemma3 BF16 greedy generation'
    def download(self,*args,**kwargs):raise RuntimeError('Downloads are forbidden; use existing exact local resources')

    def load(self,model_path,**kwargs):
        from safetensors import safe_open
        from transformers import AutoConfig,AutoTokenizer,Gemma3ForConditionalGeneration,Gemma3ImageProcessor,Gemma3Processor
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
        directory=pathlib.Path(model_path).resolve()
        if not torch.npu.is_available():raise RuntimeError('Ascend NPU is unavailable')
        self._model_name=directory.parent.name+'/'+directory.name
        if not {'config.json','tokenizer_config.json','tokenizer.json'}.issubset(self.resource_sha256):
            raise RuntimeError('Current model configuration, tokenizer and template review hashes are required')
        for rel,digest in self.resource_sha256.items():
            if hashlib.sha256((directory/rel).read_bytes()).hexdigest()!=digest:raise RuntimeError('Reviewed model resource changed: '+rel)
        if not self.runtime_sha256:raise RuntimeError('Native implementation source review is required')
        for source,digest in self.runtime_sha256.items():
            if hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()!=digest:raise RuntimeError('Native implementation changed: '+source)
        metadata=pathlib.Path(self.processor_metadata)
        if hashlib.sha256(metadata.read_bytes()).hexdigest()!=self.processor_metadata_sha256:raise RuntimeError('Reviewed image metadata changed')
        config,config_disclosure=build_reviewed_config(directory,self.reviewed_config)
        if config.architectures!=['Gemma3ForConditionalGeneration'] or getattr(config,'quantization_config',None):
            raise ValueError('This adapter requires a reviewed unquantized Gemma3 conditional checkpoint')
        config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
        with torch.device('meta'):self.model=Gemma3ForConditionalGeneration(config).to(dtype=torch.bfloat16)
        parameters=dict(self.model.named_parameters());filled=set();records=[]
        index=directory/'model.safetensors.index.json'
        if index.is_file():indexed=json.loads(index.read_text())['weight_map'];shards=sorted(set(indexed.values()))
        else:indexed=None;shards=['model.safetensors']
        for shard in shards:
            if pathlib.Path(shard).name!=shard:raise ValueError('Unexpected checkpoint shard path')
            with safe_open(str(directory/shard),framework='pt',device='cpu') as handle:
                if indexed is not None and set(handle.keys())!={key for key,value in indexed.items() if value==shard}:raise RuntimeError('Checkpoint index/header mismatch')
                for key in handle.keys():
                    name=checkpoint_name(key)
                    if name not in parameters or name in filled:raise RuntimeError('Unexpected/duplicate source parameter: '+key+' -> '+name)
                    value=handle.get_tensor(key)
                    if value.dtype!=torch.bfloat16 or list(value.shape)!=list(parameters[name].shape):raise RuntimeError('Unreviewed source dtype or shape: '+key)
                    target=value.to(self.device).contiguous();positions=sorted({0,value.numel()//2,value.numel()-1})
                    expected=value.reshape(-1)[positions];actual=target.reshape(-1)[torch.tensor(positions,device=self.device)].cpu()
                    if not torch.equal(expected,actual):raise RuntimeError('NPU weight mismatch: '+key)
                    parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False))
                    records.append({'source_file':shard,'source_key':key,'target_key':name,'shape':list(value.shape),
                                    'source_dtype':str(value.dtype),'execution_dtype':str(target.dtype),'device':str(target.device),
                                    'slice_positions':positions,'source_slice':expected.float().tolist(),'device_slice':actual.float().tolist()})
                    filled.add(name)
            logger.info('Gemma3 NPU loaded %s: %s/%s parameters',shard,len(filled),len(parameters))
        if filled!=set(parameters):raise RuntimeError('Missing checkpoint parameters: '+repr(sorted(set(parameters)-filled)))
        self.model.tie_weights()
        self.model.model.language_model.rotary_emb=Gemma3RotaryEmbedding(config.text_config).to(self.device)
        self.model.model.language_model.embed_tokens.embed_scale=torch.tensor(math.sqrt(config.text_config.hidden_size),dtype=torch.bfloat16,device=self.device)
        self.model.model.vision_tower.embeddings.position_ids=torch.arange(config.vision_config.image_size**2//config.vision_config.patch_size**2,device=self.device).expand((1,-1))
        for name,value in list(self.model.named_buffers()):
            if value.is_meta:raise RuntimeError('Unmaterialized nonparameter buffer: '+name)
        if any(p.device.type!='npu' for p in self.model.parameters()):raise RuntimeError('A model parameter is not on NPU')
        if config.tie_word_embeddings and self.model.lm_head.weight is not self.model.model.language_model.embed_tokens.weight:
            raise RuntimeError('Original tied embedding/output head relation was not restored')
        if not config.tie_word_embeddings and self.model.lm_head.weight is self.model.model.language_model.embed_tokens.weight:
            raise RuntimeError('Original separate output head was incorrectly tied')
        self.model.eval()
        self.tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True)
        image_kwargs=json.loads(metadata.read_text())
        if image_kwargs.get('size')!={'height':config.vision_config.image_size,'width':config.vision_config.image_size}:
            raise RuntimeError('Processor resolution differs from this checkpoint')
        self.image_processor=Gemma3ImageProcessor(**image_kwargs)
        template_path=directory/'chat_template.jinja'
        if template_path.is_file():
            if 'chat_template.jinja' not in self.resource_sha256:raise RuntimeError('Existing chat template must be reviewed')
            template=template_path.read_text()
        else:template=getattr(self.tokenizer,'chat_template',None)
        if not template:raise RuntimeError('Own local tokenizer or Jinja chat template is required; no substitute template')
        self.processor=Gemma3Processor(image_processor=self.image_processor,tokenizer=self.tokenizer,
                                      image_seq_length=config.mm_tokens_per_image,chat_template=template)
        if self.processor_is_own:
            if metadata.resolve()!=directory/'preprocessor_config.json':raise RuntimeError('Own processor metadata must be used')
        elif (directory/'preprocessor_config.json').is_file():raise RuntimeError('Existing own processor metadata needs explicit review')
        self.image_id=config.image_token_id;eos=getattr(config,'eos_token_id',None) or [1,106];self.eos_ids=[eos] if isinstance(eos,int) else list(eos)
        self.pad_id=self.tokenizer.pad_token_id
        if self.pad_id is None:raise RuntimeError('A local PAD token is required')
        self._active_ids=None;self._features=None;self._bridge_verified=False;self._vision_calls=0
        def observe_projection(module,args,value):
            if value.device.type!='npu' or not bool(torch.isfinite(value).all()):raise RuntimeError('Invalid NPU visual features')
            self._features=value;self._bridge_verified=False;self._vision_calls+=1
        def observe_language(module,args,values):
            if self._bridge_verified or self._features is None or values.get('inputs_embeds') is None:return
            embeds=values['inputs_embeds']
            if embeds.shape[:2]!=self._active_ids.shape:return
            actual=embeds[self._active_ids==self.image_id];expected=self._features.reshape(-1,embeds.shape[-1]).to(embeds.dtype)
            if actual.shape!=expected.shape or not torch.equal(actual,expected) or not bool(expected.abs().sum()>0):raise RuntimeError('Image-to-language embedding injection differs from projected features')
            self._bridge_verified=True
        self.model.model.multi_modal_projector.register_forward_hook(observe_projection)
        self.model.model.language_model.register_forward_pre_hook(observe_language,with_kwargs=True)
        loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2))
        snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for rel in self.resource_sha256:
            if rel!='tokenizer.json':(snapshot/rel).write_bytes((directory/rel).read_bytes())
        for source,digest in self.runtime_sha256.items():(snapshot/(pathlib.Path(source).parent.name+'_'+pathlib.Path(source).name)).write_bytes(pathlib.Path(source).read_bytes())
        (snapshot/'preprocessor_config.json').write_bytes(metadata.read_bytes())
        self.lineage={'reviewed_config_disclosure':config_disclosure,'prompt_mode':self.prompt_mode,'hardware':'Ascend 910B','devices':self.devices,'torch':torch.__version__,'torch_npu':torch_npu.__version__,
            'precision':'Actual existing BF16 tensors are used for NPU BF16 reference execution, with FP32 analytic full/sliding rotary frequencies and the derived BF16 embedding scale. Quantization-related names do not imply native low-bit execution or speedup. This adapter downloads nothing, substitutes no weights and uses no low-bit kernels',
            'protocol':'Native Gemma3 greedy generation, own tokenizer/chat template, explicit image token types, original 2048 generated-token limit. Strict explicit MC extraction with full sample denominator',
            'weight_tensor_count':len(records),'all_expected_parameters_loaded':True,'weight_loading_evidence':str(loading),
            'checkpoint_key_conversion':'Legacy language_model.model -> model.language_model; legacy vision_tower.vision_model -> model.vision_tower; projector names preserved under model; all source parameters mapped one-to-one',
            'tied_output_head_restored':bool(config.tie_word_embeddings),'image_token_count_per_image':config.mm_tokens_per_image,
            'processor_metadata_source':str(metadata),'processor_metadata_sha256':self.processor_metadata_sha256,
            'processor_metadata_disclosure':('Uses this exact checkpoint own image metadata, tokenizer and chat template; no substitute model weights' if self.processor_is_own else 'Current checkpoint omits image metadata. Reuses existing local stock Gemma3 896px processor metadata matching this vision architecture only; all weights, tokenizer and chat template are from this exact checkpoint. No downloads'),
            'chat_template_source':str(directory/'chat_template.jinja') if (directory/'chat_template.jinja').is_file() else str(directory/'tokenizer_config.json')+'::chat_template',
            'model_resource_sha256':self.resource_sha256,'model_resource_root':str(directory),'runtime_source_sha256':self.runtime_sha256,
            'architecture_source_snapshot':str(snapshot),'per_sample_evidence':str(self.trace_path),'profiler_trace':str(self.evidence/'npu_trace.json'),
            'module_placements':{n:str(next(m.parameters()).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}

    def predict(self,prompt,media_path=None):
        started=time.monotonic();self.calls+=1;self._features=None;self._bridge_verified=False;self._vision_calls=0
        entry={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None}
        try:
            media=pathlib.Path(media_path)
            with Image.open(media) as image:image=image.convert('RGB')
            if self.prompt_mode=='legacy_string_images':messages=[{'role':'user','content':prompt,'images':[str(media)]}]
            else:messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
            chat=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=self.processor(text=chat,images=image,return_tensors='pt',add_special_tokens=False,return_mm_token_type_ids=True)
            entry['chat_text']=chat;entry['source_image_sha256']=hashlib.sha256(media.read_bytes()).hexdigest()
            for name,value in inputs.items():
                if not torch.is_tensor(value):raise RuntimeError('Unexpected non-tensor model input: '+name)
                if name=='pixel_values':entry.update(pixel_shape=list(value.shape),pixel_sha256=fingerprint(value))
                else:entry[name]=value.cpu().tolist()
            inputs={k:v.to(device=self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in inputs.items()}
            self._active_ids=inputs['input_ids'];length=self._active_ids.shape[1]
            if int((self._active_ids==self.image_id).sum())!=self.model.config.mm_tokens_per_image:raise RuntimeError('Exactly one full image must reach the model')
            with torch.inference_mode():
                if not self.profile_done:
                    with contextlib.chdir(self.evidence):
                        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as profile:
                            result=self.model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True)
                            if not bool(torch.isfinite(result.logits).all()):raise RuntimeError('Nonfinite prefill logits')
                            torch.npu.synchronize()
                        profile.export_chrome_trace(str(self.evidence/'npu_trace.json'))
                    self.profile_done=True
                result=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False,
                                           eos_token_id=self.eos_ids,pad_token_id=self.pad_id,use_cache=True,
                                           return_dict_in_generate=True,output_scores=False)
            tokens=result.sequences[0,length:].cpu().tolist();raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            truncated=len(tokens)>=self.max_new_tokens and (not tokens or tokens[-1] not in self.eos_ids)
            if not self._bridge_verified or not self._vision_calls:raise RuntimeError('Actual visual generation bridge was not observed')
            entry.update(raw_output=raw,generation_token_ids=tokens,generation_count=len(tokens),truncated=truncated,
                         stop_reason='length' if truncated else 'eos' if tokens and tokens[-1] in self.eos_ids else 'other',
                         visual_embedding_injection_verified=self._bridge_verified,vision_forward_calls=self._vision_calls,
                         image_features_shape=list(self._features.shape),image_features_sha256=fingerprint(self._features))
            logger.info('Gemma3 NPU sample %s: tokens=%s truncated=%s raw=%r',self.calls,len(tokens),truncated,raw[:160])
            return raw
        except Exception:
            entry['error']=traceback.format_exc();logger.error(entry['error']);raise
        finally:
            entry['elapsed_seconds']=time.monotonic()-started
            clean=re.sub(r'\n+Answer the question by providing only the option letter.*$','',prompt or '',flags=re.S).strip()
            self.details[(clean,media_path)]=entry
            with self.trace_path.open('a',encoding='utf8') as handle:handle.write(json.dumps(entry,ensure_ascii=False)+'\n')
            self._features=None

    def batch_predict(self,batch):return [self.predict(prompt,media) for prompt,media in batch]
    def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
    def get_execution_lineage(self):return self.lineage
    def unload(self):self.model=None;torch.npu.empty_cache()
