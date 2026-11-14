"""Native NPU PaliGemma caption inference with the reviewed original caption/ROUGE protocol."""
import contextlib,datetime,hashlib,json,logging,os,pathlib,re,struct,time,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
import torch_npu
import yaml
from PIL import Image
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.paligemma_table_reference_caption_910b')

def fingerprint(value):return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tensor_record(value):return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':fingerprint(value)}
def mapped_key(key):
    for old,new in [('vision_tower.vision_model.','model.vision_tower.'),('vision_tower.','model.vision_tower.'),('multi_modal_projector.','model.multi_modal_projector.'),('language_model.model.','model.language_model.'),('language_model.lm_head.','lm_head.')]:
        if key.startswith(old):return new+key[len(old):]
    return key

class PaliGemmaTableReferenceCaption910BAdapter(IModel):
    def __init__(self,npu_devices=None,max_new_tokens=2048,vqa_prefix='',source_dtype=None,resource_sha256=None,runtime_sha256=None,weight_headers=None,
                 original_config=None,original_config_sha256=None,original_adapter_sha256=None,dataset_path=None,dataset_sha256=None,reference_report=None,reference_report_sha256=None,reference_source_sha256=None,reference_protocol=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
        self.devices=list(npu_devices if npu_devices is not None else [4])
        if len(self.devices)!=1:raise ValueError('Exactly one reviewed NPU is required')
        self.device=torch.device('npu:'+str(self.devices[0]));self.max_new_tokens=int(max_new_tokens);self.prefix=vqa_prefix;self.source_dtype=source_dtype
        if source_dtype not in ('BF16','F32'):raise ValueError('Existing checkpoint precision must be explicitly reviewed')
        self.dtype={'BF16':torch.bfloat16,'F32':torch.float32}[source_dtype]
        self.resource_sha256=dict(resource_sha256 or {});self.runtime_sha256=dict(runtime_sha256 or {});self.weight_headers=dict(weight_headers or {})
        self.original_config=original_config;self.original_config_sha256=original_config_sha256;self.original_adapter_sha256=original_adapter_sha256
        self.dataset=pathlib.Path(dataset_path);self.dataset_sha256=dataset_sha256
        self.reference_report=pathlib.Path(reference_report);self.reference_report_sha256=reference_report_sha256;self.reference_source_sha256=dict(reference_source_sha256 or {});self.reference_protocol=dict(reference_protocol or {})
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=pathlib.Path(evidence_dir).resolve()/('PaliGemma_table_reference_caption_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False)
        self.trace_path=self.evidence/'sample_execution.jsonl';self.calls=0;self.profile_done=False;self.details={};self.lineage={}
    @property
    def model_name(self):return getattr(self,'_model_name','unloaded')
    @property
    def model_type(self):return 'PaliGemmaForConditionalGeneration'
    @property
    def inference_framework(self):return 'PyTorch + torch_npu / native Transformers PaliGemma '+str(self.source_dtype)
    def download(self,*args,**kwargs):raise RuntimeError('Model and dataset downloads are forbidden')

    def load(self,model_path,**kwargs):
        from safetensors import safe_open
        from transformers import AutoConfig,AutoTokenizer,AutoImageProcessor,GenerationConfig,PaliGemmaForConditionalGeneration,PaliGemmaProcessor
        from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding
        directory=pathlib.Path(model_path).resolve()
        if not directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels'):raise ValueError('Unreviewed weight root')
        if not torch.npu.is_available():raise RuntimeError('NPU unavailable')
        torch.npu.set_device(self.device);self._model_name=directory.parent.name+'/'+directory.name
        if not {'config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json'}.issubset(self.resource_sha256) or not self.weight_headers or not self.runtime_sha256:raise RuntimeError('Complete own resource and source review required')
        for rel,sha in self.resource_sha256.items():
            if pathlib.Path(rel).name!=rel or hashlib.sha256((directory/rel).read_bytes()).hexdigest()!=sha:raise RuntimeError('Own checkpoint resource changed: '+rel)
        for rel,sha in self.runtime_sha256.items():
            if hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()!=sha:raise RuntimeError('Native source changed: '+rel)
        if (directory/'adapter_config.json').is_file() or (directory/'adapter_model.safetensors').is_file():raise RuntimeError('LoRA resources need a separate exact merge audit')
        original_path=pathlib.Path(self.original_config)
        if hashlib.sha256(original_path.read_bytes()).hexdigest()!=self.original_config_sha256:raise RuntimeError('Original root YAML changed')
        root_config=yaml.safe_load(original_path.read_text());root_adapter=root_config['model']['adapter'];root_adapter_path=pathlib.Path(root_adapter.rsplit(':',1)[0])
        if root_config['model']['name']!=self.model_name or root_config['task_type']!='image-to-text' or not root_adapter.endswith(':VLLMModel') or hashlib.sha256(root_adapter_path.read_bytes()).hexdigest()!=self.original_adapter_sha256:raise RuntimeError('Root model identity or default configuration changed')
        if not self.reference_report.is_relative_to('/mnt/afs/comp_error/workspace/reports') or hashlib.sha256(self.reference_report.read_bytes()).hexdigest()!=self.reference_report_sha256:raise RuntimeError('Existing matched reference report changed')
        historical_report=json.loads(self.reference_report.read_text());original=historical_report['pipeline_config'];old=original['model'].get('kwargs',{});adapter=original['model']['adapter'];source=pathlib.Path('/mnt/afs/comp_error')/adapter.rsplit(':',1)[0]
        if original['model']['name']!=self.model_name or original['task_type']!='image-to-text' or not adapter.endswith((':PaliGemmaMLUAdapter',':PaliGemma2MLUAdapter')) or len(historical_report['detailed_results'])!=200:raise RuntimeError('Unreviewed historical caption reference')
        if not self.reference_source_sha256:raise RuntimeError('Original reference source review required')
        for path,sha in self.reference_source_sha256.items():
            if not pathlib.Path(path).is_relative_to('/mnt/afs/comp_error/universal_eval') or hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()!=sha:raise RuntimeError('Reference recipe source changed: '+path)
        protocol={'vqa_prefix':old.get('vqa_prefix','') or '', 'mc_instruction':old.get('mc_instruction','\n\nRespond with only the letter of the correct option.') or '', 'strip_eval_instruction':bool(old.get('strip_eval_instruction',False)), 'prompt_override':old.get('prompt_override','') or '', 'extract_answer':original['algorithm'].get('kwargs',{}).get('extract_answer',True)}
        if protocol!=self.reference_protocol or self.prefix!=protocol['vqa_prefix'] or old.get('max_new_tokens',2048)!=self.max_new_tokens:raise RuntimeError('Matched original caption prompt, extraction or generation budget differs')
        if 'image-to-text-short/eval.jsonl' not in original['data']['dataset_path'] or original['algorithm']['name']!='greedy_inference':raise RuntimeError('Wrong historical task/data/algorithm')
        if not self.dataset.is_relative_to('/root/projects/comperror/workspace/datasets/protocol_snapshots') or hashlib.sha256(self.dataset.read_bytes()).hexdigest()!=self.dataset_sha256 or self.dataset_sha256!='709cf45623d2feb5cb2e5d6306a9c024e47762f8debc718a055884d3def0adbf':raise RuntimeError('Historical caption dataset snapshot changed')
        hf32={}
        for name in ('conv','matmul'):
            setting=getattr(torch.npu,name);setting.allow_hf32=False;hf32[name]=bool(setting.allow_hf32)
        if any(hf32.values()):raise RuntimeError('FP32 precision control was not applied')
        cfg=AutoConfig.from_pretrained(str(directory),local_files_only=True)
        if cfg.architectures!=['PaliGemmaForConditionalGeneration'] or getattr(cfg,'quantization_config',None) or cfg.text_config.model_type not in ('gemma','gemma2'):raise ValueError('Unreviewed PaliGemma architecture')
        if cfg.vision_config.image_size>448:raise RuntimeError('896-prefix sliding-mask and memory review pending; no blanket high-resolution approval')
        cfg._attn_implementation='eager';cfg.text_config._attn_implementation='eager';cfg.vision_config._attn_implementation='eager'
        with torch.device('meta'):self.model=PaliGemmaForConditionalGeneration(cfg)
        parameters=dict(self.model.named_parameters());buffer_types={n:t.dtype for n,t in self.model.named_buffers()};filled=set();records=[]
        index=directory/'model.safetensors.index.json';indexed=json.loads(index.read_text())['weight_map'] if index.is_file() else None;shards=sorted(set(indexed.values())) if indexed else ['model.safetensors']
        if set(shards)!=set(self.weight_headers):raise RuntimeError('Source shard inventory changed')
        for shard in shards:
            if pathlib.Path(shard).name!=shard:raise ValueError('Unexpected source shard')
            path=directory/shard;review=self.weight_headers[shard]
            with path.open('rb') as h:length=h.read(8);header=h.read(struct.unpack('<Q',length)[0])
            if path.stat().st_size!=review['size'] or hashlib.sha256(length+header).hexdigest()!=review['header_sha256']:raise RuntimeError('Reviewed checkpoint header changed')
            with safe_open(str(path),framework='pt',device='cpu') as h:
                if indexed is not None and set(h.keys())!={k for k,v in indexed.items() if v==shard}:raise RuntimeError('Source index/header mismatch')
                for key in h.keys():
                    target_key=mapped_key(key)
                    if target_key not in parameters or target_key in filled:raise RuntimeError('Unexpected/duplicate source tensor: '+key)
                    value=h.get_tensor(key)
                    if value.dtype!=self.dtype or list(value.shape)!=list(parameters[target_key].shape):raise RuntimeError('Source precision/shape changed: '+key)
                    target=value.to(self.device).contiguous();locations=sorted({0,value.numel()//2,value.numel()-1});reference=value.reshape(-1)[locations];actual=target.reshape(-1)[torch.tensor(locations,device=self.device)].cpu()
                    if not torch.equal(reference,actual):raise RuntimeError('NPU transfer mismatch: '+key)
                    parent,leaf=target_key.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));filled.add(target_key)
                    records.append({'source_file':shard,'source_key':key,'target_key':target_key,'shape':list(value.shape),'source_dtype':str(value.dtype),'execution_dtype':str(target.dtype),'device':str(target.device),'slice_positions':locations,'source_slice':reference.float().tolist(),'device_slice':actual.float().tolist()})
            log.info('PaliGemma NPU loaded %s: %s/%s source tensors',shard,len(filled),len(parameters))
        if filled!=set(parameters):raise RuntimeError('Unfilled learned parameters')
        self.model.tie_weights();tied=self.model.lm_head.weight is self.model.model.language_model.embed_tokens.weight
        if not tied or not cfg.tie_word_embeddings:raise RuntimeError('Unreviewed output-head relation')
        self.seq_len=(cfg.vision_config.image_size//cfg.vision_config.patch_size)**2
        self.model.model.vision_tower.embeddings.position_ids=torch.arange(self.seq_len,dtype=torch.int64).reshape(1,-1).to(self.device)
        embedding=self.model.model.language_model.embed_tokens;embedding.embed_scale=torch.tensor(embedding.scalar_embed_scale,dtype=buffer_types['model.language_model.embed_tokens.embed_scale']).to(self.device)
        rope_class={'gemma':GemmaRotaryEmbedding,'gemma2':Gemma2RotaryEmbedding}[cfg.text_config.model_type];self.model.model.language_model.rotary_emb=rope_class(cfg.text_config).to(self.device)
        if any(p.device!=self.device or p.dtype!=self.dtype for p in self.model.parameters()):raise RuntimeError('Not every learned parameter is on NPU at original precision')
        for name,value in self.model.named_buffers():
            if value.is_meta or value.device!=self.device or value.dtype!=buffer_types[name]:raise RuntimeError('Incorrect native analytic buffer: '+name)
        self.model.eval();self.tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True)
        ip=AutoImageProcessor.from_pretrained(str(directory),local_files_only=True,backend='pil');ip.image_seq_length=self.seq_len;self.processor=PaliGemmaProcessor(image_processor=ip,tokenizer=self.tokenizer);self.image_id=cfg.image_token_id
        if self.processor.image_token_id!=self.image_id:raise RuntimeError('Own image token mismatch')
        self.generation_config=GenerationConfig.from_pretrained(str(directory),local_files_only=True) if (directory/'generation_config.json').is_file() else GenerationConfig.from_model_config(cfg)
        self.generation_config.do_sample=False;eos=self.generation_config.eos_token_id
        if eos is None:eos=cfg.text_config.eos_token_id
        self.eos_ids=[eos] if isinstance(eos,int) else list(eos);self.generation_config.eos_token_id=self.eos_ids
        if self.generation_config.pad_token_id is None:self.generation_config.pad_token_id=self.tokenizer.pad_token_id
        self._active_ids=None;self._features=None;self._bridge=False;self._vision_calls=0;self._positions=None;self._masks=None
        def observe_features(module,args,value):
            if value.device!=self.device or value.dtype!=self.dtype or not bool(torch.isfinite(value).all()):raise RuntimeError('Invalid actual NPU image features')
            self._features=value;self._bridge=False;self._vision_calls+=1
        def observe_language(module,args,values):
            embeds=values.get('inputs_embeds')
            if self._bridge or self._features is None or embeds is None or embeds.shape[:2]!=self._active_ids.shape:return
            actual=embeds[self._active_ids==self.image_id];expected=self._features.reshape(-1,embeds.shape[-1])
            if actual.shape!=expected.shape or not torch.equal(actual,expected) or not bool(expected.abs().sum()>0):raise RuntimeError('Actual image patches did not reach the language model')
            self._bridge=True;self._positions=tensor_record(values['position_ids']);masks=values['attention_mask'];masks=masks if isinstance(masks,dict) else {'full_attention':masks};self._masks={}
            for name,mask in masks.items():
                if mask is None:raise RuntimeError('Explicit eager prefix attention mask required')
                if mask.shape[-2:]!=self._active_ids.shape[-1:]*2 or not bool((mask==0).all()):raise RuntimeError('Caption prefix must remain bidirectional for all image/prompt tokens')
                self._masks[name]=tensor_record(mask)
        self.model.model.multi_modal_projector.register_forward_hook(observe_features);self.model.model.language_model.register_forward_pre_hook(observe_language,with_kwargs=True)
        loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for rel in self.resource_sha256:
            if rel not in ('tokenizer.json','tokenizer.model'):(snapshot/rel).write_bytes((directory/rel).read_bytes())
        for src in self.runtime_sha256:(snapshot/(pathlib.Path(src).parent.name+'_'+pathlib.Path(src).name)).write_bytes(pathlib.Path(src).read_bytes())
        (snapshot/'original_root_evaluation.yaml').write_bytes(original_path.read_bytes());(snapshot/'original_reference_adapter.py').write_bytes(source.read_bytes());(snapshot/'historical_reference_report.json').write_bytes(self.reference_report.read_bytes())
        for path in self.reference_source_sha256:(snapshot/('reference_'+pathlib.Path(path).name)).write_bytes(pathlib.Path(path).read_bytes())
        self.lineage={'hardware':'Ascend 910B','devices':self.devices,'torch':torch.__version__,'torch_npu':torch_npu.__version__,
            'precision':'Every exact existing learned tensor executes on NPU at original '+self.source_dtype+' precision; no low-bit acceleration or CPU learned-layer offload. Analytic RoPE frequencies retain native CPU-constructed FP32 constants. HF32 convolution and matmul disabled within this process. No downloads or substituted model/data.',
            'source_dtype':self.source_dtype,'HF32_enabled':hf32,'protocol':'table_matched_paligemma_caption_reference','max_new_tokens':self.max_new_tokens,'vqa_prefix':self.prefix,'eos_token_ids':self.eos_ids,'all_expected_parameters_loaded':True,'weight_tensor_count':len(records),'weight_loading_evidence':str(loading),'weight_headers':self.weight_headers,'checkpoint_key_conversion':'Legacy vision_tower.vision_model -> model.vision_tower; projector -> model.multi_modal_projector; language_model.model -> model.language_model; source aliases and every learned tensor retained.',
            'tied_output_head_restored':tied,'image_seq_length':self.seq_len,'processor_backend':'PIL with own declared resize/normalization','effective_image_processor':ip.to_dict(),
            'original_evaluation_protocol':{'reference_report':str(self.reference_report),'reference_report_sha256':self.reference_report_sha256,'reference_metric':historical_report['metrics']['ROUGE-L'],'reference_protocol':self.reference_protocol,'reference_source_sha256':self.reference_source_sha256,'adapter':adapter,'original_task':'image-to-text','workbook_task':'image-text-to-text','primary_metric':'ROUGE-L','dataset_snapshot':str(self.dataset),'dataset_sha256':self.dataset_sha256,'dataset_provenance':'Exact mounted AFS1500 caption benchmark matching all original200 references and the table MLU score; original target1459 data is untouched. No MMBench substitution.','max_new_tokens':self.max_new_tokens,'root_default_config':str(original_path),'root_default_config_sha256':self.original_config_sha256,'root_default_adapter':root_adapter,'root_default_adapter_sha256':self.original_adapter_sha256,'protocol_selection':'The root default VLLM chat YAML is retained as provenance. The original table-matched full200 caption report and its PaliGemmaMLUAdapter/PaliGemma2MLUAdapter source supply the actual prefix/prompt override/instruction/extraction and budget. No alternative benchmark or score optimization.'},
            'effective_generation_config':self.generation_config.to_dict(),'text_rope_parameters':cfg.text_config.rope_parameters,'text_inv_freq':self.model.model.language_model.rotary_emb.inv_freq.detach().cpu().tolist(),'analytic_buffers':{name:tensor_record(value) for name,value in self.model.named_buffers()},'text_embedding_scale':float(embedding.embed_scale.cpu()),
            'model_resource_root':str(directory),'model_resource_sha256':self.resource_sha256,'runtime_source_sha256':self.runtime_sha256,'architecture_source_snapshot':str(snapshot),'per_sample_evidence':str(self.trace_path),'profiler_trace':str(self.evidence/'npu_trace.json'),
            'metric_protocol':'English ASCII-alphanumeric lowercasing and existing NLTK Porter stemmer, ROUGE-L F1, mean over every fixed sample including empty outputs as zero. Historical metric replay checked against complete existing200 records; no silent missing-library zero fallback.',
            'module_placements':{n:str(next(m.parameters()).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}

    def predict(self,prompt,media_path=None):
        start=time.monotonic();self.calls+=1;self._features=None;self._bridge=False;self._vision_calls=0;self._positions=None;self._masks=None
        entry={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None,'protocol':'table_matched_paligemma_caption_reference'}
        try:
            media=pathlib.Path(media_path)
            if not media.resolve().is_relative_to(self.dataset.parent):raise RuntimeError('Image outside reviewed caption snapshot')
            with Image.open(media) as im:image=im.convert('RGB')
            text=prompt;protocol=self.reference_protocol
            if protocol['prompt_override']:text=protocol['vqa_prefix']+protocol['prompt_override']
            else:
                suffix="\n\n请仔细分析图像和题目，将最终答案放在 <answer>...</answer> 标签内。只输出一个选项字母，例如 <answer>C</answer>。"
                if protocol['strip_eval_instruction'] and text.endswith(suffix):text=text[:-len(suffix)]
                text=protocol['vqa_prefix']+text+protocol['mc_instruction']
            prepared=self.processor(text=text,images=image,return_tensors='pt')
            entry.update(processor_text=text,source_image_sha256=hashlib.sha256(media.read_bytes()).hexdigest(),image_tensors={})
            for name,value in prepared.items():
                if not torch.is_tensor(value):raise RuntimeError('Unexpected non-tensor processor input')
                if name=='pixel_values':entry['image_tensors'][name]=tensor_record(value)
                else:entry[name]=value.cpu().tolist()
            labels=prepared.pop('labels',None)
            if labels is not None and not bool((labels==-100).all()):raise RuntimeError('Inference processor unexpectedly supplied nonignored training labels')
            entry['excluded_inference_fields']=['labels'] if labels is not None else []
            if int((prepared['input_ids']==self.image_id).sum())!=self.seq_len or not bool((prepared['token_type_ids']==0).all()):raise RuntimeError('Complete image and original prefix-LM inputs required')
            inputs={k:v.to(device=self.device,dtype=self.dtype) if v.is_floating_point() else v.to(self.device) for k,v in prepared.items()};self._active_ids=inputs['input_ids'];length=self._active_ids.shape[-1]
            with torch.inference_mode():
                if not self.profile_done:
                    with contextlib.chdir(self.evidence):
                        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as profile:
                            result=self.model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True)
                            if not bool(torch.isfinite(result.logits).all()):raise RuntimeError('Nonfinite actual NPU prefill logits')
                            torch.npu.synchronize(self.device)
                        profile.export_chrome_trace(str(self.evidence/'npu_trace.json'))
                    self.profile_done=True
                generated=self.model.generate(**inputs,generation_config=self.generation_config,max_new_tokens=self.max_new_tokens,use_cache=True,return_dict_in_generate=True,output_scores=False)
            tokens=generated.sequences[0,length:].cpu().tolist();raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip();truncated=len(tokens)>=self.max_new_tokens and (not tokens or tokens[-1] not in self.eos_ids)
            if not self._bridge or not self._vision_calls or list(self._features.shape)!=[1,self.seq_len,self.model.config.text_config.hidden_size] or self._positions is None or self._masks is None:raise RuntimeError('Complete actual NPU image/prefix evidence missing')
            entry.update(raw_output=raw,generation_token_ids=tokens,generation_count=len(tokens),truncated=truncated,stop_reason='length' if truncated else 'eos' if tokens and tokens[-1] in self.eos_ids else 'other',visual_embedding_injection_verified=True,vision_forward_calls=self._vision_calls,image_features_shape=list(self._features.shape),image_features_sha256=fingerprint(self._features),text_position_ids=self._positions,prefix_attention_masks=self._masks)
            log.info('PaliGemma caption NPU sample %s: tokens=%s truncated=%s raw=%r',self.calls,len(tokens),truncated,raw[:160]);return raw
        except Exception:
            entry['error']=traceback.format_exc();log.error(entry['error']);raise
        finally:
            entry['elapsed_seconds']=time.monotonic()-start;clean=re.sub(r'\n\nProvide only the final answer\.\s*$','',prompt).strip();self.details[(clean,media_path)]=entry
            with self.trace_path.open('a',encoding='utf8') as h:h.write(json.dumps(entry,ensure_ascii=False)+'\n')
            self._features=None
    def batch_predict(self,batch):return [self.predict(prompt,media) for prompt,media in batch]
    def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
    def get_execution_lineage(self):return self.lineage
    def unload(self):self.model=None;torch.npu.empty_cache()
