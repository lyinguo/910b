"""Standalone native NPU InternVL HF evaluation using each checkpoint's original inputs and tensors."""
import os,contextlib,datetime,hashlib,json,pathlib,struct,time,traceback,re,logging
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,yaml
from PIL import Image
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.internvl_hf_original_910b')

def fingerprint(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tensor_record(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fingerprint(t)}
def mapped_key(k):
    if k.startswith('language_model.model.'):return 'model.language_model.'+k[len('language_model.model.'):]
    if k.startswith('language_model.lm_head.'):return 'lm_head.'+k[len('language_model.lm_head.'):]
    if k.startswith(('vision_tower.','multi_modal_projector.')):return 'model.'+k
    return k

class InternVLHFOriginal910BAdapter(IModel):
    def __init__(self,npu_devices=None,max_new_tokens=2048,resource_sha256=None,runtime_sha256=None,weight_headers=None,original_config=None,original_config_sha256=None,original_adapter_sha256=None,dataset_path=None,dataset_sha256=None,reference_report=None,reference_report_sha256=None,source_head_sha256=None,source_embedding_sha256=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
        self.devices=list(npu_devices or []);assert len(self.devices)==1
        self.device=torch.device('npu:'+str(self.devices[0]));self.max_new_tokens=int(max_new_tokens);assert self.max_new_tokens==2048
        self.resources=dict(resource_sha256 or {});self.runtime=dict(runtime_sha256 or {});self.headers=dict(weight_headers or {})
        self.original_config=pathlib.Path(original_config);self.original_config_sha=original_config_sha256;self.original_adapter_sha=original_adapter_sha256
        self.dataset=pathlib.Path(dataset_path);self.dataset_sha=dataset_sha256;self.reference=pathlib.Path(reference_report);self.reference_sha=reference_report_sha256
        self.source_head_sha=source_head_sha256;self.source_embedding_sha=source_embedding_sha256
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('InternVL_HF_original_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False)
        self.trace=self.evidence/'sample_execution.jsonl';self.calls=0;self.profile_done=False;self.details={};self.lineage={}
    @property
    def model_name(self):return getattr(self,'_name','unloaded')
    @property
    def model_type(self):return 'InternVLForConditionalGeneration'
    @property
    def inference_framework(self):return 'Native Transformers / PyTorch + torch_npu BF16'
    def download(self,*args,**kwargs):raise RuntimeError('Model/data download prohibited')
    def load(self,model_path,**kwargs):
        from transformers import AutoConfig,AutoProcessor,InternVLForConditionalGeneration,GenerationConfig
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        from safetensors import safe_open
        directory=pathlib.Path(model_path).resolve();assert directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels')
        self._name=directory.parent.name+'/'+directory.name;assert torch.npu.is_available();torch.npu.set_device(self.device)
        for name,digest in self.resources.items():assert pathlib.Path(name).name==name and hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,name
        for path,digest in self.runtime.items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==digest,path
        assert {'config.json','preprocessor_config.json','tokenizer_config.json','chat_template.jinja'}.issubset(self.resources) and self.headers and self.runtime
        assert hashlib.sha256(self.original_config.read_bytes()).hexdigest()==self.original_config_sha
        original=yaml.safe_load(self.original_config.read_text());assert original['model']['name']==self.model_name and original['model']['adapter']=='universal_eval/model/adapters/transformers_vlm.py:TransformersVLM'
        assert original['model']['kwargs']['max_new_tokens']==2048 and original['algorithm']['kwargs']['extract_answer'] is True
        assert hashlib.sha256(pathlib.Path('universal_eval/model/adapters/transformers_vlm.py').read_bytes()).hexdigest()==self.original_adapter_sha
        assert self.dataset.is_relative_to('/root/projects/comperror/workspace/datasets') and hashlib.sha256(self.dataset.read_bytes()).hexdigest()==self.dataset_sha
        assert self.reference.is_relative_to('/mnt/afs/comp_error/workspace/reports') and hashlib.sha256(self.reference.read_bytes()).hexdigest()==self.reference_sha
        historical=json.loads(self.reference.read_text());assert historical['pipeline_config']['model']['name']==self.model_name and len(historical['detailed_results'])==200 and historical['pipeline_config']['algorithm']['kwargs']['extract_answer'] is True
        for name in ('conv','matmul'):getattr(torch.npu,name).allow_hf32=False
        raw=json.loads((directory/'config.json').read_text());cfg=AutoConfig.from_pretrained(str(directory),local_files_only=True)
        assert cfg.architectures==['InternVLForConditionalGeneration'] and cfg.text_config.model_type in ('qwen2','qwen3') and not getattr(cfg,'quantization_config',None)
        assert raw.get('tie_word_embeddings') is not True and raw['text_config'].get('tie_word_embeddings') is not True
        # Each exact source checkpoint stores different head and embedding tensors. The new composite defaultTrue must not discard its own head.
        cfg.tie_word_embeddings=False;cfg.text_config.tie_word_embeddings=False
        cfg._attn_implementation='eager';cfg.vision_config._attn_implementation='eager';cfg.text_config._attn_implementation='eager'
        with torch.device('meta'):self.model=InternVLForConditionalGeneration(cfg)
        expected=dict(self.model.named_parameters());filled=set();records=[];idx=directory/'model.safetensors.index.json';index=json.loads(idx.read_text())['weight_map'] if idx.is_file() else None
        shards=sorted(set(index.values())) if index else ['model.safetensors'];assert set(shards)==set(self.headers)
        for shard in shards:
            p=directory/shard;review=self.headers[shard]
            with p.open('rb') as f:length=f.read(8);header=f.read(struct.unpack('<Q',length)[0])
            assert p.stat().st_size==review['size'] and hashlib.sha256(length+header).hexdigest()==review['header_sha256']
            with safe_open(str(p),framework='pt',device='cpu') as handle:
                if index:assert set(handle.keys())=={k for k,v in index.items() if v==shard}
                for source_key in handle.keys():
                    target_key=mapped_key(source_key);assert target_key in expected and target_key not in filled,source_key
                    tensor=handle.get_tensor(source_key);assert tensor.dtype==torch.bfloat16 and tensor.shape==expected[target_key].shape
                    target=tensor.to(self.device).contiguous();locs=sorted({0,tensor.numel()//2,tensor.numel()-1});cpu=tensor.reshape(-1)[locs];actual=target.reshape(-1)[torch.tensor(locs,device=self.device)].cpu();assert torch.equal(cpu,actual)
                    parent,leaf=target_key.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));filled.add(target_key)
                    records.append({'source_file':shard,'source_key':source_key,'target_key':target_key,'shape':list(tensor.shape),'source_dtype':str(tensor.dtype),'execution_dtype':str(target.dtype),'device':str(target.device),'slice_positions':locs,'source_slice':cpu.float().tolist(),'device_slice':actual.float().tolist()})
            log.info('Native InternVL HF NPU loaded %s: %s/%s tensors',shard,len(filled),len(expected))
        assert filled==set(expected) and self.model.lm_head.weight is not self.model.model.language_model.embed_tokens.weight
        assert fingerprint(self.model.lm_head.weight)==self.source_head_sha and fingerprint(self.model.model.language_model.embed_tokens.weight)==self.source_embedding_sha
        cls={'qwen2':Qwen2RotaryEmbedding,'qwen3':Qwen3RotaryEmbedding}[cfg.text_config.model_type];rope=cls(cfg.text_config);self.model.model.language_model.rotary_emb=rope.to(self.device)
        assert all(p.device==self.device and p.dtype==torch.bfloat16 for p in self.model.parameters())
        assert all(not b.is_meta and b.device==self.device and b.dtype==torch.float32 for n,b in self.model.named_buffers())
        self.model.eval();self.processor=AutoProcessor.from_pretrained(str(directory),local_files_only=True);self.tokenizer=self.processor.tokenizer;self.image_id=cfg.image_token_id;assert self.processor.image_token_id==self.image_id and self.processor.image_seq_length==cfg.image_seq_length==256
        gen=GenerationConfig.from_pretrained(str(directory),local_files_only=True) if (directory/'generation_config.json').is_file() else GenerationConfig.from_model_config(cfg)
        self.model.generation_config=gen;self.generation_config,_=self.model._prepare_generation_config(None,max_new_tokens=self.max_new_tokens,do_sample=False,temperature=0.0)
        eos=self.generation_config.eos_token_id;self.eos_ids=[eos] if isinstance(eos,int) else list(eos);assert self.generation_config.do_sample is False and self.generation_config.num_beams==1
        self._ids=None;self._features=None;self._bridge=False;self._vision_calls=0;self._positions=None;self._causal=None
        def features(module,args,value):
            assert value.device==self.device and value.dtype==torch.bfloat16 and bool(torch.isfinite(value).all());self._features=value;self._bridge=False;self._vision_calls+=1
        def language(module,args,kw):
            embeds=kw.get('inputs_embeds')
            if embeds is None or self._features is None or embeds.shape[:2]!=self._ids.shape:return
            actual=embeds[self._ids==self.image_id];wanted=self._features.reshape(-1,embeds.shape[-1]);assert actual.shape==wanted.shape and torch.equal(actual,wanted) and bool(wanted.abs().sum()>0);self._bridge=True
        def positions(module,args,kw):
            value=kw.get('position_ids',args[1] if len(args)>1 else None)
            if value is not None and value.shape[-1]==self._ids.shape[-1]:self._positions=tensor_record(value)
        def causal(module,args,kw):
            mask=kw.get('attention_mask')
            if mask is not None and mask.shape[-2:]==self._ids.shape[-1:]*2:self._causal=tensor_record(mask)
        self.model.model.multi_modal_projector.register_forward_hook(features);self.model.model.language_model.register_forward_pre_hook(language,with_kwargs=True)
        self.model.model.language_model.rotary_emb.register_forward_pre_hook(positions,with_kwargs=True);self.model.model.language_model.layers[0].register_forward_pre_hook(causal,with_kwargs=True)
        loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for name in self.resources:
            if (directory/name).stat().st_size<2_000_000:(snapshot/name).write_bytes((directory/name).read_bytes())
        for path in self.runtime:(snapshot/(pathlib.Path(path).parent.name+'_'+pathlib.Path(path).name)).write_bytes(pathlib.Path(path).read_bytes())
        (snapshot/'original_config.yaml').write_bytes(self.original_config.read_bytes());(snapshot/'original_transformers_vlm.py').write_bytes(pathlib.Path('universal_eval/model/adapters/transformers_vlm.py').read_bytes());(snapshot/'historical_reference_report.json').write_bytes(self.reference.read_bytes())
        self.lineage={'hardware':'Ascend 910B','devices':self.devices,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'precision':'Every original learned tensor is native NPU BF16; analytic native RoPE buffers remain CPU-derived FP32. HF32 disabled. No low-bit acceleration, CPU learned layers, substituted weights or downloads.','source_dtype':'BF16','HF32_enabled':{n:bool(getattr(torch.npu,n).allow_hf32) for n in ('conv','matmul')},'all_expected_parameters_loaded':True,'weight_tensor_count':len(records),'weight_loading_evidence':str(loading),'weight_headers':self.headers,'model_resource_root':str(directory),'model_resource_sha256':self.resources,'runtime_source_sha256':self.runtime,'architecture_source_snapshot':str(snapshot),'per_sample_evidence':str(self.trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'max_new_tokens':2048,'image_token_id':self.image_id,'image_seq_length':256,'effective_image_processor':self.processor.image_processor.to_dict(),'effective_generation_config':self.generation_config.to_dict(),'eos_token_ids':self.eos_ids,'analytic_buffers':{n:tensor_record(b) for n,b in self.model.named_buffers()},'rope_parameters':cfg.text_config.rope_parameters,'untied_source_head_retained':True,'source_head_sha256':self.source_head_sha,'source_embedding_sha256':self.source_embedding_sha,'checkpoint_key_conversion':'Only legacy module prefixes: vision/projector -> model.*, language_model.model -> model.language_model, language_model.lm_head -> lm_head. Both different source head and embedding retained; current native composite defaultTrue is disabled. No tensor discarded.','original_config':str(self.original_config),'original_config_sha256':self.original_config_sha,'original_adapter_sha256':self.original_adapter_sha,'historical_reference_report':str(self.reference),'historical_reference_sha256':self.reference_sha,'dataset_sha256':self.dataset_sha,'protocol':'Original TransformersVLM plain-string <image> message -> own chat template -> original <IMG_CONTEXT> replacement -> own complete dynamic-patch processor; original 2048 native greedy generation settings. All reference200 input/ref records replayed independently. Explicit MC v2 extraction is separate from real raw generation.','module_placements':{n:str(next(m.parameters()).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}
    def predict(self,prompt,media_path=None):
        started=time.monotonic();self.calls+=1;self._features=None;self._bridge=False;self._vision_calls=0;self._positions=None;self._causal=None;entry={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None}
        try:
            media=pathlib.Path(media_path).resolve();assert media.is_relative_to(self.dataset.parent)
            with Image.open(media) as im:image=im.convert('RGB')
            messages=[{'role':'user','content':'<image>\n'+prompt}];chat=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
            if '<IMG_CONTEXT>' not in chat and '<image>' in chat:chat=chat.replace('<image>','<IMG_CONTEXT>')
            prepared=self.processor(text=chat,images=[image],return_tensors='pt',padding=True);assert set(prepared)=={'input_ids','attention_mask','pixel_values'}
            entry.update(messages=messages,chat=chat,source_image_sha256=hashlib.sha256(media.read_bytes()).hexdigest(),CPU_processor_tensors={k:tensor_record(v) for k,v in prepared.items()},input_ids=prepared['input_ids'].tolist(),attention_mask=prepared['attention_mask'].tolist())
            assert int((prepared['input_ids']==self.image_id).sum())==len(prepared['pixel_values'])*256 and bool(prepared['attention_mask'].all())
            inputs={k:v.to(self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in prepared.items()};self._ids=inputs['input_ids'];length=self._ids.shape[-1]
            with torch.inference_mode():
                if not self.profile_done:
                    with contextlib.chdir(self.evidence):
                        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as profile:
                            result=self.model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True);assert bool(torch.isfinite(result.logits).all());torch.npu.synchronize(self.device)
                        profile.export_chrome_trace(str(self.evidence/'npu_trace.json'))
                    self.profile_done=True
                generated=self.model.generate(**inputs,generation_config=self.generation_config,return_dict_in_generate=True,output_scores=False)
            tokens=generated.sequences[0,length:].cpu().tolist();raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip();clipped=len(tokens)>=self.max_new_tokens and (not tokens or tokens[-1] not in self.eos_ids)
            assert self._bridge and self._features is not None and self._vision_calls>=1 and self._positions is not None and self._causal is not None
            entry.update(raw_output=raw,generation_token_ids=tokens,generation_count=len(tokens),truncated=clipped,stop_reason='length' if clipped else 'eos' if tokens and tokens[-1] in self.eos_ids else 'other',visual_embedding_injection_verified=True,vision_forward_calls=self._vision_calls,image_features_shape=list(self._features.shape),image_features_sha256=fingerprint(self._features),position_ids=self._positions,causal_mask=self._causal)
            log.info('InternVL HF NPU sample %s: tiles=%s generated=%s truncated=%s raw=%r',self.calls,len(prepared['pixel_values']),len(tokens),clipped,raw[:160]);return raw
        except Exception:
            entry['error']=traceback.format_exc();log.error(entry['error']);raise
        finally:
            entry['elapsed_seconds']=time.monotonic()-started;clean=re.sub(r'\n+Answer the question by providing only the option letter.*$','',prompt or '',flags=re.S).strip();self.details[(clean,media_path)]=entry
            with self.trace.open('a') as handle:handle.write(json.dumps(entry,ensure_ascii=False)+'\n')
            self._features=None
    def batch_predict(self,batch):return [self.predict(p,m) for p,m in batch]
    def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
    def get_execution_lineage(self):return self.lineage
    def unload(self):self.model=None;torch.npu.empty_cache()
