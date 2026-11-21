"""Own saved LLaVA-Mini graph and generation; private cache and complete observers."""
import contextlib,datetime,functools,hashlib,inspect,json,logging,os,pathlib,shutil,time,traceback
import numpy as np
import torch,torch_npu
from torch import nn
from PIL import Image
from transformers import AutoTokenizer,PreTrainedTokenizerFast,CLIPImageProcessor,LlamaForCausalLM
from transformers.cache_utils import DynamicLayer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from universal_eval.model.adapters import high_nine_llava_reference as original
from universal_eval.model.llava_mini_own_saved_graph_FP16_20261122 import build_own_saved_graph
from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz
from scripts.egm8b_original_independent_cpu_910b import sha,fp,tr

log=logging.getLogger(__name__)

class OwnFourDimensionalDynamicLayer(DynamicLayer):
    def lazy_initialization(self,key_states,value_states):
        self.dtype,self.device=key_states.dtype,key_states.device
        self.keys=key_states[..., :0, :].contiguous()
        self.values=value_states[..., :0, :].contiguous()
        self.is_initialized=True

class OwnPrivateCacheLlama(LlamaForCausalLM):
    def _prepare_cache_for_generation(self,generation_config,model_kwargs,generation_mode,batch_size,max_cache_length):
        result=super()._prepare_cache_for_generation(generation_config,model_kwargs,generation_mode,batch_size,max_cache_length)
        cache=model_kwargs.get('past_key_values')
        assert cache is not None and len(cache.layers)==32
        for layer in cache.layers:
            assert type(layer) is DynamicLayer and not layer.is_initialized
            layer.__class__=OwnFourDimensionalDynamicLayer
        return result

class OwnLlavaMiniOriginalFP16(original.LlavaMiniCUDAAdapter):
    def __init__(self,protocol_file,protocol_sha256,npu_device=4,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
        assert sha(protocol_file)==protocol_sha256
        self.p=json.loads(pathlib.Path(protocol_file).read_bytes())
        self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha=protocol_sha256
        super().__init__(device='npu:'+str(npu_device),max_new_tokens=2048,torch_dtype='float16')
        self.device=torch.device(self._device);assert self.device.index!=0
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=pathlib.Path(evidence_dir).resolve()/('LlavaMini_own_original_FP16_2048_'+stamp+'_'+str(os.getpid()))
        self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'groups').mkdir()
        self.trace=self.evidence/'sample_execution.jsonl';self.details={};self.group_count=0;self.lineage={}
        self.shared_cache_source=inspect.getsource(DynamicLayer.lazy_initialization)
    @property
    def model_type(self):return 'LlavaMiniLlamaForCausalLM'
    @property
    def inference_framework(self):return 'Own saved native Transformers FP16 / Ascend910B / greedy2048'
    def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
    def load(self,model_path,**kwargs):
        from safetensors import safe_open
        p=self.p;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['own_weight_path']
        assert p['model_id']=='ICTNLP/llava-mini-llama-3.1-8b'
        for path,digest in p['runtime_and_original_source_sha256'].items():assert sha(path)==digest,path
        for name,digest in p['own_assets_SHA256'].items():assert sha(wp/name)==digest,name
        assert sha(p['own_frontend200'])==p['own_frontend200_SHA256']
        self.front=json.loads(pathlib.Path(p['own_frontend200']).read_bytes());assert len(self.front)==200
        assert torch.npu.is_available();torch.set_num_threads(4);torch.npu.set_device(self.device)
        for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
        raw=json.loads((wp/'config.json').read_bytes());self._mm_use_im_start_end=bool(raw.get('mm_use_im_start_end',False))
        model=build_own_saved_graph(raw,meta=True);model.__class__=OwnPrivateCacheLlama
        parameters=dict(model.named_parameters());assert len(parameters)==732
        loaded=[];seen=set()
        for shard,record in p['own_weight_files'].items():
            q=wp/shard;assert q.stat().st_size==record['bytes'] and sha(q)==record['SHA256']
            with safe_open(str(q),framework='pt',device='cpu') as file:
                for name in file.keys():
                    x=file.get_tensor(name);assert name in parameters and name not in seen and x.shape==parameters[name].shape and x.dtype in [torch.float16,torch.float32]
                    assert torch.isfinite(x).all(),name
                    cast=x.to(torch.float16);y=cast.to(self.device);assert fp(y)==fp(cast) and y.dtype==torch.float16
                    parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,nn.Parameter(y,requires_grad=False))
                    loaded.append(dict(source_key=name,target_key=name,shape=list(x.shape),source_dtype=str(x.dtype),source_SHA256=fp(x),original_FP16_cast_SHA256=fp(cast),actual_NPU_SHA256=fp(y),actual_dtype=str(y.dtype),actual_device=str(y.device)))
                    seen.add(name)
            log.info('Own LLaVA-Mini source/cast/actual NPU full SHA %d/732',len(loaded))
        assert seen==set(parameters)
        # Original construction creates default RoPE in F32, then model.to(F16)
        # casts BOTH registered analytic buffers. Preserve that exact order.
        model.model.rotary_emb=LlamaRotaryEmbedding(model.config,device='cpu').to(dtype=torch.float16).to(self.device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.position_ids=torch.arange(577,dtype=torch.int64,device=self.device).expand(1,-1)
        assert model.config.rope_parameters['rope_type']=='default' and model.config.rope_parameters['rope_theta']==500000.
        assert all(not t.is_meta and t.dtype==torch.float16 and t.device==self.device for t in model.parameters())
        assert all(not t.is_meta and t.device==self.device and t.dtype==(torch.int64 if n.endswith('position_ids') else torch.float16) for n,t in model.named_buffers())
        assert model.forward.__func__ is LlamaForCausalLM.forward
        model.eval();self._model=model;self._model_path=str(wp);self._model_name=p['model_id']
        try:self._tokenizer=AutoTokenizer.from_pretrained(wp,local_files_only=True)
        except Exception:self._tokenizer=PreTrainedTokenizerFast.from_pretrained(wp,local_files_only=True)
        if self._tokenizer.pad_token is None:self._tokenizer.pad_token=self._tokenizer.eos_token
        self._image_processor=CLIPImageProcessor(size={'shortest_edge':336},crop_size={'height':336,'width':336},do_resize=True,do_center_crop=True,do_rescale=True,do_normalize=True,image_mean=original.CLIP_MEAN,image_std=original.CLIP_STD)
        gen,_=model._prepare_generation_config(None,max_new_tokens=2048,do_sample=False,use_cache=True)
        assert gen.eos_token_id==[128001,128008,128009] and not gen.do_sample and gen.use_cache and gen.num_beams==1
        self.generation_config=gen.to_dict();self._dtype=torch.float16;self.hooks=[]
        self.module_kinds={};self._install_observers()
        weight_file=self.evidence/'weight_loading.json';weight_file.write_text(json.dumps(loaded,ensure_ascii=False,indent=2),encoding='utf8')
        snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for name,digest in p['runtime_and_original_source_sha256'].items():
            path=pathlib.Path(name);dest=snapshot/path.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());assert sha(dest)==digest
        (snapshot/'original_protocol.json').write_bytes(pathlib.Path(self.protocol_file).read_bytes())
        self.lineage=dict(hardware='Ascend910B',device=str(self.device),protocol_file=self.protocol_file,protocol_sha256=self.protocol_sha,weight_loading_evidence=str(weight_file),source_tensor_count=732,actual_FP16_parameter_count=732,source_dtype_counts={'F16':341,'F32':391},original_max_new_tokens=2048,original_batch_size=1,original_default_plain_RoPE_preserved=True,original_24CLIP_4prefusion_32text_128256head_preserved=True,original_analytic_FP16_buffers={n:tr(t) for n,t in model.named_buffers()},original_generation_config=self.generation_config,original_shared_cache_source_SHA256=hashlib.sha256(self.shared_cache_source.encode()).hexdigest(),private_cache_4D_empty_same_original_body=True,shared_cache_library_unmodified=True,per_sample_evidence=str(self.trace),per_group_evidence=str(self.evidence/'groups'),complete_numeric_module_kinds=self.module_kinds,profiler=str(self.evidence/'first_vision_prefill_npu_trace.json'),profiler_scope='First complete vision/prefusion and first text prefill only; not all generation steps',historical_A100_engine_or_tensor_capture_claimed=False,all_learned_forwards_native_NPU=True,precision_disclosure='Own341F16+391F32 physical source loaded by original explicit FP16 cast. Source/cast/actual NPU full SHA recorded for all732. Original registered plain RoPE inv_freq and original_inv_freq also cast FP16 before original F32 angle calculation; no FP32 model grade.',HF32_enabled={k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']})
    def _tensor(self,t):
        assert isinstance(t,torch.Tensor)
        x=t.detach().cpu().contiguous().clone();raw=x.view(torch.uint8).numpy().reshape(-1).copy()
        h=hashlib.sha256(raw.tobytes()).hexdigest();key='full_'+str(x.dtype).replace('torch.','')+'_'+'_'.join(map(str,x.shape))+'_'+h
        value=x.numpy().copy()
        if key in self.arrays:
            old=self.arrays[key];assert old.dtype==value.dtype and old.shape==value.shape and np.array_equal(old.view(np.uint8).reshape(-1),value.view(np.uint8).reshape(-1))
        else:self.arrays[key]=value
        self.occurrences+=1
        return dict(array=key,shape=list(x.shape),dtype=str(x.dtype),raw_SHA256=h,device=str(t.device))
    def _tree(self,value):
        if isinstance(value,torch.Tensor):return self._tensor(value)
        if isinstance(value,(list,tuple)):return [self._tree(v) for v in value]
        if isinstance(value,dict):return {k:self._tree(v) for k,v in value.items() if k not in ['past_key_values']}
        if value is None or isinstance(value,(bool,int,float,str)):return value
        return dict(non_tensor_type=type(value).__name__)
    def _install_observers(self):
        from transformers.models.llama.modeling_llama import LlamaRMSNorm,LlamaAttention,LlamaDecoderLayer,LlamaMLP
        from transformers.models.clip.modeling_clip import CLIPAttention,CLIPEncoderLayer,CLIPMLP
        selected=(nn.Linear,nn.Conv2d,nn.LayerNorm,nn.Embedding,nn.GELU,nn.SiLU,nn.MultiheadAttention,LlamaRMSNorm,LlamaAttention,LlamaDecoderLayer,LlamaMLP,CLIPAttention,CLIPEncoderLayer,CLIPMLP)
        for name,module in self._model.named_modules():
            if not isinstance(module,selected) and type(module).__name__ not in ['QuickGELUActivation','LlamaRotaryEmbedding','_CompressorResampler','_LlavaMiniVisionTower','CLIPVisionEmbeddings']:continue
            kind=type(module).__name__;self.module_kinds[name]=kind
            def before(mod,args,kw,n=name,k=kind):
                if not self.active:return
                call=dict(sequence=len(self.calls),module=n,kind=k,text_step=self.text_step,inputs=self._tree(args),kwargs=self._tree(kw))
                self.calls.append(call);self.stacks.setdefault(n,[]).append(call)
            def after(mod,args,kw,out,n=name):
                if not self.active:return
                call=self.stacks[n].pop();call['output']=self._tree(out);call['end_sequence']=len(self.calls)-1
            self.hooks.append(module.register_forward_pre_hook(before,with_kwargs=True))
            self.hooks.append(module.register_forward_hook(after,with_kwargs=True))
        def pre(mod,args,kw):
            if not self.active:return
            self.text_step+=1;self.steps.append(dict(step=self.text_step,inputs=self._tree(kw),cache32=[]))
        def post(mod,args,kw,out):
            if not self.active:return
            cache=out.past_key_values;assert len(cache.layers)==32
            row=self.steps[-1]
            for i,layer in enumerate(cache.layers):
                assert type(layer) is OwnFourDimensionalDynamicLayer
                k,v=layer.keys,layer.values;assert k.shape==v.shape and k.shape[1]==8 and k.shape[-1]==128
                old=self.previous_KV.get(i);unchanged=old is None or (torch.equal(k[...,:old[0].shape[-2],:],old[0]) and torch.equal(v[...,:old[1].shape[-2],:],old[1]))
                assert unchanged
                start=0 if old is None else old[0].shape[-2]
                row['cache32'].append(dict(layer=i,complete_key=tr(k),complete_value=tr(v),key_delta=self._tensor(k[...,start:,:]),value_delta=self._tensor(v[...,start:,:]),original_prefix_bitwise_unchanged=unchanged))
                self.previous_KV[i]=(k.detach().clone(),v.detach().clone())
            row['full_logits']=self._tensor(out.logits)
            row['actual_native_argmax']=out.logits[:,-1,:].argmax(-1).tolist()
            with (self.evidence/'generation_progress.jsonl').open('a',encoding='utf8') as file:file.write(json.dumps(dict(index=self.current_index,step=self.text_step,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
            if self.profile is not None:
                torch.npu.synchronize(self.device);self.profile.stop();self.profile=None
                self.profiler.export_chrome_trace(str(self.evidence/'first_vision_prefill_npu_trace.json'))
        self.hooks.append(self._model.register_forward_pre_hook(pre,with_kwargs=True));self.hooks.append(self._model.register_forward_hook(post,with_kwargs=True))
    def predict(self,prompt,media_path=None):
        assert self._model is not None and media_path and pathlib.Path(media_path).is_file()
        i=len(self.details);row=self.front[i];assert row['actual_model_prompt']==prompt and row['media_file_SHA256']==sha(media_path)
        free=shutil.disk_usage(self.evidence).free;assert free>=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes']+self.p['precheck_reserve_bytes']+self.p['per_sample_lossless_space_estimate_bytes']
        self.group_count+=1;group=self.evidence/'groups'/('%04d'%self.group_count);group.mkdir()
        self.arrays={};self.calls=[];self.steps=[];self.stacks={};self.previous_KV={};self.text_step=0;self.current_index=i;self.occurrences=0;self.active=True;self.profile=None;error=None
        self.profile=(torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,profile_memory=False,with_stack=False) if i==0 else None)
        self.profiler=self.profile
        if self.profile is not None:self.profile.start()
        try:
            image=Image.open(media_path).convert('RGB');pixels=self._image_processor(images=image,return_tensors='pt')['pixel_values'];images=pixels.to(self.device,dtype=self._dtype).unsqueeze(0)
            text=self._build_prompt_text(prompt);input_ids=self._tokenizer_image_token(text).unsqueeze(0).to(self.device)
            assert input_ids[0].tolist()==row['input_ids'] and fp(pixels)==row['pixels_F32_SHA256'] and fp(images)==row['pixels_FP16_SHA256']
            self.inputs=self._tree(dict(pixels_F32=pixels,pixels_FP16=images,input_ids=input_ids))
            with torch.inference_mode():
                compressed,fused_text=self._encode_images_mini(images,input_ids)
                embeds=self._splice(input_ids,fused_text,compressed);attention_mask=torch.ones(embeds.shape[:2],dtype=torch.long,device=self.device)
                self.fused=self._tree(dict(compressed=compressed,fused_text=fused_text,spliced_embeds=embeds,attention_mask=attention_mask))
                generated=self._model.generate(inputs_embeds=embeds,attention_mask=attention_mask,max_new_tokens=self._max_new_tokens,do_sample=False,use_cache=True)
            if hasattr(generated,'sequences'):generated=generated.sequences
            complete_generated=generated[0].tolist()
            if generated.shape[1]>self._max_new_tokens:generated=generated[:,embeds.shape[1]:]
            decoded_ids=generated[0].tolist();raw=self._tokenizer.decode(decoded_ids,skip_special_tokens=True).strip()
            assert inspect.getsource(DynamicLayer.lazy_initialization)==self.shared_cache_source
        except BaseException:
            error=traceback.format_exc();raise
        finally:
            if self.profile is not None:
                self.profile.stop();self.profile=None
                self.profiler.export_chrome_trace(str(self.evidence/'first_vision_prefill_npu_trace.json'))
            self.active=False
            archive=group/'complete_actual_values.npz';write_full_standard_npz(archive,self.arrays)
            record=dict(index=i,group=self.group_count,error=error,inputs=getattr(self,'inputs',None),fused=getattr(self,'fused',None),calls=self.calls,steps=self.steps,archive=str(archive),archive_SHA256=sha(archive),complete_copied_numeric_occurrences=self.occurrences,unique_full_NPY_values=len(self.arrays),original_full_values_shared_only_after_shape_dtype_and_every_raw_bit_equal=True)
            (group/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
        assert 1<=len(self.steps)<=2048
        eos=[128001,128008,128009];first=next((j for j,t in enumerate(complete_generated) if t in eos),None)
        execution=dict(index=i,prompt=row['prompt'],actual_model_prompt=prompt,media_path=media_path,media_SHA256=sha(media_path),raw_output=raw,generation_token_ids=complete_generated,decoded_token_ids=decoded_ids,first_EOS_index=first,generation_count=len(complete_generated),native_forward_count=len(self.steps),truncated=(first is None and len(self.steps)>=2048),empty=not raw,error=None,group_execution=str(group/'execution.json'),group_execution_SHA256=sha(group/'execution.json'),actual_device=str(self.device))
        from universal_eval.algorithm.native_explicit_mc_v2 import explicit_choice
        execution['invalid']=explicit_choice(raw) is None
        self.details[i]=execution
        with self.trace.open('a',encoding='utf8') as file:file.write(json.dumps(execution,ensure_ascii=False)+'\n')
        self.arrays={};self.previous_KV={};log.info('Own LLaVA-Mini complete sample %d / native forwards %d',i+1,len(self.steps));return raw
    def batch_predict(self,batch):return [self.predict(prompt,media) for prompt,media in batch]
    def get_sample_execution(self,prompt,media_path):
        return next((r for r in self.details.values() if r['prompt'].strip()==prompt.strip() and r['media_path']==media_path),{'error':'Missing original own execution'})
    def get_execution_lineage(self):
        import sys
        root=pathlib.Path('/root/projects/comperror');external={}
        for module in list(sys.modules.values()):
            name=getattr(module,'__file__',None)
            if not name:continue
            path=pathlib.Path(name).resolve()
            if path.is_file() and not path.is_relative_to(root):external[str(path)]=sha(path)
        self.lineage['all_external_loaded_runtime_files_sha256']=external
        return self.lineage
    def unload(self):
        for hook in getattr(self,'hooks',[]):hook.remove()
        self._model=None;self._tokenizer=None;self._image_processor=None
        import gc
        gc.collect()
        if torch.npu.is_available():torch.npu.empty_cache()
