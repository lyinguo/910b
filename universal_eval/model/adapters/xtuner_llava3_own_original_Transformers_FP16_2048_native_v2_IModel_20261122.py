"""Own Xtuner original native FP16 graph, original576 image fusion/full128257 head/2048/EOS128001."""
import datetime,hashlib,inspect,json,logging,os,pathlib,shutil,traceback
import numpy as np
import torch,torch_npu
from torch import nn
from universal_eval.model.interface import IModel
from PIL import Image
from transformers import AutoTokenizer,CLIPImageProcessor,LlavaProcessor,LlavaForConditionalGeneration,GenerationConfig
from transformers.cache_utils import DynamicLayer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from universal_eval.model.xtuner_llava3_own_saved_Transformers_graph_20261122 import build_own_meta
from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz
from scripts.egm8b_original_independent_cpu_910b import sha,fp,tr
log=logging.getLogger(__name__)
MID='xtuner/llava-llama-3-8b-v1_1-hf'
class OwnFourDimensionalDynamicLayer(DynamicLayer):
    def lazy_initialization(self,key_states,value_states):
        self.dtype,self.device=key_states.dtype,key_states.device
        self.keys=key_states[..., :0, :].contiguous()
        self.values=value_states[..., :0, :].contiguous()
        self.is_initialized=True

class OwnPrivateCacheLlama(LlavaForConditionalGeneration):
    def _prepare_cache_for_generation(self,generation_config,model_kwargs,generation_mode,batch_size,max_cache_length):
        result=super()._prepare_cache_for_generation(generation_config,model_kwargs,generation_mode,batch_size,max_cache_length)
        cache=model_kwargs.get('past_key_values')
        assert cache is not None and len(cache.layers)==32
        for layer in cache.layers:
            assert type(layer) is DynamicLayer and not layer.is_initialized
            layer.__class__=OwnFourDimensionalDynamicLayer
        return result

class OwnXtunerLlava3OriginalFP16(IModel):
    def __init__(self,protocol_file,protocol_sha256,npu_device=4,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
        assert sha(protocol_file)==protocol_sha256
        self.p=json.loads(pathlib.Path(protocol_file).read_bytes());self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha=protocol_sha256
        self.device=torch.device('npu:'+str(npu_device));assert self.device.index==4
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=pathlib.Path(evidence_dir).resolve()/('XtunerLlava3_own_original_FP16_2048_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'groups').mkdir()
        self.trace=self.evidence/'sample_execution.jsonl';self.details={};self.group_count=0;self.report_index=0;self.lineage={};self.shared_cache_source=inspect.getsource(DynamicLayer.lazy_initialization);self._model=None
    @property
    def model_type(self):return 'LlavaLlamaForCausalLM'
    @property
    def model_name(self):return MID
    @property
    def inference_framework(self):return 'Own saved native Transformers FP16 / Ascend910B / Greedy2048'
    def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
    def load(self,model_path,**kwargs):
        from safetensors import safe_open
        from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
        p=self.p;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['own_weight_path'] and p['model_id']==MID
        for path,digest in p['runtime_and_original_source_sha256'].items():assert sha(path)==digest,path
        for path,digest in p['static_evidence_SHA256'].items():assert sha(path)==digest,path
        for name,digest in p['own_assets_SHA256'].items():assert sha(wp/name)==digest,name
        assert sha(p['own_frontend200'])==p['own_frontend200_SHA256'];self.front=json.loads(pathlib.Path(p['own_frontend200']).read_bytes());assert len(self.front)==200
        ctor=json.loads(pathlib.Path(p['own_original_constructor_review']).read_bytes());assert sha(p['own_original_constructor_review'])==p['own_original_constructor_review_SHA256'] and ctor['passed'] and ctor['original_resize_before_manual_remap_order_preserved'] and ctor['native_head_and_embedding_rows_after_original_resize']==128257
        extra_path=ctor['original_extra_head_and_image_embedding_row_values'];assert sha(extra_path)==ctor['original_extra_head_and_image_embedding_row_values_SHA256']
        with ReadOnlyNPZ(extra_path) as z:extra={n:torch.from_numpy(z[n].copy()) for n in ['added_embedding','added_lm_head']}
        assert fp(extra['added_embedding'])==ctor['original_added_embedding_SHA256'] and fp(extra['added_lm_head'])==ctor['original_added_lm_head_SHA256']
        mapping=json.loads(pathlib.Path(p['own_original_legacy_mapping_meta_review']).read_bytes())['source_to_native_mapping']
        torch.npu.set_device(self.device);torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
        raw=json.loads((wp/'config.json').read_bytes());model=build_own_meta(raw);model.__class__=OwnPrivateCacheLlama;parameters=dict(model.named_parameters());assert len(parameters)==686
        loaded=[];seen=set()
        for shard,record in p['own_weight_files'].items():
            path=wp/shard;assert path.stat().st_size==record['bytes'] and sha(path)==record['SHA256']
            with safe_open(str(path),framework='pt',device='cpu') as sf:
                for name in sf.keys():
                    source=sf.get_tensor(name);target=mapping[name];assert name not in seen and target in parameters and source.dtype in [torch.float16,torch.float32] and torch.isfinite(source).all()
                    cast=source.to(torch.float16);complete=cast
                    if name in ['model.embed_tokens.weight','lm_head.weight']:
                        assert cast.shape==(128256,4096);complete=torch.cat([cast,extra['added_embedding' if name=='model.embed_tokens.weight' else 'added_lm_head']],dim=0)
                    assert complete.shape==parameters[target].shape;y=complete.to(self.device);assert fp(y)==fp(complete)
                    parent,leaf=target.rsplit('.',1);setattr(model.get_submodule(parent),leaf,nn.Parameter(y,requires_grad=False));seen.add(name)
                    loaded.append(dict(source_key=name,target_key=target,shape=list(source.shape),source_dtype=str(source.dtype),source_SHA256=fp(source),original_FP16_cast_SHA256=fp(cast),actual_NPU_original_source_rows_SHA256=fp(y[:128256] if name in ['model.embed_tokens.weight','lm_head.weight'] else y),actual_NPU_complete_parameter_SHA256=fp(y),actual_shape=list(y.shape),actual_dtype=str(y.dtype),actual_device=str(y.device),original_constructor_derived_extra_row=(name in ['model.embed_tokens.weight','lm_head.weight'])))
            log.info('Own Xtuner source/cast/actual NPU verified %d/686',len(loaded))
        assert len(seen)==686 and {v['target_key'] for v in loaded}==set(parameters)
        model.model.language_model.rotary_emb=LlamaRotaryEmbedding(model.config.text_config,device='cpu').to(self.device)
        model.model.vision_tower.embeddings.position_ids=torch.arange(577,dtype=torch.int64,device=self.device).expand(1,-1)
        assert all(t.dtype==torch.float16 and t.device==self.device and not t.is_meta for t in model.parameters())
        expected_buffers=ctor['own_graph_buffer_layout'];assert {n for n,t in model.named_buffers()}==set(expected_buffers)
        for n,t in model.named_buffers():assert str(t.dtype)==expected_buffers[n]['dtype'] and fp(t)==expected_buffers[n]['SHA256'] and t.device==self.device
        model.eval();model.generation_config=GenerationConfig.from_pretrained(wp,local_files_only=True);assert model.generation_config.eos_token_id==128001
        assert model.forward.__func__ is LlavaForConditionalGeneration.forward
        self._model=model;self._tokenizer=AutoTokenizer.from_pretrained(wp,local_files_only=True)
        if '<image>' not in self._tokenizer.get_vocab():self._tokenizer.add_special_tokens({'additional_special_tokens':['<image>']})
        assert self._tokenizer.convert_tokens_to_ids('<image>')==128256 and len(self._tokenizer)==128257
        if self._tokenizer.pad_token is None:self._tokenizer.pad_token=self._tokenizer.eos_token
        self._image_processor=CLIPImageProcessor(do_resize=True,size={'shortest_edge':336},resample=3,do_center_crop=True,crop_size={'height':336,'width':336},do_rescale=True,rescale_factor=1/255.,do_normalize=True,image_mean=[.48145466,.4578275,.40821073],image_std=[.26862954,.26130258,.27577711])
        self.processor=LlavaProcessor(image_processor=self._image_processor,tokenizer=self._tokenizer,patch_size=14,vision_feature_select_strategy='patch',num_additional_image_tokens=0)
        gen,_=model._prepare_generation_config(None,max_new_tokens=2048,do_sample=False,temperature=0.);assert gen.eos_token_id==128001 and gen.num_beams==1 and gen.use_cache and gen.repetition_penalty==1
        self.generation_config=gen.to_dict();self._dtype=torch.float16;self._max_new_tokens=2048;self.hooks=[];self.module_kinds={};self._install_observers()
        def fused_pre(mod,args,kw):
            if self.active:self.steps[-1]['native_fused_inputs']=self._tree(kw)
        self.hooks.append(model.model.language_model.register_forward_pre_hook(fused_pre,with_kwargs=True))
        weight_file=self.evidence/'weight_loading.json';weight_file.write_text(json.dumps(loaded,ensure_ascii=False,indent=2),encoding='utf-8')
        snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for n,h in p['runtime_and_original_source_sha256'].items():
            path=pathlib.Path(n);dest=snapshot/path.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());assert sha(dest)==h
        (snapshot/'original_protocol.json').write_bytes(pathlib.Path(self.protocol_file).read_bytes())
        self.lineage=dict(hardware='Ascend910B',device=str(self.device),protocol_file=self.protocol_file,protocol_sha256=self.protocol_sha,weight_loading_evidence=str(weight_file),source_tensor_count=686,actual_FP16_parameter_count=686,source_dtype_counts={'F16':682,'F32':4},original_max_new_tokens=2048,original_batch_size=10,actual_inner_batch_size=1,original_plain_F32_RoPE_buffers_preserved=True,original_CLIP24_projector2_Llama32_full128257_head_preserved=True,own_original_constructor_review=p['own_original_constructor_review'],own_original_constructor_review_SHA256=p['own_original_constructor_review_SHA256'],original_constructor_derived_extra_row_values=extra_path,original_constructor_derived_extra_row_values_SHA256=sha(extra_path),original_generation_config=self.generation_config,original_analytic_buffers={n:tr(t) for n,t in model.named_buffers()},original_shared_cache_source_SHA256=hashlib.sha256(self.shared_cache_source.encode()).hexdigest(),private_cache_4D_empty_same_original_body=True,shared_cache_library_unmodified=True,per_sample_evidence=str(self.trace),per_group_evidence=str(self.evidence/'groups'),complete_numeric_module_kinds=self.module_kinds,profiler=str(self.evidence/'first_vision_prefill_npu_trace.json'),profiler_scope='First own complete vision and text prefill only; not all generation steps',historical_A100_engine_or_tensor_capture_claimed=False,all_learned_forwards_native_NPU=True,precision_disclosure='Own682F16+4F32 physical source cast to original requestedF16; original saved resize-before-legacy-remap derives the extra head/image embedding row, full original CPU constructor proof and raw rows retained, not a foreign checkpoint. F32 RoPE buffers preserved. Actual NPU all source/cast/full parameter SHA independently checked.',HF32_enabled={'conv':False,'matmul':False})
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
                if not isinstance(module,selected) and type(module).__name__ not in ['QuickGELUActivation','LlamaRotaryEmbedding','CLIPVisionEmbeddings','LlavaMultiModalProjector']:continue
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
        reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes']+self.p['precheck_reserve_bytes']+self.p['per_sample_lossless_space_estimate_bytes'];assert shutil.disk_usage(self.evidence).free>=reserve
        self.group_count+=1;group=self.evidence/'groups'/('%04d'%self.group_count);group.mkdir();self.arrays={};self.calls=[];self.steps=[];self.stacks={};self.previous_KV={};self.text_step=0;self.current_index=i;self.occurrences=0;self.active=True;error=None;self.inputs=None;self.fused=None
        self.profile=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,profile_memory=False,with_stack=False) if i==0 else None;self.profiler=self.profile
        if self.profile is not None:self.profile.start()
        try:
            image=Image.open(media_path).convert('RGB');messages=[{'role':'user','content':'<image>\n'+prompt}];text=self._tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False);assert text==row['text']
            original=self.processor(images=[image],text=text,return_tensors='pt',padding=True);assert original['input_ids'][0].tolist()==row['input_ids'] and fp(original['pixel_values'])==row['pixels_F32_SHA256'];inputs={n:t.to(self.device) for n,t in original.items()}
            self.inputs=self._tree(dict(pixels_F32=inputs['pixel_values'],input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask']))
            with torch.inference_mode():generated=self._model.generate(**inputs,max_new_tokens=2048,do_sample=False,temperature=0.)
            if hasattr(generated,'sequences'):generated=generated.sequences
            assert generated[0,:inputs['input_ids'].shape[1]].tolist()==row['input_ids'];complete_generated=generated[0,inputs['input_ids'].shape[1]:].tolist();raw=self._tokenizer.decode(complete_generated,skip_special_tokens=True).strip()
            assert inspect.getsource(DynamicLayer.lazy_initialization)==self.shared_cache_source
        except BaseException:error=traceback.format_exc();raise
        finally:
            if self.profile is not None:
                self.profile.stop();self.profile=None;self.profiler.export_chrome_trace(str(self.evidence/'first_vision_prefill_npu_trace.json'))
            self.active=False;archive=group/'complete_actual_values.npz';write_full_standard_npz(archive,self.arrays)
            record=dict(index=i,group=self.group_count,error=error,inputs=self.inputs,fused=self.fused,calls=self.calls,steps=self.steps,archive=str(archive),archive_SHA256=sha(archive),complete_copied_numeric_occurrences=self.occurrences,unique_full_NPY_values=len(self.arrays),original_full_values_shared_only_after_shape_dtype_and_every_raw_bit_equal=True);(group/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        assert 1<=len(self.steps)==len(complete_generated)<=2048;first=next((j for j,t in enumerate(complete_generated) if t==128001),None)
        execution=dict(index=i,prompt=row['prompt'],actual_model_prompt=prompt,media_path=media_path,media_SHA256=sha(media_path),raw_output=raw,generation_token_ids=complete_generated,decoded_token_ids=complete_generated,first_EOS_index=first,generation_count=len(complete_generated),native_forward_count=len(self.steps),truncated=first is None and len(self.steps)>=2048,empty=not raw,error=None,group_execution=str(group/'execution.json'),group_execution_SHA256=sha(group/'execution.json'),actual_device=str(self.device))
        from universal_eval.algorithm.native_explicit_mc_v2 import explicit_choice
        execution['invalid']=explicit_choice(raw) is None;self.details[i]=execution
        with self.trace.open('a',encoding='utf-8') as file:file.write(json.dumps(execution,ensure_ascii=False)+'\n')
        self.arrays={};self.previous_KV={};log.info('Own Xtuner complete sample %d, native forwards %d',i+1,len(self.steps));return raw
    def batch_predict(self,batch):return [self.predict(prompt,media) for prompt,media in batch]
    def get_sample_execution(self,prompt,media_path):
        e=self.details[self.report_index];assert e['media_path']==media_path and prompt.strip() in [e['prompt'].strip(),e['actual_model_prompt'].strip()];self.report_index+=1;return e
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
