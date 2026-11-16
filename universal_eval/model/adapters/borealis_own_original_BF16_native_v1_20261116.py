"""Borealis own BF16 checkpoint, native Gemma3 SDPA, original Greedy2048.
All 883 original mixed-storage tensors are individually cast as the own BF16
configuration specifies, then whole-value checked on the NPU. No foreign model
weights, processor, vocabulary or CPU model score. Original historical vLLM
engine bytes were not captured; this boundary is retained in every report.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,logging,os,shutil,subprocess,inspect,importlib,re,traceback
import numpy as np
import torch,torch_npu
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig,AutoProcessor,GenerationConfig,Gemma3ForConditionalGeneration
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from universal_eval.model.borealis_own_20261116.observer import BorealisLosslessObserver,arr,bs,save_npz,write,sha,tsha
MID='NbAiLab/borealis-4b-instruct-preview';LOG=logging.getLogger(__name__)
def js(p):return json.loads(Path(p).read_bytes())
def checkpoint_name(name):
    for a,b in [('language_model.model.','model.language_model.'),('language_model.lm_head.','lm_head.'),('vision_tower.vision_model.','model.vision_tower.'),('multi_modal_projector.','model.multi_modal_projector.')]:
        if name.startswith(a):return b+name[len(a):]
    assert name.startswith(('model.language_model.','model.vision_tower.','model.multi_modal_projector.','lm_head.')),name
    return name
def load_own_model(W,p,device):
    cfg=AutoConfig.from_pretrained(str(W),local_files_only=True)
    assert cfg.dtype==cfg.text_config.dtype==cfg.vision_config.dtype==torch.bfloat16
    assert (cfg.text_config.num_hidden_layers,cfg.text_config.hidden_size,cfg.text_config.num_key_value_heads,cfg.text_config.head_dim,cfg.text_config.vocab_size)==(34,2560,4,256,262208)
    assert cfg.text_config.use_bidirectional_attention is False and cfg.text_config.sliding_window==1024
    cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
    with torch.device('meta'):model=Gemma3ForConditionalGeneration(cfg).to(dtype=torch.bfloat16)
    named=dict(model.named_parameters());index=js(W/'model.safetensors.index.json')['weight_map'];mapping={checkpoint_name(k):k for k in index};assert len(named)==len(index)==len(mapping)==883 and set(mapping)==set(named)
    records={}
    for name,rec in p['own_weight_files'].items():assert (W/name).stat().st_size==rec['bytes'] and sha(W/name)==rec['SHA256']
    for shard in sorted(set(index.values())):
        with safe_open(str(W/shard),framework='pt',device='cpu') as sf:
            assert set(sf.keys())=={k for k,v in index.items() if v==shard}
            for key in sorted(sf.keys()):
                value=sf.get_tensor(key).contiguous();name=checkpoint_name(key);assert value.dtype in [torch.float32,torch.bfloat16] and value.shape==named[name].shape
                cast=value.to(torch.bfloat16).contiguous();target=cast.to(device).contiguous();assert tsha(target)==tsha(cast) and torch.equal(target.cpu(),cast)
                parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False))
                records[key]=dict(source_name=key,target_name=name,source_dtype=str(value.dtype),shape=list(value.shape),source_SHA256=tsha(value),cast_dtype='torch.bfloat16',cast_SHA256=tsha(cast),actual_NPU_SHA256=tsha(target),actual_NPU_device=str(target.device),source_file=shard)
        LOG.info('Borealis own checkpoint loaded %s, %d/883 complete source/cast/NPU tensors',shard,len(records))
    model.tie_weights();assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    # Recreate the native analytic RoPE in its intrinsic FP32 constructor dtype.
    # Meta .to(BF16) must not quantize the original inverse frequencies.
    rope=Gemma3RotaryEmbedding(cfg.text_config)
    model.model.language_model.rotary_emb=rope
    model.model.language_model.embed_tokens.embed_scale=torch.tensor(2560**.5,dtype=torch.bfloat16)
    model.model.vision_tower.embeddings.position_ids=torch.arange(4096,dtype=torch.int64).expand((1,-1))
    buffers={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tsha(t)) for n,t in model.named_buffers()}
    assert len(buffers)==6 and not any(t.is_meta for t in model.buffers())
    model=model.to(device).eval();actual_buffers={}
    for n,t in model.named_buffers():assert t.device==device and tsha(t)==buffers[n]['SHA256'];actual_buffers[n]=dict(buffers[n],actual_NPU_SHA256=tsha(t))
    assert all(t.device==device and t.dtype==torch.bfloat16 for t in model.parameters())
    model.generation_config=GenerationConfig.from_pretrained(str(W),local_files_only=True);assert model.generation_config.eos_token_id==[106,1,106] and model.generation_config.pad_token_id==0
    aliases=[g for g in [['model.language_model.embed_tokens.weight','lm_head.weight']] if model.lm_head.weight is model.model.language_model.embed_tokens.weight]
    proof=dict(model_id=MID,active_weights=records,own_output_head_alias=dict(source_name=mapping['model.language_model.embed_tokens.weight'],target_name='lm_head.weight',tied_parameter_identity=True,actual_NPU_SHA256=tsha(model.lm_head.weight)),native_config_derived_nonpersistent_buffers=actual_buffers,CPU_initializer_buffers=buffers,actual_native_aliases=aliases,own_generation_config=model.generation_config.to_dict(),actual_attention_implementations={'text':cfg.text_config._attn_implementation,'vision':cfg.vision_config._attn_implementation},learned_values_only_from_own_checkpoint=True,no_A100_tensor_capture_claimed=True,own_original_mixed_storage_cast_to_own_config_BF16=True)
    return model,proof
class BorealisOwnBF16(BorealisLosslessObserver):
    @property
    def model_name(self):return MID
    def load(self,model_path,**kwargs):
        assert sha(self.protocol_file)==self.protocol_sha256;self.p=js(self.protocol_file);self.W=Path(model_path);assert self.p['model_id']==MID and str(self.W)==self.p['weight_path']
        for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
        for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
        for n,h in self.p['own_runtime_assets_SHA256'].items():assert sha(self.W/n)==h,n
        assert sha(self.p['dataset_path'])==self.p['dataset_SHA256'];q=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert q.returncode==0 and 'No running processes found in NPU 4' in q.stdout
        self.device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
        stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('Borealis4B_own_original_BF16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
        self.front=[json.loads(s) for s in Path(self.p['own_frontend200']).read_text().splitlines()];assert len(self.front)==200
        self.processor=AutoProcessor.from_pretrained(str(self.W),local_files_only=True);self.tokenizer=self.processor.tokenizer;assert self.processor.chat_template and not self.processor.image_processor.do_pan_and_scan
        self.model,proof=load_own_model(self.W,self.p,self.device);self.weights=self.home/'own883_source_cast_actual_NPU.json';write(self.weights,proof)
        self.stop_ids=[106,1,106];self.outer_batches=[];self.outer_dispatches=[];self.audio_tower_forward_calls=0;self.index=self.report_index=0;self.details=[];self.handles=[];self.active=self.failed=self.profile_done=False;self.profiler=None;self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self._install()
        libraries={}
        for name in ['torch','torch_npu','transformers','transformers.models.gemma3.modeling_gemma3','transformers.models.siglip.modeling_siglip','transformers.models.gemma3.processing_gemma3','transformers.cache_utils','transformers.masking_utils','transformers.generation.utils','transformers.generation.logits_process','transformers.activations']:
            path=inspect.getfile(importlib.import_module(name));libraries[path]=sha(path)
        self.lineage=dict(hardware='Ascend910B',devices=[4],runtime='Own original BF16 native HF Gemma3 SDPA, original own task and Greedy2048',runtime_dtype='BF16',configured_outer_batch_size=10,actual_inner_batch_size=1,max_new_tokens=2048,max_request_model_length=16384,use_cache=True,cache_implementation='own native Gemma3 dynamic/sliding KV for saved vLLM persistent cache',EOS=self.stop_ids,head_size=262208,LM_layers=34,vision_layers=27,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha256,evidence_root=str(self.home),weight_evidence=str(self.weights),weight_evidence_SHA256=sha(self.weights),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),profiler_scope='First actual own vision and native prefill only',runtime_library_source_SHA256=libraries,own_template_used=True,historical_source_boundary=self.p['historical_source_boundary'],no_A100_tensor_capture_claimed=True)
    def _install(self):
        def root_pre(m,args,kw):
            if not self.active:return
            self.frame=dict(step=len(self.frames),layers=[]);self.frame_refs={}
            for k in ['input_ids','attention_mask','position_ids','cache_position','token_type_ids']:
                if torch.is_tensor(kw.get(k)):self.frame[k]=self._ref(kw[k],f'f{len(self.frames)}_'+k)
        def root_post(m,args,kw,out):
            if not self.active:return
            row=out.logits[0,-1];assert row.device==self.device and row.dtype==torch.bfloat16 and tuple(row.shape)==(262208,)
            assert torch.isfinite(row).all(),'Nonfinite own BF16 head; preserve failed original stage'
            self.heads.append(arr(row));self.frame.update(head_dtype=str(row.dtype),full_head_shape=list(out.logits.shape),head_argmax=int(row.argmax().item()),native_cache_class=type(out.past_key_values).__name__,cache=self._capture_cache(out.past_key_values));self.frames.append(self.frame)
            if self.profiler is not None and len(self.frames)==1:self.profiler.stop();self.profiler=None;self.profile_done=True
        self.handles.extend([self.model.register_forward_pre_hook(root_pre,with_kwargs=True),self.model.register_forward_hook(root_post,with_kwargs=True)])
        for name,module in self.model.named_modules():
            if re.fullmatch(r'model.language_model.layers.\d+',name) or re.fullmatch(r'model.vision_tower.encoder.layers.\d+',name) or name in ['model.vision_tower','model.multi_modal_projector']:
                def stage(m,args,kw,out,name=name):
                    if not self.active:return
                    self.counts[name]=self.counts.get(name,0)+1
                    if name in self.seen:return
                    self.seen.add(name);t=out.last_hidden_state if hasattr(out,'last_hidden_state') else out[0] if isinstance(out,(list,tuple)) else out
                    assert torch.is_tensor(t) and torch.isfinite(t).all(),name
                    key='stage_'+name.replace('.','_');self.stages[key]=self._record(key,t,self.index<3)
                self.handles.append(module.register_forward_hook(stage,with_kwargs=True))
            if re.fullmatch(r'model.language_model.layers.\d+',name):
                def layer_pre(m,args,kw,name=name):
                    if not self.active:return
                    data=inspect.signature(m.forward).bind_partial(*args,**kw).arguments;rec=dict(name=name)
                    for k in ['attention_mask','position_embeddings','position_ids']:rec[k]=self._ref(data.get(k),f'f{len(self.frames)}_'+name.replace('.','_')+'_'+k)
                    self.frame['layers'].append(rec)
                self.handles.append(module.register_forward_pre_hook(layer_pre,with_kwargs=True))
        def text_pre(m,args,kw):
            if self.active and not self.frames:
                for k,v in kw.items():
                    if torch.is_tensor(v):self.stages['injection_'+k]=self._record('injection_'+k,v,self.index<3)
        def embeds(m,args,out):
            if self.active and not self.frames:self.stages['original_text_embeds_before_multimodal_scatter']=self._record('original_text_embeds_before_multimodal_scatter',out,self.index<3)
        def head_pre(m,args):
            if self.active:self.frame['own_head_input']=self._ref(args[0][0,-1],f'f{len(self.frames)}_own_head_input')
        self.handles.extend([self.model.model.language_model.register_forward_pre_hook(text_pre,with_kwargs=True),self.model.model.language_model.embed_tokens.register_forward_hook(embeds),self.model.lm_head.register_forward_pre_hook(head_pre)])

    def batch_predict(self, batch):
        assert 1 <= len(batch) <= 10
        self.outer_batches.append(len(batch))
        dispatch = dict(start_index=self.index, requested_count=len(batch),
                        requested_indices=list(range(self.index, self.index+len(batch))), status='started')
        self.outer_dispatches.append(dispatch)
        try:
            result = [self.predict(prompt, media) for prompt, media in batch]
            dispatch.update(status='completed', completed_count=len(result))
            return result
        except BaseException:
            dispatch.update(status='failed', committed_count=self.index-dispatch['start_index'], error=traceback.format_exc())
            raise

    def get_execution_lineage(self):
        # The pipeline collects this before inference. Completed mutable facts
        # belong in the referenced final manifest, not an early empty snapshot.
        self.lineage['actual_batch_manifest'] = self.lineage['final_trace_manifest']
        return self.lineage

    def get_sample_execution(self, prompt, media_path):
        if self.report_index == 0:
            paths = [p for p in (self.home/'profiler').rglob('*') if p.is_file()]
            profile = {str(p):sha(p) for p in paths}
            completed = [rec for rec in self.outer_dispatches if rec['status']=='completed']
            write(self.home/'completed_trace.json', dict(samples=self.index,failed=self.failed,
                caption_audio_tower_forward_calls=self.audio_tower_forward_calls,
                sample_trace_SHA256=sha(self.trace), first_real_vision_prefill_all_profiler_files_SHA256=profile,
                outer_batch_dispatches=self.outer_dispatches,
                actual_outer_batch_sizes=[rec['requested_count'] for rec in completed],
                actual_outer_batch_indices=[rec['requested_indices'] for rec in completed],
                actual_single_model_forward_batch_size=1))
        i=self.report_index;self.report_index+=1
        assert i<len(self.details)
        return self.details[i]

    def _predict_original_compute_and_capture(self,prompt,media_path=None):
        i=self.index;ref=self.front[i];text=ref['formatted_text'];assert prompt==ref['original_model_prompt'] and sha(media_path)==ref['image_SHA256'] and text==ref['formatted_text']
        inputs=self.processor(text=text,images=[Image.open(media_path).convert('RGB')],return_tensors='pt',padding=True,add_special_tokens=False,return_mm_token_type_ids=True)
        inputs={k:v.to(dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in inputs.items()}
        for k,v in inputs.items():assert tsha(v)==ref['inputs'][k]['SHA256'] and list(v.shape)==ref['inputs'][k]['shape'] and str(v.dtype)==ref['inputs'][k]['dtype']
        assert ref['generation']=={'max_new_tokens':2048,'do_sample':False} and inputs['input_ids'].shape[1]+2048<=16384
        reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
        self.arrays={};self.frames=[];self.heads=[];self.stages={};self.counts={};self.seen=set();self.cache_previous={};self.cache_cpu={};self.cache_members={};self.active=True
        try:
            actual={k:v.to(self.device) for k,v in inputs.items()}
            for k,v in actual.items():self._record('input_'+k,v)
            if not self.profile_done:
                self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')),record_shapes=True);self.profiler.start()
            with torch.inference_mode():generated=self.model.generate(**actual,**ref['generation'])
            self.active=False;tokens=generated[0].cpu().tolist();prefix=inputs['input_ids'][0].tolist();n=len(self.frames);assert tokens[:len(prefix)]==prefix and len(tokens)-len(prefix)==n and 0<n<=2048
            assert not set(tokens[len(prefix):-1]).intersection(self.stop_ids) and (n==2048 or tokens[-1] in self.stop_ids)
            raw=self.processor.batch_decode(generated[:,len(prefix):],skip_special_tokens=True)[0].strip();self.arrays['all_original_full_head_logits']=np.stack(self.heads);self.arrays['generated_tokens']=np.array(tokens,dtype=np.int64)
            final={str(g):dict(actual_NPU_SHA256=tsha(t),reconstructed_complete_SHA256=bs(self.cache_cpu[g])) for g,t in self.cache_previous.items()};assert all(x['actual_NPU_SHA256']==x['reconstructed_complete_SHA256'] for x in final.values())
            assert all(self.counts.get('model.language_model.layers.'+str(k))==n for k in range(34)) and all(self.counts.get('model.vision_tower.encoder.layers.'+str(k))==1 for k in range(27)) and self.counts.get('model.multi_modal_projector')==1
            size=sum(a.nbytes for a in self.arrays.values());assert shutil.disk_usage(self.home).free>reserve+size
            archive=self.home/f'sample_{i:05d}.npz';save_npz(archive,self.arrays)
            record=dict(index=i,actual_model_prompt=prompt,actual_model_text=text,media_path=str(media_path),media_SHA256=ref['image_SHA256'],reference=ref['reference'],input_ids=prefix,prompt_length=len(prefix),generated_tokens=tokens,raw_output=raw,error=None,actual_native_forwards=n,frames=self.frames,activation_prefill=self.stages,activation_counts=self.counts,cache_layout=self.cache_layout,cache_group_members={str(g):v for g,v in self.cache_members.items()},final_native_cache_groups_SHA256=final,original_budget_hit=n==2048 and tokens[-1] not in self.stop_ids,stop_ids=self.stop_ids,archive=str(archive),archive_SHA256=sha(archive),archive_bytes=archive.stat().st_size,lossless_uncompressed_bytes=size)
            rp=self.home/f'sample_{i:05d}.json';write(rp,record);detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=n,generated_tokens=tokens,raw_output=raw,error=None)
            with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
            self.details.append(detail);self.index+=1;self.arrays={};self.heads=[];self.cache_previous={};self.cache_cpu={};LOG.info('Gemma3n actual sample %d/200, native original2048 forwards=%d, exact lossless arrays=%d bytes',self.index,n,archive.stat().st_size);return raw
        except BaseException:
            self.active=False;self.failed=True
            if self.profiler is not None:self.profiler.stop();self.profiler=None
            error=traceback.format_exc();rp=self.home/f'sample_{i:05d}_failed.json';observed_tokens=locals().get('tokens');observed_raw=locals().get('raw');partial_archive=None;partial_archive_error=None
            try:
                if self.heads:self.arrays['all_partial_original_full_head_logits']=np.stack(self.heads)
                if observed_tokens is not None:self.arrays['observed_generated_tokens']=np.asarray(observed_tokens,dtype=np.int64)
                if self.arrays:
                    partial_archive=self.home/f'sample_{i:05d}_failed_partial_values.npz';save_npz(partial_archive,self.arrays)
            except BaseException:partial_archive_error=traceback.format_exc()
            failure=dict(index=i,actual_model_prompt=prompt,actual_model_text=text,media_path=str(media_path),media_SHA256=ref['image_SHA256'],reference=ref['reference'],input_ids=inputs['input_ids'].tolist(),generated_tokens=observed_tokens,raw_output=observed_raw,committed_native_forwards=len(self.frames),frames=self.frames,activation_prefill=self.stages,activation_counts=self.counts,error=error,partial_archive=str(partial_archive) if partial_archive and partial_archive.exists() else None,partial_archive_SHA256=sha(partial_archive) if partial_archive and partial_archive.exists() else None,partial_archive_write_error=partial_archive_error,qualification_rejected=True)
            write(rp,failure);detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(self.frames),generated_tokens=observed_tokens,raw_output=observed_raw,error=error)
            with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
            self.details.append(detail);self.index+=1;raise
