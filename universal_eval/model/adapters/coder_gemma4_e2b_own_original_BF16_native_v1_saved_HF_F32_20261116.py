"""Own Gemma4 E2B BF16 native model, original adapter branch and Greedy2048.
All 1951 own BF16 unquantized checkpoint tensors are active and accounted for.
This is the saved HF default-template/F32-input branch, not a native4bit claim. No downloaded packages,
borrowed model assets or CPU score. Historical vLLM engine bytes were not saved.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,json,hashlib,logging,inspect,importlib,re,shutil,subprocess,traceback
import numpy as np
import torch,torch_npu
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig,AutoProcessor,AutoModelForImageTextToText,GenerationConfig
from universal_eval.model.coder_gemma4_e2b_own_20261116.observer import Gemma4LosslessObserver,arr,bs,save_npz,write,sha,tsha
MID='coder3101/gemma-4-E2B-it-qat-q4_0-unquantized-heretic';LOG=logging.getLogger(__name__)
def js(p):return json.loads(Path(p).read_bytes())
def jsonable_loading_info(value):
    """Lossless deterministic normalization of HF loading metadata, not tensors."""
    if isinstance(value,dict):return {str(k):jsonable_loading_info(v) for k,v in value.items()}
    if isinstance(value,(set,frozenset)):return sorted(jsonable_loading_info(v) for v in value)
    if isinstance(value,(list,tuple)):return [jsonable_loading_info(v) for v in value]
    if value is None or isinstance(value,(str,bool,int,float)):return value
    raise TypeError('Unreviewed loading metadata type: '+type(value).__name__)
def load_own_model(W,p,device):
    c=AutoConfig.from_pretrained(str(W),local_files_only=True);assert c.dtype==c.text_config.dtype==c.vision_config.dtype==torch.bfloat16
    assert (c.text_config.num_hidden_layers,c.text_config.num_kv_shared_layers,c.text_config.hidden_size,c.text_config.vocab_size,c.text_config.head_dim,c.text_config.global_head_dim)==(35,20,1536,262144,256,512)
    assert c.text_config.use_bidirectional_attention is None and c.text_config.sliding_window==512 and c.vision_config.num_hidden_layers==16
    for n,v in p['own_weight_files'].items():assert (W/n).stat().st_size==v['bytes'] and sha(W/n)==v['SHA256']
    model,info=AutoModelForImageTextToText.from_pretrained(str(W),config=c,local_files_only=True,dtype=torch.bfloat16,output_loading_info=True)
    unused=set()
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs') and set(info.get('unexpected_keys',[]))<=unused,info
    hdr=js(p['own_source_headers']);persistent=set(model.state_dict());assert set(hdr)-persistent==unused and persistent-set(hdr)=={'lm_head.weight'} and model.lm_head.weight is model.model.language_model.embed_tokens.weight
    cpu_buffers={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tsha(t)) for n,t in model.named_buffers() if n not in persistent}
    model=model.eval().to(device);state=model.state_dict();active={};ignored={}
    for shard in p['own_weight_files']:
        with safe_open(str(W/shard),framework='pt',device='cpu') as sf:
            assert set(sf.keys())==set(hdr)
            for n in sf.keys():
                raw=sf.get_tensor(n).contiguous();assert raw.dtype==torch.bfloat16 and list(raw.shape)==hdr[n]['shape'];rec=dict(name=n,shape=list(raw.shape),source_dtype=str(raw.dtype),source_SHA256=tsha(raw),source_file=shard)
                if n in unused:ignored[n]=rec;continue
                target=state[n];assert target.device==device and target.dtype==torch.bfloat16 and target.shape==raw.shape and tsha(target)==rec['source_SHA256'],n
                active[n]=dict(rec,actual_NPU_device=str(target.device),actual_NPU_SHA256=tsha(target))
    assert len(active)==1951 and len(ignored)==0 and set(active)|set(ignored)==set(hdr)
    buffers={}
    for n,t in model.named_buffers():
        if n not in persistent:assert t.device==device and tsha(t)==cpu_buffers[n]['SHA256'];buffers[n]=dict(cpu_buffers[n],actual_NPU_SHA256=tsha(t))
    assert all(t.device==device for t in model.parameters())
    model.generation_config=GenerationConfig.from_pretrained(str(W),local_files_only=True);assert model.generation_config.eos_token_id==[1,106,50] and model.generation_config.pad_token_id==0
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight and tuple(model.lm_head.weight.shape)==(262144,1536)
    proof=dict(model_id=MID,active_weights=active,original_unused_shared_KV_source=ignored,own_output_head_alias=dict(source_name='model.language_model.embed_tokens.weight',target_name='lm_head.weight',tied_parameter_identity=True,actual_NPU_SHA256=tsha(model.lm_head.weight)),CPU_initializer_buffers=cpu_buffers,native_config_derived_nonpersistent_buffers=buffers,own_generation_config=model.generation_config.to_dict(),original_HF_loading_information=jsonable_loading_info(info),actual_native_aliases=[['model.language_model.embed_tokens.weight','lm_head.weight']],own_all_clipped_linear_statistics_preserved=True,learned_values_only_from_own_checkpoint=True,no_A100_tensor_capture_claimed=True)
    return model,proof
class CoderGemma4E2BOwnBF16(Gemma4LosslessObserver):
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
        stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('CoderGemma4E2B_own_original_BF16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
        self.front=[json.loads(s) for s in Path(self.p['own_frontend200']).read_text().splitlines()];assert len(self.front)==200
        self.processor=AutoProcessor.from_pretrained(str(self.W),local_files_only=True);self.tokenizer=self.processor.tokenizer;assert self.processor.chat_template and self.processor.image_processor.max_soft_tokens==280
        self.model,proof=load_own_model(self.W,self.p,self.device);self.weights=self.home/'own1951_all_source_actual_NPU_unused0.json';write(self.weights,proof)
        self.stop_ids=[1,106,50];self.outer_batches=[];self.outer_dispatches=[];self.audio_tower_forward_calls=0;self.index=self.report_index=0;self.details=[];self.handles=[];self.active=self.failed=self.profile_done=False;self.profiler=None;self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self._install()
        libraries={}
        for name in ['torch','torch_npu','transformers','transformers.models.gemma4.modeling_gemma4','transformers.models.gemma4.processing_gemma4','transformers.cache_utils','transformers.masking_utils','transformers.generation.utils','transformers.generation.logits_process','transformers.activations']:
            path=inspect.getfile(importlib.import_module(name));libraries[path]=sha(path)
        self.lineage=dict(hardware='Ascend910B',devices=[4],runtime='Own saved HF BF16 native Gemma4, original default template/F32 inputs/Greedy2048',runtime_dtype='BF16',configured_outer_batch_size=10,actual_inner_batch_size=1,max_new_tokens=2048,native_config_position_limit=131072,original_explicit_max_model_len=None,actual_input_floating_dtype="float32",use_cache=True,cache_implementation='own native Gemma4 physical15 plus shared20 dynamic/sliding KV for saved vLLM persistent cache',EOS=self.stop_ids,head_size=262144,LM_layers=35,vision_layers=16,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha256,evidence_root=str(self.home),weight_evidence=str(self.weights),weight_evidence_SHA256=sha(self.weights),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),profiler_scope='First actual own vision and native prefill only',runtime_library_source_SHA256=libraries,own_template_used=True,historical_source_boundary=self.p['historical_source_boundary'],no_A100_tensor_capture_claimed=True)
    def _install(self):
        def no_audio(m,args,kw):
            self.audio_tower_forward_calls+=1;raise RuntimeError('Original image-only task may not execute the audio tower')
        self.handles.append(self.model.model.audio_tower.register_forward_pre_hook(no_audio,with_kwargs=True))
        def pre(m,args,kw):
            if not self.active:return
            self.frame=dict(step=len(self.frames),layers=[]);self.frame_refs={};self.shared_latest={}
            for k in ['input_ids','attention_mask','position_ids','cache_position','mm_token_type_ids','image_position_ids']:
                if torch.is_tensor(kw.get(k)):self.frame[k]=self._ref(kw[k],f'f{len(self.frames)}_'+k)
        def post(m,args,kw,out):
            if not self.active:return
            row=out.logits[0,-1];assert row.device==self.device and row.dtype==torch.bfloat16 and tuple(row.shape)==(262144,) and torch.isfinite(row).all()
            self.heads.append(arr(row));self.frame.update(head_dtype=str(row.dtype),full_head_shape=list(out.logits.shape),head_argmax=int(row.argmax().item()),native_cache_class=type(out.past_key_values).__name__,cache=self._capture_cache(out.past_key_values));self.frames.append(self.frame)
            if self.profiler is not None and len(self.frames)==1:self.profiler.stop();self.profiler=None;self.profile_done=True
        self.handles.extend([self.model.register_forward_pre_hook(pre,with_kwargs=True),self.model.register_forward_hook(post,with_kwargs=True)])
        for name,module in self.model.named_modules():
            if re.fullmatch(r'model.language_model.layers.\d+',name) or re.fullmatch(r'model.vision_tower.encoder.layers.\d+',name) or name in ['model.vision_tower','model.embed_vision']:
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
                    data=inspect.signature(m.forward).bind_partial(*args,**kw).arguments;rec=dict(name=name,shared_kv_from=m.self_attn.layer_type if m.self_attn.is_kv_shared_layer else None)
                    for k in ['attention_mask','position_embeddings','position_ids','per_layer_input']:rec[k]=self._ref(data.get(k),f'f{len(self.frames)}_'+name.replace('.','_')+'_'+k)
                    if m.self_attn.is_kv_shared_layer:
                        assert set(data['shared_kv_states'])=={'full_attention','sliding_attention'};self.shared_latest=dict(data['shared_kv_states'])
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
        inputs=self.processor(text=text,images=[Image.open(media_path).convert('RGB')],return_tensors='pt',padding=True)
        assert all(v.dtype==torch.float32 for v in inputs.values() if torch.is_tensor(v) and v.is_floating_point()),'Saved HF F32 processor inputs may not be rounded early'
        for k,v in inputs.items():assert tsha(v)==ref['inputs'][k]['SHA256'] and list(v.shape)==ref['inputs'][k]['shape'] and str(v.dtype)==ref['inputs'][k]['dtype']
        assert ref['generation']=={'max_new_tokens':2048,'do_sample':False,'temperature':0.0} and inputs['input_ids'].shape[1]+2048<=self.model.config.text_config.max_position_embeddings
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
            assert all(self.counts.get('model.language_model.layers.'+str(k))==n for k in range(35)) and all(self.counts.get('model.vision_tower.encoder.layers.'+str(k))==1 for k in range(16)) and self.counts.get('model.embed_vision')==1
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
