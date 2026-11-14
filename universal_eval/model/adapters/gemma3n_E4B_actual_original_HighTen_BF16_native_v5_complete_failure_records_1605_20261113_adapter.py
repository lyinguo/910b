"""Own E4B original HighTen BF16/2048/serial1 deployment.

Uses unchanged native Gemma3n text/audio functions and the original weight-free
MobileNet operator source; every learned value is loaded from the own original
checkpoint. This is reference BF16 inference, not a native low-bit claim.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,os,logging,re,inspect,shutil,subprocess,traceback,zipfile,csv,importlib
import numpy as np
import torch,torch_npu
from PIL import Image
from transformers import AutoProcessor
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.gemma3n_E4B_HighTen_own_parameterized_BF16_vision_native_v1_after_reboot_20261113 import load_own_original_native_model,tsha,sha
MID='google/gemma-3n-E4B';LOG=logging.getLogger(__name__)
def js(p):return json.loads(Path(p).read_bytes())
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False)
def arr(t):
    t=t.detach().cpu().contiguous();return t.view(torch.uint16).numpy() if t.dtype==torch.bfloat16 else t.numpy()
def bs(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save_npz(p,values):
    with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for k,a in values.items():
            with z.open(k+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(a),allow_pickle=False)
def record_pre_generation_failure(home,trace,index,prompt,media_path,reference,error):
    """Record a real rejected request without claiming tokens or NPU forwards."""
    path=Path(home)/f'sample_{index:05d}_failed_before_NPU_generation.json'
    failure=dict(index=index,actual_model_prompt=prompt,media_path=str(media_path),reference=reference,generated_tokens=None,raw_output=None,error=error,actual_native_forwards=0,genuine_NPU_forward_started=False,qualification_rejected=True,failure_phase='before_original_generate')
    write(path,failure)
    detail=dict(index=index,actual_execution_file=str(path),actual_execution_SHA256=sha(path),actual_model_prompt=prompt,actual_native_forwards=0,generated_tokens=None,raw_output=None,error=error)
    with Path(trace).open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
    return detail

class Gemma3nOriginalHighTenE4B(IModel):
    def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
        assert protocol['model_id']==MID and npu_devices==[4]
        self.protocol_file=Path(protocol_file);self.protocol_sha256=protocol_sha256;self.devices=npu_devices
    @property
    def model_name(self):return MID
    @property
    def model_type(self):return 'vlm'
    @property
    def inference_framework(self):return 'Transformers'
    def download(self,*args,**kwargs):raise RuntimeError('Only the own original existing weights are authorized')
    def load(self,model_path,**kwargs):
        assert sha(self.protocol_file)==self.protocol_sha256;self.p=js(self.protocol_file);self.W=Path(model_path);assert str(self.W)==self.p['weight_path'] and self.p['model_id']==MID
        for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
        for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
        for n,h in self.p['own_runtime_assets_SHA256'].items():assert sha(self.W/n)==h,n
        assert sha(self.p['dataset_path'])==self.p['dataset_SHA256']
        query=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert query.returncode==0 and 'No running processes found in NPU 4' in query.stdout
        self.device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
        stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('Gemma3nE4B_actual_original_Greedy_HighTen_BF16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
        self.front=js(self.p['actual_AA_frontend200'])['samples'];assert len(self.front)==200 and js(self.p['independent_R_CPU200'])['passed']
        self.processor=AutoProcessor.from_pretrained(str(self.W),local_files_only=True,trust_remote_code=True);self.tokenizer=self.processor.tokenizer
        self.image_token=self.tokenizer.decode([js(self.W/'config.json')['image_token_id']],skip_special_tokens=False);assert not self.tokenizer.chat_template
        self.model,weight_proof=load_own_original_native_model(self.W,self.p['vision_graph'],self.p['vision_graph_layout'],self.p['actual_AA_control_dir'],self.device,self.p['original_native_buffers'],None)
        self.weights=self.home/'own1631_source_cast_actual_NPU_and_unused45_source.json';write(self.weights,weight_proof)
        assert self.model.generation_config.eos_token_id==1 and self.model.generation_config.pad_token_id==0 and self.model.generation_config.cache_implementation=='hybrid'
        self.audio_tower_forward_calls=0;self.index=0;self.report_index=0;self.details=[];self.handles=[];self.active=False;self.failed=False;self.profiler=None;self.profile_done=False
        self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self._install()
        self.lineage=dict(hardware='Ascend910B',devices=self.devices,runtime='Own original native Gemma3n text/audio plus original parameterized MobileNet eval operators',runtime_dtype='BF16',actual_outer_batch_size=1,actual_inner_batch_size=1,max_new_tokens=2048,original_generate_overrides={'max_new_tokens':2048,'do_sample':False},use_cache=True,cache_implementation='hybrid',EOS=[1],head_size=262400,per_layer_input_vocab=262144,LM_layers=35,KV_shared_layers=list(range(20,35)),protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha256,evidence_root=str(self.home),weight_evidence=str(self.weights),weight_evidence_SHA256=sha(self.weights),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),first_real_vision_and_prefill_profiler_dir=str(self.home/'profiler'),profiler_scope='First original real vision and full native text prefill only',activation_retention='All first-prefill native35 LM, own4 original vision stages, projector and actual input injection values for first3; all full heads and actual masks/positions; every physical/native shared KV buffer losslessly preserved via complete initial values plus exact verified growth/roll/changed-slot updates and final full SHA',historical_source_boundary=self.p['historical_source_boundary'],original_report_task='image-text-to-text',workbook_B_task='image-text-to-text',no_model_data_package_download_or_fake_timm=True)
        libraries={}
        for name in ['torch','torch_npu','transformers','transformers.models.gemma3n.modeling_gemma3n','transformers.models.gemma3n.processing_gemma3n','transformers.cache_utils','transformers.masking_utils','transformers.generation.utils','transformers.generation.logits_process','transformers.activations']:
            path=inspect.getfile(importlib.import_module(name));libraries[path]=sha(path)
        self.lineage['runtime_library_source_SHA256']=libraries
    def _record(self,name,t,retain=True):
        assert t.device==self.device;v=t.detach().cpu().contiguous();meta=dict(shape=list(v.shape),dtype=str(v.dtype),SHA256=tsha(v))
        if retain:assert name not in self.arrays;self.arrays[name]=arr(v)
        return meta
    def _ref(self,t,name):
        if t is None:return None
        if isinstance(t,(tuple,list)):return [self._ref(x,name+'_'+str(i)) for i,x in enumerate(t)]
        assert torch.is_tensor(t);ident=(id(t),t.data_ptr(),tuple(t.shape),str(t.dtype))
        if ident not in self.frame_refs:self.frame_refs[ident]=dict(array=name,**self._record(name,t))
        return self.frame_refs[ident]
    def _capture_cache(self,cache):
        assert cache is not None;entries={};groups={}
        for i,layer in enumerate(cache.layers):
            for field in ['keys','values']:
                t=getattr(layer,field,None)
                if torch.is_tensor(t) and t.numel():
                    assert t.device==self.device and t.dtype==torch.bfloat16 and t.shape[:2]==(1,2) and t.shape[-1]==256
                    name=f'physical_{i}_{field}';kind=(type(layer).__name__,'physical');groups.setdefault((kind,tuple(t.shape)),[]).append((name,t));entries[name]=dict(shape=list(t.shape),cache_class=type(cache).__name__,layer_class=type(layer).__name__,dtype=str(t.dtype))
        assert len(entries)==40
        for i,(key,value) in sorted(self.shared_latest.items()):
            assert i in [18,19]
            for field,t in [('keys',key),('values',value)]:
                name=f'shared_full_native_{i}_{field}';groups.setdefault((('shared_full',str(i)),tuple(t.shape)),[]).append((name,t));entries[name]=dict(shape=list(t.shape),cache_class='original_forward_shared_kv_states',layer_class=str(i),dtype=str(t.dtype))
        assert len(self.shared_latest)==2 and len(entries)==44
        ordered=sorted(groups,key=str);layout=[([n for n,t in groups[k]],k[0]) for k in ordered]
        if not self.frames:self.cache_layout=entries;self.cache_members={g:names for g,(names,kind) in enumerate(layout)};self.cache_kinds=[kind for names,kind in layout]
        assert {g:names for g,(names,kind) in enumerate(layout)}==self.cache_members and [kind for names,kind in layout]==self.cache_kinds
        current={};observations=[]
        for g,k in enumerate(ordered):
            packed=torch.stack([t for n,t in groups[k]]).contiguous();current[g]=packed;prefix=f'f{len(self.frames)+1}_cache_{g}'
            if not self.frames:
                a=arr(packed);self.arrays[prefix+'_initial']=a;self.cache_cpu[g]=a.copy();obs=dict(group=g,kind='initial',array=prefix+'_initial',SHA256=bs(a),shape=list(a.shape),dtype=str(packed.dtype))
            else:
                prev=self.cache_previous[g];growth=packed.shape[:3]==prev.shape[:3] and packed.shape[4:]==prev.shape[4:] and packed.shape[3]==prev.shape[3]+1
                if growth and torch.equal(packed[:,:,:,:-1].contiguous().view(torch.int16),prev.view(torch.int16)):
                    a=arr(packed[:,:,:,-1:]);self.arrays[prefix+'_append']=a;self.cache_cpu[g]=np.concatenate([self.cache_cpu[g],a],axis=3);obs=dict(group=g,kind='native_length_growth_one',array=prefix+'_append',SHA256=bs(a),shape=list(packed.shape),all_previous_NPU_values_checked_bit_identical=True)
                elif packed.shape==prev.shape and packed.shape[3]>1 and torch.equal(packed[:,:,:,:-1].contiguous().view(torch.int16),prev[:,:,:,1:].contiguous().view(torch.int16)):
                    a=arr(packed[:,:,:,-1:]);self.arrays[prefix+'_roll_append']=a;self.cache_cpu[g]=np.concatenate([self.cache_cpu[g][:,:,:,1:],a],axis=3);obs=dict(group=g,kind='native_exact_sliding_roll_one',array=prefix+'_roll_append',SHA256=bs(a),shape=list(packed.shape),all_retained_NPU_window_values_checked_bit_identical=True)
                elif packed.shape!=prev.shape:
                    a=arr(packed);self.arrays[prefix+'_complete_shape_change']=a;self.cache_cpu[g]=a.copy();obs=dict(group=g,kind='complete_native_shape_change',array=prefix+'_complete_shape_change',SHA256=bs(a),shape=list(packed.shape))
                else:
                    different=packed.reshape(-1).view(torch.int16)!=prev.reshape(-1).view(torch.int16);ids=torch.nonzero(different,as_tuple=False).flatten();assert ids.numel()==int(different.sum().item());obs=dict(group=g,kind='exact_changed_slots',changed_count=ids.numel(),shape=list(packed.shape),all_NPU_values_checked=True)
                    if ids.numel():
                        ix=ids.cpu().numpy();a=arr(packed.reshape(-1).index_select(0,ids));self.arrays[prefix+'_indices']=ix;self.arrays[prefix+'_values']=a;self.cache_cpu[g].reshape(-1)[ix]=a;obs.update(indices_array=prefix+'_indices',values_array=prefix+'_values',indices_SHA256=bs(ix),values_SHA256=bs(a))
                obs['reconstructed_complete_SHA256']=bs(self.cache_cpu[g])
            observations.append(obs)
        self.cache_previous=current;return observations
    def _install(self):
        def reject_unexpected_audio_tower(m,args,kw):
            self.audio_tower_forward_calls+=1
            raise RuntimeError('Original image-only HighTen inputs never invoke audio_tower; preserve failure if this changes')
        self.handles.append(self.model.model.audio_tower.register_forward_pre_hook(reject_unexpected_audio_tower,with_kwargs=True))
        def pre(m,args,kw):
            if not self.active:return
            self.frame=dict(step=len(self.frames),layers=[]);self.frame_refs={};self.shared_latest={}
            for key in ['input_ids','attention_mask','position_ids','cache_position','token_type_ids']:
                if torch.is_tensor(kw.get(key)):self.frame[key]=self._ref(kw[key],f'f{len(self.frames)+1}_'+key)
        def post(m,args,kw,out):
            if not self.active:return
            row=out.logits[0,-1];assert row.device==self.device and row.dtype==torch.bfloat16 and row.shape==(262400,);self.heads.append(arr(row));self.frame.update(head_dtype=str(row.dtype),full_head_shape=list(out.logits.shape),head_argmax=int(row.argmax().item()),native_cache_class=type(out.past_key_values).__name__,cache=self._capture_cache(out.past_key_values));self.frames.append(self.frame)
            if self.profiler is not None and len(self.frames)==1:self.profiler.stop();self.profiler=None;self.profile_done=True
        self.handles.extend([self.model.register_forward_pre_hook(pre,with_kwargs=True),self.model.register_forward_hook(post,with_kwargs=True)])
        def vision(values,names):
            if not self.active:return
            for name,t in zip(['last_hidden_state',*names],values):self.stages['vision_'+name]=self._record('vision_'+name.replace('.','_'),t,self.index<3)
        self.model.model.vision_tower.observer=vision
        for name,module in self.model.named_modules():
            if re.fullmatch(r'model.language_model.layers.\d+',name) or name=='model.embed_vision':
                def stage(m,args,kw,out,name=name):
                    if not self.active:return
                    self.counts[name]=self.counts.get(name,0)+1
                    if name=='model.embed_vision' and torch.is_tensor(kw.get('inputs_embeds')):
                        assert 'vision_projected' not in self.stages;self.stages['vision_projected']=self._record('vision_projected',out,self.index<3)
                    if name in self.seen:return
                    self.seen.add(name);t=out[0] if isinstance(out,(list,tuple)) else out;key='stage_'+name.replace('.','_');self.stages[key]=self._record(key,t,self.index<3)
                self.handles.append(module.register_forward_hook(stage,with_kwargs=True))
            if re.fullmatch(r'model.language_model.layers.\d+',name):
                def layer_pre(m,args,kw,name=name):
                    if not self.active:return
                    data=inspect.signature(m.forward).bind_partial(*args,**kw).arguments;record=dict(name=name,shared_kv_from=m.self_attn.kv_shared_layer_index if m.self_attn.is_kv_shared_layer else None)
                    for k in ['attention_mask','position_embeddings','position_ids','per_layer_input']:
                        if k in data:record[k]=self._ref(data[k],f'f{len(self.frames)+1}_'+name.replace('.','_')+'_'+k)
                    if m.self_attn.is_kv_shared_layer:
                        values=data['shared_kv_states'];assert set(values)=={18,19};self.shared_latest=dict(values);record['shared_kv_storage']={str(k):[dict(pointer=t.data_ptr(),shape=list(t.shape),dtype=str(t.dtype)) for t in pair] for k,pair in values.items()}
                    self.frame['layers'].append(record)
                self.handles.append(module.register_forward_pre_hook(layer_pre,with_kwargs=True))
        def text_input(m,args,kw):
            if self.active and not self.frames:
                for k,v in kw.items():
                    if torch.is_tensor(v):self.stages['injection_'+k]=self._record('injection_'+k,v,self.index<3)
        self.handles.append(self.model.model.language_model.register_forward_pre_hook(text_input,with_kwargs=True))
        def token_embeds(m,args,out):
            if self.active and not self.frames:self.stages['original_text_embeds_before_multimodal_scatter']=self._record('original_text_embeds_before_multimodal_scatter',out,self.index<3)
        self.handles.append(self.model.model.language_model.embed_tokens.register_forward_hook(token_embeds))
    def _predict_original_compute_and_capture(self,prompt,media_path=None):
        i=self.index;ref=self.front[i];text='<image_soft_token>\n'+prompt;assert prompt==ref['original_model_prompt'] and sha(media_path)==ref['image_SHA256'] and text==ref['formatted_text']
        inputs=self.processor(text=text,images=[Image.open(media_path).convert('RGB')],return_tensors='pt',padding=True)
        inputs={k:v.to(dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in inputs.items()}
        for k,v in inputs.items():assert tsha(v)==ref['inputs'][k]['SHA256'] and list(v.shape)==ref['inputs'][k]['shape'] and str(v.dtype)==ref['inputs'][k]['dtype']
        assert ref['generation']=={'max_new_tokens':2048,'do_sample':False}
        reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
        self.arrays={};self.frames=[];self.heads=[];self.stages={};self.counts={};self.seen=set();self.cache_previous={};self.cache_cpu={};self.cache_members={};self.active=True
        try:
            actual={k:v.to(self.device) for k,v in inputs.items()}
            for k,v in actual.items():self._record('input_'+k,v)
            if not self.profile_done:
                self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')),record_shapes=True);self.profiler.start()
            with torch.inference_mode():generated=self.model.generate(**actual,**ref['generation'])
            self.active=False;tokens=generated[0].cpu().tolist();prefix=inputs['input_ids'][0].tolist();n=len(self.frames);assert tokens[:len(prefix)]==prefix and len(tokens)-len(prefix)==n and 0<n<=2048
            assert 1 not in tokens[len(prefix):-1] and (n==2048 or tokens[-1]==1)
            raw=self.processor.batch_decode(generated[:,len(prefix):],skip_special_tokens=True)[0].strip();self.arrays['all_original_full_head_logits']=np.stack(self.heads);self.arrays['generated_tokens']=np.array(tokens,dtype=np.int64)
            final={str(g):dict(actual_NPU_SHA256=tsha(t),reconstructed_complete_SHA256=bs(self.cache_cpu[g])) for g,t in self.cache_previous.items()};assert all(x['actual_NPU_SHA256']==x['reconstructed_complete_SHA256'] for x in final.values())
            assert all(self.counts.get('model.language_model.layers.'+str(k))==n for k in range(35)) and self.counts['model.embed_vision']==n+1 and 'vision_projected' in self.stages and {k for k in self.stages if k.startswith('vision_')}==({'vision_'+name for name in ['last_hidden_state',*self.model.model.vision_tower.m['stage_outputs']]}|{'vision_projected'})
            size=sum(a.nbytes for a in self.arrays.values());assert shutil.disk_usage(self.home).free>reserve+size
            archive=self.home/f'sample_{i:05d}.npz';save_npz(archive,self.arrays)
            record=dict(index=i,actual_model_prompt=prompt,actual_model_text=text,media_path=str(media_path),media_SHA256=ref['image_SHA256'],reference=ref['reference'],input_ids=prefix,prompt_length=len(prefix),generated_tokens=tokens,raw_output=raw,error=None,actual_native_forwards=n,frames=self.frames,activation_prefill=self.stages,activation_counts=self.counts,cache_layout=self.cache_layout,cache_group_members={str(g):v for g,v in self.cache_members.items()},final_native_cache_groups_SHA256=final,original_budget_hit=n==2048 and tokens[-1]!=1,stop_ids=[1],archive=str(archive),archive_SHA256=sha(archive),archive_bytes=archive.stat().st_size,lossless_uncompressed_bytes=size)
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
    def predict(self,prompt,media_path=None):
        before=self.index
        try:return self._predict_original_compute_and_capture(prompt,media_path)
        except BaseException:
            # The original function already records failures inside generate.
            # Only pre-generation validation/space failures were previously absent.
            if self.index==before:
                assert len(self.details)==before and not self.active
                self.failed=True
                reference=self.front[before]['reference'] if before<len(self.front) else None
                detail=record_pre_generation_failure(self.home,self.trace,before,prompt,media_path,reference,traceback.format_exc())
                self.details.append(detail);self.index+=1
            raise
    def batch_predict(self,batch):
        assert 1<=len(batch)<=1;return [self.predict(p,m) for p,m in batch]
    def get_sample_execution(self,prompt,media_path):
        if self.report_index==0:
            paths=[p for p in (self.home/'profiler').rglob('*') if p.is_file()];profile={str(p):sha(p) for p in paths};write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,caption_audio_tower_forward_calls=self.audio_tower_forward_calls,sample_trace_SHA256=sha(self.trace),first_real_vision_prefill_all_profiler_files_SHA256=profile))
        i=self.report_index;self.report_index+=1;assert i<len(self.details);return self.details[i]
    def get_execution_lineage(self):return self.lineage
    def unload(self):
        for h in getattr(self,'handles',[]):h.remove()
        self.model=None;self.cache_previous={};torch.npu.empty_cache()
