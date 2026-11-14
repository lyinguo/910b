"""Own original T5Gemma2 BF16/SDPA/canonical64 and unmodified native hybrid cache.

The cache observer stores initial complete physical buffers plus exact changed-slot
values. It preserves the actual native hybrid configuration even when native HF resolves DynamicCache; it compares every element on NPU and retains final complete NPU SHA; it
does not alter any cache, attention, precision or generation operation.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,gzip,hashlib,inspect,logging,os,re,shutil,subprocess,traceback,zipfile,importlib
import numpy as np
import torch
import torch_npu
from safetensors import safe_open
from transformers import T5Gemma2ForConditionalGeneration
import transformers.models.t5gemma2.modeling_t5gemma2 as native_T5
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.t5gemma2_own_original_canonical_frontend_v3_own_tokenizer_config_mask_flag_0123_20261113 import OwnCanonicalFrontend,canonical_text
MID='google/t5gemma-2-1b-1b';LOG=logging.getLogger(__name__)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def arr(t):
    x=t.detach().cpu().contiguous();return x.view(torch.uint16).numpy() if x.dtype==torch.bfloat16 else x.numpy()
def bs(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False)
def npz(p,values):
    with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for k,v in values.items():
            with z.open(k+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(v),allow_pickle=False)
class T5Gemma2OriginalHighTenNative(IModel):
    def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
        assert protocol['model_id']==MID and len(npu_devices)==1
        self.protocol_file=Path(protocol_file);self.protocol_sha256=protocol_sha256;self.devices=npu_devices
    @property
    def model_name(self):return MID
    @property
    def model_type(self):return 'vlm'
    @property
    def inference_framework(self):return 'Transformers'
    def download(self,*args,**kwargs):raise RuntimeError('Only own original existing checkpoint is authorized')
    def load(self,model_path,**kwargs):
        assert sha(self.protocol_file)==self.protocol_sha256;self.p=json.loads(self.protocol_file.read_bytes());self.W=Path(model_path)
        assert self.p['model_id']==MID and str(self.W)==self.p['weight_path']
        for n,h in self.p['own_runtime_assets_SHA256'].items():assert sha(self.W/n)==h,n
        for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
        for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
        assert sha(self.W/'model.safetensors')==self.p['own_weight']['SHA256'] and sha(native_T5.__file__)=='f9d2a5752e56b2b160a75a9728b0a436a70f2212d4ecb2fde1e9f0d8934f3e9c'
        probe=subprocess.run(['npu-smi','info'],capture_output=True,text=True);assert probe.returncode==0 and 'No running processes found in NPU '+str(self.devices[0]) in probe.stdout
        self.device=torch.device('npu:'+str(self.devices[0]));torch.npu.set_device(self.device);torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
        stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('T5Gemma2_original_HighTen_BF16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
        self.frontend=OwnCanonicalFrontend(self.W);self.tokenizer=self.frontend.tokenizer
        cfg=json.loads((self.W/'tokenizer_config.json').read_bytes());assert cfg['clean_up_tokenization_spaces'] is False and cfg['add_bos_token'] is True and cfg['add_eos_token'] is False
        raw=gzip.decompress(Path(self.p['actual_AA_frontend200_gzip']).read_bytes());assert hashlib.sha256(raw).hexdigest()==self.p['actual_AA_frontend200_raw_SHA256'];self.front=json.loads(raw)['samples'];assert len(self.front)==200
        self.cpu=json.loads(Path(self.p['independent_R_CPU200']).read_bytes());assert self.cpu['passed'] and self.cpu['sample_count']==200
        self.aa=json.loads(Path(self.p['actual_AA_metadata']).read_bytes());refs=json.loads(Path(self.p['actual_AA_source_manifest']).read_bytes());assert len(refs)==1119
        self.model=T5Gemma2ForConditionalGeneration.from_pretrained(str(self.W),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval().to(self.device)
        assert len(self.model.model.encoder.text_model.layers)==len(self.model.model.decoder.layers)==26 and len(self.model.model.encoder.vision_tower.encoder.layers)==27
        assert self.model.lm_head.out_proj.weight.shape==(262144,1152) and self.model.generation_config.to_dict()==self.aa['actual_generation_config']
        assert self.model.generation_config.cache_implementation=='hybrid' and self.model.generation_config.eos_token_id==1
        state=self.model.state_dict();weights={}
        with safe_open(str(self.W/'model.safetensors'),framework='pt',device='cpu') as f:
            assert set(f.keys())=={r['name'] for r in refs}
            for r in refs:
                src=f.get_tensor(r['name']);v=state[r['target_name']]
                assert list(src.shape)==r['shape']==list(v.shape) and str(src.dtype)==r['source_dtype'] and v.device==self.device and v.dtype==torch.bfloat16
                assert tsha(src)==r['source_SHA256'] and tsha(src.bfloat16())==r['cast_SHA256']==r['actual_CUDA_SHA256']==tsha(v),r['name']
                weights[r['name']]=dict(r,actual_NPU_SHA256=tsha(v),actual_NPU_device=str(v.device))
        aliases={}
        for name,t in self.model.named_parameters(remove_duplicate=False):aliases.setdefault(str((t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()))),[]).append(name)
        assert sorted(sorted(x) for x in aliases.values())==sorted(sorted(x) for x in self.aa['actual_parameter_aliases'])
        self.weight_proof=self.home/'own1119_source_cast_actual_NPU.json';write(self.weight_proof,dict(weights=weights,aliases=list(aliases.values())))
        self.index=0;self.report_index=0;self.details=[];self.handles=[];self.active=False;self.failed=False;self.profiler=None;self.profile_done=False
        self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self.profile=self.home/'npu_trace.json';self._install()
        import transformers,transformers.cache_utils,transformers.masking_utils,transformers.generation.utils
        libraries={inspect.getfile(m):sha(inspect.getfile(m)) for m in [torch,torch_npu,transformers,transformers.cache_utils,transformers.masking_utils,transformers.generation.utils,native_T5]}
        dependencies=json.loads(Path(self.p['actual_AA_native_dependency_manifest']).read_bytes())
        for name,rec in dependencies.items():
            module=importlib.import_module(name);path=inspect.getfile(module);assert sha(path)==rec['SHA256'],(name,'actual AA/R functional native source mismatch')
            libraries[path]=sha(path)
        self.lineage=dict(hardware='Ascend910B',devices=self.devices,runtime='own original native HF T5Gemma2 canonical serial1',runtime_dtype='BF16',max_new_tokens=64,actual_outer_batch_size=1,actual_inner_batch_size=1,attention='original native encoder bidirectional/sliding + decoder merged self/cross SDPA',use_cache=True,cache_implementation='hybrid',head_size=262144,tokenizer_size=262144,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha256,evidence_root=str(self.home),weight_evidence=str(self.weight_proof),weight_evidence_SHA256=sha(self.weight_proof),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),first_real_vision_and_prefill_profiler=str(self.profile),profiler_scope='first native vision, full text encoder and first decoder prefill only',runtime_library_source_SHA256=libraries,activation_retention='all79 first-prefill stage SHA/shape/dtype/count for every sample, full values first3; all full heads, all actual positions/masks and lossless physical self/cross cache initial + native growth/shape changes/exact NPU-checked changed slots at every step; final full actual NPU buffer SHA',historical_source_boundary=self.p['historical_source_boundary'],own_original_mask_special_runtime_flag_from_tokenizer_config=True)
    def _record(self,name,value,retain=True):
        assert value.device==self.device
        v=value.detach().cpu().contiguous();meta=dict(shape=list(v.shape),dtype=str(v.dtype),SHA256=tsha(v))
        if retain:assert name not in self.arrays;self.arrays[name]=arr(v)
        return meta
    def _refs(self,value,prefix,cache):
        if value is None:return None
        values=value if isinstance(value,(list,tuple)) else [value];out=[]
        for j,t in enumerate(values):
            assert torch.is_tensor(t);ident=(id(t),t.data_ptr(),tuple(t.shape),str(t.dtype))
            if ident not in cache:
                key=prefix+'_'+str(j);cache[ident]=dict(array=key,**self._record(key,t))
            out.append(cache[ident])
        return out
    def _capture_cache(self,cache):
        assert cache is not None;groups={};entries={}
        for label,c in [('self',cache.self_attention_cache),('cross',cache.cross_attention_cache)]:
            assert len(c.layers)==26
            for i,layer in enumerate(c.layers):
                for field in ['keys','values']:
                    t=getattr(layer,field);assert t.device==self.device and t.dtype==torch.bfloat16 and t.shape[:2]==(1,1) and t.shape[-1]==256
                    name=f'{label}_{i}_{field}';key=(label,type(c).__name__,type(layer).__name__,tuple(t.shape[:2]),t.shape[-1],str(t.dtype));groups.setdefault(key,[]).append((name,t));entries[name]=dict(cache_class=type(c).__name__,layer_class=type(layer).__name__,shape=list(t.shape),dtype=str(t.dtype))
        keys=sorted(groups,key=str)
        if not self.frames:
            self.group_keys=keys;self.cache_members={g:[n for n,t in groups[k]] for g,k in enumerate(keys)};self.cache_layout=entries
        assert keys==self.group_keys and set(entries)==set(self.cache_layout)
        for name,entry in entries.items():
            original=self.cache_layout[name];assert {k:v for k,v in entry.items() if k!='shape'}=={k:v for k,v in original.items() if k!='shape'}
        observations=[];new_previous={}
        for g,k in enumerate(keys):
            members=groups[k];assert [n for n,t in members]==self.cache_members[g]
            assert len({tuple(t.shape) for n,t in members})==1
            packed=torch.stack([t for n,t in members]).contiguous();new_previous[g]=packed
            if not self.frames:
                value=arr(packed);self.arrays[f'cache_group_{g}_initial']=value;self.cache_cpu[g]=value.copy();obs=dict(group=g,kind='initial',array=f'cache_group_{g}_initial',shape=list(packed.shape),dtype=str(packed.dtype),SHA256=bs(value),complete_physical_buffer_retained=True)
            else:
                old=self.cache_previous[g]
                if packed.shape!=old.shape:
                    growth=packed.shape[:3]==old.shape[:3] and packed.shape[4:]==old.shape[4:] and packed.shape[3]==old.shape[3]+1
                    prefix_equal=growth and torch.equal(packed[:,:,:,:-1].contiguous().view(torch.int16),old.view(torch.int16))
                    if prefix_equal:
                        value=arr(packed[:,:,:,-1:]);a=f'f{len(self.frames)+1}_cache_group_{g}_append';self.arrays[a]=value;self.cache_cpu[g]=np.concatenate((self.cache_cpu[g],value),axis=3);obs=dict(group=g,kind='native_length_growth_one',shape=list(packed.shape),dtype=str(packed.dtype),append_array=a,append_SHA256=bs(value),all_previous_NPU_values_checked_bit_identical=True)
                    else:
                        value=arr(packed);a=f'f{len(self.frames)+1}_cache_group_{g}_complete_shape_change';self.arrays[a]=value;self.cache_cpu[g]=value.copy();obs=dict(group=g,kind='complete_native_shape_change',shape=list(packed.shape),dtype=str(packed.dtype),array=a,SHA256=bs(value),all_current_native_values_retained=True)
                else:
                    old_bits=old.reshape(-1).view(torch.int16);bits=packed.reshape(-1).view(torch.int16);different=bits!=old_bits;idx=torch.nonzero(different,as_tuple=False).flatten();assert int(different.sum().item())==idx.numel()
                    obs=dict(group=g,kind='exact_changed_slots',shape=list(packed.shape),dtype=str(packed.dtype),changed_count=idx.numel(),all_native_NPU_elements_compared_exact=True)
                    if idx.numel():
                        ids=idx.cpu().numpy();values=arr(packed.reshape(-1).index_select(0,idx));a=f'f{len(self.frames)+1}_cache_group_{g}_indices';b=f'f{len(self.frames)+1}_cache_group_{g}_values';self.arrays[a]=ids;self.arrays[b]=values;self.cache_cpu[g].reshape(-1)[ids]=values;obs.update(indices_array=a,values_array=b,indices_SHA256=bs(ids),values_SHA256=bs(values))
                obs['reconstructed_complete_SHA256']=bs(self.cache_cpu[g])
            observations.append(obs)
        self.cache_previous=new_previous;return observations
    def _install(self):
        def root_pre(m,args,kw):
            if not self.active:return
            self.frame=dict(step=len(self.frames),attention=[]);self.frame_cache={}
            for key in ['decoder_input_ids','input_ids','attention_mask','decoder_attention_mask','position_ids','cache_position']:
                t=kw.get(key)
                if torch.is_tensor(t):self.frame[key]=dict(array=f'f{len(self.frames)+1}_{key}',**self._record(f'f{len(self.frames)+1}_{key}',t));self.frame[key+'_values']=t.cpu().tolist() if t.numel()<20000 else None
        def head(m,args,out):
            if self.active:
                row=out[0,-1];assert row.shape==(262144,) and row.dtype==torch.bfloat16 and row.device==self.device;self.heads.append(arr(row));self.frame.update(argmax=int(row.argmax().item()),head_shape=list(out.shape),head_dtype=str(out.dtype))
        def root_post(m,args,kw,out):
            if not self.active:return
            self.frame['cache']=self._capture_cache(out.past_key_values);self.frame['native_cache_class']=type(out.past_key_values).__name__;self.frames.append(self.frame)
            if self.profiler is not None and len(self.frames)==1:self.profiler.stop();self.profiler.export_chrome_trace(str(self.profile));self.profiler=None;self.profile_done=True
        self.handles.extend([self.model.register_forward_pre_hook(root_pre,with_kwargs=True),self.model.lm_head.register_forward_hook(head),self.model.register_forward_hook(root_post,with_kwargs=True)])
        for name,module in self.model.named_modules():
            if re.search(r'(?:layers|layer)\.\d+$',name):
                def stage(m,args,out,name=name):
                    if not self.active:return
                    self.counts[name]=self.counts.get(name,0)+1
                    if name in self.seen:return
                    self.seen.add(name);v=out[0] if isinstance(out,(list,tuple)) else out;key='stage_'+re.sub('[^a-zA-Z0-9_]','_',name);self.stages[key]=self._record(key,v,self.index<3)
                self.handles.append(module.register_forward_hook(stage))
                if name.startswith(('model.encoder.text_model.layers.','model.decoder.layers.')):
                    def layer_pre(m,args,kw,name=name):
                        if not self.active:return
                        data=inspect.signature(m.forward).bind_partial(*args,**kw).arguments;encoder=name.startswith('model.encoder.');cache=self.encoder_cache if encoder else self.frame_cache;obs=dict(layer_name=name)
                        for key in ['attention_mask','merged_attention_mask','position_ids','position_embeddings']:
                            if key in data:obs[key]=self._refs(data[key],('enc' if encoder else 'f'+str(len(self.frames)+1))+'_'+name.replace('.','_')+'_'+key,cache)
                        if encoder:self.encoder_observations.append(obs)
                        else:self.frame['attention'].append(obs)
                    self.handles.append(module.register_forward_pre_hook(layer_pre,with_kwargs=True))
    def predict(self,prompt,media_path=None):
        i=self.index;ref=self.front[i];inputs,text,expanded=self.frontend.encode(prompt,media_path);assert text==ref['actual_model_text'] and sha(media_path)==ref['media_SHA256']
        for key,value in inputs.items():assert tsha(value)==ref['inputs'][key]['SHA256'],(i,key)
        reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
        self.arrays={};self.frames=[];self.heads=[];self.stages={};self.counts={};self.seen=set();self.encoder_cache={};self.encoder_observations=[];self.cache_previous={};self.cache_cpu={};self.cache_members={};self.cache_layout={};self.group_keys=[]
        self.active=True
        try:
            actual={k:v.to(self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in inputs.items()}
            for key,value in actual.items():self._record('input_'+key,value);assert tsha(value)==(ref['inputs'][key]['original_BF16_input_SHA256'] if key=='pixel_values' else ref['inputs'][key]['SHA256'])
            if not self.profile_done:self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True);self.profiler.start()
            with torch.inference_mode():generated=self.model.generate(**actual,max_new_tokens=64,do_sample=False)
            self.active=False;tokens=generated[0].cpu().tolist();n=len(self.frames);assert tokens[0]==2 and tokens[1:]==[f['argmax'] for f in self.frames] and 0<n<=64
            assert 1 not in tokens[1:-1] and (n==64 or tokens[-1]==1)
            raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip();self.arrays['all_original_full_head_logits']=np.stack(self.heads)
            final_cache={str(g):dict(actual_complete_NPU_SHA256=tsha(t),reconstructed_complete_SHA256=bs(self.cache_cpu[g])) for g,t in self.cache_previous.items()};assert all(x['actual_complete_NPU_SHA256']==x['reconstructed_complete_SHA256'] for x in final_cache.values())
            assert len(self.stages)==len(self.counts)==79 and len(self.encoder_observations)==26
            total=sum(x.nbytes for x in self.arrays.values());assert shutil.disk_usage(self.home).free>reserve+total
            archive=self.home/f'sample_{i:05d}.npz';npz(archive,self.arrays)
            record=dict(index=i,actual_model_prompt=prompt,actual_model_text=text,expanded_text=expanded,media_path=str(media_path),media_SHA256=ref['media_SHA256'],input_ids=inputs['input_ids'].tolist(),generated_tokens=tokens,raw_output=raw,reference=ref['reference'],error=None,actual_native_forwards=n,frames=self.frames,activation_prefill=self.stages,activation_counts=self.counts,encoder_attention_observations=self.encoder_observations,cache_layout=self.cache_layout,cache_group_members={str(g):v for g,v in self.cache_members.items()},final_native_cache_groups_SHA256=final_cache,original_budget_hit=n==64 and tokens[-1]!=1,stop_ids=[1],archive=str(archive),archive_SHA256=sha(archive),archive_bytes=archive.stat().st_size,lossless_uncompressed_bytes=total)
            rp=self.home/f'sample_{i:05d}.json';write(rp,record);detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=n,generated_tokens=tokens,raw_output=raw,error=None)
            with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
            self.details.append(detail);self.index+=1;LOG.info('T5Gemma2 actual sample %d/200, native original64 forwards=%d, exact lossless cache/head bytes=%d',self.index,n,archive.stat().st_size)
            self.arrays={};self.heads=[];self.cache_cpu={};self.cache_previous={};return raw
        except BaseException:
            self.active=False;self.failed=True
            if self.profiler is not None:
                self.profiler.stop();self.profiler.export_chrome_trace(str(self.profile));self.profiler=None
            error=traceback.format_exc();rp=self.home/f'sample_{i:05d}_failed.json'
            write(rp,dict(index=i,actual_model_prompt=prompt,actual_model_text=text,media_path=str(media_path),reference=ref['reference'],committed_native_forwards=len(self.frames),generated_tokens=[f['argmax'] for f in self.frames],error=error))
            detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(self.frames),generated_tokens=[f['argmax'] for f in self.frames],raw_output=None,error=error)
            with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
            self.details.append(detail);self.index+=1;raise
    def batch_predict(self,batch):
        assert len(batch)==1;return [self.predict(p,m) for p,m in batch]
    def get_sample_execution(self,prompt,media_path):
        i=self.report_index;self.report_index+=1
        if i==0:write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),first_prefill_profiler_SHA256=sha(self.profile)))
        assert i<len(self.details);return self.details[i]
    def get_execution_lineage(self):return self.lineage
    def unload(self):
        for h in getattr(self,'handles',[]):h.remove()
        self.model=None;self.cache_previous={};torch.npu.empty_cache()
