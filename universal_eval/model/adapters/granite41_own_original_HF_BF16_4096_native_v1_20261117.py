"""Own Granite4.1 HF/BF16/4096, full native cache/head and image provenance.

Original remote model mathematics are imported unchanged from own source.
Observation hooks call original operators; no alternate checkpoint or score.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys,os,json,hashlib,inspect,importlib,re,shutil,subprocess,traceback,types
import numpy as np
import torch,torch_npu
from PIL import Image
from safetensors import safe_open
from transformers import GenerationConfig
from universal_eval.model.interface import IModel
from universal_eval.model.coder_gemma4_e2b_own_20261116.observer import arr,bs,sha,tsha,write,save_npz
from universal_eval.model.granite41_own_original_source_20261117.configuration import Granite4VisionConfig
from universal_eval.model.granite41_own_original_source_20261117.modeling import Granite4VisionForConditionalGeneration
from universal_eval.model.granite41_own_original_source_20261117.processing import Granite4VisionProcessor
MID='ibm-granite/granite-vision-4.1-4b'
def js(p):return json.loads(Path(p).read_bytes())
def load_own(W,p,device):
    c=Granite4VisionConfig.from_pretrained(W,local_files_only=True)
    assert (c.text_config.num_hidden_layers,c.text_config.hidden_size,c.text_config.vocab_size,c.text_config.num_key_value_heads)==(40,2560,100352,8)
    assert c.text_config.embedding_multiplier==12 and c.text_config.residual_multiplier==.22 and c.text_config.attention_multiplier==.015625 and c.text_config.logits_scaling==10
    headers=js(p['own_source_headers']);mapping=js(p['own_source_to_native_name_map'])['mapping'];assert len(headers)==len(mapping)==1083 and len(set(mapping.values()))==1083
    with torch.device('meta'):model=Granite4VisionForConditionalGeneration(c)
    state={};records={}
    for shard,info in p['own_weight_files'].items():
        assert (W/shard).stat().st_size==info['bytes'] and sha(W/shard)==info['SHA256']
        with safe_open(str(W/shard),framework='pt',device='cpu') as sf:
            for n in sf.keys():
                raw=sf.get_tensor(n).contiguous();assert raw.dtype==torch.bfloat16 and list(raw.shape)==headers[n]['shape']
                target=mapping[n];assert target not in state;state[target]=raw;records[n]=dict(source_file=shard,source_dtype=str(raw.dtype),shape=list(raw.shape),source_SHA256=tsha(raw),native_name=target)
    assert set(records)==set(headers);expected=model.state_dict();missing=set(expected)-set(state);assert missing=={'lm_head.weight'} and model.lm_head.weight is model.model.language_model.embed_tokens.weight
    state['lm_head.weight']=state['model.language_model.embed_tokens.weight'];assert set(state)==set(expected)
    for name,t in state.items():assert t.shape==expected[name].shape
    model.load_state_dict(state,strict=True,assign=True);model.tie_weights()
    # Native rotary inverse frequencies are config-derived nonpersistent buffers.
    # Rebuild only meta buffers with their own class initializer, without weights.
    for module in model.modules():
        if any(t.device.type=='meta' for t in module.buffers(recurse=False)):
            assert type(module).__name__ in ['GraniteRotaryEmbedding','SiglipVisionEmbeddings']
            if type(module).__name__=='GraniteRotaryEmbedding':reference=type(module)(config=module.config,device=torch.device('cpu'))
            else:reference=type(module)(module.config)
            for name,t in reference.named_buffers(recurse=False):module.register_buffer(name,t.clone(),persistent=name not in reference._non_persistent_buffers_set)
    assert all(t.device.type!='meta' for t in list(model.parameters())+list(model.buffers()))
    buffers={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tsha(t)) for n,t in model.named_buffers() if n not in model.state_dict()}
    model.eval().to(device);actual=model.state_dict()
    for n,rec in records.items():
        t=actual[rec['native_name']];assert t.device==device and t.dtype==torch.bfloat16 and list(t.shape)==rec['shape'] and tsha(t)==rec['source_SHA256'],n
        rec.update(actual_NPU_device=str(t.device),actual_NPU_SHA256=tsha(t))
    for n,t in model.named_buffers():
        if n in buffers:assert t.device==device and tsha(t)==buffers[n]['SHA256'];buffers[n]['actual_NPU_SHA256']=tsha(t)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    model.generation_config=GenerationConfig.from_pretrained(W,local_files_only=True);assert model.generation_config.eos_token_id==100257
    return model,dict(model_id=MID,all_own_source_tensors=records,own_source_count=1083,own_tied_head_source='model.language_model.embed_tokens.weight',own_tied_head_actual_NPU_SHA256=tsha(model.lm_head.weight),native_config_buffers=buffers,own_generation_config=model.generation_config.to_dict(),no_foreign_or_random_weight=True)

class Granite41OwnOriginalBF16(IModel):
    def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
        assert protocol['model_id']==MID and npu_devices==[4];self.protocol_file=Path(protocol_file);self.protocol_sha256=protocol_sha256
    @property
    def model_name(self):return MID
    @property
    def model_type(self):return 'vlm'
    @property
    def inference_framework(self):return 'Transformers'
    def download(self,*a,**kw):raise RuntimeError('Own existing assets only')
    def _ref(self,t,name):
        if t is None:return None
        if isinstance(t,(tuple,list)):return [self._ref(v,name+'_'+str(i)) for i,v in enumerate(t)]
        assert torch.is_tensor(t) and t.device==self.device
        key=(t.data_ptr(),tuple(t.shape),str(t.dtype))
        if key not in self.refs:
            a=arr(t).copy();assert name not in self.arrays;self.arrays[name]=a;self.refs[key]=dict(array=name,shape=list(t.shape),dtype=str(t.dtype),SHA256=bs(a))
        return self.refs[key]
    def load(self,model_path,**kwargs):
        assert sha(self.protocol_file)==self.protocol_sha256;self.p=js(self.protocol_file);p=self.p;W=self.W=Path(model_path);assert p['model_id']==MID and str(W)==p['weight_path']
        for n,h in p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
        for n,h in p['static_evidence_SHA256'].items():assert sha(n)==h,n
        for n,h in p['own_runtime_assets_SHA256'].items():assert sha(W/n)==h,n
        assert sha(p['dataset_path'])==p['dataset_SHA256'];q=subprocess.run(['npu-smi','info'],capture_output=True,text=True);assert q.returncode==0 and 'No running processes found in NPU 4' in q.stdout
        self.device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
        stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('Granite41_own_HF_BF16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
        self.model,weights=load_own(W,p,self.device);self.weights=self.home/'own1083_all_actual_NPU_sources.json';write(self.weights,weights)
        self.processor=Granite4VisionProcessor.from_pretrained(W,local_files_only=True);self.front=[json.loads(s) for s in Path(p['own_frontend200']).read_text().splitlines()];assert len(self.front)==200
        self.index=self.report_index=0;self.details=[];self.dispatch=[];self.handles=[];self.active=self.failed=self.profile_done=False;self.profiler=None;self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self._install()
        runtime={}
        for name in ['torch','torch_npu','transformers','transformers.models.granite.modeling_granite','transformers.models.siglip.modeling_siglip','transformers.models.blip_2.modeling_blip_2','transformers.cache_utils','transformers.masking_utils','transformers.generation.utils','transformers.generation.logits_process']:
            f=Path(inspect.getfile(importlib.import_module(name)));runtime[str(f)]=sha(f)
        self.lineage=dict(hardware='Ascend910B',devices=[4],runtime='Own original HF Granite4Vision BF16',runtime_dtype='BF16',max_new_tokens=4096,configured_outer_batch_size=10,actual_inner_batch_size=1,EOS=[100257],head_size=100352,LM_layers=40,vision_layers=27,use_cache=True,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha256,evidence_root=str(self.home),weight_evidence=str(self.weights),weight_evidence_SHA256=sha(self.weights),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),runtime_library_source_SHA256=runtime,historical_source_boundary=p['historical_source_boundary'],A100_historical_tensor_capture_claimed=False,original_default_chat_template_kwargs_empty=True,profiler_scope='First actual vision and prefill only')
    def _capture_cache(self,cache):
        assert cache is not None and len(cache.layers)==40;current=[]
        for layer in cache.layers:
            for name in ['keys','values']:
                t=getattr(layer,name);assert t.device==self.device and t.dtype==torch.bfloat16 and t.shape[:2]==(1,8) and t.shape[-1]==64;current.append(t)
        packed=torch.stack(current);name='cache_step_'+str(len(self.frames));before=self.cache_previous
        if before is None:
            a=arr(packed).copy();self.cache_cpu=a;kind='initial'
        else:
            assert packed.shape[:3]==before.shape[:3] and packed.shape[4:]==before.shape[4:] and packed.shape[3]==before.shape[3]+1
            assert torch.equal(packed[:,:,:,:-1].contiguous().view(torch.int16),before.view(torch.int16)),'All previous native KV values must be unchanged'
            a=arr(packed[:,:,:,-1:]).copy();self.cache_cpu=np.concatenate([self.cache_cpu,a],axis=3);kind='append_one'
        self.arrays[name]=a;assert tsha(packed)==bs(self.cache_cpu);self.cache_previous=packed
        return dict(kind=kind,array=name,shape=list(packed.shape),dtype=str(packed.dtype),SHA256=bs(a),all_previous_values_bitwise_preserved=True,reconstructed_complete_SHA256=bs(self.cache_cpu),actual_complete_NPU_SHA256=tsha(packed))
    def _install(self):
        def pre(m,args,kw):
            if not self.active:return
            self.refs={};self.frame=dict(step=len(self.frames),layers=[])
            for k in ['input_ids','attention_mask','position_ids','cache_position']:
                if torch.is_tensor(kw.get(k)):self.frame[k]=self._ref(kw[k],'step_'+str(len(self.frames))+'_'+k)
        def post(m,args,kw,out):
            if not self.active:return
            row=out.logits[0,-1];assert row.device==self.device and row.dtype==torch.bfloat16 and tuple(row.shape)==(100352,) and torch.isfinite(row).all()
            self.frame.update(full_head=self._ref(row,'head_'+str(len(self.frames))),head_argmax=int(row.argmax()),cache=self._capture_cache(out.past_key_values));self.frames.append(self.frame)
            if self.profiler is not None and len(self.frames)==1:self.profiler.stop();self.profiler=None;self.profile_done=True
        self.handles.extend([self.model.register_forward_pre_hook(pre,with_kwargs=True),self.model.register_forward_hook(post,with_kwargs=True)])
        def head(m,args):
            if self.active:self.frame['head_input']=self._ref(args[0][0,-1],'head_input_'+str(len(self.frames)))
        self.handles.append(self.model.lm_head.register_forward_pre_hook(head))
        for name,module in self.model.named_modules():
            if re.fullmatch(r'model.language_model.layers.\d+',name):
                def layer(m,args,kw,name=name):
                    if not self.active:return
                    data=inspect.signature(m.forward).bind_partial(*args,**kw).arguments;rec=dict(name=name)
                    for field in ['position_ids','position_embeddings','attention_mask']:rec[field]=self._ref(data.get(field),'s'+str(len(self.frames))+'_'+name.replace('.','_')+'_'+field)
                    self.frame['layers'].append(rec)
                self.handles.append(module.register_forward_pre_hook(layer,with_kwargs=True))
            if re.fullmatch(r'model.vision_tower.encoder.layers.\d+',name) or re.fullmatch(r'model.(layerwise_projectors|spatial_projectors).\d+',name):
                def stage(m,args,kw,out,name=name):
                    if not self.active:return
                    self.counts[name]=self.counts.get(name,0)+1
                    if name in self.stage:return
                    t=out.last_hidden_state if hasattr(out,'last_hidden_state') else out[0] if isinstance(out,(list,tuple)) else out;assert torch.is_tensor(t) and t.device==self.device and torch.isfinite(t).all()
                    self.stage[name]=dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tsha(t))
                    # Full small-sample activation exemplars; all native KV/head
                    # values remain lossless for every sample and every step.
                    if self.index<3:self.stage[name]['values']=self._ref(t,'activation_'+name.replace('.','_'))
                self.handles.append(module.register_forward_hook(stage,with_kwargs=True))
        original=self.model.model.get_image_features
        def image_features(inst,*args,**kw):
            result=original(*args,**kw)
            if self.active:self.images=[dict(target_layer=int(i),features=[self._ref(t,'vision_injection_'+str(j)+'_'+str(k)) for k,t in enumerate(v)]) for j,(i,v) in enumerate(result)]
            return result
        self.model.model.get_image_features=types.MethodType(image_features,self.model.model)
    def _predict(self,prompt,media_path):
        i=self.index;ref=self.front[i];assert prompt==ref['original_model_prompt'] and sha(media_path)==ref['image_SHA256'];inputs=self.processor(text=ref['formatted_text'],images=[Image.open(media_path).convert('RGB')],return_tensors='pt',padding=True)
        for k,t in inputs.items():assert list(t.shape)==ref['inputs'][k]['shape'] and str(t.dtype)==ref['inputs'][k]['dtype'] and tsha(t)==ref['inputs'][k]['SHA256']
        reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
        self.arrays={};self.refs={};self.frames=[];self.stage={};self.counts={};self.images=[];self.cache_cpu=None;self.cache_previous=None
        actual={k:t.to(self.device) for k,t in inputs.items()}
        inputrefs={k:self._ref(t,'input_'+k) for k,t in actual.items()};self.active=True
        try:
            if not self.profile_done:
                self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')),record_shapes=True);self.profiler.start()
            with torch.inference_mode():generated=self.model.generate(**actual,max_new_tokens=4096,do_sample=False,temperature=0.0)
            self.active=False;tokens=generated[0].cpu().tolist();prefix=inputs['input_ids'][0].tolist();n=len(self.frames);assert tokens[:len(prefix)]==prefix and len(tokens)-len(prefix)==n and 0<n<=4096 and 100257 not in tokens[len(prefix):-1] and (n==4096 or tokens[-1]==100257)
            raw=self.processor.batch_decode(generated[:,len(prefix):],skip_special_tokens=True)[0].strip();assert all(self.counts.get('model.vision_tower.encoder.layers.'+str(j))==1 for j in range(27));assert len(self.images)==8
            size=sum(a.nbytes for a in self.arrays.values());assert shutil.disk_usage(self.home).free>reserve+size
            archive=self.home/f'sample_{i:05d}.npz';save_npz(archive,self.arrays)
            rec=dict(index=i,error=None,raw_output=raw,reference=ref['reference'],actual_model_prompt=prompt,formatted_text=ref['formatted_text'],media_path=str(media_path),input_ids=prefix,inputs=inputrefs,generated_tokens=tokens,actual_native_forwards=n,frames=self.frames,full_vision_injection_features=self.images,activation_prefill=self.stage,activation_counts=self.counts,archive=str(archive),archive_SHA256=sha(archive),archive_bytes=archive.stat().st_size,lossless_uncompressed_bytes=size,original_budget_hit=n==4096 and tokens[-1]!=100257)
        except BaseException:
            self.active=False;self.failed=True;failure=dict(index=i,error=traceback.format_exc(),actual_native_forwards=len(self.frames),raw_output=None,actual_model_prompt=prompt,reference=ref['reference']);write(self.home/f'sample_{i:05d}_failure.json',failure);raise
        finally:
            if self.profiler is not None:self.profiler.stop();self.profiler=None
        rp=self.home/f'sample_{i:05d}.json';write(rp,rec);detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),raw_output=raw,error=None,actual_native_forwards=n,generated_tokens=tokens)
        with self.trace.open('a',encoding='utf8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
        self.details.append(detail);self.index+=1;self.arrays={};self.cache_previous=self.cache_cpu=None;return raw
    def predict(self,prompt,media_path=None):return self._predict(prompt,media_path)
    def batch_predict(self,batch):
        assert 1<=len(batch)<=10;row=dict(indices=list(range(self.index,self.index+len(batch))),status='started');self.dispatch.append(row)
        try:values=[self.predict(*item) for item in batch];row['status']='completed';return values
        except BaseException:row['status']='failed';row['error']=traceback.format_exc();raise
    def get_execution_lineage(self):return self.lineage
    def get_sample_execution(self,prompt,media_path):
        if self.report_index==0:
            files={str(p):sha(p) for p in (self.home/'profiler').rglob('*') if p.is_file()};runtime=set()
            for module in list(sys.modules.values()):
                name=getattr(module,'__file__',None)
                if name:
                    f=Path(name)
                    if f.is_file() and f.suffix in ['.py','.so']:runtime.add(f.resolve())
            for line in Path('/proc/self/maps').read_text().splitlines():
                name=line.split()[-1]
                if name.startswith('/') and ('.so' in name or 'python3' in name) and Path(name).is_file():runtime.add(Path(name).resolve())
            write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),outer_dispatches=self.dispatch,first_real_vision_prefill_all_profiler_files_SHA256=files,all_actual_imported_source_and_mapped_library_SHA256={str(f):sha(f) for f in sorted(runtime)}))
        i=self.report_index;self.report_index+=1;return self.details[i]
    def unload(self):
        for h in getattr(self,'handles',[]):h.remove()
        self.model=None;self.cache_previous=None;torch.npu.empty_cache()
