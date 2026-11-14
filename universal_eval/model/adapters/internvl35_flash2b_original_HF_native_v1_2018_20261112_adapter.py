"""Own InternVL3.5 Flash2B: original BF16/eager/visual gating and native cached generation.

Only the optional, unused timm import is excluded from the own source package.
All original weights, quantile, compression, decoder and generation math remain native.
Lossless KV prefix + append deltas retain every cache value without recopying prefixes.
"""
from contextlib import ExitStack
from datetime import datetime
from functools import wraps
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib,inspect,json,logging,os,re,shutil,subprocess,traceback,zipfile
import numpy as np
import torch
import torch_npu
from safetensors import safe_open
from transformers import AutoTokenizer
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.internvl35_flash2b_own_frontend_v3_original_TV2_uint8_F32_1518_20261112 import OwnFlash2BInputs
from universal_eval.model.internvl35_flash2b_own_native_2018_20261112.configuration_internvl_chat import InternVLChatConfig
from universal_eval.model.internvl35_flash2b_own_native_2018_20261112.modeling_internvl_chat import InternVLChatModel

MID='OpenGVLab/InternVL3_5-2B-Flash'
LOG=logging.getLogger(__name__)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def array(value):
    value=value.detach().cpu().contiguous()
    return value.view(torch.uint16).numpy() if value.dtype==torch.bfloat16 else value.numpy()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False)
def save_npz(path,values):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,value in values.items():
            with z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(value),allow_pickle=False)

class InternVL35Flash2BOriginalHFNative(IModel):
    def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
        assert protocol['model_id']==MID and len(npu_devices)==1
        self.protocol_file=Path(protocol_file);self.protocol_sha=protocol_sha256;self.devices=npu_devices
    @property
    def model_name(self):return MID
    @property
    def model_type(self):return 'vlm'
    @property
    def inference_framework(self):return 'Transformers'
    def download(self,model_name,save_dir=None):raise RuntimeError('Only the original own local checkpoint is authorized')
    def load(self,model_path,**kwargs):
        assert sha(self.protocol_file)==self.protocol_sha
        self.p=json.loads(self.protocol_file.read_bytes());self.W=Path(model_path)
        assert self.p['model_id']==MID and str(self.W)==self.p['weight_path']
        for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
        for n,rec in self.p['own_weight_files'].items():assert sha(self.W/n)==rec['SHA256'],n
        for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
        for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
        probe=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
        assert probe.returncode==0 and ('No running processes found in NPU '+str(self.devices[0])) in probe.stdout,'Original target card is occupied; never share another process card'
        self.device=torch.device('npu:'+str(self.devices[0]));torch.npu.set_device(self.device);torch.set_num_threads(4)
        torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
        now=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
        self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('InternVL35Flash2B_original_BF16_'+now+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
        self.inputs=OwnFlash2BInputs(self.W)
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.W),local_files_only=True,trust_remote_code=False)
        assert len(self.tokenizer)==151678 and self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')==151671
        cfg=InternVLChatConfig(**json.loads((self.W/'config.json').read_bytes()))
        assert cfg.vision_config.drop_path_rate==0.0 and cfg.use_backbone_lora==cfg.use_llm_lora==0
        original_init=InternVLChatModel.__init__
        @wraps(original_init)
        def original_post_init_compat(instance,config,*a,**kw):
            original_init(instance,config,*a,**kw)
            if not hasattr(instance,'all_tied_weights_keys'):instance.post_init()
        InternVLChatModel.__init__=original_post_init_compat
        try:
            self.model=InternVLChatModel.from_pretrained(str(self.W),config=cfg,dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True,use_flash_attn=False).eval().to(self.device)
        finally:InternVLChatModel.__init__=original_init
        self.model.img_context_token_id=151671
        lm=self.model.language_model
        assert self.model.flash_mode and self.model.flash_relative_threshold==.1 and self.model.flash_absolute_threshold==.5
        assert lm.config._attn_implementation=='eager' and all(not x.attn.use_flash_attn for x in self.model.vision_model.encoder.layers)
        assert len(lm.model.layers)==28 and len(self.model.vision_model.encoder.layers)==24 and lm.lm_head.weight.shape==(151936,2048)
        aa=json.loads(Path(self.p['actual_AA_metadata']).read_bytes())
        for k in ['vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','rms_norm_eps','layer_types','rope_parameters','attention_bias','attention_dropout','sliding_window','use_sliding_window']:
            assert getattr(lm.config,k,None)==aa['actual_config']['llm_config'].get(k),k
        assert lm.generation_config.to_dict()==aa['actual_generation_config'],'Actual generation defaults differ from own AA control'
        expected={x['name']:x for x in json.loads(Path(self.p['actual_AA_source_manifest']).read_bytes())}
        params=dict(self.model.named_parameters(remove_duplicate=False));buffers=dict(self.model.named_buffers(remove_duplicate=False));assert set(params)==set(expected) and len(params)==718
        loaded={};aliases={}
        with safe_open(str(self.W/'model.safetensors'),framework='pt',device='cpu') as f:
            assert set(f.keys())==set(expected)
            for n in f.keys():
                src=f.get_tensor(n);ref=expected[n];actual=params.get(n,buffers.get(n))
                assert list(src.shape)==ref['shape']==list(actual.shape) and str(src.dtype)==ref['source_dtype']
                assert tensor_sha(src)==ref['source_SHA256'] and tensor_sha(src.bfloat16())==ref['cast_SHA256']==ref['actual_CUDA_SHA256'],n
                assert actual.device==self.device and actual.dtype==torch.bfloat16 and tensor_sha(actual)==ref['cast_SHA256'],n
                loaded[n]=dict(ref,actual_NPU_SHA256=tensor_sha(actual),actual_NPU_device=str(actual.device))
                key=str((actual.untyped_storage().data_ptr(),actual.storage_offset(),tuple(actual.shape),tuple(actual.stride())))
                aliases.setdefault(key,[]).append(n)
        assert sorted(sorted(v) for v in aliases.values())==sorted(sorted(v) for v in aa['actual_parameter_aliases'])
        self.weight_proof=self.home/'own718_source_cast_actual_NPU.json';write(self.weight_proof,dict(weights=loaded,aliases=list(aliases.values())))
        self.front=json.loads(Path(self.p['actual_AA_frontend200']).read_bytes())['samples'];self.cpu=json.loads(Path(self.p['independent_R_CPU200']).read_bytes())['records'];assert len(self.front)==len(self.cpu)==200
        self.index=0;self.report_index=0;self.details=[];self.handles=[];self.active=False;self.failed=False;self.profile_done=False;self.profiler=None
        self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self.profile=self.home/'npu_trace.json'
        self._install()
        libs={}
        import transformers,transformers.generation.utils,transformers.masking_utils,transformers.models.qwen3.modeling_qwen3
        for mod in [torch,torch_npu,transformers,transformers.generation.utils,transformers.masking_utils,transformers.models.qwen3.modeling_qwen3]:
            path=inspect.getfile(mod);libs[path]=sha(path)
        self.lineage=dict(hardware='Ascend910B',devices=self.devices,runtime='own native HF InternVL flash visual-compression + Qwen3',runtime_dtype='BF16',max_new_tokens=2048,actual_outer_batch_size=1,actual_inner_batch_size=1,attention='own current actual AA eager vision and text; flash_mode means visual token compression',use_cache=True,head_size=151936,tokenizer_size=151678,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,evidence_root=str(self.home),weight_evidence=str(self.weight_proof),weight_evidence_SHA256=sha(self.weight_proof),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),first_real_vision_and_prefill_profiler=str(self.profile),profiler_scope='first native vision plus first native LM prefill only',runtime_library_source_SHA256=libs,source_dependency_boundary='only original zero-drop-path unused timm import excluded; no stub, no AFS or shared dependency change',activation_retention='every sample all52 prefill activation SHA/shape/dtype/count; full52 arrays first3; all samples full native KV prefill and append deltas with exact unchanged-prefix checks, all full head rows and positions/masks/gating/compression retained',historical_source_boundary=self.p['historical_source_boundary'])
        LOG.info('Own718 original BF16/eager Flash2B loaded on %s; native .1/.5 visual compression/cacheTrue/2048',self.device)
    def _record(self,name,value,retain=True):
        assert torch.is_tensor(value) and value.device==self.device,(name,getattr(value,'device',None))
        v=value.detach().cpu().contiguous();raw=v.reshape(-1).view(torch.uint8).numpy().tobytes();meta=dict(shape=list(v.shape),dtype=str(v.dtype),SHA256=hashlib.sha256(raw).hexdigest())
        if retain:
            assert name not in self.arrays,name
            self.arrays[name]=v.view(torch.uint16).numpy() if v.dtype==torch.bfloat16 else v.numpy()
        return meta
    def _install(self):
        lm=self.model.language_model
        def lm_pre(m,args,kw):
            if not self.active:return
            assert not args
            n=len(self.frames);self.frame=dict(step=n,attention=[],input_ids=None,position_ids=None);self.tensor_cache={}
            for key in ['input_ids','position_ids','cache_position','attention_mask']:
                val=kw.get(key)
                if torch.is_tensor(val):
                    self.frame[key]=self._record('f%d_%s'%(n+1,key),val)
                    if key in ['input_ids','position_ids','cache_position']:self.frame[key+'_values']=val.detach().cpu().tolist()
            if n==0:
                self.activations['prefill_inputs_embeds']=self._record('prefill_inputs_embeds',kw['inputs_embeds']);self.compressed_length=kw['inputs_embeds'].shape[1]
        def head_hook(m,args,out):
            if self.active:
                row=out[0,-1];assert row.shape==(151936,) and row.dtype==torch.bfloat16 and row.device==self.device
                self.logits.append(array(row));self.frame['head_shape']=list(out.shape);self.frame['head_dtype']=str(row.dtype);self.frame['argmax']=int(row.argmax().item())
        def lm_post(m,args,kw,out):
            if not self.active:return
            n=len(self.frames);length=self.compressed_length+n;cache=out.past_key_values
            assert cache is not None and len(cache.layers)==28
            pieces=[];newpairs=[]
            for i,layer in enumerate(cache.layers):
                k,v=layer.keys,layer.values
                assert k.shape==v.shape==(1,8,length,128) and k.dtype==v.dtype==torch.bfloat16 and k.device==v.device==self.device
                if n:
                    pk,pv=self.previous_pairs[i]
                    assert torch.equal(k[:,:,:-1],pk) and torch.equal(v[:,:,:-1],pv),(n,i,'native cache prefix changed')
                    pieces.extend((k[0,:,-1:],v[0,:,-1:]))
                else:pieces.extend((k[0],v[0]))
                newpairs.append((k,v))
            packed=torch.stack(pieces).reshape(28,2,8,-1,128)
            if n:self.deltas.append(array(packed))
            else:self.arrays['native_cache_prefill']=array(packed)
            self.previous_pairs=newpairs;self.frame['cache_seq_len']=length;self.frame['unchanged_prefix_exact_checked_layers']=28 if n else 0;self.frames.append(self.frame)
            if self.profiler is not None and n==0:
                self.profiler.stop();self.profiler.export_chrome_trace(str(self.profile));self.profiler=None;self.profile_done=True
        self.handles.extend([lm.register_forward_pre_hook(lm_pre,with_kwargs=True),lm.lm_head.register_forward_hook(head_hook),lm.register_forward_hook(lm_post,with_kwargs=True)])
        for name,module in self.model.named_modules():
            if re.fullmatch(r'(vision_model.encoder.layers|language_model.model.layers)\.\d+',name):
                def stage_hook(m,args,out,name=name):
                    if not self.active:return
                    self.counts[name]=self.counts.get(name,0)+1
                    if self.frames:return
                    value=out[0] if isinstance(out,(tuple,list)) else out
                    key='stage_'+re.sub('[^a-zA-Z0-9_]','_',name);self.activations[key]=self._record(key,value,retain=self.index<3)
                self.handles.append(module.register_forward_hook(stage_hook))
        for i,layer in enumerate(lm.model.layers):
            def attn_pre(m,args,kw,i=i):
                if not self.active:return
                obs={'layer':i}
                for key in ['attention_mask','position_ids','position_embeddings']:
                    values=kw.get(key)
                    if values is None:obs[key]=None;continue
                    values=values if isinstance(values,(tuple,list)) else [values];refs=[]
                    for j,value in enumerate(values):
                        assert torch.is_tensor(value),(key,type(value))
                        ident=(id(value),value.data_ptr(),tuple(value.shape),str(value.dtype))
                        if ident not in self.tensor_cache:
                            name='f%d_attn_%s_%d_%d'%(len(self.frames)+1,key,i,j);meta=self._record(name,value);self.tensor_cache[ident]=dict(array=name,**meta)
                        refs.append(self.tensor_cache[ident])
                    obs[key]=refs
                self.frame['attention'].append(obs)
            self.handles.append(layer.self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True))
        extract=self.model.extract_feature_flash
        def extract_observed(*args,**kw):
            result=extract(*args,**kw)
            if self.active:
                for i,value in enumerate(result):self.activations['own_visual64_256_gate_'+str(i)]=self._record('own_visual64_256_gate_'+str(i),value)
            return result
        self.model.extract_feature_flash=extract_observed
        compress=self.model.compress_visual_tokens_in_sentence
        def compress_observed(*args,**kw):
            result=compress(*args,**kw)
            if self.active:
                for i,value in enumerate(result):self.activations['own_actual_compressed_inputs_mask_keep_'+str(i)]=self._record('own_actual_compressed_inputs_mask_keep_'+str(i),value)
                gate=kw['gate_result'];self.gate_decisions=gate.detach().cpu().tolist();self.activations['actual_gate_decisions']=self._record('actual_gate_decisions',gate)
            return result
        self.model.compress_visual_tokens_in_sentence=compress_observed
    def predict(self,prompt,media_path=None):
        i=self.index;ref=self.front[i];cpu=self.cpu[i]
        assert prompt==ref['actual_model_prompt'] and sha(media_path)==ref['media_file_SHA256']
        inputs=self.inputs.prepare(prompt,media_path)
        assert inputs['text']==ref['text'] and inputs['input_ids'].tolist()==ref['input_ids']
        for key,expected in [('pixels_F32','pixels_F32_SHA256'),('pixels_BF16','pixels_BF16_SHA256'),('input_ids','input_ids_SHA256'),('attention_mask','attention_mask_SHA256')]:
            assert tensor_sha(inputs[key])==ref[expected]==cpu[expected],(i,key)
        reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes']
        assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes'],'Original lossless evidence space gate'
        self.arrays={'input_ids':inputs['input_ids'].numpy(),'pixels_BF16':array(inputs['pixels_BF16'])};self.frames=[];self.logits=[];self.deltas=[];self.activations={};self.counts={};self.previous_pairs=[];self.gate_decisions=[]
        self.active=True
        try:
            if not self.profile_done:
                self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True);self.profiler.start()
            with torch.no_grad():
                generated=self.model.generate(pixel_values=inputs['pixels_BF16'].to(self.device),input_ids=inputs['input_ids'].to(self.device),attention_mask=inputs['attention_mask'].to(self.device),do_sample=False,max_new_tokens=2048)
            self.active=False
            tokens=generated[0].detach().cpu().tolist();assert tokens==[x['argmax'] for x in self.frames] and 0<len(tokens)<=2048
            assert (len(tokens)==2048 or tokens[-1]==151645) and all(t!=151645 for t in tokens[:-1]),'Original natural EOS/budget boundary'
            raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            self.arrays['all_original_full_head_logits']=np.stack(self.logits)
            self.arrays['native_cache_decode_delta']=np.concatenate(self.deltas,axis=3) if self.deltas else np.empty((28,2,8,0,128),dtype=np.uint16)
            final_cache=[[tensor_sha(v) for v in pair] for pair in self.previous_pairs]
            assert len(self.frames)==len(self.logits) and len(self.activations)>=52+3+4 and all(x==1 for k,x in self.counts.items() if k.startswith('vision_model.'))
            uncompressed=sum(v.nbytes for v in self.arrays.values());assert shutil.disk_usage(self.home).free>reserve+uncompressed,'Original complete lossless write reserve'
            ap=self.home/('sample_%05d.npz'%i);save_npz(ap,self.arrays)
            record=dict(index=i,actual_model_prompt=prompt,query=inputs['text'],media_path=str(media_path),media_SHA256=ref['media_file_SHA256'],input_ids=inputs['input_ids'][0].tolist(),generated_tokens=tokens,raw_output=raw,reference=ref['reference'],error=None,actual_native_forwards=len(self.frames),frames=self.frames,activation_prefill=self.activations,activation_counts=self.counts,gate_decisions=self.gate_decisions,original_budget_hit=len(tokens)==2048 and tokens[-1]!=151645,stop_ids=[151645],compressed_length=self.compressed_length,final_cache_SHA256=final_cache,archive=str(ap),archive_SHA256=sha(ap),archive_bytes=ap.stat().st_size,lossless_uncompressed_bytes=uncompressed)
            rp=self.home/('sample_%05d.json'%i);write(rp,record)
            detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(self.frames),generated_tokens=tokens,raw_output=raw,error=None)
            with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
            self.details.append(detail);self.index+=1
            LOG.info('Flash2B actual sample %d/%d, %d native forwards, compressed input %d, archive %d bytes',i+1,200,len(self.frames),self.compressed_length,ap.stat().st_size)
            self.arrays={};self.logits=[];self.deltas=[];self.previous_pairs=[]
            return raw
        except BaseException:
            self.active=False;self.failed=True
            if self.profiler is not None:
                self.profiler.stop();self.profiler=None
            write(self.home/('sample_%05d_failed.json'%i),dict(index=i,committed_native_forwards=len(self.frames),error=traceback.format_exc()));raise
    def batch_predict(self,batch):
        assert len(batch)==1,'Original actual HF serial1 is required'
        return [self.predict(prompt,path) for prompt,path in batch]
    def get_sample_execution(self,prompt,media_path):
        i=self.report_index;self.report_index+=1
        if i==0:write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),first_prefill_profiler_SHA256=sha(self.profile)))
        assert i<len(self.details)
        return self.details[i]
    def get_execution_lineage(self):return self.lineage
    def unload(self):
        for h in getattr(self,'handles',[]):h.remove()
        self.model=None;self.previous_pairs=[];torch.npu.empty_cache()
