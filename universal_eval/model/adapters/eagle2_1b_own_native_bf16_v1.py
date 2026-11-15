"""Own Eagle2-1B: saved eager BF16/256/outer10-inner1, complete native NPU evidence.

No peft/video stub, online A100 dependency, altered tokens, or CPU learned forward.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,json,hashlib,inspect,logging,subprocess,traceback,shutil,zipfile,re
import torch,torch_npu,numpy as np
from safetensors import safe_open
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM,Qwen2RotaryEmbedding
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.eagle2_1b_own_frontend_v1 import OwnEagle2Inputs,tensor_sha
from universal_eval.model.eagle2_1b_own_native_20261115.configuration_eagle2_5_vl import Eagle2_5_VLConfig
from universal_eval.model.eagle2_1b_own_native_20261115.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration

MID='nvidia/Eagle2-1B';LOG=logging.getLogger(__name__)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array(x):
    x=x.detach().cpu().contiguous();return x.view(torch.uint16).numpy() if x.dtype==torch.bfloat16 else x.numpy()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False)
def save_npz(p,values):
    with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for k,v in values.items():
            with z.open(k+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(v),allow_pickle=False)

class Eagle2OwnNativeBF16(IModel):
    def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
        assert protocol['model_id']==MID and len(npu_devices)==1
        self.protocol_file=Path(protocol_file);self.protocol_sha=protocol_sha256;self.devices=npu_devices
    @property
    def model_name(self):return MID
    @property
    def model_type(self):return 'vlm'
    @property
    def inference_framework(self):return 'Transformers'
    def download(self,*a,**kw):raise RuntimeError('Only own existing weights are authorized')
    def load(self,model_path,**kwargs):
        assert sha(self.protocol_file)==self.protocol_sha;self.p=json.loads(self.protocol_file.read_bytes());self.W=Path(model_path)
        assert self.p['model_id']==MID and str(self.W)==self.p['weight_path']
        for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
        for n,r in self.p['own_weight_files'].items():assert sha(self.W/n)==r['SHA256'],n
        for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
        for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
        info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert info.returncode==0 and ('No running processes found in NPU '+str(self.devices[0])) in info.stdout
        self.device=torch.device('npu:'+str(self.devices[0]));torch.npu.set_device(self.device);torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
        self.inputs=OwnEagle2Inputs(self.W);self.tokenizer=self.inputs.tokenizer
        cfg=Eagle2_5_VLConfig(**json.loads((self.W/'config.json').read_bytes()));cfg._attn_implementation=cfg.vision_config._attn_implementation=cfg.text_config._attn_implementation='eager'
        assert cfg.use_backbone_lora==cfg.use_llm_lora==0 and cfg.text_config.eos_token_id==151645 and cfg.text_config.hidden_size==896 and cfg.text_config.num_hidden_layers==24
        with torch.device('meta'):
            vision=SiglipVisionModel(cfg.vision_config);lm=Qwen2ForCausalLM(cfg.text_config);self.model=Eagle2_5_VLForConditionalGeneration(cfg,vision_model=vision,language_model=lm)
        expected=json.loads(Path(self.p['own_source_manifest']).read_bytes());weights={}
        with safe_open(str(self.W/'model.safetensors'),framework='pt',device='cpu') as f:
            assert set(f.keys())==set(expected) and {v['runtime_name'] for v in expected.values()}==set(dict(self.model.named_parameters(remove_duplicate=False)))
            for n in f.keys():
                t=f.get_tensor(n);assert tensor_sha(t)==expected[n]['source_SHA256'] and t.dtype==torch.bfloat16
                weights[expected[n]['runtime_name']]=t.to(self.device)
        self.model.load_state_dict(weights,strict=True,assign=True);del weights
        lm=self.model.language_model
        if cfg.text_config.tie_word_embeddings:
            assert expected['language_model.lm_head.weight']['source_SHA256']==expected['language_model.model.embed_tokens.weight']['source_SHA256']
            lm.lm_head.weight=lm.model.embed_tokens.weight
        # Nonlearned own HF construction: exact position ids and default analytic RoPE.
        rotary=Qwen2RotaryEmbedding(cfg.text_config,device='cpu');assert rotary.rope_type=='default'
        buffers={}
        for name,t in list(self.model.named_buffers(remove_duplicate=False)):
            if name.endswith('.position_ids'):v=torch.arange(1024,dtype=torch.int64).expand((1,-1))
            elif name.endswith(('.inv_freq','.original_inv_freq')):v=rotary.inv_freq.detach().cpu()
            else:raise AssertionError(('Unexpected native buffer',name,t.shape,t.dtype))
            assert v.shape==t.shape;parent,_,key=name.rpartition('.');module=self.model.get_submodule(parent);setattr(module,key,v.to(self.device));buffers[name]=dict(shape=list(v.shape),dtype=str(v.dtype),CPU_analytic_SHA256=tensor_sha(v),actual_NPU_SHA256=tensor_sha(getattr(module,key)))
        lm.model.rotary_emb.original_inv_freq=lm.model.rotary_emb.inv_freq
        self.model.eval();assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters())
        assert lm.config._attn_implementation==self.model.vision_model.config._attn_implementation=='eager'
        now=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('Eagle2_1B_own_BF16_'+now+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
        actual={n:dict(ref,actual_NPU_SHA256=tensor_sha(dict(self.model.named_parameters(remove_duplicate=False))[ref['runtime_name']]),actual_NPU_device=str(self.device)) for n,ref in expected.items()}
        assert all(x['actual_NPU_SHA256']==x['cast_SHA256'] for x in actual.values());self.weight_proof=self.home/'all_own_source_actual_NPU.json';write(self.weight_proof,dict(weights=actual,buffers=buffers,own_tied_head=cfg.text_config.tie_word_embeddings))
        self.front=json.loads(Path(self.p['own_frontend200']).read_bytes());assert len(self.front)==200
        self.index=0;self.report_index=0;self.details=[];self.handles=[];self.active=False;self.failed=False;self.profile_done=False;self.profiler=None;self.batches=[]
        self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self.profile=self.home/'npu_trace.json';self._install()
        libs={}
        import transformers,transformers.generation.utils,transformers.masking_utils,transformers.models.qwen2.modeling_qwen2,transformers.models.siglip.modeling_siglip
        for mod in [torch,torch_npu,transformers,transformers.generation.utils,transformers.masking_utils,transformers.models.qwen2.modeling_qwen2,transformers.models.siglip.modeling_siglip]:
            path=inspect.getfile(mod);libs[path]=sha(path)
        self.lineage=dict(hardware='Ascend910B',devices=self.devices,runtime='Own Eagle2 Siglip27 + Qwen2_24 native eager',runtime_dtype='BF16',max_new_tokens=256,actual_outer_batch_size=10,actual_inner_batch_size=1,attention='saved original adapter forces eager; native eager unchanged',use_cache=True,head_size=151674,tokenizer_size=151674,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,evidence_root=str(self.home),weight_evidence=str(self.weight_proof),weight_evidence_SHA256=sha(self.weight_proof),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),first_real_vision_and_prefill_profiler=str(self.profile),profiler_scope='first native vision plus first native LM prefill only',runtime_library_source_SHA256=libs,source_dependency_boundary='Only unused peft import excluded from own source; no LoRA enabled, no stub or package installation',activation_retention='Every actual full head row, every native cache prefix/append, every position/mask; 51 layer prefill values first3 and full stage SHA metadata all samples',historical_source_boundary=self.p['historical_source_boundary'])
    def predict(self,prompt,media_path=None):
        i=self.index;self.frames=[];self.active=False
        try:
            ref=self.front[i];assert prompt==ref['actual_model_prompt'] and sha(media_path)==ref['media_file_SHA256'];v=self.inputs.prepare(prompt,media_path)
            assert v['text']==ref['text'] and v['input_ids'][0].tolist()==ref['input_ids']
            for k in ['input_ids','attention_mask','pixels_F32','pixels_BF16']:assert tensor_sha(v[k])==ref[k+'_SHA256'],(i,k)
            reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
            self.arrays={'input_ids':v['input_ids'].numpy(),'pixels_BF16':array(v['pixels_BF16'])};self.logits=[];self.head_inputs=[];self.deltas=[];self.activations={};self.counts={};self.previous_pairs=[];self.active=True
            if not self.profile_done:
                self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')));self.profiler.start()
            with torch.no_grad():generated=self.model.generate(pixel_values=v['pixels_BF16'].to(self.device),input_ids=v['input_ids'].to(self.device),attention_mask=v['attention_mask'].to(self.device),do_sample=False,max_new_tokens=256,temperature=0.0)
            self.active=False;tokens=generated[0].detach().cpu().tolist();assert tokens==[f['argmax'] for f in self.frames] and 0<len(tokens)<=256
            assert (len(tokens)==256 or tokens[-1]==151645) and all(t!=151645 for t in tokens[:-1])
            raw=self.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            self.arrays['all_original_full_head_logits']=np.stack(self.logits);self.arrays['all_head_inputs']=np.stack(self.head_inputs);self.arrays['native_cache_decode_delta']=np.concatenate(self.deltas,axis=3) if self.deltas else np.empty((24,2,2,0,64),dtype=np.uint16)
            final_cache=[[tensor_sha(v) for v in pair] for pair in self.previous_pairs];assert len(self.frames)==len(self.logits) and len(self.activations)>=51+2
            uncompressed=sum(v.nbytes for v in self.arrays.values());assert shutil.disk_usage(self.home).free>reserve+uncompressed
            ap=self.home/('sample_%05d.npz'%i);save_npz(ap,self.arrays);rp=self.home/('sample_%05d.json'%i)
            rec=dict(index=i,actual_model_prompt=prompt,query=v['text'],media_path=str(media_path),media_SHA256=ref['media_file_SHA256'],input_ids=v['input_ids'][0].tolist(),generated_tokens=tokens,raw_output=raw,reference=ref['reference'],error=None,actual_native_forwards=len(self.frames),frames=self.frames,activation_prefill=self.activations,activation_counts=self.counts,original_budget_hit=len(tokens)==256 and tokens[-1]!=151645,stop_ids=[151645],compressed_length=self.compressed_length,final_cache_SHA256=final_cache,archive=str(ap),archive_SHA256=sha(ap),archive_bytes=ap.stat().st_size,lossless_uncompressed_bytes=uncompressed);write(rp,rec)
            detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(self.frames),generated_tokens=tokens,raw_output=raw,error=None)
            with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
            self.details.append(detail);self.index+=1;self.arrays={};self.logits=[];self.head_inputs=[];self.deltas=[];self.previous_pairs=[];LOG.info('Eagle2-1B actual sample %d/200, %d forwards, archive %d bytes',i+1,len(tokens),ap.stat().st_size);return raw
        except BaseException:
            self.active=False;self.failed=True
            if self.profiler is not None:self.profiler.stop();self.profiler=None
            write(self.home/('sample_%05d_failed.json'%i),dict(index=i,committed_native_forwards=len(self.frames),error=traceback.format_exc()));raise
    def batch_predict(self,batch):
        assert 1<=len(batch)<=10
        record=dict(indices=list(range(self.index,self.index+len(batch))),status='running');self.batches.append(record)
        try:
            result=[self.predict(prompt,path) for prompt,path in batch];record['status']='completed';return result
        except BaseException:record['status']='failed';raise
    def get_sample_execution(self,prompt,media_path):
        i=self.report_index;self.report_index+=1
        if i==0:write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),actual_outer_batches=self.batches,first_prefill_profiler_SHA256=sha(self.profile),first_real_vision_prefill_profiler_SHA256={str(p):sha(p) for p in (self.home/'profiler').rglob('*') if p.is_file()}))
        return self.details[i]
    def get_execution_lineage(self):return self.lineage
    def unload(self):
        for h in getattr(self,'handles',[]):h.remove()
        self.model=None;self.previous_pairs=[];torch.npu.empty_cache()

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
                    row=out[0,-1];assert row.shape==(151674,) and row.dtype==torch.bfloat16 and row.device==self.device
                    self.logits.append(array(row));self.head_inputs.append(array(args[0][0,-1]));self.frame['head_SHA256']=tensor_sha(row);self.frame['head_shape']=list(out.shape);self.frame['head_dtype']=str(row.dtype);self.frame['argmax']=int(row.argmax().item())
            def lm_post(m,args,kw,out):
                if not self.active:return
                n=len(self.frames);length=self.compressed_length+n;cache=out.past_key_values
                assert cache is not None and len(cache.layers)==24
                pieces=[];newpairs=[]
                for i,layer in enumerate(cache.layers):
                    k,v=layer.keys,layer.values
                    assert k.shape==v.shape==(1,2,length,64) and k.dtype==v.dtype==torch.bfloat16 and k.device==v.device==self.device
                    if n:
                        pk,pv=self.previous_pairs[i]
                        assert torch.equal(k[:,:,:-1],pk) and torch.equal(v[:,:,:-1],pv),(n,i,'native cache prefix changed')
                        pieces.extend((k[0,:,-1:],v[0,:,-1:]))
                    else:pieces.extend((k[0],v[0]))
                    newpairs.append((k,v))
                packed=torch.stack(pieces).reshape(24,2,2,-1,64)
                if n:self.deltas.append(array(packed))
                else:self.arrays['native_cache_prefill']=array(packed)
                self.previous_pairs=newpairs;self.frame['cache_seq_len']=length;self.frame['unchanged_prefix_exact_checked_layers']=24 if n else 0;self.frames.append(self.frame)
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
            extract=self.model.extract_feature
            def observe_own_visual(*args,**kw):
                result=extract(*args,**kw)
                if self.active:self.activations['own_visual_features']=self._record('own_visual_features',result)
                return result
            self.model.extract_feature=observe_own_visual

