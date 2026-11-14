"""Own Mono-InternVL visual-expert HF BF16 reference on Ascend, original serial decode."""
from datetime import datetime
from functools import wraps
import hashlib,json,logging,os,shutil,sys,traceback,zipfile
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import torch
import torch_npu
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig,AutoModel,AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from universal_eval.model.interface import IModel

LOG=logging.getLogger(__name__)
MID='OpenGVLab/Mono-InternVL-2B'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def array(t):
    t=t.detach().cpu().contiguous()
    return t.view(torch.uint16).numpy() if t.dtype==torch.bfloat16 else t.numpy()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False)
def save_npz(p,values):
    with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,value in values.items():
            with z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(value),allow_pickle=False)

class MonoInternVLOriginalNative(IModel):
    def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
        self.protocol_file=Path(protocol_file);self.protocol_sha=protocol_sha256
        self.mid=protocol['model_id'];self.devices=npu_devices
    @property
    def model_name(self):return MID
    @property
    def model_type(self):return 'vlm'
    @property
    def inference_framework(self):return 'Transformers'
    def download(self,model_name,save_dir=None):raise RuntimeError('Original own local checkpoint required')
    def load(self,model_path,**kwargs):
        assert self.mid==MID and sha(self.protocol_file)==self.protocol_sha
        self.p=js(self.protocol_file);self.W=Path(model_path)
        assert str(self.W)==self.p['weight_path'] and len(self.devices)==1
        self.device=torch.device('npu:'+str(self.devices[0]));torch.npu.set_device(self.device);torch.set_num_threads(4)
        for kind in ('matmul','conv'):getattr(torch.npu,kind).allow_hf32=False
        for name,digest in self.p['own_assets_SHA256'].items():assert sha(self.W/name)==digest,name
        for name,rec in self.p['own_weight_files'].items():assert sha(self.W/name)==rec['SHA256'],name
        for path,digest in self.p['static_evidence_SHA256'].items():assert sha(path)==digest,path
        now=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
        self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('MonoInternVL2B_original_BF16_'+now+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
        os.environ.update(HF_MODULES_CACHE=str(self.home/'own_hf_modules'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        # The dynamic module cache is selected at transformers import time.
        import transformers.dynamic_module_utils as dm
        dm.HF_MODULES_CACHE=str(self.home/'own_hf_modules')
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.W),trust_remote_code=True,local_files_only=True)
        from universal_eval.model.mono_internvl_own_native_2109_20261111.configuration_internvl_chat import InternVLChatConfig
        from universal_eval.model.mono_internvl_own_native_2109_20261111.modeling_internvl_chat import InternVLChatModel
        cfg=InternVLChatConfig(**js(self.W/'config.json'));cls=InternVLChatModel
        assert cfg.use_backbone_lora==cfg.use_llm_lora==0
        original_init=cls.__init__
        @wraps(original_init)
        def init(instance,config,*a,**kw):
            rs=getattr(config.llm_config,'rope_scaling',None)
            if isinstance(rs,dict) and 'type' not in rs:config.llm_config.rope_scaling=None
            original_init(instance,config,*a,**kw)
            if not hasattr(instance,'all_tied_weights_keys'):instance.post_init()
        cls.__init__=init
        # Same own original HF loader and BF16 conversion. No fake CUDA or remote-code stubs.
        self.model=cls.from_pretrained(str(self.W),config=cfg,dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True).eval().to(self.device)
        self.rope_buffers={}
        for name,module in self.model.named_modules():
            inv=getattr(module,'inv_freq',None)
            if inv is None or not torch.is_tensor(inv) or not inv.is_floating_point():continue
            dim=inv.numel()*2;base=float(getattr(module,'base',10000.))
            fresh=1.0/(base**(torch.arange(0,dim,2).float()/dim))
            if not torch.allclose(inv.detach().float().cpu(),fresh,rtol=1e-3,atol=1e-6):module.inv_freq=fresh.to(device=inv.device,dtype=inv.dtype)
            for k in ('cos_cached','sin_cached'):
                if hasattr(module,k):delattr(module,k)
            module.max_seq_len_cached=0
            self.rope_buffers[name]=dict(base=base,dtype=str(module.inv_freq.dtype),SHA256=tensor_sha(module.inv_freq),shape=list(module.inv_freq.shape))
        assert len(self.rope_buffers)==24 and all(x['base']==1000000 for x in self.rope_buffers.values())
        expected={x['name']:x for x in js(self.p['actual_AA_source_manifest'])};params=dict(self.model.named_parameters(remove_duplicate=False))
        assert len(params)==253 and set(params)==set(expected)
        loaded={}
        for filename in self.p['own_weight_files']:
            with safe_open(self.W/filename,framework='pt',device='cpu') as f:
                for name in f.keys():
                    src=f.get_tensor(name);ref=expected[name];actual=params[name]
                    assert list(src.shape)==ref['shape']==list(actual.shape) and str(src.dtype)==ref['source_dtype']
                    assert tensor_sha(src)==ref['source_SHA256'] and tensor_sha(src.bfloat16())==ref['cast_SHA256']==ref['actual_CUDA_SHA256']
                    assert actual.device==self.device and actual.dtype==torch.bfloat16 and tensor_sha(actual)==ref['cast_SHA256'],name
                    loaded[name]=dict(ref,actual_NPU_SHA256=tensor_sha(actual),actual_NPU_device=str(actual.device))
        assert len({x.data_ptr() for x in params.values()})==253
        self.weight_proof=self.home/'own253_source_cast_actual_NPU.json';write(self.weight_proof,dict(weights=loaded,rope_buffers=self.rope_buffers,aliases=[]))
        self.ref=js(self.p['independent_R_CPU200'])['records'];assert len(self.ref)==200
        assert self.model.template=='internlm2-chat' and self.model.num_image_token==256 and self.model.config.max_dynamic_patch==24
        assert type(self.model.language_model.model.layers[0].attention).__name__=='InternLM2Attention'
        self.handles=[];self._install();self.index=0;self.report_index=0;self.details=[];self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False)
        self.profile=self.home/'npu_trace.json';self.failed=False;self.profile_done=False
        self.lineage=dict(hardware='Ascend910B',devices=self.devices,runtime_dtype='BF16',runtime='own Mono native InternVisionPatch + InternLM2 visual-expert HF manual greedy',actual_outer_batch_size=1,actual_inner_batch_size=1,max_new_tokens=2048,attention='own original eager',use_cache='24-layer legacy tuple KV; image expert only original prefill',evidence_root=str(self.home),protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,weight_evidence=str(self.weight_proof),weight_evidence_SHA256=sha(self.weight_proof),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),first_real_vision_and_prefill_profiler=str(self.profile),token354_policy='own tokenizer unchanged; no input354; generated354 retained and explicitly audited',profiler_scope='first original vision and decoder prefill only',activation_array_retention='all24 LM prefill arrays first3; all200 layer activation SHA/shape/dtype/count; all200 complete KV/head/vision/connector/merged-input arrays unchanged')
        LOG.info('Mono own253 BF16 loaded on %s, original visual-expert/serial1/2048',self.device)
    def _record(self,name,value,retain=True):
        assert value.device==self.device,(name,value.device)
        meta=dict(shape=list(value.shape),dtype=str(value.dtype),SHA256=tensor_sha(value))
        if retain and name not in self.arrays:self.arrays[name]=array(value)
        return meta
    def _install(self):
        lm=self.model.language_model
        def lm_pre(m,args,kw):
            assert not args
            n=len(self.frames);pos=kw['position_ids'];self.frame=dict(step=n,positions=pos.detach().cpu().tolist(),input_ids=None if kw.get('input_ids') is None else kw['input_ids'].detach().cpu().tolist(),actual_visual_token_mask=None,attention=[],expert_calls={})
            if n==0:
                self.activations['merged_embeddings']=self._record('merged_embeddings',kw['inputs_embeds'])
                visual=kw['visual_token_mask'];self.frame['actual_visual_token_mask']=self._record('visual_token_mask',visual)
        def lm_post(m,args,kw,out):
            assert out.logits.device==self.device and out.logits.shape[-1]==92553
            row=out.logits[0,-1];self.logits.append(array(row));self.frame['full_logits_shape']=list(out.logits.shape);self.frame['logits_dtype']=str(out.logits.dtype);self.frame['argmax']=int(row.argmax().item())
            cache=out.past_key_values;end=len(self.ids)+len(self.frames);assert len(cache)==24
            block=[]
            for k,v in cache:
                assert k.shape==v.shape==(1,8,end,128) and k.dtype==v.dtype==torch.bfloat16
                block.extend((k[0] if not self.frames else k[0,:,-1:,:],v[0] if not self.frames else v[0,:,-1:,:]))
            value=torch.stack(block).reshape(24,2,8,-1,128)
            if not self.frames:self.arrays['native_cache_prefill']=array(value)
            else:self.deltas.append(array(value))
            self.last_cache=cache;self.frame['cache_seq_len']=end;self.frames.append(self.frame)
        self.handles.extend((lm.register_forward_pre_hook(lm_pre,with_kwargs=True),lm.register_forward_hook(lm_post,with_kwargs=True)))
        def observation(name,retain):
            def hook(m,args,out):
                self.counts[name]=self.counts.get(name,0)+1
                if self.frames:return
                if not torch.is_tensor(out):out=out[0] if isinstance(out,(tuple,list)) else out.last_hidden_state
                self.activations[name]=self._record(name,out,retain or self.index<3)
            return hook
        self.handles.append(self.model.vision_model.register_forward_hook(observation('vision_patch',True)))
        self.handles.append(self.model.mlp1.register_forward_hook(observation('connector',True)))
        for i,layer in enumerate(lm.model.layers):
            self.handles.append(layer.register_forward_hook(observation('LM_layer_%02d'%i,self.p.get('retain_all_prefill_layers',False))))
            def attention_pre(m,args,kw,i=i):
                mask=kw.get('attention_mask');mn=None
                if mask is not None:mn='causal_mask_'+tensor_sha(mask);self._record(mn,mask)
                self.frame['attention'].append(dict(layer=i,mask=mn,positions=kw['position_ids'].detach().cpu().tolist(),past_length=0 if kw.get('past_key_value') is None else kw['past_key_value'][0].shape[2]))
            self.handles.append(layer.attention.register_forward_pre_hook(attention_pre,with_kwargs=True))
            for suffix in ('feed_forward','feed_forward_ve'):
                def expert_pre(m,args,i=i,suffix=suffix):
                    key='%02d_%s'%(i,suffix);self.frame['expert_calls'][key]=list(args[0].shape)
                self.handles.append(getattr(layer,suffix).register_forward_pre_hook(expert_pre))
    def _pixels(self,path):
        with Image.open(path) as im:image=im.convert('RGB')
        ratios=sorted({(i,j) for n in range(1,25) for i in range(1,n+1) for j in range(1,n+1) if i*j<=24},key=lambda x:x[0]*x[1])
        best=(1,1);delta=float('inf');ratio=image.width/image.height
        for target in ratios:
            diff=abs(ratio-target[0]/target[1])
            if diff<delta:best=target;delta=diff
            elif diff==delta and image.width*image.height>.5*448*448*target[0]*target[1]:best=target
        def resize(im,size):
            w,h=size
            if im.width!=w:im=im.resize((w,im.height),Image.Resampling.BICUBIC)
            if im.height!=h:im=im.resize((im.width,h),Image.Resampling.BICUBIC)
            return im
        full=resize(image,(448*best[0],448*best[1]));tiles=[full.crop((j%best[0]*448,j//best[0]*448,(j%best[0]+1)*448,(j//best[0]+1)*448)) for j in range(best[0]*best[1])]
        if len(tiles)>1:tiles.append(resize(image,(448,448)))
        raw=np.stack([np.asarray(x,dtype=np.uint8).transpose(2,0,1) for x in tiles]);x=torch.from_numpy(raw.copy()).float()/255
        return (x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    def predict(self,prompt,media_path=None):
        i=self.index;ref=self.ref[i];assert prompt==ref['actual_pipeline_prompt'] and Path(media_path).resolve()==Path(ref['R_media_path']).resolve()
        assert sha(media_path)==ref['media_SHA256'];pixel=self._pixels(media_path);assert tensor_sha(pixel)==ref['CPU_F32_pixels_SHA256'] and tensor_sha(pixel.bfloat16())==ref['actual_BF16_pixels_SHA256']
        conv=sys.modules[type(self.model).__module__].get_conv_template(self.model.template);conv.system_message=self.model.system_message
        question=prompt if '<image>' in prompt else '<image>\n'+prompt;conv.append_message(conv.roles[0],question);conv.append_message(conv.roles[1],None)
        query=conv.get_prompt().replace('<image>','<img>'+'<IMG_CONTEXT>'*256*pixel.shape[0]+'</img>',1)
        ids=self.tokenizer(query,return_tensors='pt')['input_ids'];assert query==ref['query'] and ids[0].tolist()==ref['input_ids']
        self.ids=ids[0].tolist();self.model.img_context_token_id=self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>');eos=self.tokenizer.convert_tokens_to_ids(conv.sep)
        assert eos==ref['actual_sep_eos'] and self.tokenizer.eos_token_id==2 and 354 not in self.ids
        self.arrays={'input_ids':ids.numpy(),'pixels_BF16':array(pixel.bfloat16())};self.frames=[];self.logits=[];self.deltas=[];self.activations={};self.counts={};self.last_cache=None
        assert shutil.disk_usage(self.home).free>4*1024**3+len(self.ids)*24*2*8*128*2*3,'Lossless original evidence space reserve'
        x=ids.to(self.device);pixel=pixel.to(device=self.device,dtype=torch.bfloat16)
        kw=dict(pixel_values=pixel,input_ids=x,attention_mask=torch.ones_like(x),position_ids=torch.arange(x.shape[1],device=self.device).unsqueeze(0),image_flags=torch.ones(pixel.shape[0],dtype=torch.long,device=self.device),use_cache=True,return_dict=True)
        gen=[];error=None
        try:
            with torch.no_grad():
                if not self.profile_done:
                    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:out=self.model(**kw)
                    prof.export_chrome_trace(str(self.profile));self.profile_done=True
                else:out=self.model(**kw)
                past=out.past_key_values;logits=out.logits[:,-1];cur=len(self.ids)
                for _ in range(2048):
                    nxt=logits.argmax(-1,keepdim=True);token=int(nxt[0,0].item())
                    if token in (eos,2):break
                    gen.append(token)
                    assert shutil.disk_usage(self.home).free>4*1024**3+sum(v.nbytes for v in self.arrays.values())+sum(v.nbytes for v in self.deltas),'Lossless original evidence reserve'
                    out=self.model.language_model(inputs_embeds=self.model.language_model.get_input_embeddings()(nxt),attention_mask=None,position_ids=torch.tensor([[cur]],device=self.device),past_key_values=past,use_cache=True,return_dict=True)
                    past=out.past_key_values;logits=out.logits[:,-1];cur+=1
            raw=self.tokenizer.decode(gen,skip_special_tokens=True).split(conv.sep)[0].strip()
            self.arrays['all_original_full_head_logits']=np.stack(self.logits)
            self.arrays['native_cache_decode_delta']=np.concatenate(self.deltas,axis=3) if self.deltas else np.empty((24,2,8,0,128),dtype=np.uint16)
            final=[[tensor_sha(v) for v in pair] for pair in self.last_cache]
            record=dict(index=i,actual_model_prompt=prompt,query=query,media_path=str(media_path),media_SHA256=ref['media_SHA256'],input_ids=self.ids,generated_tokens=gen,raw_output=raw,error=None,actual_native_forwards=len(self.frames),frames=self.frames,activation_prefill=self.activations,activation_counts=self.counts,final_cache_SHA256=final,stop_ids=[eos,2],original_budget_hit=len(gen)==2048,generated_token354_positions=[j for j,t in enumerate(gen) if t==354])
            archive=self.home/('sample_%05d.npz'%i);save_npz(archive,self.arrays);record.update(archive=str(archive),archive_SHA256=sha(archive))
            rp=self.home/('sample_%05d.json'%i);write(rp,record)
            detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(self.frames),generated_tokens=gen,raw_output=raw,error=None)
            with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
            self.details.append(detail);self.index+=1
            LOG.info('Mono actual sample %d completed, %d native forwards, generated %d, native visual expert true',i,len(self.frames),len(gen))
            self.arrays={};self.deltas=[];self.logits=[];self.last_cache=None
            return raw
        except Exception:
            self.failed=True;write(self.home/('sample_%05d_failed.json'%i),dict(index=i,committed_native_forwards=len(self.frames),error=traceback.format_exc()));raise
    def batch_predict(self,batch):
        assert len(batch)==1
        return [self.predict(prompt,media) for prompt,media in batch]
    def get_sample_execution(self,prompt,media_path):
        i=self.report_index;self.report_index+=1
        if i==0:write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),first_prefill_profiler_SHA256=sha(self.profile)))
        assert i<len(self.details)
        return self.details[i]
    def get_execution_lineage(self):return self.lineage
    def unload(self):
        for h in getattr(self,'handles',[]):h.remove()
        self.model=None;self.last_cache=None;torch.npu.empty_cache()
