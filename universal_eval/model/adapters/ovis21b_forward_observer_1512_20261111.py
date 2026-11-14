"""Read-only tensor and execution observation for the own SmolVLM2 checkpoint."""
import functools,hashlib,json,re,zipfile
from pathlib import Path
import numpy as np
import torch
def raw_array(t):
    t=t.detach().to('cpu').contiguous()
    return t.view(torch.uint8).numpy().tobytes(),t
def digest_tensor(t):
    raw,c=raw_array(t)
    return {'shape':list(c.shape),'dtype':str(c.dtype),'SHA256':hashlib.sha256(raw).hexdigest()}
def source_cuda_or_npu_manifest(model,weight):
    from safetensors import safe_open
    state=model.state_dict()
    result={};used=set()
    for file in sorted(Path(weight).glob('*.safetensors')):
        with safe_open(file,framework='pt',device='cpu') as f:
            for key in f.keys():
                source=f.get_tensor(key)
                options=[key,key.replace('model.text_model.','model.language_model.'),key.replace('text_model.','language_model.')]
                possible=[k for k in dict.fromkeys(options) if k in state]
                assert len(possible)==1,(key,possible)
                actual_name=possible[0];actual=state[actual_name]
                assert actual.device.type in ('cuda','npu') and source.shape==actual.shape,(key,actual.device,source.shape,actual.shape)
                expected=source.to(dtype=actual.dtype)
                source_record=digest_tensor(source);expected_record=digest_tensor(expected);actual_record=digest_tensor(actual)
                assert expected_record==actual_record,(key,'actual loaded parameter differs from own source cast')
                result[key]={'source':source_record,'cast':expected_record,'actual_parameter_name':actual_name,
                    'actual_device_type':actual.device.type,'physical_storage':actual.untyped_storage().data_ptr(),
                    'physical_storage_offset':actual.storage_offset()}
                used.add(actual_name)
    assert used==set(state),('unmatched state',sorted(set(state)-used))
    aliases={}
    for n,t in model.named_parameters(remove_duplicate=False):
        aliases.setdefault((str(t.device),t.untyped_storage().data_ptr()),[]).append(n)
    return {'source_tensor_count':len(result),'source_to_actual':result,
        'physical_aliases':[v for v in aliases.values() if len(v)>1],
        'actual_parameter_dtypes':sorted({str(t.dtype) for t in model.parameters()})}
class ForwardObserver:
    def __init__(self,model):
        self.model=model;self.enabled=False;self.capture_generation=False;self.capture_dense=True;self.saved_keys=set();self.index=None;self.steps=[];self.outputs=[];self.stage=[];self.arrays={};self.handles=[];self.current=None
        self.handles.append(model.register_forward_pre_hook(self.before,with_kwargs=True))
        self.handles.append(model.register_forward_hook(self.after,with_kwargs=True))
        for name,module in model.named_modules():
            if type(module).__name__ in ('AIMv2Block','Qwen2DecoderLayer','AIMv2Attention'):
                self.handles.append(module.register_forward_pre_hook(self.layer_before(name),with_kwargs=True))
                self.handles.append(module.register_forward_hook(self.layer_after(name),with_kwargs=True))
        original=model.generate
        @functools.wraps(original)
        def generate(*args,**kwargs):
            if not (self.enabled or self.capture_generation):return original(*args,**kwargs)
            call={k:v for k,v in kwargs.items() if not isinstance(v,torch.Tensor)}
            result=original(*args,**kwargs)
            ids=result.detach().cpu().tolist()
            prefix=0
            self.outputs.append({'index':self.index,'input_token_ids':[],
                 'generated_token_ids':ids[0][prefix:],'generation_arguments':call,
                 'all_generated_sequence_length':len(ids[0]),'input_length':prefix})
            return result
        model.generate=generate
    def tensor(self,t,store=True,force_store=False):
        rec=digest_tensor(t)
        if store and (self.capture_dense or force_store):
            k=rec['SHA256']+'_'+str(t.dtype).replace('torch.','')
            if k not in self.arrays and k not in self.saved_keys:
                c=t.detach().to('cpu').contiguous()
                self.arrays[k]=c.view(torch.uint16).numpy().copy() if c.dtype==torch.bfloat16 else c.numpy().copy()
            rec['array_key']=k
        return rec
    def simple(self,v):
        if isinstance(v,torch.Tensor):return self.tensor(v,force_store=not v.is_floating_point())
        if isinstance(v,(list,tuple)):return [self.simple(t) for t in v]
        if v is None or isinstance(v,(int,float,bool,str)):return v
        return {'class':type(v).__name__}
    def before(self,module,args,kw):
        if not self.enabled:return
        self.current={'index':self.index,'forward_index':len([x for x in self.steps if x['index']==self.index]),
            'arguments':{k:self.simple(v) for k,v in kw.items() if k!='past_key_values'}}
        pkv=kw.get('past_key_values')
        self.current['cache_before_length']=int(pkv.get_seq_length()) if pkv is not None else 0
    def after(self,module,args,kw,out):
        if not self.enabled:return
        assert self.current is not None
        logits=out.logits
        assert logits.shape[-1]==151936 and torch.isfinite(logits).all()
        self.current['whole_logits']=self.tensor(logits,force_store=True)
        self.current['last_position_argmax']=logits[:,-1,:].float().argmax(-1).detach().cpu().tolist()
        self.current['whole_vocabulary_size']=int(logits.shape[-1])
        cache=getattr(out,'past_key_values',None);records=[]
        if cache is not None:
            prev=self.current['cache_before_length']
            for i,l in enumerate(cache.layers):
                k=getattr(l,'keys',None);v=getattr(l,'values',None)
                if k is None:records.append({'layer':i,'K':None,'V':None});continue
                records.append({'layer':i,'K_full':digest_tensor(k),'V_full':digest_tensor(v),
                    'K_delta':self.tensor(k[:,:,prev:,:]),'V_delta':self.tensor(v[:,:,prev:,:]),
                    'delta_start':prev})
        self.current['native_cache']=records;self.steps.append(self.current);self.current=None
    def layer_before(self,name):
        def hook(m,args,kw):
            if self.enabled:
                self.stage.append({'index':self.index,'forward_index':len([s for s in self.steps if s['index']==self.index]),
                    'module':name,'class':type(m).__name__,'phase':'before',
                    'position_and_mask':{k:self.simple(v) for k,v in kw.items()
                        if k in ('attention_mask','position_ids','position_embeddings','cache_position')}})
        return hook
    def layer_after(self,name):
        def hook(m,args,kw,out):
            if self.enabled:
                v=out[0] if isinstance(out,tuple) else out
                if hasattr(v,'last_hidden_state'):v=v.last_hidden_state
                assert isinstance(v,torch.Tensor),(name,type(v))
                self.stage.append({'index':self.index,'forward_index':len([s for s in self.steps if s['index']==self.index]),
                    'module':name,'class':type(m).__name__,'phase':'after','output':self.tensor(v)})
        return hook
    def save(self,path):
        path=Path(path)
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for k,v in self.arrays.items():
                with archive.open(k+'.npy','w') as f:np.lib.format.write_array(f,v,allow_pickle=False)
        return {'path':str(path),'bytes':path.stat().st_size,
            'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'array_count':len(self.arrays)}

