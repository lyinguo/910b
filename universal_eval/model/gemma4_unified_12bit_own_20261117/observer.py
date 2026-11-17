"""Pass-through full-head/native-cache observer for own genuine HF Unified12B.

No learned operation is replaced. Every saved cache state is reconstructed
losslessly from its actual NPU prefill and appends/evictions. Prefix equality
is tested on NPU before retaining an append. This is storage, not pruning.
"""
from pathlib import Path
import hashlib,json,shutil,functools
import numpy as np
import torch

DTYPES={'torch.bfloat16':np.uint16,'torch.float16':np.uint16,'torch.float32':np.uint32,'torch.int64':np.int64,'torch.int32':np.int32,'torch.bool':np.bool_}

def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def fingerprint(t):
    a=t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy();h=hashlib.sha256()
    for i in range(0,a.size,8*1024*1024):h.update(memoryview(a[i:i+8*1024*1024]))
    return h.hexdigest()
def record(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=fingerprint(t),device=str(t.device))
def array(t):
    t=t.detach().cpu().contiguous();a=t.view(torch.uint8).numpy().reshape(-1).view(DTYPES[str(t.dtype)]).reshape(tuple(t.shape));return a.copy()

class FullObserver:
    def __init__(self,model,home,device):
        self.model=model;self.home=Path(home);self.device=device;self.handles=[];self.active=False;self.profile_done=False
        self.handles.append(model.lm_head.register_forward_pre_hook(self.head))
        self.handles.append(model.model.embed_vision.register_forward_hook(self.vision))
        self.handles.append(model.model.language_model.register_forward_pre_hook(self.language,with_kwargs=True))
        self.handles.append(model.model.language_model.rotary_emb.register_forward_hook(self.rope,with_kwargs=True))
        for i,layer in enumerate(model.model.language_model.layers):
            def hidden(module,args,out,i=i):
                if self.active and not self.frames and self.index<3:self.prefill_layers[str(i)]=self.save(f'actual_LM_prefill_layer{i}',out if torch.is_tensor(out) else out[0])
            self.handles.append(layer.register_forward_hook(hidden))
        self.original_forward=model.forward
        @functools.wraps(self.original_forward)
        def observed(*args,**kwargs):
            if not self.active:return self.original_forward(*args,**kwargs)
            assert shutil.disk_usage(self.home).free>4.5*1024**3+self.uncompressed_bytes,'Original3GiB hard+1GiB log+0.5GiB preflight and pending archive guard'
            self.current=dict(root_inputs={k:self.save('root_'+k,v) for k,v in kwargs.items() if torch.is_tensor(v)},masks={},rope={})
            self.updates={};cache=kwargs.get('past_key_values');assert cache is not None,'Original generate must provide its native DynamicCache'
            if not getattr(cache,'_own_value_observer_installed',False):
                original_update=cache.update
                @functools.wraps(original_update)
                def update(keys,values,layer_idx,*a,**kw):
                    result=original_update(keys,values,layer_idx,*a,**kw)
                    if self.active:self.cache_update(layer_idx,keys,values,result)
                    return result
                cache.update=update;cache._own_value_observer_installed=True
            if not self.profile_done:
                import torch_npu
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],record_shapes=False,profile_memory=False,with_stack=False) as prof:out=self.original_forward(*args,**kwargs)
                prof.export_chrome_trace(str(self.home/'npu_trace.json'));self.profile_done=True
            else:out=self.original_forward(*args,**kwargs)
            self.after(out);return out
        model.forward=observed
        self.original_processors=model._get_logits_processor
        def processors(*args,**kwargs):
            result=self.original_processors(*args,**kwargs)
            if self.active:
                self.processors=[]
                for item in result:
                    values={}
                    for k,v in vars(item).items():
                        if torch.is_tensor(v):values[k]=dict(values=v.detach().cpu().tolist(),dtype=str(v.dtype))
                        elif isinstance(v,torch.device):values[k]=str(v)
                        elif isinstance(v,(str,int,float,bool,list,type(None))):values[k]=v
                        else:raise TypeError((type(item).__name__,k,type(v)))
                    self.processors.append(dict(name=type(item).__name__,attributes=values))
            return result
        model._get_logits_processor=processors
    def begin(self,index,inputs):
        assert not self.active;self.index=index;self.arrays={};self.uncompressed_bytes=0;self.frames=[];self.previous=None;self.processors=[];self.visual=None;self.language_prefill=None;self.prefill_layers={};self.counter=0;self.inputs={k:self.save('original_'+k,v) for k,v in inputs.items()};self.active=True
    def save(self,name,t):
        key=f'{self.counter:07}_{name}';self.counter+=1;self.arrays[key]=array(t);self.uncompressed_bytes+=self.arrays[key].nbytes;return dict(key=key,**record(t))
    def head(self,module,args):
        if self.active:self.current['head_hidden']=self.save('full_head_input',args[0][0,-1])
    def vision(self,module,args,out):
        if self.active:self.visual=self.save('actual_full_vision_projection',out)
    def language(self,module,args,kwargs):
        if not self.active:return
        masks=kwargs.get('attention_mask')
        if isinstance(masks,dict):
            for k,v in masks.items():
                assert v is None or torch.is_tensor(v),(k,type(v));self.current['masks'][k]=None if v is None else self.save('mask_'+k,v)
        elif masks is not None:self.current['masks']['single']=self.save('mask_single',masks)
        if not self.frames:
            v=kwargs.get('inputs_embeds');assert v is not None;self.language_prefill=self.save('actual_injected_LM_prefill',v)
    def rope(self,module,args,kwargs,out):
        if not self.active:return
        pos=kwargs.get('position_ids',args[1] if len(args)>1 else None);kind=kwargs.get('layer_type',args[2] if len(args)>2 else None);assert pos is not None and kind in ['sliding_attention','full_attention']
        self.current['rope'][kind]=dict(position_ids=self.save('actual_rope_position_'+kind,pos),cos=self.save('actual_rope_cos_'+kind,out[0]),sin=self.save('actual_rope_sin_'+kind,out[1]))
    def cache_update(self,index,keys,values,result):
        assert index not in self.updates and len(result)==2
        k,v=result;assert k.dtype==v.dtype==torch.bfloat16 and k.device==v.device==self.device
        if self.previous is None:
            self.updates[index]=dict(mode='actual_prefill',K=self.save(f'full_attention_prefill_layer{index}_K',k),V=self.save(f'full_attention_prefill_layer{index}_V',v),attention_shape=list(k.shape),drop=0,prefix_unchanged_on_NPU=True)
        else:
            old_k,old_v=self.previous[index];assert keys.shape[-2]==values.shape[-2]==1
            assert k.shape[-2]==old_k.shape[-2]+1 and torch.equal(k[:,:,:-1],old_k) and torch.equal(v[:,:,:-1],old_v),('Actual returned attention prefix differs',index)
            assert torch.equal(k[:,:,-1:],keys) and torch.equal(v[:,:,-1:],values)
            self.updates[index]=dict(mode='actual_append_with_exact_eviction',K=self.save(f'actual_append_layer{index}_K',keys),V=self.save(f'actual_append_layer{index}_V',values),attention_shape=list(k.shape),prefix_unchanged_on_NPU=True)
    def after(self,out):
        logits=out.logits[0,-1];assert logits.device==self.device and logits.dtype==torch.bfloat16 and logits.numel()==262144 and bool(torch.isfinite(logits).all());self.current['head_logits']=self.save('all_original_262144_head_logits',logits)
        cache=out.past_key_values;assert cache is not None and len(cache.layers)==48 and set(self.updates)==set(range(48));new=[];states=[]
        for i,layer in enumerate(cache.layers):
            k,v=layer.keys,layer.values;assert k.device==v.device==self.device and k.dtype==v.dtype==torch.bfloat16 and k.shape==v.shape
            n=k.shape[-2];seen=int(layer.get_seq_length());item=dict(layer=i,cache_class=type(layer).__name__,shape=list(k.shape),dtype=str(k.dtype),device=str(k.device),logical_seen=seen)
            item.update(self.updates[i]);item['retained_cache_length']=n;item['attention_returned_length']=item['attention_shape'][-2]
            if self.previous is not None:
                old_k,old_v=self.previous[i];before=old_k.shape[-2];drop=before+1-n;assert 0<=drop<=before and k.data_ptr()!=old_k.data_ptr() and v.data_ptr()!=old_v.data_ptr(),('Unexpected in-place or nonappend cache',i)
                keep=n-1;assert bool(torch.equal(old_k[:,:,drop:,:],k[:,:,:keep,:])) and bool(torch.equal(old_v[:,:,drop:,:],v[:,:,:keep,:])),('NPU actual prefix changed',i)
                item['drop']=drop
            states.append(item);new.append((k,v))
        self.current['cache']=states;self.frames.append(self.current);self.previous=new
    def finish(self,raw_decoded,raw_output,generated):
        self.active=False;assert len(generated)==len(self.frames) and self.previous is not None
        final=[dict(layer=i,K=record(k),V=record(v)) for i,(k,v) in enumerate(self.previous)]
        path=self.home/f'sample_{self.index:05}.npz';assert not path.exists();assert shutil.disk_usage(self.home).free>4.5*1024**3+self.uncompressed_bytes;np.savez_compressed(path,**self.arrays)
        value=dict(index=self.index,archive=str(path),archive_bytes=path.stat().st_size,archive_SHA256=file_sha(path),input_values=self.inputs,visual=self.visual,language_prefill=self.language_prefill,prefill_LM_layers=self.prefill_layers,frames=self.frames,logits_processors=self.processors,final_native_cache=final,raw_decoded=raw_decoded,raw_output=raw_output,generation_token_ids=generated,actual_native_forwards=len(self.frames),all_full_head_rows_and_native48KV_values_retained=True,lossless_cache_prefill_append_eviction_not_value_pruning=True)
        f=self.home/f'sample_{self.index:05}.json';assert not f.exists();f.write_bytes(json.dumps(value,ensure_ascii=False).encode());self.arrays={};self.previous=None;return dict(index=self.index,actual_execution_file=str(f),actual_execution_SHA256=hashlib.sha256(f.read_bytes()).hexdigest(),archive=str(path),archive_SHA256=value['archive_SHA256'],archive_bytes=value['archive_bytes'],actual_native_forwards=len(self.frames),error=None,raw_output=raw_output,generation_token_ids=generated)
    def preserve_failure(self,error,traceback_text):
        self.active=False;path=self.home/f'failed_sample_{self.index:05}.npz';assert not path.exists();archive_error=None
        try:
            assert shutil.disk_usage(self.home).free>4*1024**3+self.uncompressed_bytes,'Failure archive cannot violate original hard/log reserve'
            np.savez_compressed(path,**self.arrays)
        except Exception as failure:archive_error=repr(failure)
        value=dict(index=self.index,error=repr(error),traceback=traceback_text,frames=self.frames,current_incomplete_frame=getattr(self,'current',None),input_values=self.inputs,visual=self.visual,language_prefill=self.language_prefill,archive=str(path) if path.exists() else None,archive_SHA256=file_sha(path) if path.exists() else None,archive_bytes=path.stat().st_size if path.exists() else 0,archive_write_error=archive_error,actual_completed_native_forwards=len(self.frames),failed_run_not_accepted=True)
        out=self.home/f'failed_sample_{self.index:05}.json';assert not out.exists();out.write_bytes(json.dumps(value,ensure_ascii=False).encode());return value
