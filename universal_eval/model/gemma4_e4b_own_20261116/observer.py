"""Own Gemma4 E4B complete lossless physical24/shared18 native KV observer.
Serialization and observation only; no learned tensor or model math replacement.
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
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def tsha(t):
    v=t.detach().cpu().contiguous();return hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
MID='google/gemma-4-E4B-it';LOG=logging.getLogger(__name__)
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

class Gemma4LosslessObserver(IModel):
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
                    assert t.device==self.device and t.dtype==torch.bfloat16 and t.shape[:2]==(1,2) and t.shape[-1] in [256,512]
                    name=f'physical_{i}_{field}';kind=(type(layer).__name__,'physical');groups.setdefault((kind,tuple(t.shape)),[]).append((name,t));entries[name]=dict(shape=list(t.shape),cache_class=type(cache).__name__,layer_class=type(layer).__name__,dtype=str(t.dtype))
        assert len(entries)==48
        for i,(key,value) in sorted(self.shared_latest.items()):
            assert i in ['full_attention','sliding_attention']
            for field,t in [('keys',key),('values',value)]:
                name=f'shared_full_native_{i}_{field}';groups.setdefault((('shared_full',str(i)),tuple(t.shape)),[]).append((name,t));entries[name]=dict(shape=list(t.shape),cache_class='original_forward_shared_kv_states',layer_class=str(i),dtype=str(t.dtype))
        assert len(self.shared_latest)==2 and len(entries)==52
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
    def unload(self):
        for h in getattr(self,'handles',[]):h.remove()
        self.model=None;self.cache_previous={};torch.npu.empty_cache()
