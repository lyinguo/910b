"""Read-only complete tensor observation for Kanana's original native modules.

Every tensor occurrence is independently copied in full. Equal complete raw
values may share one standard NPY member; all occurrences retain their own
semantic references, dtype, shape and SHA. This module changes no forward,
generation, cache update, model precision or numerical acceptance threshold.
"""
from collections.abc import Mapping
from pathlib import Path
import hashlib,json,os,shutil
import numpy as np
import torch
from universal_eval.utils.complete_standard_Fortran_NPZ_values_20261117 import write_full_standard_npz


def logical_raw(array):
    return np.ascontiguousarray(array).reshape(-1).view(np.uint8).tobytes()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


class CompleteNativeTensorJournal:
    """One complete sample, with immutable values and no retained device views."""
    def __init__(self,home,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3):
        root=Path('/root/projects/comperror/workspace/diagnostics').resolve()
        self.home=Path(home).resolve()
        if not self.home.is_relative_to(root):
            raise ValueError('Capture must remain inside its own diagnostics directory')
        self.home.mkdir(parents=True,exist_ok=False)
        self.hard_reserve=int(hard_reserve_bytes)
        self.log_reserve=int(log_reserve_bytes)
        if self.hard_reserve<3*1024**3 or self.log_reserve<1024**3:
            raise ValueError('Original hard and log reserves must not be reduced')
        self.index=None;self.arrays={};self.descriptors={};self.events=[];self.tensor_occurrences=0
        self.full_raw_compares_for_shared_occurrences=0

    def begin(self,index):
        if self.index is not None:
            raise RuntimeError('An active sample cannot be overwritten')
        if not isinstance(index,int) or index<0:
            raise ValueError('A genuine nonnegative sample index is required')
        if (self.home/f'sample_{index:05d}.npz').exists() or (self.home/f'sample_{index:05d}.json').exists():
            raise FileExistsError('An existing sample must never be overwritten')
        self.index=index;self.arrays={};self.descriptors={};self.events=[];self.tensor_occurrences=0
        self.full_raw_compares_for_shared_occurrences=0

    def tensor(self,value):
        if self.index is None:
            raise RuntimeError('No active sample')
        if value.device.type=='meta' or value.layout!=torch.strided:
            raise TypeError('A real complete strided tensor value is required')
        # copy=True applies even to CPU input: no alias survives later mutation.
        cpu=value.detach().to(device='cpu',copy=True).contiguous()
        logical_dtype=str(cpu.dtype);shape=list(cpu.shape)
        if cpu.dtype==torch.bfloat16:
            array=cpu.view(torch.uint16).numpy().copy()
        else:
            try:array=cpu.numpy().copy()
            except (TypeError,RuntimeError) as exc:
                raise TypeError(f'Unsupported exact capture dtype: {cpu.dtype}') from exc
        if list(array.shape)!=shape or array.dtype.hasobject:
            raise AssertionError('Logical rank/shape/dtype must remain complete')
        raw=logical_raw(array)
        descriptor=dict(dtype=logical_dtype,storage_dtype=array.dtype.str,shape=shape,
                        raw_C_order_SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        encoded=json.dumps(descriptor,sort_keys=True,separators=(',',':')).encode()
        key='tensor_'+hashlib.sha256(encoded).hexdigest()
        if key in self.arrays:
            # A matching digest alone never authorizes sharing a captured value.
            assert self.descriptors[key]==descriptor
            assert self.arrays[key].shape==array.shape and self.arrays[key].dtype==array.dtype
            assert logical_raw(self.arrays[key])==raw
            self.full_raw_compares_for_shared_occurrences+=1
        else:
            self.arrays[key]=array;self.descriptors[key]=descriptor
        self.tensor_occurrences+=1
        return dict(kind='tensor',key=key,**descriptor)

    def pack(self,value,_active=None):
        if isinstance(value,torch.Tensor):return self.tensor(value)
        if value is None or isinstance(value,(str,bool,int,float)):return value
        if isinstance(value,torch.dtype):return dict(kind='torch.dtype',value=str(value))
        if isinstance(value,torch.device):return dict(kind='torch.device',value=str(value))
        if isinstance(value,type):return dict(kind='class',value=value.__module__+'.'+value.__qualname__)
        active=set() if _active is None else _active
        ident=id(value)
        if ident in active:raise TypeError('Unexpected cyclic value in native observation')
        active.add(ident)
        try:
            if isinstance(value,Mapping):
                if not all(isinstance(k,str) for k in value):raise TypeError('Native mappings require exact string keys')
                return dict(kind='mapping',class_name=type(value).__module__+'.'+type(value).__qualname__,
                            values={k:self.pack(v,active) for k,v in value.items()})
            if isinstance(value,(tuple,list)):
                return dict(kind='tuple' if isinstance(value,tuple) else 'list',values=[self.pack(v,active) for v in value])
            # Real HF Cache and layer instances only. No surrogate cache or stub.
            if type(value).__module__=='transformers.cache_utils':
                from transformers.cache_utils import Cache,CacheLayerMixin
                if isinstance(value,(Cache,CacheLayerMixin)):
                    return dict(kind='native_cache_state',class_name=type(value).__module__+'.'+type(value).__qualname__,
                                attributes={k:self.pack(v,active) for k,v in vars(value).items()})
            raise TypeError('Unsupported original native value: '+type(value).__module__+'.'+type(value).__qualname__)
        finally:active.remove(ident)

    def record(self,name,value,phase='value',call_id=None):
        if self.index is None:raise RuntimeError('No active sample')
        row=dict(event_index=len(self.events),name=str(name),phase=str(phase),call_id=call_id,value=self.pack(value))
        self.events.append(row)
        return row

    def finish(self,extra=None):
        if self.index is None:raise RuntimeError('No active sample')
        if not self.events or not self.arrays:raise RuntimeError('Empty captures cannot be accepted')
        # An uncompressed bound protects the entire write plus the original
        # hard/log reserves. This does not replace model sample/200 guards.
        complete_bytes=sum(a.nbytes+4096 for a in self.arrays.values())
        free=shutil.disk_usage(self.home).free
        if free<self.hard_reserve+self.log_reserve+complete_bytes:
            raise RuntimeError(f'Complete value write capacity refused: free={free}, required={self.hard_reserve+self.log_reserve+complete_bytes}')
        index=self.index;array_file=self.home/f'sample_{index:05d}.npz'
        write_full_standard_npz(array_file,self.arrays)
        # Independently reopen every complete standard NPY member, including
        # scalar and zero-element arrays, before releasing the CPU copies.
        with np.load(array_file,allow_pickle=False) as reopened:
            assert set(reopened.files)==set(self.arrays)
            for key,array in self.arrays.items():
                actual=reopened[key]
                assert actual.dtype==array.dtype and actual.shape==array.shape
                assert logical_raw(actual)==logical_raw(array)
        manifest=dict(schema='kanana_complete_native_values_v1',sample_index=index,
            full_value_file=str(array_file),full_value_file_SHA256=file_sha(array_file),
            tensors=self.descriptors,events=self.events,tensor_occurrences=self.tensor_occurrences,
            unique_complete_arrays=len(self.arrays),independent_full_copies_for_every_tensor_occurrence=True,
            full_raw_compares_for_shared_occurrences=self.full_raw_compares_for_shared_occurrences,
            all_complete_standard_NPY_values_reopened_and_raw_bits_equal=True,
            no_sampling_truncation_precision_math_or_semantic_reference_reduction=True,extra=extra or {})
        path=self.home/f'sample_{index:05d}.json'
        with path.open('x',encoding='utf8') as stream:json.dump(manifest,stream,ensure_ascii=False,indent=2);stream.write('\n')
        result=dict(path=str(path),SHA256=file_sha(path),full_value_file=str(array_file),full_value_file_SHA256=manifest['full_value_file_SHA256'],
                    sample_index=index,tensor_occurrences=self.tensor_occurrences,unique_complete_arrays=len(self.arrays))
        self.index=None;self.arrays={};self.descriptors={};self.events=[]
        return result


class KananaCompleteNativeModuleObserver:
    """Only observational PyTorch hooks; every original callable is retained."""
    def __init__(self,journal):
        self.journal=journal;self.handles=[];self.stack={};self.counter=0;self.observed_names=[]

    def _register_native_module(self,label,module):
        # Used by strict full-topology installation and isolated unit tests.
        def pre(mod,args,kwargs):
            if self.journal.index is None:return
            call=self.counter;self.counter+=1;self.stack.setdefault(label,[]).append(call)
            self.journal.record(label,dict(args=args,kwargs=kwargs),phase='forward_pre',call_id=call)
        def post(mod,args,kwargs,output):
            if self.journal.index is None:return
            if not self.stack.get(label):raise RuntimeError('Missing original pre-hook occurrence')
            call=self.stack[label].pop()
            self.journal.record(label,output,phase='forward_post',call_id=call)
        self.handles.append(module.register_forward_pre_hook(pre,with_kwargs=True))
        self.handles.append(module.register_forward_hook(post,with_kwargs=True))
        self.observed_names.append(label)

    def attach(self,model):
        if self.handles:raise RuntimeError('Observer is already attached')
        names=dict(model.named_modules())
        required=['vision_model','vision_model.patch_embed','abstractor','abstractor.net.0','abstractor.net.1','abstractor.net.2',
                  'language_model','language_model.model','language_model.lm_head']
        required += [f'vision_model.blocks.{i}' for i in range(32)]
        required += [f'language_model.model.layers.{i}' for i in range(32)]
        if any(n not in names for n in required):raise ValueError('The original complete Kanana topology is required')
        if model.language_model.config.vocab_size!=128259:raise ValueError('The own full 128259 head is required')
        self.original_forward_functions={n:getattr(m.forward,'__func__',m.forward) for n,m in names.items()}
        for name,module in names.items():
            label=name or '<root>'
            self._register_native_module(label,module)
        assert all(getattr(m.forward,'__func__',m.forward) is self.original_forward_functions[n] for n,m in names.items())
        return dict(all_original_forward_functions_identical=True,observed_native_modules=len(names),
                    complete_32_vision_and_32_Llama_layers_and_full_head_registered=True,not_runtime_model_acceptance=True)

    def detach(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
        if any(self.stack.values()):raise RuntimeError('An incomplete native forward remains in the capture')

