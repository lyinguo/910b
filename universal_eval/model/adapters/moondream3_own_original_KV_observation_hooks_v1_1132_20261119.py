"""Candidate observation hooks for Moondream3's own physical KV implementation.

No wrapper performs model math, replaces a cache, or changes returned tensors.
Install before a private job starts, restore after the job, and reject any cache
reallocation. Full model and independent numerical verification are still needed.
"""
from pathlib import Path
import functools,hashlib,inspect,shutil,types
import numpy as np
import torch
from universal_eval.model.adapters.moondream3_own_complete_physical_KV_journal_v1_1128_20261119 import CompletePhysicalKVJournal

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class OriginalPhysicalKVObservation:
    def __init__(self,model,home,*,original_source_path,original_source_SHA256,min_free_bytes):
        self.model=model
        self.original_source=Path(original_source_path).resolve()
        assert sha(self.original_source)==original_source_SHA256
        assert self.original_source.name=='moondream.py'
        assert model.config.text.n_layers==24 and model.config.text.max_context==4096
        assert model.config.text.n_heads==model.config.text.n_kv_heads==32
        assert model.config.text.dim==2048 and model.use_flex_decoding is True
        assert str(model.device)=='npu:4'
        assert isinstance(min_free_bytes,int) and min_free_bytes>=int(6.5*1024**3)
        self.min_free_bytes=min_free_bytes;self.blocks=list(model.text.blocks)
        assert len(self.blocks)==24
        self.cache_objects=[b.kv_cache for b in self.blocks]
        self.handles=[];self.installed=False;self.original_update_calls=0;self.original_prefix_load_calls=0
        self.journal=CompletePhysicalKVJournal(home)

    def _guard(self):
        assert shutil.disk_usage('/root/projects/comperror').free>=self.min_free_bytes
        assert [id(b.kv_cache) for b in self.blocks]==[id(c) for c in self.cache_objects], \
            'Original physical cache was replaced while observing'

    def _bits(self,t):
        assert t.device==self.model.device and t.dtype==torch.bfloat16
        # Independent full device copy precedes any equal-value sharing.
        return t.detach().cpu().contiguous().view(torch.uint16).numpy().copy()

    def _positions(self,t):
        assert t.device==self.model.device and t.dtype==torch.long
        return t.detach().cpu().contiguous().numpy().copy()

    def _bind(self,obj,name,callback):
        existed=name in obj.__dict__;prior=obj.__dict__.get(name)
        self.handles.append((obj,name,existed,prior))
        setattr(obj,name,types.MethodType(callback,obj))

    def install(self):
        assert not self.installed
        self._guard()
        assert Path(inspect.getsourcefile(type(self.model))).resolve()==self.original_source
        assert Path(inspect.getsourcefile(type(self.model).load_encoded_image)).resolve()==self.original_source
        try:
            for layer,cache in enumerate(self.cache_objects):
                assert Path(inspect.getsourcefile(type(cache))).resolve()==self.original_source
                assert type(cache).__name__=='KVCache'
                self.journal.register_initial(layer,self._bits(cache.k_cache),self._bits(cache.v_cache))
                original=cache.update
                def make_callback(layer,original):
                    @functools.wraps(original)
                    def observed(cache,pos_ids,k,v):
                        self._guard()
                        pos=self._positions(pos_ids);kb=self._bits(k);vb=self._bits(v)
                        result=original(pos_ids,k,v)
                        assert result[0] is cache.k_cache and result[1] is cache.v_cache
                        self.journal.record_actual_write(layer,pos,kb,vb,
                            self._bits(cache.k_cache),self._bits(cache.v_cache),
                            kind='original_KVCache_update')
                        self.original_update_calls+=1
                        return result
                    return observed
                self._bind(cache,'update',make_callback(layer,original))
            original_load=self.model.load_encoded_image
            @functools.wraps(original_load)
            def observed_load(model,encoded_image):
                self._guard()
                assert len(encoded_image.caches)==24
                prefixes=[(self._bits(k),self._bits(v)) for k,v in encoded_image.caches]
                result=original_load(encoded_image)
                for layer,(cache,(kb,vb)) in enumerate(zip(self.cache_objects,prefixes)):
                    assert kb.shape==vb.shape and kb.shape[2]==encoded_image.pos
                    pos=np.arange(kb.shape[2],dtype=np.int64)
                    self.journal.record_actual_write(layer,pos,kb,vb,
                        self._bits(cache.k_cache),self._bits(cache.v_cache),
                        kind='original_load_encoded_image_prefix')
                self.original_prefix_load_calls+=1
                return result
            self._bind(self.model,'load_encoded_image',observed_load)
            self.installed=True
        except BaseException:
            self.restore();raise
        return self

    def restore(self):
        while self.handles:
            obj,name,existed,prior=self.handles.pop()
            if existed:setattr(obj,name,prior)
            else:delattr(obj,name)
        self.installed=False

    def finalize(self):
        assert self.installed
        self._guard()
        return dict(self.journal.finalize(),original_update_calls=self.original_update_calls,
                    original_prefix_load_calls=self.original_prefix_load_calls,
                    original_source_SHA256=sha(self.original_source),
                    original_cache_math_unmodified=True,actual_model_NF_passed=False)
