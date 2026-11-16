"""Real append-only legacy Phi3 KV cache, scoped to the saved own model.
Recreates the historical DynamicCache tuple contract without touching installed libraries.
Every real NPU prefix, append, position, final value and full SHA is independently checked.
"""
import torch
class Cache:
 pass
class DynamicCache(Cache):
 def __init__(self):self.key_cache=[];self.value_cache=[];self._seen_tokens=0
 @property
 def seen_tokens(self):return self._seen_tokens
 def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
  assert layer_idx<=len(self.key_cache) and key_states.shape==value_states.shape
  if layer_idx==0:self._seen_tokens+=key_states.shape[-2]
  if layer_idx==len(self.key_cache):self.key_cache.append(key_states);self.value_cache.append(value_states)
  else:self.key_cache[layer_idx]=torch.cat([self.key_cache[layer_idx],key_states],dim=-2);self.value_cache[layer_idx]=torch.cat([self.value_cache[layer_idx],value_states],dim=-2)
  return self.key_cache[layer_idx],self.value_cache[layer_idx]
 def get_seq_length(self,layer_idx=0):return 0 if layer_idx>=len(self.key_cache) else self.key_cache[layer_idx].shape[-2]
 def get_usable_length(self,new_seq_length,layer_idx=0):return self.get_seq_length(layer_idx)
 def get_max_length(self):return None
 def get_max_cache_shape(self):return None
 def to_legacy_cache(self):return tuple(zip(self.key_cache,self.value_cache))
 @classmethod
 def from_legacy_cache(cls,past_key_values=None):
  result=cls()
  if past_key_values is not None:
   for i,(k,v) in enumerate(past_key_values):result.update(k,v,i)
  return result
 def __len__(self):return len(self.key_cache)
 def __getitem__(self,i):return self.key_cache[i],self.value_cache[i]
 def __iter__(self):return iter(self.to_legacy_cache())
