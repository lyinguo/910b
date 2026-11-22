"""Lossless full tensor occurrences, complete-array aliasing and real cache descriptors."""
from collections.abc import Mapping
import hashlib
import numpy as np
import torch

def raw_array(t):
 t=t.detach().cpu().contiguous().clone()
 return t.view(torch.uint16).numpy().copy() if t.dtype==torch.bfloat16 else t.numpy().copy()

class CompleteValueTree:
 def __init__(self,device):self.device=torch.device(device);self.arrays={};self.occurrences=0
 def tensor(self,t):
  assert torch.is_tensor(t) and t.device==self.device
  a=raw_array(t);digest=hashlib.sha256(a.tobytes()).hexdigest();dtype=str(t.dtype);shape=list(t.shape)
  key='full_'+dtype.replace('torch.','')+'_'+'_'.join(str(v) for v in shape)+'_'+digest
  if key in self.arrays:
   old=self.arrays[key];assert old.shape==a.shape and old.dtype==a.dtype and old.tobytes()==a.tobytes()
  else:self.arrays[key]=a
  self.occurrences+=1
  return dict(tensor=key,shape=shape,dtype=dtype,SHA256=digest)
 def encode(self,value,layer_index=None,cache_output=False):
  from transformers.cache_utils import Cache
  if torch.is_tensor(value):return self.tensor(value)
  if isinstance(value,Cache):
   lengths=[int(value.get_seq_length(i)) for i in range(len(value.layers))]
   if cache_output:
    pairs=[]
    for l in value.layers:
     assert l.keys is not None and l.values is not None
     pairs.append([self.tensor(l.keys),self.tensor(l.values)])
    return dict(cache_full_pairs=pairs,cache_layer_lengths=lengths)
   assert layer_index is not None and lengths[layer_index]==0,('Only genuine original fresh-request prefill cache is admitted',layer_index,lengths)
   return dict(original_fresh_layer_cache=layer_index,cache_layer_lengths=lengths)
  if isinstance(value,Mapping):return dict(mapping=[[str(k),self.encode(v,layer_index,cache_output)] for k,v in value.items()])
  if isinstance(value,tuple):return dict(tuple=[self.encode(v,layer_index,cache_output) for v in value])
  if isinstance(value,list):return dict(list=[self.encode(v,layer_index,cache_output) for v in value])
  if isinstance(value,torch.dtype):return dict(torch_dtype=str(value))
  if value is None or type(value) in [bool,int,float,str]:return value
  raise TypeError('Uncaptured native value type: '+str(type(value)))

def reconstruct(tree,archive,cpu_config=None):
 if not isinstance(tree,dict):return tree
 if 'tensor' in tree:
  a=np.array(archive[tree['tensor']],copy=True,order='C');assert list(a.shape)==tree['shape'] and hashlib.sha256(a.tobytes()).hexdigest()==tree['SHA256'];t=torch.from_numpy(a)
  want=getattr(torch,tree['dtype'].split('.')[-1]);t=t.view(torch.bfloat16) if want==torch.bfloat16 else t
  assert t.dtype==want;return t
 if 'tuple' in tree:return tuple(reconstruct(x,archive,cpu_config) for x in tree['tuple'])
 if 'list' in tree:return [reconstruct(x,archive,cpu_config) for x in tree['list']]
 if 'mapping' in tree:return {k:reconstruct(v,archive,cpu_config) for k,v in tree['mapping']}
 if 'torch_dtype' in tree:return getattr(torch,tree['torch_dtype'].split('.')[-1])
 if 'original_fresh_layer_cache' in tree:
  from transformers.cache_utils import DynamicCache
  assert cpu_config is not None and tree['cache_layer_lengths'][tree['original_fresh_layer_cache']]==0
  return DynamicCache(config=cpu_config)
 if 'cache_full_pairs' in tree:return [[reconstruct(v,archive,cpu_config) for v in pair] for pair in tree['cache_full_pairs']]
 raise ValueError('Unknown complete value schema: '+str(tree.keys()))

def compare_reference(actual,tree,archive,checks,label,relative_rmse=.02):
 """Compare every full output value; CPU references never determine the grade."""
 from transformers.cache_utils import Cache
 if isinstance(tree,dict) and 'tensor' in tree:
  want=reconstruct(tree,archive);assert torch.is_tensor(actual) and actual.device.type=='cpu' and actual.dtype==want.dtype and actual.shape==want.shape,(label,'layout')
  if actual.is_floating_point():
   a=actual.detach().float().numpy().astype(np.float64);b=want.float().numpy().astype(np.float64);assert np.isfinite(a).all() and np.isfinite(b).all(),(label,'nonfinite')
   error=a-b;rmse=float(np.sqrt(np.mean(error**2))) if a.size else 0.;scale=float(np.sqrt(np.mean(a**2))) if a.size else 0.;ratio=rmse/max(scale,1e-6);maximum=float(np.max(np.abs(error))) if a.size else 0.
   assert ratio<=relative_rmse,(label,'relative_RMSE',ratio,maximum)
   checks.append(dict(name=label,shape=list(want.shape),dtype=str(want.dtype),all_values=int(want.numel()),relative_RMSE=ratio,max_absolute_error=maximum,CPU_reference_not_grade=True))
  else:
   assert torch.equal(actual,want),(label,'nonfloating_values');checks.append(dict(name=label,shape=list(want.shape),dtype=str(want.dtype),all_values=int(want.numel()),exact=True))
  return
 if isinstance(tree,dict) and 'cache_full_pairs' in tree:
  assert isinstance(actual,Cache) and len(actual.layers)==len(tree['cache_full_pairs'])
  for i,pair in enumerate(tree['cache_full_pairs']):
   compare_reference(actual.layers[i].keys,pair[0],archive,checks,label+'/K'+str(i),relative_rmse);compare_reference(actual.layers[i].values,pair[1],archive,checks,label+'/V'+str(i),relative_rmse)
  return
 if isinstance(tree,dict) and 'mapping' in tree:
  assert isinstance(actual,Mapping) and list(actual)==[k for k,v in tree['mapping']],(label,'mapping')
  for k,v in tree['mapping']:compare_reference(actual[k],v,archive,checks,label+'/'+k,relative_rmse)
  return
 for kind,typ in [('tuple',tuple),('list',list)]:
  if isinstance(tree,dict) and kind in tree:
   assert isinstance(actual,typ) and len(actual)==len(tree[kind]),(label,kind)
   for i,(a,b) in enumerate(zip(actual,tree[kind])):compare_reference(a,b,archive,checks,label+'/'+str(i),relative_rmse)
   return
 assert actual==tree,(label,'scalar')
