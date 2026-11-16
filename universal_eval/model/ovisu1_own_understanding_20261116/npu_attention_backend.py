"""Real CANN varlen vision attention and original F32 non-interleaved rotary math."""
import math,torch
def apply_rotary_emb(x,cos,sin,interleaved=False,inplace=False,**kwargs):
 assert not interleaved and not inplace and not kwargs and x.dtype==cos.dtype==sin.dtype==torch.float32
 assert x.ndim==4 and x.shape[0]==1 and cos.shape==sin.shape and cos.shape[0]==x.shape[1] and cos.shape[1]*2==x.shape[-1]
 c=torch.cat([cos,cos],-1)[None,:,None,:];s=torch.cat([sin,sin],-1)[None,:,None,:];a,b=x.chunk(2,-1)
 return x*c+torch.cat([-b,a],-1)*s
def flash_attn_varlen_func(q,k,v,cuq,cuk,maxq,maxk,dropout_p=0.0,softmax_scale=None,causal=False,**kwargs):
 import torch_npu
 assert q.device.type==k.device.type==v.device.type=='npu' and q.dtype==k.dtype==v.dtype==torch.bfloat16 and dropout_p==0 and not causal and not kwargs
 assert q.shape==k.shape==v.shape and cuq.tolist()==cuk.tolist() and q.shape[0]==int(cuq[-1]) and int((cuq[1:]-cuq[:-1]).max())==maxq==maxk
 return torch_npu.npu_fusion_attention(q,k,v,q.shape[1],input_layout='TND',scale=softmax_scale or 1/math.sqrt(q.shape[-1]),keep_prob=1.0,pre_tockens=2147483647,next_tockens=2147483647,actual_seq_qlen=cuq[1:].tolist(),actual_seq_kvlen=cuk[1:].tolist())[0]
