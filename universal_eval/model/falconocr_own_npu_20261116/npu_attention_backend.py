"""CANN storage and kernels for Falcon's unchanged attention predicate and sinks.

The predicate is evaluated for every actual Q/KV position.  This replaces the
CUDA BlockMask storage/compiled FlexAttention backend, never model outputs.
"""
import copy,math
import torch
import torch_npu

class CANNPredicateMask:
    BLOCK_SIZE=(128,128)
    def __init__(self,mask_mod,batch_size,q_length,kv_length,device):
        self.mask_mod=mask_mod;self.batch_size=batch_size
        self.seq_lengths=(q_length,kv_length);self.device=torch.device(device)
        assert self.device.type=='npu' and batch_size==1
    def __getitem__(self,index):
        assert len(index)==3 and index[:2]==(slice(None),slice(None)) and isinstance(index[2],int) and index[2]>=0
        return copy.copy(self)
    def allowed(self,q_length,kv_length):
        assert self.seq_lengths==(q_length,kv_length)
        q=torch.arange(q_length,device=self.device).view(1,-1,1)
        k=torch.arange(kv_length,device=self.device).view(1,1,-1)
        b=torch.arange(self.batch_size,device=self.device).view(-1,1,1)
        h=torch.zeros((),device=self.device,dtype=torch.long)
        allow=self.mask_mod(b,h,q,k).bool().unsqueeze(1)
        assert allow.shape==(self.batch_size,1,q_length,kv_length)
        return allow.contiguous()

def falcon_attention_npu(q,k,v,mask):
    assert q.device.type==k.device.type==v.device.type=='npu'
    assert q.dtype==k.dtype==v.dtype==torch.bfloat16 and q.shape[1]==16 and q.shape[-1]==64
    allow=mask.allowed(q.shape[2],k.shape[2])
    result=torch_npu.npu_fusion_attention(q.contiguous(),k.contiguous(),v.contiguous(),head_num=16,input_layout='BNSD',atten_mask=(~allow).contiguous(),scale=1/math.sqrt(64),keep_prob=1.0,pre_tockens=2147483647,next_tockens=2147483647,sparse_mode=0)
    out,maximum,total=result[:3]
    assert out.dtype==torch.bfloat16 and maximum.dtype==total.dtype==torch.float32
    assert maximum.shape[:3]==total.shape[:3]==q.shape[:3] and maximum.shape[-1]==total.shape[-1]==8
    lse=maximum[...,0]+torch.log(total[...,0])
    assert lse.shape==q.shape[:3] and bool(torch.isfinite(lse).all())
    return out,lse

def falcon_squared_relu_gate_npu(packed):
    """Original Triton loads/promotes BF16 to F32 and casts only at the store."""
    assert packed.device.type=='npu' and packed.dtype==torch.bfloat16
    gate=packed[...,0::2].float();up=packed[...,1::2].float()
    return (torch.relu(gate).square()*up).to(packed.dtype)
