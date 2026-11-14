"""Independent deterministic diagnostic: original F32 angles, FP64 libm, explicit BF16 RNE.

No model forward or NPU function is replaced. BF16 inverse frequencies remain
the original CPU-parsed FP32 constants. FP64 exactly represents each product
of these F32 values with the benchmark's integer positions; a single cast
rounds it to the original FP32 angle. Scalar FP64 libm is only the independent
trigonometric reference, followed by explicit round-to-nearest-even BF16 bits.
"""
import math,numpy as np,torch

def _bf16_rne(values):
 x=np.asarray(values,dtype=np.float32);assert np.isfinite(x).all();bits=x.view(np.uint32);rounded=((bits.astype(np.uint64)+np.uint64(0x7fff)+((bits>>16)&1).astype(np.uint64))>>16).astype(np.uint16);return torch.from_numpy(rounded.copy()).view(torch.bfloat16)

def independent_text_rope(position4,inv_freq32):
 assert position4.dtype==torch.int64 and inv_freq32.dtype==torch.float32
 pos=position4[1:].numpy().astype(np.float64);freq=(pos[...,None]*inv_freq32.numpy().astype(np.float64)).astype(np.float32);mix=freq[0].copy();mix[...,1:60:3]=freq[1,...,1:60:3];mix[...,2:60:3]=freq[2,...,2:60:3];angles=np.concatenate((mix,mix),axis=-1);shape=angles.shape
 cos=np.fromiter((math.cos(float(x)) for x in angles.ravel()),dtype=np.float64,count=angles.size).reshape(shape);sin=np.fromiter((math.sin(float(x)) for x in angles.ravel()),dtype=np.float64,count=angles.size).reshape(shape)
 return _bf16_rne(cos),_bf16_rne(sin)
