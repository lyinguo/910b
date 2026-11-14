"""Independent own Mllama additive masks and original Llama3 FP32 frequencies."""
import math,torch,numpy as np
from scripts.cosmos_reason2_stable_rope_reference_910b import _bf16_rne

def inverse_frequencies():
    base=1/(500000.**(torch.arange(0,128,2,dtype=torch.int64).float()/128))
    wavelength=2*math.pi/base
    low=torch.where(wavelength>8192,base/8,base)
    blend=(8192/wavelength-1)/3
    return torch.where((wavelength>=2048)&(wavelength<=8192),(1-blend)*low/8+blend*low,low)

def text_rope(positions):
    inv=inverse_frequencies();angles=(positions.numpy().astype(np.float64)[...,None]*inv.numpy().astype(np.float64)).astype(np.float32)
    angles=np.concatenate([angles,angles],axis=-1)
    cos=np.fromiter((math.cos(float(x)) for x in angles.ravel()),dtype=np.float64,count=angles.size).reshape(angles.shape)
    sin=np.fromiter((math.sin(float(x)) for x in angles.ravel()),dtype=np.float64,count=angles.size).reshape(angles.shape)
    return _bf16_rne(cos),_bf16_rne(sin)

def original_vision_mask(aspect_mask):
    # Preserve the native original pairwise-invalid outer product. It is not
    # replaced with a conventional key-only padding mask.
    b=aspect_mask.shape[0];real=aspect_mask.reshape(b,4,1).expand(b,4,1608).clone().bfloat16();real[:,:,-7:]=0
    invalid=(1-real).reshape(b,6432)
    result=invalid[:,:,None]*invalid[:,None,:]*torch.finfo(torch.bfloat16).min
    return result[:,None]
