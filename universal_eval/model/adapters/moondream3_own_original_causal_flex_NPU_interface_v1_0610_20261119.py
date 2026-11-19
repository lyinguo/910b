"""Execute only Moondream's original one-token causal Flex interface on NPU.

This is a real attention implementation using the original file's existing SDPA
operation. The original mask_mod is evaluated for every physical cache element
on NPU. No key/value truncation, cache mutation, scale, precision, or mask-value
change is permitted. This file does not load weights or register a shared patch.
"""
import math
import torch
import torch.nn.functional as F

def original_dense_mask(query, key, value, block_mask):
    assert query.device.type == key.device.type == value.device.type == 'npu'
    assert query.device == key.device == value.device
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert tuple(query.shape) == (1, 32, 1, 64)
    assert tuple(key.shape) == tuple(value.shape) == (1, 32, 4096, 64)
    assert block_mask is not None and tuple(block_mask.seq_lengths) == (1, 4096)
    fn = block_mask.mask_mod
    closure = dict(zip(fn.__code__.co_freevars, [v.cell_contents for v in fn.__closure__]))
    assert set(closure) == {'mask_mod', 'offset'}
    base, offset = closure['mask_mod'], closure['offset']
    assert base.__module__ == 'universal_eval.model.moondream3_own_original_source_20261119.moondream'
    assert base.__name__ == 'causal_mask'
    assert offset.device == query.device and offset.numel() == 1 and offset.dtype == torch.int64
    assert 730 <= int(offset.item()) < 4096
    b = torch.zeros((1, 1, 1, 1), dtype=torch.int64, device=query.device)
    h = torch.arange(32, dtype=torch.int64, device=query.device).reshape(1, 32, 1, 1)
    q = torch.zeros((1, 1, 1, 1), dtype=torch.int64, device=query.device)
    k = torch.arange(4096, dtype=torch.int64, device=query.device).reshape(1, 1, 1, 4096)
    dense = fn(b, h, q, k).expand(1, 32, 1, 4096).contiguous()
    assert dense.dtype == torch.bool and dense.device == query.device
    assert torch.equal(dense, (k <= offset).expand_as(dense))
    return dense

def original_causal_flex_attention(query, key, value, *, block_mask=None, **kwargs):
    if kwargs:
        raise ValueError('Only the exact original Moondream one-token causal interface is supported')
    dense = original_dense_mask(query, key, value, block_mask)
    # Both original FlexAttention and this original SDPA operation use 1/sqrt(D),
    # no dropout, and equal Q/KV head counts. Physical 4096-key storage is retained.
    return F.scaled_dot_product_attention(query, key, value, attn_mask=dense,
        dropout_p=0.0, is_causal=False, enable_gqa=False)
