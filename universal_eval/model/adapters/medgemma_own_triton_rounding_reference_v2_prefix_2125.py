"""Reference for the observed MedGemma PrefixLM Triton attention boundaries."""
import torch
from torch.nn import functional as F


def medgemma_original_prefix_attention(query, key, value, allowed, use_2d, window):
    """Run the original full-KV tile span with the caller's multimodal visibility."""
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert query.shape[1:] == (8, 256) and key.shape[1:] == value.shape[1:] == (4, 256)
    assert key.shape == value.shape and allowed.shape == (len(query), len(key))
    assert allowed.dtype == torch.bool and window in (0, 1024)
    assert use_2d or len(query) == 1
    qh = query.transpose(0, 1).float()
    kh = key.repeat_interleave(2, dim=1).transpose(0, 1).float()
    vh = value.repeat_interleave(2, dim=1).transpose(0, 1).float()
    nkeys, nq = len(key), len(query)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * 0.0625
    scores.masked_fill_(~allowed[None], float("-inf"))
    tile = 32 if use_2d or window == 1024 else 16
    target_segments = 1 if use_2d else 16
    tiles_per_segment = (nkeys + tile * target_segments - 1) // (tile * target_segments)
    extent = tiles_per_segment * tile
    segments = (nkeys + extent - 1) // extent
    tiles = F.pad(scores, (0, extent * segments - nkeys), value=float("-inf"))
    tiles = tiles.reshape(8, nq, segments, tiles_per_segment, tile)

    # USE_MM_PREFIX extends the loop to the full existing KV length and disables
    # sliding-window tile pruning. Individual elements still obey the actual mask.
    # A fully masked initial tile uses the native softmax_step fallback of zero.
    maxima = tiles.amax(dim=-1)
    maxima[..., :1] = torch.where(torch.isneginf(maxima[..., :1]), 0.0, maxima[..., :1])
    running = torch.cummax(maxima, dim=-1).values
    combined_max = running[..., -1].amax(dim=-1)
    weights = (tiles - running[..., None]).exp().to(torch.bfloat16).float()
    weights = weights * (running - combined_max[..., None, None]).exp()[..., None]
    weights = weights.reshape(8, nq, -1)[..., :nkeys]
    denominator = (scores - combined_max[..., None]).exp().sum(dim=-1, keepdim=True)
    assert torch.isfinite(denominator).all() and (denominator > 0).all()
    return (torch.matmul(weights, vh) / denominator).to(torch.bfloat16).transpose(0, 1)
