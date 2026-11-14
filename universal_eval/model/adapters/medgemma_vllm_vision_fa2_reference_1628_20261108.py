"""MedGemma FA2 vision attention reference for the observed original BF16 path."""
import torch


def medgemma_visual_fa2(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Preserve reverse K tiles, FP32 online sums, and the BF16 P operand of PV."""
    assert query.shape == key.shape == value.shape
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert query.device == key.device == value.device
    batch, heads, length, dimension = query.shape
    assert batch > 0 and (heads, length, dimension) == (16, 4096, 72)

    packed_shape = (batch * heads, length, dimension)
    query, key, value = (tensor.reshape(packed_shape).contiguous() for tensor in (query, key, value))
    scale_log2 = torch.tensor(dimension ** -0.5, device=query.device, dtype=torch.float32) * 1.4426950408889634
    row_shape = (batch * heads, length, 1)
    maximum = torch.full(row_shape, float("-inf"), device=query.device, dtype=torch.float32)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros(packed_shape, device=query.device, dtype=torch.float32)

    for begin in range(length - 64, -1, -64):
        end = begin + 64
        scores = torch.bmm(query, key[:, begin:end].transpose(1, 2), out_dtype=torch.float32)
        new_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        rescale = torch.exp2((maximum - new_maximum) * scale_log2)
        probability = torch.exp2(torch.addcmul(-new_maximum * scale_log2, scores, scale_log2))
        denominator = denominator * rescale + probability.sum(-1, keepdim=True)
        contribution = torch.bmm(probability.bfloat16(), value[:, begin:end], out_dtype=torch.float32)
        numerator = numerator * rescale + contribution
        maximum = new_maximum

    return (numerator / denominator).bfloat16().reshape(batch, heads, length, dimension)
