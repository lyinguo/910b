from .npu_attention_backend import CANNPredicateMask
import torch
from torch import Tensor as T
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

# ---------------------------------------------------------------------------
# Two compiled variants of flex_attention
# ---------------------------------------------------------------------------
# _decode:  fullgraph=True, static shapes.
#           Used for decode steps (S_q == 1) where shapes are fixed and
#           the call will be captured inside a CUDA graph.  fullgraph=True
#           avoids graph breaks that would corrupt the capture.
#
# _prefill: dynamic=True, symbolic shapes.
#           Used for prefill steps (S_q > 1) where the sequence length
#           varies per image.  dynamic=True lets one compiled graph handle
#           all lengths without recompilation.  Prefill is never inside a
#           CUDA graph, so symbolic shape guards are fine.
compiled_flex_attn_decode = torch.compile(flex_attention, fullgraph=True)
compiled_flex_attn_prefill = torch.compile(flex_attention, dynamic=True)


def offset_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    """Get a mask mod function with an offset applied to the query positions."""

    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


def get_causal_mask_mod() -> _mask_mod_signature:
    """Causal mask that prevents attention to future tokens."""

    def _causal_mask(b: T, h: T, q_idx: T, kv_idx: T) -> T:
        return q_idx >= kv_idx

    return _causal_mask


def get_document_mask_mod(batch: T, eos_id: int) -> _mask_mod_signature:
    """Document mask: prevents attention across document boundaries (token IDs [B, S])."""
    eos_mask = batch == eos_id
    eos_mask[:, -1] = True
    cumulative_mask = torch.cumsum(torch.where(eos_mask, 1, 0), dim=1)
    sequence_indices = torch.zeros_like(cumulative_mask, dtype=torch.int32)
    sequence_indices[:, 1:] = cumulative_mask[:, :-1]

    def document_mask(b: T, h: T, q_idx: T, kv_idx: T) -> T:
        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]

    return document_mask


def get_non_left_pad_mask_mod(batch: T, pad_id: int) -> _mask_mod_signature:
    """Prevent model from attending to the left-padded token required for correct batch inference."""

    non_pad_mask_id = torch.cumsum(batch != pad_id, dim=1)

    # Left-most pad tokens have cumulative id == 0.
    def mask_mod(b, h, q_idx, kv_idx):
        return non_pad_mask_id[b, kv_idx] > 0

    return mask_mod


def get_image_prefix_mask_mod(
    batch: T, soi_id: int, eoi_id: int
) -> _mask_mod_signature:
    """Image-prefix mask: tokens between SOI and EOI attend only within same image."""
    soi_mask = batch == soi_id
    eoi_mask = batch == eoi_id
    acc_soi_mask = torch.cumsum(soi_mask, dim=1)
    acc_eoi_mask = torch.cumsum(eoi_mask, dim=1)
    img_mask = (acc_soi_mask - acc_eoi_mask) > 0
    img_indices = acc_soi_mask * img_mask

    def image_prefix_mask_mod(b, h, q_idx, kv_idx):
        is_img_tokens = img_mask[b, q_idx] & img_mask[b, kv_idx]
        is_same_image = img_indices[b, q_idx] == img_indices[b, kv_idx]
        return is_img_tokens & is_same_image

    return image_prefix_mask_mod


_compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=True
)  # reduce-overhead mode breaks manual CUDA graph capture (private streams)

# MLU 兼容：torch.compile/Inductor 与 flex_attention 编译路径在 MLU 上不可用。
# 检测 MLU 后，create_attention_mask 直接走 eager create_block_mask。
def _is_mlu() -> bool:
    try:
        import torch_mlu  # noqa: F401
        return hasattr(torch, "mlu") and torch.mlu.is_available()
    except Exception:
        return False


_MLU_MODE = _is_mlu()


@torch.inference_mode()
def create_attention_mask(*args, **kwargs) -> BlockMask:
    """Compiled for large masks; inference_mode avoids grad_mode recompiles."""
    if 'device' in kwargs and torch.device(kwargs['device']).type == 'npu':
        return CANNPredicateMask(args[0], args[1], args[3], args[4], kwargs['device'])
    if _MLU_MODE:
        # MLU: 不编译，直接构建 BlockMask（内部为 eager torch 算子）
        return create_block_mask(*args, **kwargs)
    return _compiled_create_block_mask(*args, **kwargs)


@torch.inference_mode()
def flex_attn_eager(xq, xk, xv, block_mask):
    """flex_attention 的 eager 等价实现（MLU 兼容）。

    通过直接求值 block_mask.mask_mod 得到稠密布尔掩码，
    再用标准 softmax 注意力计算 output 与 logsumexp(lse)。
    返回 (output, lse)，与 flex_attention(return_aux=AuxRequest(lse=True)) 对齐。
    """
    B, H, Q, D = xq.shape
    KV = xk.shape[2]
    device = xq.device
    mask_mod = block_mask.mask_mod
    q_idx = torch.arange(Q, device=device).view(1, -1, 1)
    kv_idx = torch.arange(KV, device=device).view(1, 1, -1)
    b_idx = torch.arange(B, device=device).view(-1, 1, 1)
    h_idx = torch.zeros((), device=device, dtype=torch.long)
    dense = mask_mod(b_idx, h_idx, q_idx, kv_idx)  # 广播 -> (B, Q, KV)
    dense = dense.unsqueeze(1)  # (B, 1, Q, KV)
    scores = torch.matmul(xq, xk.transpose(-2, -1)) / (D ** 0.5)
    scores = scores.masked_fill(~dense.bool(), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # (B, H, Q)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, xv)
    return out, lse


def create_batch_attention_mask(
    input_batch: T,
    *,
    pad_token_id: int,
    eos_token_id: int,
    soi_token_id: int,
    eoi_token_id: int,
    max_len: int | None = None,
) -> BlockMask:
    """Build the combined FlexAttention mask for the batch engine.

    Composes causal + document + non-left-pad + image-prefix masks.
    """
    B, S = input_batch.size()
    block_causal_mask_mod = and_masks(
        get_causal_mask_mod(),
        get_document_mask_mod(input_batch, eos_token_id),
        get_non_left_pad_mask_mod(input_batch, pad_token_id),
    )
    image_prefix_mask_mod = get_image_prefix_mask_mod(
        batch=input_batch,
        soi_id=soi_token_id,
        eoi_id=eoi_token_id,
    )
    mask_mod = or_masks(image_prefix_mask_mod, block_causal_mask_mod)
    max_len = max_len or S
    return create_attention_mask(mask_mod, B, None, max_len, max_len, device=input_batch.device)
