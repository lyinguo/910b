import einops as E
import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for rotary embeddings with given dimensions.

    MLU 兼容补丁：原实现返回 complex64（torch.polar），寒武纪 MLU 不支持
    complex dtype（RuntimeError: Torch mlu only support integer/floating/bool）。
    此处改为返回等价的实数张量 [S, D//2, 2]（最后一维为 cos/sin），
    数学上与 complex 形式完全等价。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return torch.stack([cos, sin], dim=-1)  # [S, D//2, 2] 实数


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1D rotary embedding（实数实现，MLU 兼容）

    freqs_cis: (B, S, D//2, 2)，最后一维为 cos/sin。
    等价于原来的 complex 乘法：(a+ib)*(c+is) = (ac-bs) + i(as+bc)，
    其中 a/b 为输入相邻元素对 (even/odd)。
    """
    assert freqs_cis.ndim == 4, (
        "Freqs_cis must be indexed by position ids already and has shape (B,S,D,2)"
    )
    cos = E.rearrange(freqs_cis[..., 0], "b s d -> b s 1 d")
    sin = E.rearrange(freqs_cis[..., 1], "b s d -> b s 1 d")

    def _rot(x):
        xf = x.float()
        even = xf[..., 0::2]
        odd = xf[..., 1::2]
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
        out = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        return out

    xq_out = _rot(xq)
    xk_out = _rot(xk)
    return xq_out.type_as(xq), xk_out.type_as(xk)


###### 2D golden rope
"""
Dimension key:
    B: batch size
    S: number of tokens per sample, Seqlen
    T: Number of selected Tokens
    P: pos_dim
    h: n_heads
    d: head_dim
    F: num_freqs == head_dim // 2
"""


def apply_golden_freqs_cis_to_visual_pos(freqs_hFP, pos_BSP) -> torch.Tensor:
    """
    This function is applied once per input batch, and the cached
    freqs_cis is passed through to all layers.
    MLU 兼容：返回实数 (T, H, F, 2)（cos/sin），不再使用 torch.polar。
    """
    # 1. Boolean mask → integer indices (no unbacked shapes)
    img_mask_BS = E.reduce(~torch.isnan(pos_BSP), 'b s p -> b s', reduction='all')
    idx_b, idx_s = torch.nonzero(img_mask_BS, as_tuple=True)   # each shape: (N,)

    # 2. Gather the positional tensor for those tokens
    pos_tP = pos_BSP[idx_b, idx_s].float() # (N, p)

    # 3. Project positions onto the frequency table → angles θ
    theta_thF = torch.einsum("tp,hfp->thf", pos_tP, freqs_hFP.float())  # (t, h, f)

    # 4. 单位圆上的 cos/sin（实数）
    cos_thF = torch.cos(theta_thF)
    sin_thF = torch.sin(theta_thF)
    return torch.stack([cos_thF, sin_thF], dim=-1)  # (T, H, F, 2)


def apply_golden_rotary_emb(input_BShd, freqs_cis_thF, pos_BSP) -> torch.Tensor:
    """
    Rotates *only* the image tokens in `input_BShd`.  No boolean indexing,
    so it is safe for Torch‑Inductor.
    freqs_cis_thF: (T, H, F, 2) 实数 cos/sin。
    """
    img_mask_BS = E.reduce(~torch.isnan(pos_BSP), 'b s p -> b s', reduction='all')
    idx_b, idx_s = torch.nonzero(img_mask_BS, as_tuple=True)  # (N,)

    input_thd = input_BShd[idx_b, idx_s].float()  # (N, h, d)
    x_even = input_thd[..., 0::2]  # (N, h, F)
    x_odd = input_thd[..., 1::2]   # (N, h, F)

    cos_thF = freqs_cis_thF[..., 0]
    sin_thF = freqs_cis_thF[..., 1]

    # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    rot_even = x_even * cos_thF - x_odd * sin_thF
    rot_odd = x_even * sin_thF + x_odd * cos_thF

    output_real = torch.empty_like(input_thd)
    output_real[..., 0::2] = rot_even
    output_real[..., 1::2] = rot_odd
    output_real = output_real.type_as(input_BShd)

    output_BShd = input_BShd.clone()
    output_BShd[idx_b, idx_s] = output_real

    return output_BShd


def apply_3d_rotary_emb(
    xq: torch.Tensor,  # (B, S, H, D)
    xk: torch.Tensor,  # (B, S, H, D)
    freqs_cis: torch.Tensor,
    freqs_cis_2d: torch.Tensor | None,
    pos_hw: torch.Tensor | None,  # (B,S,3)
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_t, xq_hw = xq.chunk(chunks=2, dim=-1)
    xk_t, xk_hw = xk.chunk(chunks=2, dim=-1)
    B, S, H, D = xq.shape

    xq_t, xk_t = apply_rotary_emb(xq_t, xk_t, freqs_cis)
    if freqs_cis_2d is not None and pos_hw is not None:
        xq_hw = apply_golden_rotary_emb(xq_hw, freqs_cis_2d, pos_hw)
        xk_hw = apply_golden_rotary_emb(xk_hw, freqs_cis_2d, pos_hw)

    xq_out = torch.concat([xq_t, xq_hw], dim=-1).type_as(xq)
    xk_out = torch.concat([xk_t, xk_hw], dim=-1).type_as(xk)
    return xq_out, xk_out
