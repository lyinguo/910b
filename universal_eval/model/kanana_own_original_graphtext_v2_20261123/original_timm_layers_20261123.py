"""Exact original active timm normalization classes and position resampling."""
from typing import List,Optional
import logging,math
import torch
from torch import nn
from torch.nn import functional as F
from .original_fast_norm_20261123 import is_fast_norm,fast_layer_norm
from .original_timm_FX_20261123 import register_notrace_function
_logger=logging.getLogger(__name__)
class LayerNorm(nn.LayerNorm):
    """ LayerNorm w/ fast norm option
    """
    _fast_norm: torch.jit.Final[bool]

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    _fast_norm: torch.jit.Final[bool]

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

@torch.fx.wrap
@register_notrace_function
def resample_abs_pos_embed(
        posemb: torch.Tensor,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb
