from pathlib import Path

import einops as E
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from PIL import Image
from torch import Tensor as T
from torch import nn
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
)
from transformers import AutoTokenizer, PreTrainedModel

from .attention import (
    _MLU_MODE,
    compiled_flex_attn_decode,
    compiled_flex_attn_prefill,
    create_batch_attention_mask,
    flex_attn_eager,
    offset_mask_mod,
)
from .configuration_falcon_ocr import FalconOCRConfig
from .processing_falcon_ocr import load_image, process_batch
from .rope import (
    apply_3d_rotary_emb,
    apply_golden_freqs_cis_to_visual_pos,
    precompute_freqs_cis,
)


CATEGORY_PROMPTS = {
    "plain": "Extract the text content from this image.",
    "formula": "Extract the formula content from this image.",
    "table": "Extract the table content from this image.",
    "text": "Extract the text content from this image.",
    "caption": "Extract the caption content from this image.",
    "footnote": "Extract the footnote content from this image.",
    "list-item": "Extract the list-item content from this image.",
    "page-footer": "Extract the page-footer content from this image.",
    "page-header": "Extract the page-header content from this image.",
    "section-header": "Extract the section-header content from this image.",
    "title": "Extract the title content from this image.",
}

LAYOUT_TO_OCR_CATEGORY: dict[str, str | None] = {
    "text": "text",
    "table": "table",
    "formula": "formula",
    "caption": "caption",
    "footnote": "footnote",
    "list-item": "list-item",
    "title": "title",
    "header": "text",
    "footer": "page-footer",
    "number": "text",
    "figure_title": "caption",
    "paragraph_title": "section-header",
    "doc_title": "title",
    "reference_content": "text",
    "reference": "text",
    "abstract": "text",
    "aside_text": "text",
    "content": "text",
    "formula_number": "text",
    "vision_footnote": "footnote",
    "algorithm": "text",
    "page-footer": "page-footer",
    "page-header": "page-header",
    "section-header": "section-header",
    # Skip — no text to extract
    "image": None,
    "picture": None,
    "figure": None,
    "chart": None,
    "seal": None,
}

_LAYOUT_TARGET_H, _LAYOUT_TARGET_W = 800, 800
_MIN_CROP_DIM = 16

def _box_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _intersection_area(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _containment_ratio(small, large):
    area = _box_area(small)
    if area <= 0:
        return 0.0
    return _intersection_area(small, large) / area


def _filter_nested_detections(detections: list[dict], containment_threshold: float = 0.8) -> list[dict]:
    """Remove any box that is mostly contained within a strictly larger box."""
    areas = [_box_area(d["bbox"]) for d in detections]
    keep = []
    for i, det in enumerate(detections):
        is_nested = False
        for j, other in enumerate(detections):
            if i == j:
                continue
            if areas[j] <= areas[i]:
                continue
            if _containment_ratio(det["bbox"], other["bbox"]) > containment_threshold:
                is_nested = True
                break
        if not is_nested:
            keep.append(det)
    return keep


# Attention

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, S, H, D = x.shape
    if n_rep == 1:
        return x
    return torch.unsqueeze(x, dim=3).expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)


class Attention(nn.Module):
    def __init__(self, config: FalconOCRConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.n_rep = config.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim or config.dim // config.n_heads
        self.q_dim = config.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.wqkv = nn.Linear(config.dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.sinks = nn.Parameter(torch.empty((config.n_heads,)))

    def _pre_attention_qkv(self, x) -> tuple[T, T, T]:
        qkv = self.wqkv(F.rms_norm(x, (x.size(-1),)))
        xq, xk, xv = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        xq = E.rearrange(xq, "b s (h d) -> b s h d", d=self.head_dim)
        xk = E.rearrange(xk, "b s (h d) -> b s h d", d=self.head_dim)
        xv = E.rearrange(xv, "b s (h d) -> b s h d", d=self.head_dim)
        xq = F.rms_norm(xq, (xq.size(-1),))
        xk = F.rms_norm(xk, (xk.size(-1),))
        xk = repeat_kv(xk, n_rep=self.n_rep)
        xv = repeat_kv(xv, n_rep=self.n_rep)
        return xq, xk, xv

    def _post_attention(self, output: T, lse: T) -> T:
        # Sink-based scaling: sigmoid(lse - sinks) * output
        # equivalent to prepending a sink token to the input
        sinks_BHS = self.sinks.view(1, -1, 1)
        sink_scale = torch.sigmoid(lse - sinks_BHS)
        output = (output * sink_scale.unsqueeze(-1)).to(output.dtype)
        output = output.permute(0, 2, 1, 3).contiguous().flatten(2)
        return self.wo(output)

    def compile_attention(self, *, dynamic: bool = True, mode: str = "default"):
        self._pre_attention_qkv = torch.compile(self._pre_attention_qkv, dynamic=dynamic, mode=mode)
        self._post_attention = torch.compile(self._post_attention, dynamic=dynamic, mode=mode)

    def forward(
        self, x: T, attention_masks: BlockMask, freqs_cis: T,
        freqs_cis_2d: T | None = None, pos_hw: T | None = None,
        kv_cache=None, input_pos=None, batch_idx=None,
        flex_attn_kernel_options=None,
    ):
        xq, xk, xv = self._pre_attention_qkv(x)
        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d, pos_hw)
        xq = E.rearrange(xq, "b s h d -> b h s d")
        xk = E.rearrange(xk, "b s h d -> b h s d")
        xv = E.rearrange(xv, "b s h d -> b h s d")
        xk, xv = kv_cache.insert_kv(self.layer_id, xk, xv, input_pos=input_pos, batch_idx=batch_idx)
        # MLU 兼容：flex_attention 编译路径不可用，走 eager 等价实现
        if _MLU_MODE:
            output, lse = flex_attn_eager(xq, xk, xv, attention_masks)
            return self._post_attention(output, lse)
        flex_fn = compiled_flex_attn_decode if xq.shape[2] == 1 else compiled_flex_attn_prefill
        output, aux_output = flex_fn(xq, xk, xv, block_mask=attention_masks, return_aux=AuxRequest(lse=True))
        return self._post_attention(output, aux_output.lse)


# FeedForward

@triton.jit
def _squared_relu_gate_kernel(
    packed_ptr, out_ptr, n_rows, n_cols,
    in_row_stride, in_col_stride, out_row_stride, out_col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = n_rows * n_cols
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    rows = offsets // n_cols
    cols = offsets % n_cols
    gate_idx = rows * in_row_stride + (2 * cols) * in_col_stride
    up_idx = rows * in_row_stride + (2 * cols + 1) * in_col_stride
    out_idx = rows * out_row_stride + cols * out_col_stride
    gate = tl.load(packed_ptr + gate_idx, mask=mask)
    up = tl.load(packed_ptr + up_idx, mask=mask)
    gate = tl.where(gate > 0, gate, 0.0)
    out = gate * gate * up
    tl.store(out_ptr + out_idx, out, mask=mask)


def squared_relu_gate(packed: T, hidden_dim: int) -> T:
    """Processes interleaved [gate, up, gate, up, ...] from w13; output = ReLU(gate)^2 * up.

    MLU 兼容：triton kernel 无法在 MLU 上启动，改用等价的 eager torch 实现。
    """
    if _MLU_MODE:
        gate = packed[..., 0::2]
        up = packed[..., 1::2]
        return (F.relu(gate) ** 2) * up
    packed_2d = packed.flatten(0, -2)
    n_rows = packed_2d.shape[0]
    n_cols = hidden_dim
    out_2d = torch.empty((n_rows, n_cols), device=packed.device, dtype=packed.dtype)
    n = n_rows * n_cols
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _squared_relu_gate_kernel[grid](
        packed_2d, out_2d, n_rows, n_cols,
        packed_2d.stride(0), packed_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        BLOCK_SIZE=1024,
    )
    return out_2d.view(*packed.shape[:-1], hidden_dim)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.rms_norm(x, (x.size(-1),))
        w13_out = self.w13(x)
        return self.w2(squared_relu_gate(w13_out, self.hidden_dim))


# TransformerBlock

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: FalconOCRConfig):
        super().__init__()
        self.attention = Attention(config, layer_id)
        self.feed_forward = FeedForward(config.dim, config.ffn_dim)

    def compile(self, *, dynamic: bool = True, mode: str = "default"):
        self.feed_forward = torch.compile(self.feed_forward, dynamic=dynamic, mode=mode)
        self.attention.compile_attention(dynamic=dynamic, mode=mode)
        return self

    def forward(
        self, x: T, freqs_cis: T, freqs_cis_2d: T | None = None,
        pos_hw: T | None = None, attention_masks=None, kv_cache=None,
        input_pos=None, batch_idx=None, flex_attn_kernel_options=None,
    ):
        B, S, D = x.shape
        x = x + self.attention(
            x, freqs_cis=freqs_cis, freqs_cis_2d=freqs_cis_2d, pos_hw=pos_hw,
            attention_masks=attention_masks, kv_cache=kv_cache,
            input_pos=input_pos, batch_idx=batch_idx,
            flex_attn_kernel_options=flex_attn_kernel_options,
        )
        out = x + self.feed_forward(x)
        return out.reshape(B, S, D)


# KV Cache

class KVCache:
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, num_layers):
        self.kv_shape = (num_layers, 2, max_batch_size, n_heads, max_seq_length, head_dim)
        self.kv_cache = None
        self.pos = 0
        self.pos_t: T | None = None

    def reset(self):
        self.pos = 0
        self.pos_t = None

    def get_pos(self):
        return self.pos

    def set_pos_t(self, pos_t):
        self.pos_t = pos_t

    def increment_and_get_pos_t(self):
        assert self.pos_t is not None
        self.pos_t += 1
        return self.pos_t

    def insert_kv(self, layer_id: int, k: T, v: T, **kwargs):
        del kwargs
        assert self.pos_t is not None
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        self.kv_cache[layer_id, 0, :, :, t0:t1] = k
        self.kv_cache[layer_id, 1, :, :, t0:t1] = v
        key_view = self.kv_cache[layer_id, 0, :, :, :t1]
        value_view = self.kv_cache[layer_id, 1, :, :, :t1]
        if layer_id == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# Sampling

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=0.0, top_k=None):
    assert temperature >= 0.0
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


# Main Model

class FalconOCRForCausalLM(PreTrainedModel):
    config_class = FalconOCRConfig
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, config: FalconOCRConfig):
        super().__init__(config)
        img_in_dim = config.temporal_patch_size * config.spatial_patch_size ** 2 * config.channel_size
        self.img_projector = nn.Linear(img_in_dim, config.dim, bias=False)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.layers = nn.ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, config)

        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        rope_dim = config.head_dim // 2
        freqs_cis = precompute_freqs_cis(rope_dim, config.max_seq_len, config.rope_theta)
        freqs_cis_golden = torch.empty((config.n_heads, rope_dim // 2, 2), dtype=torch.float)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.register_buffer("freqs_cis_golden", freqs_cis_golden, persistent=True)

        self._weights_fused = False
        self._is_compiled = False

        self.post_init()

    # Weight management

    def _ensure_device_buffers(self):
        """Recompute non-persistent buffers that HF meta-device loading may discard."""
        if self._weights_fused:
            return
        device = self.tok_embeddings.weight.device
        c = self.config
        rope_dim = c.head_dim // 2
        freqs_cis = precompute_freqs_cis(rope_dim, c.max_seq_len, c.rope_theta).to(device)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        if self.freqs_cis_golden.device != device:
            self.freqs_cis_golden = self.freqs_cis_golden.to(device)
        self._weights_fused = True

    def compile_model(self):
        if self._is_compiled:
            return
        torch._inductor.config.triton.cudagraphs = False
        for layer in self.layers.values():
            layer.compile(dynamic=True, mode="default")
        self._is_compiled = True

    # Tokenizer

    def _get_tokenizer(self):
        if not hasattr(self, "_tokenizer"):
            import os
            path = self.config._name_or_path
            is_local = os.path.exists(path)
            self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=is_local, trust_remote_code=True)
            for token_name, token in self._tokenizer.special_tokens_map.items():
                if isinstance(token, str):
                    setattr(self._tokenizer, token_name, token)
                    setattr(
                        self._tokenizer, token_name + "_id",
                        self._tokenizer.convert_tokens_to_ids(token),
                    )
        return self._tokenizer

    # Attention mask

    def get_attention_mask(self, input_batch: T, max_len: int | None = None):
        return create_batch_attention_mask(
            input_batch,
            pad_token_id=self._pad_token_id,
            eos_token_id=self.config.eos_id,
            soi_token_id=self.config.image_cls_token_id,
            eoi_token_id=self.config.img_end_id,
            max_len=max_len,
        )

    # Embedding helpers

    def _scatter_img_tokens_with_projector(self, h_BSD, pixel_patches_NLC, pixel_masks_NTHW, tokens_BS):
        B, S, D = h_BSD.shape
        pixel_patch_mask = E.reduce(
            pixel_masks_NTHW,
            "n (t pt) (h ph) (w pw) -> (n t h w)",
            reduction="any",
            pt=self.config.temporal_patch_size,
            ph=self.config.spatial_patch_size,
            pw=self.config.spatial_patch_size,
        )
        pixel_patches_flat = E.rearrange(pixel_patches_NLC, "n p c -> (n p) c")
        valid_patches = pixel_patches_flat[pixel_patch_mask]
        valid_feats = self.img_projector(valid_patches)
        img_mask_h_BSD = E.repeat(tokens_BS == self.config.img_id, "b s -> b s d", d=D)
        assert valid_feats.numel() == img_mask_h_BSD.sum()
        return torch.masked_scatter(h_BSD, img_mask_h_BSD, valid_feats)

    # Core forward

    def forward(
        self,
        tokens: T,
        attention_mask: BlockMask,
        kv_cache,
        rope_pos_t: T | None = None,
        rope_pos_hw: T | None = None,
        pixel_values: T | None = None,
        pixel_mask: T | None = None,
    ):
        B, S = tokens.size()
        c = self.config
        block_mask = attention_mask

        T_pos = kv_cache.get_pos()
        is_prefill = S != 1

        if is_prefill:
            assert rope_pos_t is not None and rope_pos_hw is not None
            pos_t = rope_pos_t[:, T_pos:T_pos + S].long()
            kv_cache.pos_t = pos_t[:, -1:]
            freqs_cis = self.freqs_cis[pos_t]
            rope_pos_hw = rope_pos_hw[:, T_pos:T_pos + S]
            freqs_cis_golden = apply_golden_freqs_cis_to_visual_pos(self.freqs_cis_golden, rope_pos_hw)
            block_mask.seq_lengths = (S, S)
        else:
            pos_t = kv_cache.increment_and_get_pos_t()
            freqs_cis = self.freqs_cis[pos_t]
            freqs_cis_golden = None
            block_idx = T_pos // block_mask.BLOCK_SIZE[0]
            block_mask = block_mask[:, :, block_idx]
            block_mask.seq_lengths = (S, T_pos + S)
            block_mask.mask_mod = offset_mask_mod(attention_mask.mask_mod, offset=T_pos)

        h_BSD = self.tok_embeddings(tokens)

        if pixel_values is not None:
            assert pixel_mask is not None
            pixel_values = pixel_values.to(self.dtype)
            pixel_mask = pixel_mask.to(self.dtype)
            pixel_patches_NLC = E.rearrange(
                pixel_values,
                "n (t pt) (h ph) (w pw) c -> n (t h w) (pt ph pw c)",
                pt=c.temporal_patch_size, ph=c.spatial_patch_size, pw=c.spatial_patch_size,
            )
            h_BSD = self._scatter_img_tokens_with_projector(h_BSD, pixel_patches_NLC, pixel_mask, tokens)

        for layer in self.layers.values():
            h_BSD = layer(
                h_BSD, freqs_cis=freqs_cis, freqs_cis_2d=freqs_cis_golden,
                pos_hw=rope_pos_hw, attention_masks=block_mask, kv_cache=kv_cache,
            )

        h_BSD = self.norm(h_BSD)
        logits_BSV = self.output(h_BSD)
        return logits_BSV

    # Layout detection

    def _load_layout_model(self, layout_model: str = "PaddlePaddle/PP-DocLayoutV3_safetensors"):
        if hasattr(self, "_layout_model"):
            return
        import torchvision.transforms.functional as tvF
        from transformers import AutoModelForObjectDetection, PPDocLayoutV3ImageProcessorFast

        self._layout_processor = PPDocLayoutV3ImageProcessorFast.from_pretrained(layout_model)
        self._layout_det_model = AutoModelForObjectDetection.from_pretrained(
            layout_model, torch_dtype=torch.float16,
        ).to(self.device).eval()
        self._layout_id2label = self._layout_det_model.config.id2label
        self._tvF = tvF

    @torch.inference_mode()
    def _run_layout_detection(
        self, images: list[Image.Image], threshold: float = 0.5,
    ) -> list[list[dict]]:
        """Run PP-DocLayoutV3 on a batch of PIL images, return per-image detections."""
        device = self.device
        tvF = self._tvF

        target_sizes = torch.tensor([img.size[::-1] for img in images])
        tensors = [tvF.pil_to_tensor(img) for img in images]

        # GPU-accelerated resize + normalize
        result = torch.empty(
            len(tensors), 3, _LAYOUT_TARGET_H, _LAYOUT_TARGET_W,
            dtype=torch.float16, device=device,
        )
        size_groups: dict[tuple[int, int], list[int]] = {}
        for i, t in enumerate(tensors):
            size_groups.setdefault((t.shape[1], t.shape[2]), []).append(i)

        for shape, indices in size_groups.items():
            batch = torch.stack([tensors[i] for i in indices])
            batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            batch = F.interpolate(
                batch, size=(_LAYOUT_TARGET_H, _LAYOUT_TARGET_W),
                mode="bicubic", align_corners=False, antialias=False,
            )
            batch = (batch.clamp_(0, 255) / 255.0).to(torch.float16)
            for j, idx in enumerate(indices):
                result[idx] = batch[j]
            del batch

        outputs = self._layout_det_model(pixel_values=result)
        del result

        # Postprocess on GPU
        logits = outputs.logits
        boxes = outputs.pred_boxes
        order_logits = outputs.order_logits

        box_centers, box_dims = boxes.split(2, dim=-1)
        boxes_xyxy = torch.cat([box_centers - 0.5 * box_dims, box_centers + 0.5 * box_dims], dim=-1)

        img_h, img_w = target_sizes.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(device, dtype=boxes_xyxy.dtype)
        boxes_xyxy = boxes_xyxy * scale[:, None, :]

        num_queries = logits.shape[1]
        num_classes = logits.shape[2]
        scores = logits.sigmoid()
        scores_flat, index = scores.flatten(1).topk(num_queries, dim=-1)
        labels = index % num_classes
        box_indices = index // num_classes
        boxes_xyxy = boxes_xyxy.gather(dim=1, index=box_indices.unsqueeze(-1).expand(-1, -1, 4))

        order_seqs = self._layout_processor._get_order_seqs(order_logits)
        order_seqs = order_seqs.gather(dim=1, index=box_indices)

        batch_results = []
        for s, l, b, o in zip(scores_flat, labels, boxes_xyxy, order_seqs):
            mask = s >= threshold
            o_valid = o[mask]
            _, indices_sorted = o_valid.sort()

            detections = []
            for si, li, bi in zip(s[mask][indices_sorted], l[mask][indices_sorted], b[mask][indices_sorted]):
                detections.append({
                    "category": self._layout_id2label[li.item()],
                    "bbox": [round(x, 2) for x in bi.tolist()],
                    "score": round(si.item(), 4),
                })
            batch_results.append(detections)

        return batch_results

    # Core batch decode (shared by generate & generate_with_layout)

    def _generate_batch(
        self,
        image_prompt_pairs: list[tuple],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        min_dimension: int,
        max_dimension: int,
        seed: int | None,
    ) -> list[str]:
        """Core autoregressive decode for a list of (image, prompt) pairs."""
        device = self.device
        tokenizer = self._get_tokenizer()
        self._pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
        stop_token_ids = [self.config.eos_id, tokenizer.convert_tokens_to_ids("<|end_of_query|>")]

        batch_inputs = process_batch(
            tokenizer, self.config, image_prompt_pairs,
            max_length=4096, min_dimension=min_dimension, max_dimension=max_dimension,
        )
        batch_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_inputs.items()}

        tokens = batch_inputs["tokens"]
        B, L = tokens.size()
        block_size = 128
        S = (L + max_new_tokens + block_size - 1) // block_size * block_size
        assert S <= self.config.max_seq_len

        rng = torch.Generator(device).manual_seed(seed) if seed is not None else None

        kv_cache = KVCache(
            max_batch_size=B, max_seq_length=S, n_heads=self.config.n_heads,
            head_dim=self.config.head_dim, num_layers=self.config.n_layers,
        )

        padded_tokens = torch.full((B, S), self._pad_token_id, dtype=tokens.dtype, device=device)
        padded_tokens[:, :L] = tokens

        attention_mask = self.get_attention_mask(padded_tokens, max_len=S)

        logits_BSV = self.forward(
            tokens=tokens, rope_pos_t=batch_inputs["pos_t"], rope_pos_hw=batch_inputs["pos_hw"],
            attention_mask=attention_mask, kv_cache=kv_cache,
            pixel_values=batch_inputs["pixel_values"], pixel_mask=batch_inputs["pixel_mask"],
        )

        stop_ids = torch.tensor(stop_token_ids).to(device)
        should_stop_B = torch.full((B,), False, dtype=torch.bool, device=device)
        generated_ids: list[list[int]] = [[] for _ in range(B)]

        while not torch.all(should_stop_B) and (pos := kv_cache.get_pos()) < S:
            tokens_B1 = sample_next_token(logits_BSV[:, -1], rng, temperature, top_k)

            if torch.any(should_stop_B):
                tokens_B1 = tokens_B1.clone()
                tokens_B1[should_stop_B, :] = self._pad_token_id
            padded_tokens[:, pos] = tokens_B1[:, -1]

            for b in range(B):
                if not should_stop_B[b]:
                    generated_ids[b].append(tokens_B1[b, 0].item())

            logits_BSV = self.forward(
                tokens=tokens_B1, attention_mask=attention_mask, kv_cache=kv_cache,
            )

            hit_stop_B = torch.isin(tokens_B1, stop_ids).any(dim=-1)
            should_stop_B = should_stop_B.logical_or(hit_stop_B)

        results = []
        for b in range(B):
            text = tokenizer.decode(generated_ids[b], skip_special_tokens=False)
            text = text.replace("<|end_of_query|>", "").replace("<|end_of_text|>", "").strip()
            results.append(text)

        return results

    # Main API: generate

    @torch.inference_mode()
    def generate(
        self,
        images,
        *,
        category: str | list[str] = "plain",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_k: int | None = None,
        min_dimension: int = 64,
        max_dimension: int = 1024,
        compile: bool = True,
        seed: int | None = 42,
    ) -> list[str]:
        """
        Extract text from document images.

        Args:
            images: Single PIL Image (or path/URL) or list of them.
            category: OCR category — one of "plain", "text", "table", "formula",
                "caption", "footnote", "list-item", "page-footer", "page-header",
                "section-header", "title". Can be a single string (applied to all
                images) or a list (one per image).
            max_new_tokens: Maximum generation steps.
            temperature: Sampling temperature (0.0 = greedy).
            top_k: Top-k sampling (None = disabled).
            min_dimension: Min image side after resize.
            max_dimension: Max image side after resize.
            compile: Whether to torch.compile on first call.
            seed: Random seed for reproducibility (None = non-deterministic).

        Returns:
            List of extracted text strings, one per image.
        """
        self._ensure_device_buffers()
        if compile:
            self.compile_model()

        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        if isinstance(category, str):
            category = [category] * len(images)
        assert len(images) == len(category), "Must provide one category per image"

        image_prompt_pairs = []
        for img, cat in zip(images, category):
            instruction = CATEGORY_PROMPTS.get(cat.strip().lower(), CATEGORY_PROMPTS["plain"])
            prompt = f"<|image|>{instruction}\n<|OCR_PLAIN|>"
            image_prompt_pairs.append((img, prompt))

        return self._generate_batch(
            image_prompt_pairs,
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            min_dimension=min_dimension, max_dimension=max_dimension, seed=seed,
        )

    # Main API: generate_with_layout

    @torch.inference_mode()
    def generate_with_layout(
        self,
        images,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_k: int | None = None,
        min_dimension: int = 64,
        max_dimension: int = 1024,
        compile: bool = True,
        seed: int | None = 42,
        layout_threshold: float = 0.3,
        layout_batch_size: int = 4,
        ocr_batch_size: int = 32,
        containment_threshold: float = 0.8,
        layout_model: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
    ) -> list[list[dict]]:
        """
        Run layout detection then OCR on each detected region.

        Args:
            images: Single PIL Image (or path/URL) or list of them.
            max_new_tokens: Maximum generation steps per crop.
            temperature: Sampling temperature (0.0 = greedy).
            top_k: Top-k sampling (None = disabled).
            min_dimension: Min crop side after resize for OCR.
            max_dimension: Max crop side after resize for OCR.
            compile: Whether to torch.compile on first call.
            seed: Random seed for reproducibility.
            layout_threshold: Confidence threshold for layout detections.
            layout_batch_size: Batch size for layout detection.
            ocr_batch_size: Batch size for OCR generation (chunks crops).
            containment_threshold: Drop formula boxes >threshold contained in text boxes.
            layout_model: HuggingFace model ID for layout detection.

        Returns:
            Per-image list of detections, each a dict with keys:
            ``category``, ``bbox`` [x1,y1,x2,y2], ``score``, ``text``.
        """
        self._ensure_device_buffers()
        if compile:
            self.compile_model()
        self._load_layout_model(layout_model)

        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        pil_images = [load_image(img).convert("RGB") for img in images]

        # --- Layout detection (batched) ---
        all_layout_dets: list[list[dict]] = []
        for i in range(0, len(pil_images), layout_batch_size):
            batch_imgs = pil_images[i : i + layout_batch_size]
            dets = self._run_layout_detection(batch_imgs, threshold=layout_threshold)
            all_layout_dets.extend(dets)

        # --- Filter nested boxes (e.g. inline formulas inside text) ---
        all_layout_dets = [
            _filter_nested_detections(dets, containment_threshold)
            for dets in all_layout_dets
        ]

        # --- Build crops + track origin ---
        flat_crops: list[tuple[Image.Image, str]] = []
        crop_origins: list[tuple[int, int]] = []  # (image_idx, det_idx)

        for img_idx, (pil_img, dets) in enumerate(zip(pil_images, all_layout_dets)):
            if not dets or (len(dets) == 1 and dets[0]["category"].strip().lower() == "image"):
                prompt = f"<|image|>{CATEGORY_PROMPTS['plain']}\n<|OCR_PLAIN|>"
                flat_crops.append((pil_img, prompt))
                crop_origins.append((img_idx, -1))
                continue

            img_w, img_h = pil_img.size
            for det_idx, det in enumerate(dets):
                cat_key = det["category"].strip().lower()
                ocr_cat = LAYOUT_TO_OCR_CATEGORY.get(cat_key)
                if ocr_cat is None:
                    continue

                x1, y1, x2, y2 = det["bbox"]
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(img_w, int(x2 + 0.5))
                y2 = min(img_h, int(y2 + 0.5))
                cw, ch = x2 - x1, y2 - y1
                if cw < _MIN_CROP_DIM or ch < _MIN_CROP_DIM:
                    continue
                short, long = sorted((cw, ch))
                resized_short = short * (max_dimension / long) if long > max_dimension else short
                if resized_short < _MIN_CROP_DIM:
                    continue

                crop = pil_img.crop((x1, y1, x2, y2))
                instruction = CATEGORY_PROMPTS.get(ocr_cat, CATEGORY_PROMPTS["plain"])
                prompt = f"<|image|>{instruction}\n<|OCR_PLAIN|>"
                flat_crops.append((crop, prompt))
                crop_origins.append((img_idx, det_idx))

        # --- OCR in chunks ---
        flat_texts: list[str] = []
        for i in range(0, max(len(flat_crops), 1), ocr_batch_size):
            chunk = flat_crops[i : i + ocr_batch_size]
            if not chunk:
                break
            texts = self._generate_batch(
                chunk,
                max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                min_dimension=min_dimension, max_dimension=max_dimension, seed=seed,
            )
            flat_texts.extend(texts)

        # --- Reassemble per-image results ---
        results: list[list[dict]] = [[] for _ in range(len(pil_images))]
        for (img_idx, det_idx), text in zip(crop_origins, flat_texts):
            if det_idx == -1:
                img_w, img_h = pil_images[img_idx].size
                results[img_idx].append({
                    "category": "plain",
                    "bbox": [0, 0, img_w, img_h],
                    "score": 1.0,
                    "text": text,
                })
            else:
                det = all_layout_dets[img_idx][det_idx]
                results[img_idx].append({
                    "category": det["category"],
                    "bbox": det["bbox"],
                    "score": det["score"],
                    "text": text,
                })

        return results
