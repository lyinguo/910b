"""Independent Cognitive vLLM-math implementation for an Ascend reference adapter.

This module is not an approved adapter. Scheduler, tokenizer/detokenizer, report
delivery and independent verification belong to the model adapter still under
review. No scores or workbook writes are provided here. It deliberately does
not import the older HF Gemma3 adapter or the optional CUDA/vLLM packages.

Architecture/math reference: the exact installed A100 vLLM Gemma3, SigLIP and
IR layernorm sources captured in the model's diagnostic manifest. Packed QKV
and gate/up, unrounded FP32 fused residual normalization, a BF16 RoPE cache,
and text-only embedding scaling are represented explicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import torch
from torch.nn import functional as F


def tensor_sha(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gemma_rms(value: torch.Tensor, weight: torch.Tensor,
              residual: torch.Tensor | None = None, epsilon: float = 1e-6):
    """vLLM IR: normalize the unrounded F32 sum, return BF16 residual separately.

    Gemma's weight+1 is computed in F32. Converting that weight to BF16 before
    multiplication, or rounding the residual before its variance, changes the
    original operation. These expressions preserve the source IR arithmetic;
    backend reduction/kernel differences still require numerical disclosure.
    """
    dtype = value.dtype
    summed = value.float()
    if residual is not None:
        summed = summed + residual.float()
    variance = summed.pow(2).mean(-1, keepdim=True)
    normalized = summed * torch.rsqrt(variance + epsilon)
    output = (normalized * (weight.float() + 1.0)).to(dtype)
    return output if residual is None else (output, summed.to(dtype))


def gelu_tanh_and_mul(packed: torch.Tensor) -> torch.Tensor:
    """Preserve the genuine CUDA fused op's BF16 intermediate GELU rounding.

    The genuine CUDA probe found 5848/40960 differences for a final-only F32
    cast, whereas BF16 GELU followed by BF16 multiply matched every probe value.
    This is an NPU reference to those original semantics, not CUDA acceleration
    or a claim that the NPU GELU kernel is bit-identical to the CUDA kernel.
    """
    gate, up = packed.chunk(2, dim=-1)
    return F.gelu(gate, approximate='tanh') * up


class RotaryCache:
    """Full original vLLM cache extent; frequencies are computed on the NPU.

    No A100/CPU cache is copied in to force numerical agreement. Global linear
    scaling has 131072*8 rows; local rotation has 131072 rows. The source native
    CUDA op evaluates products in F32 with one final cast to the query dtype.
    """
    def __init__(self, device: torch.device, *, base: float, factor: float):
        self.device, self.base, self.factor = device, base, factor
        self.max_original_positions = 131072
        frequency = 1.0 / (
            base ** (torch.arange(0, 256, 2, dtype=torch.float32, device=device) / 256)
        )
        positions = torch.arange(self.max_original_positions * factor,
                                 dtype=torch.float32, device=device)
        if factor != 1.0:
            positions = positions / factor
        angles = torch.einsum('i,j->ij', positions, frequency)
        self.cache = torch.cat([angles.cos(), angles.sin()], dim=-1).bfloat16()
        assert self.cache.shape == (int(self.max_original_positions * factor), 256)

    def apply(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert value.ndim == 3 and value.shape[0] == positions.numel()
        cos, sin = self.cache[positions].chunk(2, dim=-1)
        cos, sin = cos[:, None, :].float(), sin[:, None, :].float()
        left, right = value.float().chunk(2, dim=-1)
        return torch.cat([left * cos - right * sin,
                          right * cos + left * sin], dim=-1).to(value.dtype)


class SourceWeights:
    """Fresh whole-file/source/cast/actual-NPU SHA checks for all 883 tensors.

    No optional missing keys, uninitialized learned parameters, library tie()
    operation or sampling of a few weight positions is permitted. The source
    q/k/v and gate/up slices populate one packed physical NPU allocation.
    """
    def __init__(self, root: Path, manifest: dict, device: torch.device,
                 progress: Callable[[dict], None] | None = None):
        from safetensors import safe_open
        assert device.type == 'npu', 'Learned model execution must be on an Ascend NPU'
        self.root, self.device = root.resolve(), device
        self.values: dict[str, torch.Tensor] = {}
        self.records: list[dict] = []
        index = {}
        # This model's authentic checkpoint is one safetensors file without an
        # index JSON. Read its actual headers; never manufacture resource files.
        for source_file in sorted(manifest['source_files']):
            if not source_file.endswith('.safetensors'):
                continue
            with safe_open(str(self.root / source_file), framework='pt', device='cpu') as header:
                for key in header.keys():
                    assert key not in index, ('Duplicate learned tensor', key)
                    index[key] = source_file
        expected = manifest['source_tensors']
        assert set(index) == set(expected) and len(expected) == 883
        before = {}
        for name, reviewed in manifest['source_files'].items():
            path = self.root / name
            assert path.is_file() and file_sha(path) == reviewed['SHA256'], name
            stat = path.stat()
            before[name] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        self.original_files = before
        visited, slices = set(), {}
        for shard in sorted(set(index.values())):
            assert Path(shard).name == shard
            with safe_open(str(self.root / shard), framework='pt', device='cpu') as handle:
                assert set(handle.keys()) == {key for key in index if index[key] == shard}
                for key in sorted(handle.keys()):
                    raw, reviewed = handle.get_tensor(key), expected[key]
                    assert key not in visited and str(raw.dtype) == reviewed['dtype'] == 'torch.bfloat16'
                    assert list(raw.shape) == reviewed['shape'] and tensor_sha(raw) == reviewed['SHA256'], key
                    target, start, total = self.packed_mapping(key, expected)
                    if target not in self.values:
                        shape = list(raw.shape)
                        if total is not None:
                            shape[0] = total
                        self.values[target] = torch.empty(shape, dtype=torch.bfloat16, device=device)
                    tensor = self.values[target]
                    actual = tensor if start is None else tensor[start:start + raw.shape[0]]
                    assert actual.shape == raw.shape
                    actual.copy_(raw)
                    actual_sha = tensor_sha(actual)
                    assert actual_sha == reviewed['SHA256'] and torch.equal(actual.cpu(), raw), key
                    self.records.append(dict(source=key, source_file=shard,
                        source_dtype=str(raw.dtype), source_shape=list(raw.shape),
                        source_SHA256=reviewed['SHA256'], cast_SHA256=reviewed['SHA256'],
                        actual_NPU_SHA256=actual_sha, packed_parameter=target,
                        packed_slice=None if start is None else [start, start + raw.shape[0]],
                        physical_shape=list(tensor.shape), device=str(tensor.device)))
                    slices.setdefault(target, []).append((0 if start is None else start,
                                                         raw.shape[0], key))
                    visited.add(key)
            if progress:
                progress({'phase': 'weight_load', 'source_tensors': len(visited),
                          'expected_source_tensors': 883, 'shard': shard})
        assert visited == set(expected)
        for name, covered in slices.items():
            end = 0
            for start, length, key in sorted(covered):
                assert start == end, ('Packed overlap or gap', name, key, start, end)
                end += length
            assert end == self.values[name].shape[0]
        assert not any(name.endswith('lm_head.weight') for name in expected)
        assert self.values['language_model.model.embed_tokens.weight'].shape == (262208, 2560)
        self.physical = [dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                              SHA256=tensor_sha(value), device=str(value.device))
                         for name, value in sorted(self.values.items())]
        self.assert_resources_unchanged()

    @staticmethod
    def packed_mapping(key: str, expected: dict):
        for part, tag in [('.q_proj.', 'q'), ('.k_proj.', 'k'), ('.v_proj.', 'v')]:
            if part in key:
                qsize = expected[key.replace(part, '.q_proj.')]['shape'][0]
                ksize = expected[key.replace(part, '.k_proj.')]['shape'][0]
                return (key.replace(part, '.qkv_proj.'),
                        {'q': 0, 'k': qsize, 'v': qsize + ksize}[tag], qsize + 2 * ksize)
        for part, tag in [('.gate_proj.', 0), ('.up_proj.', 1)]:
            if part in key:
                width = expected[key]['shape'][0]
                return key.replace(part, '.gate_up_proj.'), tag * width, 2 * width
        return key, None, None

    def assert_resources_unchanged(self):
        for name, original in self.original_files.items():
            stat = (self.root / name).stat()
            assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == original, name

    def __getitem__(self, key):
        return self.values[key]


class CognitiveNativeMath:
    """Packed native architecture; no scheduler/output/approval shortcuts.

    ``text_step`` takes actual scheduled request chunks and their absolute
    positions. All active requests share packed GEMMs. Attention is evaluated
    per logical request against its own cache, never across request boundaries.
    This is an explicit logical KV reference, not CUDA paged-cache acceleration.
    A separate adapter must qualify scheduling, encoder/MM caches and sampling.
    """
    def __init__(self, weights: SourceWeights, config: dict,
                 observe: Callable[[str, torch.Tensor, dict], None] | None = None):
        text, vision = config['text_config'], config['vision_config']
        assert config['architectures'] == ['Gemma3ForConditionalGeneration']
        assert (text['num_hidden_layers'], text['hidden_size'], text['intermediate_size'],
                text['num_attention_heads'], text['num_key_value_heads'], text['head_dim'],
                text['vocab_size']) == (34, 2560, 10240, 8, 4, 256, 262208)
        assert (vision['num_hidden_layers'], vision['hidden_size'], vision['intermediate_size'],
                vision['num_attention_heads'], vision['image_size'], vision['patch_size']) == (27, 1152, 4304, 16, 896, 14)
        assert text['rms_norm_eps'] == vision['layer_norm_eps'] == 1e-6
        assert text['sliding_window'] == 1024 and text['max_position_embeddings'] == 131072
        assert text['hidden_activation'] == vision['hidden_act'] == 'gelu_pytorch_tanh'
        assert text['use_bidirectional_attention'] is False and config['mm_tokens_per_image'] == 256
        assert text.get('attn_logit_softcapping') is None and text.get('final_logit_softcapping') is None
        self.w, self.device, self.observe = weights, weights.device, observe
        self.layer_types = list(text['layer_types'])
        assert self.layer_types == ['full_attention' if (i+1) % 6 == 0 else 'sliding_attention' for i in range(34)]
        self.rope = {'sliding_attention': RotaryCache(self.device, base=10000.0, factor=1.0),
                     'full_attention': RotaryCache(self.device, base=1000000.0, factor=8.0)}
        self.scale = torch.tensor(math.sqrt(2560), dtype=torch.bfloat16, device=self.device)
        self.vision_positions = torch.arange(4096, dtype=torch.int64, device=self.device)
        self.cache: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = [{} for _ in range(34)]
        self.forward_ordinal = 0

    def emit(self, name, value, **metadata):
        assert value.device == self.device
        if self.observe:
            self.observe(name, value, metadata)
        return value

    def linear(self, value, prefix):
        return F.linear(value, self.w[prefix + '.weight'], self.w.values.get(prefix + '.bias'))

    def layer_norm(self, value, prefix):
        return F.layer_norm(value, (value.shape[-1],), self.w[prefix + '.weight'],
                            self.w[prefix + '.bias'], 1e-6)

    @torch.inference_mode()
    def vision(self, pixels):
        assert pixels.device == self.device and pixels.dtype == torch.bfloat16
        assert tuple(pixels.shape[1:]) == (3, 896, 896)
        prefix = 'vision_tower.vision_model'
        self.emit('vision_actual_pixels', pixels)
        value = F.conv2d(pixels, self.w[prefix + '.embeddings.patch_embedding.weight'],
                         self.w[prefix + '.embeddings.patch_embedding.bias'], stride=14)
        value = value.flatten(2).transpose(1, 2)
        value = value + F.embedding(self.vision_positions,
                                   self.w[prefix + '.embeddings.position_embedding.weight'])
        for layer in range(27):
            name = prefix + '.encoder.layers.' + str(layer)
            residual = value
            normalized = self.layer_norm(value, name + '.layer_norm1')
            qkv = self.linear(normalized, name + '.self_attn.qkv_proj')
            query, key, val = [t.reshape(t.shape[0], 4096, 16, 72).transpose(1, 2)
                               for t in qkv.chunk(3, -1)]
            attention = F.scaled_dot_product_attention(query, key, val,
                dropout_p=0.0, is_causal=False, scale=72**-0.5)
            attention = attention.transpose(1, 2).reshape_as(normalized)
            value = self.linear(attention, name + '.self_attn.out_proj') + residual
            residual = value
            hidden = self.linear(self.layer_norm(value, name + '.layer_norm2'), name + '.mlp.fc1')
            hidden = self.emit('vision_activation_%02d' % layer,
                              F.gelu(hidden, approximate='tanh'), original_activation='gelu_pytorch_tanh')
            value = self.linear(hidden, name + '.mlp.fc2') + residual
        value = self.layer_norm(value, prefix + '.post_layernorm')
        self.emit('vision_encoder_output', value)
        pooled = F.avg_pool2d(value.transpose(1, 2).reshape(value.shape[0], 1152, 64, 64).contiguous(),
                             kernel_size=4, stride=4).flatten(2).transpose(1, 2)
        normed = gemma_rms(pooled, self.w['multi_modal_projector.mm_soft_emb_norm.weight'])
        projected = normed @ self.w['multi_modal_projector.mm_input_projection_weight']
        assert tuple(projected.shape) == (pixels.shape[0], 256, 2560)
        return self.emit('visual_projector_output', projected)

    @torch.inference_mode()
    def embeddings(self, tokens: torch.Tensor, visuals: torch.Tensor | None,
                   positions: torch.Tensor, image_start_position: int | None = None):
        assert tokens.device == self.device and tokens.dtype == torch.int64 and tokens.ndim == 1
        value = F.embedding(tokens, self.w['language_model.model.embed_tokens.weight']) * self.scale
        # Generated image-token IDs are ordinary text embedding lookups when
        # the runner supplies no multimodal embeddings at decode time. The
        # full original vocabulary is never masked to prevent these outputs.
        image = tokens == 262144 if visuals is not None else torch.zeros_like(tokens, dtype=torch.bool)
        if visuals is None:
            assert not bool(image.any())
        else:
            assert visuals.shape == (256, 2560) and image_start_position is not None
            selected = positions[image] - image_start_position
            assert int(selected.numel()) == int(image.sum()) and bool(((selected >= 0) & (selected < 256)).all())
            injected = visuals[selected]
            value[image] = injected
            assert torch.equal(value[image], injected), 'Visual features must not be multiplied by text sqrt'
        return value

    @torch.inference_mode()
    def text_step(self, requests: list[dict]) -> torch.Tensor:
        """Each item: id, token_ids, absolute positions, optional visual features.

        A request chunk must follow its actual past length exactly. The caller
        supplies scheduler chunks; this module never silently serializes a full
        model batch or invents a chunking/OOM fallback policy.
        """
        assert requests and len({r['id'] for r in requests}) == len(requests)
        lengths = [r['token_ids'].numel() for r in requests]
        assert min(lengths) > 0
        positions = torch.cat([r['positions'] for r in requests])
        value = torch.cat([self.embeddings(r['token_ids'], r.get('visuals'), r['positions'], r.get('image_start_position')) for r in requests])
        assert positions.dtype == torch.int64 and positions.device == self.device
        residual = None
        self.emit('text_input_embeddings', value, ordinal=self.forward_ordinal,
                  request_ids=[r['id'] for r in requests], lengths=lengths,
                  positions=positions.cpu().tolist())
        for layer in range(34):
            prefix = 'language_model.model.layers.' + str(layer)
            if residual is None:
                residual = value
                normalized = gemma_rms(value, self.w[prefix + '.input_layernorm.weight'])
            else:
                normalized, residual = gemma_rms(value, self.w[prefix + '.input_layernorm.weight'], residual)
            qkv = self.linear(normalized, prefix + '.self_attn.qkv_proj')
            query, key, val = qkv.split([2048, 1024, 1024], -1)
            query = gemma_rms(query.reshape(-1, 8, 256), self.w[prefix + '.self_attn.q_norm.weight'])
            key = gemma_rms(key.reshape(-1, 4, 256), self.w[prefix + '.self_attn.k_norm.weight'])
            rotary = self.rope[self.layer_types[layer]]
            query, key = rotary.apply(query, positions), rotary.apply(key, positions)
            val = val.reshape(-1, 4, 256)
            # All packed QKV writes precede attention. Genuine vLLM can share
            # full prefix blocks with another request scheduled in THIS SAME
            # wave, including a source row sorted after the cached suffix row.
            # Resolve the request DAG before any attention reads those blocks.
            chunks={};offset=0
            for request,length in zip(requests,lengths):
                rid=request['id'];k=key[offset:offset+length];v=val[offset:offset+length]
                prefix_source=request.get('prefix_source_request_id')
                prefix_length=int(request.get('prefix_token_count',0))
                chunks[rid]=(k,v,request,prefix_source,prefix_length)
                self.emit('KV_delta_key_%02d' % layer,k,request_id=rid,
                    positions=request['positions'].cpu().tolist(),ordinal=self.forward_ordinal,
                    prefix_source_request_id=prefix_source,prefix_token_count=prefix_length)
                self.emit('KV_delta_value_%02d' % layer,v,request_id=rid,
                    positions=request['positions'].cpu().tolist(),ordinal=self.forward_ordinal,
                    prefix_source_request_id=prefix_source,prefix_token_count=prefix_length)
                offset+=length
            pending=set(chunks)
            while pending:
                progress=False
                for rid in list(pending):
                    k,v,request,source,n=chunks[rid]
                    if source is not None:
                        assert rid not in self.cache[layer] and n>0 and source!=rid
                        if source in pending:continue
                        source_k,source_v=self.cache[layer][source]
                        assert n<=source_k.shape[0]
                        past=(source_k[:n],source_v[:n])
                    else:past=self.cache[layer].get(rid)
                    past_length=0 if past is None else past[0].shape[0]
                    expected_positions=torch.arange(past_length,past_length+k.shape[0],dtype=torch.int64,device=self.device)
                    assert torch.equal(request['positions'],expected_positions),('Past/position mismatch',layer,rid)
                    if past is not None:k,v=torch.cat([past[0],k]),torch.cat([past[1],v])
                    self.cache[layer][rid]=(k,v);pending.remove(rid);progress=True
                assert progress,'Cyclic or missing real prefix source'
            outputs,offset=[],0
            for request,length in zip(requests,lengths):
                rid=request['id'];q=query[offset:offset+length]
                k,v=self.cache[layer][rid];past_length=k.shape[0]-length
                key_positions = torch.arange(k.shape[0], device=self.device)
                mask = request['positions'][:, None] >= key_positions[None, :]
                if self.layer_types[layer] == 'sliding_attention':
                    mask &= request['positions'][:, None] - key_positions[None, :] < 1024
                attended = F.scaled_dot_product_attention(q.transpose(0, 1)[None],
                    k.repeat_interleave(2, dim=1).transpose(0, 1)[None],
                    v.repeat_interleave(2, dim=1).transpose(0, 1)[None],
                    attn_mask=mask[None, None], dropout_p=0.0, is_causal=False, scale=256**-0.5)
                outputs.append(attended[0].transpose(0, 1).reshape(length, 2048))
                self.emit('attention_visibility_%02d' % layer, mask, request_id=rid,
                          past_length=past_length, key_length=k.shape[0],
                          layer_type=self.layer_types[layer], q_heads=8, kv_heads=4)
                offset += length
            attention = self.linear(torch.cat(outputs), prefix + '.self_attn.o_proj')
            attention = gemma_rms(attention, self.w[prefix + '.post_attention_layernorm.weight'])
            hidden, residual = gemma_rms(attention, self.w[prefix + '.pre_feedforward_layernorm.weight'], residual)
            packed = self.linear(hidden, prefix + '.mlp.gate_up_proj')
            hidden = self.emit('language_activation_%02d' % layer, gelu_tanh_and_mul(packed),
                               original_activation='gelu_pytorch_tanh_and_mul')
            hidden = self.linear(hidden, prefix + '.mlp.down_proj')
            value = gemma_rms(hidden, self.w[prefix + '.post_feedforward_layernorm.weight'])
        value, residual = gemma_rms(value, self.w['language_model.model.norm.weight'], residual)
        last_positions = torch.tensor([sum(lengths[:i+1])-1 for i in range(len(lengths))],
                                      dtype=torch.int64, device=self.device)
        # Genuine vLLM selects scheduled last hidden states before its full BF16 head.
        # Actual A100 compute_logits remains BF16; sampling may view these same
        # values in F32, but the model head must not be relabelled as F32.
        # The tied learned head remains the complete source262208 rows, including
        # the original padded IDs. No vocabulary resize/mask or label constraint.
        logits = F.linear(value[last_positions], self.w['language_model.model.embed_tokens.weight'])
        assert logits.dtype == torch.bfloat16 and logits.shape == (len(requests), 262208) and bool(torch.isfinite(logits).all())
        self.emit('full_vocabulary_logits', logits, ordinal=self.forward_ordinal,
                  request_ids=[r['id'] for r in requests], argmax=logits.argmax(-1).cpu().tolist())
        self.forward_ordinal += 1
        return logits

    def release_requests(self, request_ids: list[str]):
        for cache in self.cache:
            for rid in request_ids:
                assert rid in cache
                del cache[rid]
