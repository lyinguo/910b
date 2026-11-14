"""Observe the native Large BF16 vision encoder and full-prefix greedy decoding."""

import functools
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def host_array(tensor):
    host = tensor.detach().cpu().contiguous()
    if host.dtype == torch.bfloat16:
        host = host.view(torch.uint16)
    return host.numpy().copy()


def bf16_float(bits):
    assert bits.dtype == np.uint16
    return (bits.astype(np.uint32) << 16).view(np.float32)


def array_record(array):
    return {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
    }


def tensor_record(tensor):
    return {**array_record(host_array(tensor)), 'dtype': str(tensor.dtype)}


def native_attention(model):
    return {
        'root': model.config._attn_implementation,
        'vision': model.encoder.config._attn_implementation,
        'text': model.decoder.config._attn_implementation,
    }


class PostPenaltyObservation(LogitsProcessor):
    def __init__(self, trace):
        self.trace = trace

    def __call__(self, input_ids, scores):
        self.trace.observe_scores(input_ids, scores)
        return scores


class NativePix2Trace:
    def __init__(self, model, device, first_forward=None):
        self.model = model
        self.device = torch.device(device)
        self.first_forward = first_forward
        self.active = False
        self.handles = []
        self.activation_names = []
        self.hook(model.encoder, self.observe_encoder)
        self.hook(model, self.observe_forward)
        self.hook(model.decoder.lm_head, self.observe_head)
        for axis in ('row', 'column'):
            module = getattr(model.encoder.embeddings, f'{axis}_embedder')
            self.hook(module, self.observe_coordinate(axis))
        self.hook(model.encoder.encoder.layer[0].attention, self.observe_vision_bias)
        first = model.decoder.layer[0]
        self.hook(first.self_attention.attention, self.observe_self)
        self.hook(first.self_attention.attention.relative_attention_bias, self.observe_buckets)
        for index, layer in enumerate(model.decoder.layer):
            self.hook(layer.encoder_decoder_attention.attention, self.observe_cross(index))
        hidden_layers = {'encoder.embeddings', 'encoder.layernorm', 'decoder.final_layer_norm'}
        hidden_layers.update(f'encoder.encoder.layer.{index}' for index in range(18))
        hidden_layers.update(f'decoder.layer.{index}' for index in range(18))
        for name, module in model.named_modules():
            if name.endswith('.act'):
                assert type(module).__name__ == 'NewGELUActivation', (name, type(module))
                self.activation_names.append(name)
                self.hook(module, self.observe_activation(name))
            elif name in hidden_layers:
                self.hook(module, self.observe_hidden(name))
        assert len(self.activation_names) == 36
        self.original_processors = model._get_logits_processor

        @functools.wraps(self.original_processors)
        def observe_processors(*args, **kwargs):
            processors = self.original_processors(*args, **kwargs)
            if self.active:
                self.processor_types = [type(item).__name__ for item in processors]
                assert self.processor_types == ['RepetitionPenaltyLogitsProcessor']
                assert processors[0].penalty == 1.2 and processors[0].prompt_ignore_length is None
                processors.append(PostPenaltyObservation(self))
            return processors

        model._get_logits_processor = observe_processors

    def hook(self, module, function):
        self.handles.append(module.register_forward_hook(function, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.model._get_logits_processor = self.original_processors

    def begin(self, capture_layers=False):
        assert not self.active
        self.active = True
        self.capture_layers = capture_layers
        self.encoder = None
        self.head = None
        self.arrays = {}
        self.raw_logits = []
        self.penalized_logits = []
        self.steps = []
        self.current = {}
        self.cross_counts = [0] * 18
        self.activation_counts = dict.fromkeys(self.activation_names, 0)
        self.processor_types = []

    def observe_encoder(self, module, args, kwargs, output):
        if self.active:
            assert self.encoder is None
            self.encoder = output.last_hidden_state
            assert self.encoder.shape == (1, 2048, 1536) and self.encoder.device == self.device
            assert self.encoder.dtype == torch.bfloat16
            self.arrays['encoder'] = host_array(self.encoder)
            self.arrays['encoder_mask'] = host_array(kwargs['attention_mask'])

    def observe_coordinate(self, axis):
        def capture(module, args, kwargs, output):
            if self.active:
                assert args[0].device == self.device and args[0].dtype == torch.int64
                self.arrays[f'{axis}_positions'] = host_array(args[0])
        return capture

    def observe_vision_bias(self, module, args, kwargs, output):
        if self.active:
            bias = output[1]
            row = bias[:, :1, :1, :]
            assert bias.device == self.device and torch.equal(bias, row.expand_as(bias))
            self.vision_bias_shape = list(bias.shape)
            self.arrays['vision_mask_bias_row'] = host_array(row)

    def observe_buckets(self, module, args, kwargs, output):
        if self.active:
            assert args[0].device == self.device
            self.current['relative_buckets'] = tensor_record(args[0])

    def observe_self(self, module, args, kwargs, output):
        if self.active:
            assert kwargs.get('past_key_values') is None
            mask = kwargs['mask']
            self.current['causal_mask'] = None if mask is None else tensor_record(mask)
            self.current['self_position_bias'] = tensor_record(output[1])

    def observe_cross(self, index):
        def capture(module, args, kwargs, output):
            if self.active:
                encoder = kwargs['key_value_states']
                assert encoder.data_ptr() == self.encoder.data_ptr() and encoder.device == self.device
                assert kwargs.get('past_key_values') is None
                self.cross_counts[index] += 1
                if index == 0:
                    mask = kwargs['mask']
                    bias = output[1]
                    row = bias[:, :1, :1, :]
                    assert bias.device == self.device and torch.equal(bias, row.expand_as(bias))
                    if mask is None:
                        assert np.all(self.arrays['encoder_mask'] == 1), 'Native omitted mask with padded patches'
                    else:
                        assert mask.device == self.device
                    self.current['cross_mask'] = None if mask is None else tensor_record(mask)
                    self.current['cross_bias_shape'] = list(bias.shape)
                    self.current['cross_bias_row'] = tensor_record(row)
                    if not self.steps:
                        self.arrays['cross_input'] = host_array(encoder)
                        self.arrays['cross_bias_first_row'] = host_array(row)
        return capture

    def observe_activation(self, name):
        def capture(module, args, kwargs, output):
            if self.active:
                assert output.device == self.device and output.dtype == torch.bfloat16
                self.activation_counts[name] += 1
                if self.capture_layers and not self.steps:
                    self.arrays[f'act_input.{name}'] = host_array(args[0][:, :4])
                    self.arrays[f'act_output.{name}'] = host_array(output[:, :4])
        return capture

    def observe_hidden(self, name):
        def capture(module, args, kwargs, output):
            if self.active and self.capture_layers and not self.steps:
                hidden = output[0] if isinstance(output, tuple) else output
                self.arrays[f'hidden.{name}'] = host_array(hidden[:, :4])
        return capture

    def observe_head(self, module, args, kwargs, output):
        if self.active:
            assert output.device == self.device and output.dtype == torch.bfloat16
            self.head = output

    def observe_forward(self, module, args, kwargs, output):
        if not self.active:
            return
        prefix = kwargs['decoder_input_ids']
        assert kwargs['use_cache'] is False and kwargs.get('past_key_values') is None
        assert output.past_key_values is None and output.logits.data_ptr() == self.head.data_ptr()
        assert self.head.shape == (1, prefix.shape[1], 50244)
        self.current['head_shape'] = list(self.head.shape)
        self.current['prefix'] = host_array(prefix)[0].tolist()
        self.current['past_key_values_is_none'] = True
        logits = host_array(self.head[0, -1])
        assert np.isfinite(bf16_float(logits)).all()
        self.raw_logits.append(logits)
        self.head = None
        if not self.steps and self.first_forward is not None:
            self.first_forward()

    def observe_scores(self, prefix, scores):
        assert self.active and scores.shape == (1, 50244) and scores.device == self.device
        assert scores.dtype == torch.float32 and host_array(prefix)[0].tolist() == self.current['prefix']
        values = host_array(scores[0])
        assert np.isfinite(values).all()
        self.penalized_logits.append(values)
        self.current['selected_argmax'] = int(values.argmax())
        self.steps.append(self.current)
        self.current = {}

    def finish(self, generated):
        self.active = False
        sequence = host_array(generated)[0].tolist()
        count = len(self.steps)
        assert 1 <= count <= 2048 and len(sequence) == count + 1 and sequence[0] == 0
        assert sequence[-1] == 1 or count == 2048
        assert 1 not in sequence[:-1]
        assert sequence[1:] == [step['selected_argmax'] for step in self.steps]
        assert all(step['prefix'] == sequence[:index + 1] for index, step in enumerate(self.steps))
        assert len(self.raw_logits) == len(self.penalized_logits) == count
        assert self.cross_counts == [count] * 18
        assert all(times == (count if name.startswith('decoder.') else 1) for name, times in self.activation_counts.items())
        assert np.array_equal(self.arrays['encoder'], self.arrays['cross_input'])
        self.arrays.update(raw_head_logits=np.stack(self.raw_logits), native_penalized_logits=np.stack(self.penalized_logits))
        return {
            'generation_sequence_ids': sequence,
            'generation_token_count': count,
            'truncated': count == 2048 and sequence[-1] != 1,
            'budget_hit': count == 2048,
            'stop_reason': 'eos' if sequence[-1] == 1 else 'length',
            'steps': self.steps,
            'vision_bias_shape': self.vision_bias_shape,
            'cross_input_storage_matches_all_18_layers': self.cross_counts,
            'activation_call_counts': self.activation_counts,
            'native_logit_processors': self.processor_types,
            'device': str(self.device),
            'arrays': {name: array_record(array) for name, array in self.arrays.items()},
        }
