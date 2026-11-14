"""Independently verify the native BF16 Large report, inputs and score."""
from decimal import Decimal
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
from PIL import Image
from tokenizers import Tokenizer
import torch
from transformers import AutoConfig, Pix2StructForConditionalGeneration

from verify_TextCapsBase_A100_FP16_disclosed_v3_0527_20261107 import (
    R, OUTPUT, REPORT_NAMES, file_sha, independent_patches, read_json, read_lines, record, rouge_l,
)

MODEL = 'google/pix2struct-textcaps-large'
ATTENTION = {'root': 'eager', 'vision': 'eager', 'text': 'eager'}
GENERATION = {'max_new_tokens': 2048, 'do_sample': False, 'repetition_penalty': 1.2}
RELATIVE = 'decoder.layer.0.self_attention.attention.relative_attention_bias.weight'
ALL_VALID = [31, 122, 124, 180]
CONTROLS = [0, 1, 2, 31, 122, 124, 174, 180]
LAYERS = {*(f'encoder.encoder.layer.{i}.mlp.act' for i in range(18)),
          *(f'decoder.layer.{i}.mlp.DenseReluDense.act' for i in range(18))}
HIDDEN = {'encoder.embeddings', 'encoder.layernorm', 'decoder.final_layer_norm',
          *(f'encoder.encoder.layer.{i}' for i in range(18)), *(f'decoder.layer.{i}' for i in range(18))}
ABBREVIATION = re.compile(r'(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|No|Jr|Sr|U\.S|e\.g|i\.e|approx|Fig|Vol)$', re.I)
BYTE_FALLBACK = re.compile(r'<0x[0-9A-Fa-f]{2}>')


def caption(raw):
    cleaned = ' '.join(BYTE_FALLBACK.sub(' ', raw).split())
    for punctuation in re.finditer(r'[.!?](?=\s|$)', cleaned):
        prefix = cleaned[:punctuation.end()]
        if not ABBREVIATION.search(prefix.split()[-1].rstrip('.!?')):
            return cleaned, prefix
    return cleaned, cleaned


def bf16_bits(tensor):
    return tensor.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()


def bf16_float(bits):
    assert bits.dtype == np.uint16
    return (bits.astype(np.uint32) << 16).view(np.float32)


def native_record(array, dtype):
    return {**record(array), 'dtype': dtype}


def independent_inputs(path):
    with Image.open(path) as image:
        flat, mask = independent_patches(image.convert('RGB'))
    return flat.numpy(), bf16_bits(flat), mask.numpy()


def own_config(weight):
    config = AutoConfig.from_pretrained(weight, local_files_only=True)
    assert config.vision_config.num_hidden_layers == config.text_config.num_hidden_layers == 18
    assert config.vision_config.hidden_size == config.text_config.hidden_size == 1536
    assert config.vision_config.num_attention_heads == config.text_config.num_heads == 24
    assert config.vision_config.d_ff == config.text_config.d_ff == 3968
    assert config.text_config.vocab_size == 50244
    assert config.text_config.relative_attention_num_buckets == 32
    assert config.text_config.relative_attention_max_distance == 128
    assert not config.tie_word_embeddings and not config.text_config.tie_word_embeddings
    return config


def check_weights(weight, p, lineage, device):
    path = weight / 'pytorch_model.bin'
    assert path.stat().st_size == p['checkpoint']['bytes'] and file_sha(path) == p['checkpoint']['sha256']
    config = own_config(weight)
    with torch.device('meta'):
        model = Pix2StructForConditionalGeneration(config)
    shapes = {name: list(value.shape) for name, value in model.named_parameters(remove_duplicate=False)}
    assert len(shapes) == 423 and not list(model.buffers())
    loaded = read_json(lineage['weight_loading_evidence'])
    aa = read_json(p['source_weight_manifest'])
    source = torch.load(path, weights_only=True, map_location='cpu', mmap=zipfile.is_zipfile(path))
    assert loaded.keys() == aa.keys() == source.keys() == shapes.keys()
    for name, tensor in source.items():
        assert tensor.dtype == torch.bfloat16 and list(tensor.shape) == shapes[name], name
        expected = native_record(tensor.contiguous().view(torch.uint16).numpy(), 'torch.bfloat16')
        assert aa[name] == {'source': expected, 'actual_CUDA': expected}, name
        assert loaded[name] == {**aa[name], 'actual_NPU': expected, 'device': device}, name
    return source[RELATIVE].float().numpy()


def self_masks(bias):
    assert bias.shape == (32, 24)
    minimum = torch.finfo(torch.bfloat16).min

    @lru_cache(maxsize=128)
    def expected(length):
        rows = np.arange(length, dtype=np.int64)
        difference = rows[:, None] - rows[None, :]
        distance = np.maximum(difference, 0)
        large = 16 + np.floor(np.log(np.maximum(distance, 16) / 16) / math.log(8) * 16).astype(np.int64)
        buckets = np.where(distance < 16, distance, np.minimum(31, large))
        mask = np.where(difference < 0, np.float32(minimum), np.float32(0.0))[None, None]
        positions = bias[buckets].transpose(2, 0, 1)[None]
        return (native_record(buckets, 'torch.int64'),
                native_record(bf16_bits(torch.from_numpy(mask)), 'torch.bfloat16'),
                native_record(bf16_bits(torch.from_numpy(positions + mask)), 'torch.bfloat16'))
    return expected


def scalar_table(p, device):
    receipt = read_json(p['scalar_receipt'])
    assert receipt['device'] == device and receipt['learned_model_forward_count'] == 0
    assert receipt['normal_multiply_exact'] and receipt['normal_divide_max_ULP_against_CPU_F32'] <= 1
    assert file_sha(receipt['path']) == receipt['sha256'] == p['evidence_sha256'][receipt['path']]
    with np.load(receipt['path'], allow_pickle=False) as measured:
        values = bf16_float(np.arange(65536, dtype=np.uint16))
        finite = np.isfinite(values)
        assert np.array_equal(measured['finite'], finite) and int(finite.sum()) == receipt['finite_BF16_encodings'] == 65280
        x = values[finite]
        div, mul = measured['positive_divide_F32'], measured['multiply_F32']
        assert div.dtype == mul.dtype == np.float32 and div.shape == mul.shape == x.shape
        normal = (abs(x) >= np.finfo(np.float32).tiny * 2) & (abs(x) <= np.finfo(np.float32).max / 2)
        assert int(normal.sum()) == receipt['normal_safe_magnitude_encodings']
        ulp = np.abs(div[normal].view(np.int32).astype(np.int64) -
                     (x[normal] / np.float32(1.2)).view(np.int32).astype(np.int64))
        assert int(ulp.max()) == receipt['normal_divide_max_ULP_against_CPU_F32']
        assert np.count_nonzero(ulp) == receipt['normal_divide_different_encodings']
        assert np.array_equal(mul[normal], x[normal] * np.float32(1.2))
        table = measured['signed_native_penalty_table']
        assert table.dtype == np.float32 and table.shape == (65536,)
        assert np.array_equal(table[finite].view(np.uint32), np.where(x < 0, mul, div).view(np.uint32))
    return table, receipt


def check_trace(path, item, patches, mask, device, expected_self, save_layers, table=None):
    sequence = item['generation_sequence_ids']
    count = item['generation_token_count']
    assert 1 <= count <= 2048 and len(sequence) == count + 1 and sequence[0] == 0
    assert all(type(value) is int and 0 <= value < 50244 for value in sequence)
    assert 1 not in sequence[:-1] and (sequence[-1] == 1 or count == 2048)
    assert item['budget_hit'] == (count == 2048) and item['truncated'] == (count == 2048 and sequence[-1] != 1)
    assert item['stop_reason'] == ('eos' if sequence[-1] == 1 else 'length')
    assert item['device'] == device and item['vision_bias_shape'] == [1, 24, 2048, 2048]
    assert item['cross_input_storage_matches_all_18_layers'] == [count] * 18
    assert item['native_logit_processors'] == ['RepetitionPenaltyLogitsProcessor']
    assert item['activation_call_counts'] == {name: count if name.startswith('decoder.') else 1 for name in LAYERS}
    assert len(item['steps']) == count
    cross = np.where(mask == 1, np.float32(0), np.float32(torch.finfo(torch.bfloat16).min))[:, None, None, :]
    cross_bits = bf16_bits(torch.from_numpy(cross))
    keys = {'encoder', 'cross_input', 'encoder_mask', 'row_positions', 'column_positions', 'vision_mask_bias_row',
            'cross_bias_first_row', 'raw_head_logits', 'native_penalized_logits'}
    if save_layers:
        keys.update(prefix + name for prefix in ('act_input.', 'act_output.') for name in LAYERS)
        keys.update('hidden.' + name for name in HIDDEN)
    with np.load(path, allow_pickle=False) as archive:
        assert len(archive.files) == len(keys) and set(archive.files) == set(item['arrays']) == keys
        arrays = {name: archive[name] for name in keys}
    for name, array in arrays.items():
        assert record(array) == item['arrays'][name], name
        assert np.isfinite(bf16_float(array) if array.dtype == np.uint16 else array).all(), name
    encoder = arrays['encoder']
    assert encoder.dtype == np.uint16 and encoder.shape == (1, 2048, 1536)
    assert np.array_equal(encoder, arrays['cross_input']) and arrays['cross_input'].dtype == np.uint16
    assert arrays['encoder_mask'].dtype == np.float32 and np.array_equal(arrays['encoder_mask'], mask)
    for axis, index in [('row', 0), ('column', 1)]:
        assert arrays[axis + '_positions'].dtype == np.int64
        assert np.array_equal(arrays[axis + '_positions'], bf16_float(patches[:, :, index]).astype(np.int64))
    assert arrays['vision_mask_bias_row'].dtype == np.float32
    assert np.array_equal(arrays['vision_mask_bias_row'], 1.0 - mask[:, None, None, :])
    assert arrays['cross_bias_first_row'].dtype == np.uint16 and np.array_equal(arrays['cross_bias_first_row'], cross_bits)
    if save_layers:
        for name in LAYERS:
            shape = (1, 1 if name.startswith('decoder.') else 4, 3968)
            for prefix in ('act_input.', 'act_output.'):
                assert arrays[prefix + name].shape == shape and arrays[prefix + name].dtype == np.uint16
        for name in HIDDEN:
            shape = (1, 1 if name.startswith('decoder.') else 4, 1536)
            assert arrays['hidden.' + name].shape == shape and arrays['hidden.' + name].dtype == np.uint16
    before, after = arrays['raw_head_logits'], arrays['native_penalized_logits']
    assert before.dtype == np.uint16 and after.dtype == np.float32 and before.shape == after.shape == (count, 50244)
    division_only = reciprocal_only = scalar_only = omitted = 0
    for step, row in enumerate(item['steps'], 1):
        buckets, causal, bias = expected_self(step)
        assert row['prefix'] == sequence[:step] and row['past_key_values_is_none']
        assert row['head_shape'] == [1, step, 50244] and row['relative_buckets'] == buckets
        assert row['causal_mask'] == causal and row['self_position_bias'] == bias
        assert row['cross_bias_shape'] == [1, 24, step, 2048]
        assert row['cross_bias_row'] == native_record(cross_bits, 'torch.bfloat16')
        if row['cross_mask'] is None:
            assert np.all(mask == 1), 'Native cross mask omitted despite padded patches'
            omitted += 1
        else:
            assert row['cross_mask'] == native_record(np.broadcast_to(cross_bits, (1, 1, step, 2048)), 'torch.bfloat16')
        raw, result = bf16_float(before[step - 1]), after[step - 1]
        seen = np.unique(sequence[:step])
        unmodified = np.ones(50244, dtype=bool)
        unmodified[seen] = False
        assert np.array_equal(raw[unmodified].view(np.uint32), result[unmodified].view(np.uint32))
        original, adjusted = raw[seen], result[seen]
        neg = original < 0
        assert np.array_equal(adjusted[neg], original[neg] * np.float32(1.2))
        pos, positive_out = original[~neg], adjusted[~neg]
        divides = positive_out == pos / np.float32(1.2)
        reciprocals = positive_out == pos * np.float32(1.0 / 1.2)
        if table is None:
            assert device.startswith('cuda:') and (divides | reciprocals).all()
        else:
            assert device.startswith('npu:')
            assert np.array_equal(adjusted.view(np.uint32), table[before[step - 1, seen]].view(np.uint32))
        division_only += int((divides & ~reciprocals).sum())
        reciprocal_only += int((reciprocals & ~divides).sum())
        scalar_only += int((~(divides | reciprocals)).sum())
        assert row['selected_argmax'] == int(result.argmax()) == sequence[step]
    return (division_only, reciprocal_only, scalar_only), omitted


def check_disclosure(p, lineage, count):
    if 'numerical_disclosure' not in p:
        assert count == 3 and 'numerical_disclosure' not in lineage
        return {}
    disclosure = read_json(p['numerical_disclosure'])
    assert file_sha(p['numerical_disclosure']) == p['numerical_disclosure_sha256']
    assert lineage['numerical_disclosure'] == disclosure and disclosure['numerical_review_verified']
    assert disclosure['own_BF16_vision_parameter_count'] == 167 and disclosure['cpu_FP32_is_diagnostic_only']
    assert disclosure['all_real_prefix_scalar_values_bit_equal']
    evidence = disclosure['evidence_sha256']
    assert len(evidence) >= 10
    for path, digest in evidence.items():
        assert file_sha(path) == digest, path
    return {**evidence, p['numerical_disclosure']: p['numerical_disclosure_sha256']}


def verify(folder, count):
    assert count in (3, 200) and {path.name for path in folder.iterdir()} == REPORT_NAMES
    assert folder.parent.parent == R / 'workspace/reports/google_pix2struct-textcaps-large'
    assert folder.parent.name == 'image-to-text' and re.fullmatch(r'\d{8}_\d{6}', folder.name)
    report = read_json(folder / 'eval_report.json')
    cfg = report['pipeline_config']
    model = cfg['model']
    p = model['kwargs']['protocol']
    device = 'npu:' + str(model['kwargs']['npu_devices'][0])
    assert model['name'] == p['model_id'] == MODEL and cfg['task_type'] == 'image-to-text'
    assert cfg['sample_count'] == count and cfg['sample_range'] == f'0:{count - 1}'
    assert cfg['output'] == OUTPUT and not model['download'] and not cfg['data']['download']
    assert cfg['data']['dataset_path'] == p['dataset_path'] and not cfg['data']['filter_empty']
    assert cfg['metrics']['include_failed_predictions'] and cfg['metrics']['override'] == ['ROUGE-L']
    assert cfg['algorithm'] == {'name': 'greedy_inference', 'batch_size': 1,
                                'kwargs': {'extract_answer': False, 'truncate_first_sentence': False}}
    assert p['truncate_first_sentence'] and p['source_dtype'] == p['execution_dtype'] == 'BF16'
    assert p['generation_kwargs'] == GENERATION and p['original_requested_max_new_tokens'] == 2048
    assert p['effective_max_patches'] == 2048 and p['serialized_max_patches'] == 4096
    assert Decimal(str(cfg['baseline']['score'])) == Decimal(p['A100_D']) == Decimal('.1801')
    assert cfg['baseline']['primary_metric'] == 'ROUGE-L' and cfg['baseline']['gap_mode'] == 'relative'
    assert Decimal(str(cfg['baseline']['threshold'])) == Decimal('.05')
    lineage = read_json(folder / 'operator_lineage.json')['actual_execution']
    assert lineage['protocol_review'] == p and lineage['hardware'] == 'Ascend 910B'
    assert lineage['source_dtype'] == lineage['execution_dtype'] == 'BF16'
    assert lineage['all_expected_parameters_loaded'] and lineage['weight_tensor_count'] == lineage['untied_parameter_count'] == 423
    assert lineage['source_cast_changed'] == lineage['source_cast_underflow'] == 0
    assert lineage['HF32_enabled'] == {'conv': False, 'matmul': False} and set(lineage['module_placements'].values()) == {device}
    assert lineage['native_cache_disabled'] and lineage['native_logit_observation_returns_original_scores']
    assert lineage['actual_attention'] == p['actual_A100_attention'] == ATTENTION
    assert set(lineage['activation_paths']) == LAYERS
    gen = lineage['effective_generation_config']
    assert gen == p['effective_generation_config'] and gen['use_cache'] is False
    assert gen['num_beams'] == 1 and gen['eos_token_id'] == 1 and gen['decoder_start_token_id'] == gen['pad_token_id'] == 0
    assert all(gen[key] == value for key, value in GENERATION.items())
    external = dict(p['evidence_sha256'])
    for path, digest in {**p['runtime_sha256'], **external}.items():
        assert file_sha(path) == digest, path
    diagnostics = check_disclosure(p, lineage, count)
    table, scalar = scalar_table(p, device)
    weight = Path(model['model_path']).resolve()
    assert str(weight) == p['weight_path'] and '/'.join(weight.parts[-2:]) == MODEL
    for name, digest in p['own_resource_sha256'].items():
        assert Path(name).name == name and file_sha(weight / name) == digest, name
    bias = check_weights(weight, p, lineage, device)
    expected = self_masks(bias)
    tokenizer = Tokenizer.from_file(str(weight / 'tokenizer.json'))
    old = read_json(p['reference_report'])
    originals = old['detailed_results']
    assert len(originals) == 200 and [int(v['index']) for v in originals] == list(range(200))
    assert all(v['raw_output'] == v['prediction'] for v in originals)
    original_score = math.fsum(rouge_l(row['prediction'], row['reference']) for row in originals) / 200
    assert math.isclose(original_score, old['metrics']['ROUGE-L'], rel_tol=1e-14)
    aa = read_json(p['A100_controls_review'])
    assert aa['actual_returncode'] == 0 and aa['model_id'] == MODEL and aa['device'] == 'cuda:0'
    controls = {row['index']: row for row in aa['controls']}
    assert sorted(controls) == CONTROLS and aa['native_control_header']['effective_generation'] == gen
    assert aa['native_control_header']['attention'] == ATTENTION
    data, cpu = read_lines(p['dataset_path']), read_json(p['CPU200_reference'])
    assert len(data) == 1459 and len(cpu) == 200
    trace_file = Path(lineage['per_sample_evidence'])
    evidence_root = trace_file.parent
    assert evidence_root.parent == R / OUTPUT['evidence_dir'] and evidence_root.name.startswith('TextCapsLarge_A100_BF16_1025_')
    inputs, outputs, trace = (read_lines(folder / 'sample_inputs.jsonl'), read_lines(folder / 'sample_outputs.jsonl'), read_lines(trace_file))
    details = report['detailed_results']
    for rows in (inputs, outputs, details):
        assert len(rows) == count and [row['index'] for row in rows] == list(range(count))
    assert len(trace) == count and [row['original_index'] for row in trace] == list(range(count))
    media_seen = set()
    empty, invalid, truncated, differences, changed, scores, before_scores = [], [], [], [], [], [], []
    omitted_indices, omitted_steps, penalty_counts = [], 0, np.zeros(3, dtype=np.int64)
    aa_raw_differences, aa_step_count = [], 0
    for i, (entry, output, detail, item) in enumerate(zip(inputs, outputs, details, trace)):
        original, row, ref = originals[i], data[i], cpu[i]
        assert entry['prompt'] == detail['prompt'] == row['prompt'] == item['prompt'] == original['prompt']
        assert entry['reference'] == output['reference'] == detail['reference'] == row['reference'] == original['reference']
        assert output['extra']['execution'] == detail['extra']['execution'] == item
        assert item['error'] is None and output['error'] is None and item['image_only'] and i == ref['index']
        media = Path(entry['media_path']).resolve()
        assert media == (Path(p['dataset_path']).parent / row['media_path']).resolve() == Path(item['media_path']).resolve()
        assert media not in media_seen and file_sha(media) == ref['image_SHA256'] == item['source_image_sha256']
        media_seen.add(media)
        flat, patches, mask = independent_inputs(media)
        expected_inputs = {'flattened_patches': native_record(patches, 'torch.bfloat16'), 'attention_mask': native_record(mask, 'torch.float32')}
        assert {'flattened_patches': native_record(flat, 'torch.float32'), 'attention_mask': expected_inputs['attention_mask']} == ref['CPU_F32_input']
        assert item['native_inputs'] == ref['model_native_inputs'] == expected_inputs
        archive = Path(item['native_arrays_path'])
        assert archive == evidence_root / 'native_samples' / f'{i:06d}.npz'
        assert file_sha(archive) == item['native_arrays_sha256']
        external[str(archive)] = item['native_arrays_sha256']
        penalty, omitted = check_trace(archive, item, patches, mask, device, expected, i < 3, table)
        penalty_counts += penalty
        if omitted:
            omitted_indices.append(i)
            omitted_steps += omitted
        raw = tokenizer.decode(item['generation_sequence_ids'], skip_special_tokens=True).strip()
        cleaned, scored = caption(raw)
        assert raw == item['raw_before_cleanup'] and cleaned == item['caption_after_byte_cleanup']
        assert scored == output['raw_output'] == detail['raw_output'] == output['prediction'] == detail['prediction'] == item['raw_output']
        if i in controls:
            control = controls[i]
            captured = control['capture']
            assert control['archived_raw'] == original['raw_output'] and control['image_SHA256'] == ref['image_SHA256']
            assert captured['model_native_inputs'] == expected_inputs
            native = captured['native_trace']
            aa_raw = tokenizer.decode(native['generation_sequence_ids'], skip_special_tokens=True).strip()
            assert aa_raw == control['raw_before_cleanup'] and caption(aa_raw)[1] == control['own_first_sentence_output']
            assert external[control['arrays_npz_path']] == captured['arrays_npz_sha256']
            check_trace(control['arrays_npz_path'], native, patches, mask, 'cuda:0', expected, True)
            aa_step_count += native['generation_token_count']
            if (raw, scored) != (aa_raw, control['own_first_sentence_output']):
                aa_raw_differences.append(i)
        if scored != original['raw_output']:
            differences.append(i)
        if raw != scored:
            changed.append(i)
        if not scored:
            empty.append(i)
        elif not any(char.isalnum() for char in scored):
            invalid.append(i)
        if item['truncated']:
            truncated.append(i)
        scores.append(rouge_l(scored, output['reference']))
        before_scores.append(rouge_l(raw, output['reference']))
    assert omitted_indices == (ALL_VALID if count == 200 else [])
    assert report['statistics']['total_samples'] == count and report['statistics']['error_count'] == 0
    assert report['summary']['primary_metric'] == 'ROUGE-L' and set(report['metrics']) == {'ROUGE-L'}
    score = report['metrics']['ROUGE-L']
    independently_scored = math.fsum(scores) / count
    assert math.isclose(independently_scored, score, rel_tol=1e-14)
    baseline = Decimal(p['A100_D'])
    relative = abs(Decimal(str(score)) - baseline) / abs(baseline)
    assert report['summary']['passed'] == (relative <= Decimal('.05'))
    profiler = read_json(lineage['profiler_trace'])
    events = profiler if isinstance(profiler, list) else profiler['traceEvents']
    hardware = {event['pid'] for event in events if event.get('name') == 'process_name'
                and event.get('args', {}).get('name') == 'Ascend Hardware'}
    kernels = sum(event.get('pid') in hardware and event.get('ph') == 'X'
                  and str(event.get('args', {}).get('Task Type', '')).startswith('KERNEL') for event in events)
    assert kernels > 100 and lineage['profiler_scope'] == 'First actual native encoder and first decoder prefill of sample 0'
    snapshot = evidence_root / 'source_snapshot'
    for name, digest in p['project_source_sha256'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert file_sha(R / name) == file_sha(snapshot / name) == digest
    assert read_json(snapshot / 'protocol.json') == p
    configs = list((snapshot / 'configs').rglob('*.yaml'))
    assert len(configs) == 1 and read_json(configs[0])['model'] == cfg['model']
    external.update(diagnostics)
    for path in sorted(evidence_root.rglob('*')):
        if path.is_file() and str(path) not in external:
            external[str(path)] = file_sha(path)
    return {
        'passed': True, 'report_dir': str(folder), 'sample_count': count, 'historical_record_reuse': False,
        'metrics': {'ROUGE-L': score}, 'ROUGE-L': score, 'independent_ROUGE-L': independently_scored,
        'original_A100_independent_ROUGE-L': original_score, 'A100_D': p['A100_D'],
        'relative_error_to_A100_table_D': float(relative),
        'formal_alignment_within_5pct': count == 200 and relative <= Decimal('.05'),
        'report_sha256': {name: file_sha(folder / name) for name in sorted(REPORT_NAMES)},
        'external_evidence_sha256': external, 'TextCapsLarge_numerical_diagnostic_evidence_sha256': diagnostics,
        'source_dtype': 'BF16', 'execution_dtype': 'BF16', 'source_tensor_count': 423, 'untied_parameter_count': 423,
        'actual_activation_module_count': len(LAYERS), 'actual_attention': ATTENTION,
        'native_cache_disabled': True, 'actual_full_vocabulary_size': 50244,
        'full_vocab_greedy_steps': sum(row['generation_token_count'] for row in trace),
        'independent_CPU_image_replays': count, 'actual_A100_native_control_steps': aa_step_count,
        'actual_A100_all_valid_mask_controls': ALL_VALID, 'A100_control_raw_difference_indices': aa_raw_differences,
        'native_fully_valid_image_cross_mask_omitted_indices': omitted_indices,
        'native_fully_valid_image_cross_mask_omitted_steps': omitted_steps,
        'native_F32_repetition_penalty_rounding': {
            'NPU_division_only_values': int(penalty_counts[0]), 'NPU_reciprocal_only_values': int(penalty_counts[1]),
            'NPU_exact_measured_scalar_only_values': int(penalty_counts[2]), 'NPU_original_scalar_probe': scalar,
        },
        'error_indices': [], 'empty_indices': empty, 'invalid_indices': invalid, 'truncated_indices': truncated,
        'raw_difference_indices_vs_selected_original_A100': differences, 'first_sentence_changed_indices': changed,
        'raw_before_cleanup_independent_ROUGE-L': math.fsum(before_scores) / count,
        'sample_ROUGE_L_F1': scores, 'NPU_profiler_kernel_events': kernels, 'profiler_scope': lineage['profiler_scope'],
        'protocol': 'Own BF16, image-only eager, serial one, 2048 tokens, repetition penalty 1.2, native full-prefix and no cache, own cleanup and first sentence.',
    }


if __name__ == '__main__':
    torch.set_num_threads(4)
    print(json.dumps(verify(Path(sys.argv[1]).resolve(), int(sys.argv[2])), ensure_ascii=False, indent=2))
