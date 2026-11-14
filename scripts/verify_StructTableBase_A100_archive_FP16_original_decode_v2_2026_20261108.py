"""Independently verify the original image-only StructTable Base inference protocol."""
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
from safetensors import safe_open
from tokenizers import Tokenizer
import torch
from transformers import AutoConfig, GenerationConfig, Pix2StructForConditionalGeneration
from verify_TextCapsBase_A100_FP16_disclosed_v3_0527_20261107 import (
    ATTENTION, GENERATION, OUTPUT, RELATIVE, REPORT_NAMES, file_sha, independent_patches,
    read_json, read_lines, record, rouge_l, self_masks,
)

R = Path(__file__).resolve().parent.parent
MODEL = 'InternScience/StructTable-base'
VOCAB = 77078
TOKENIZER_IDS = 77078
ALL_VALID = [31, 122, 124, 180]


ORIGINAL_CLEANUP = (
    (' .', '.'),
    (' ?', '?'),
    (' !', '!'),
    (' ,', ','),
    (" ' ", "'"),
    (" n't", "n't"),
    (" 'm", "'m"),
    (" 's", "'s"),
    (" 've", "'ve"),
    (" 're", "'re"),
)


def original_cleanup(caption):
    for before, after in ORIGINAL_CLEANUP:
        caption = caption.replace(before, after)
    return caption.strip()


def check_trace(path, item, patches, mask, device, layers, expected_self, save_layers, penalty_table=None):
    sequence = item['generation_sequence_ids']
    count = item['generation_token_count']
    assert 1 <= count <= 512 and len(sequence) == count + 1 and sequence[0] == 0
    assert all(type(v) is int and 0 <= v < 77078 for v in sequence)
    assert 1 not in sequence[:-1] and (sequence[-1] == 1 or count == 512)
    assert item['truncated'] == (count == 512 and sequence[-1] != 1)
    assert item['budget_hit'] == (count == 512) and item['stop_reason'] == ('eos' if sequence[-1] == 1 else 'length')
    assert item['device'] == device and item['vision_bias_shape'] == [1, 12, 2048, 2048]
    assert item['cross_input_storage_matches_all_12_layers'] == [count] * 12
    assert item['native_logit_processors'] == ['RepetitionPenaltyLogitsProcessor']
    assert item['activation_call_counts'] == {name: count if name.startswith('decoder.') else 1 for name in layers}
    assert len(item['steps']) == count
    cross_row = np.where(mask == 1, 0.0, -65504.0).astype(np.float16)[:, None, None, :]
    cross_row_record = record(cross_row)
    with np.load(path, allow_pickle=False) as archive:
        assert len(archive.files) == len(item['arrays']) == (84 if save_layers else 9)
        assert set(archive.files) == set(item['arrays'])
        for name, description in item['arrays'].items():
            array = archive[name]
            assert record(array) == description and np.isfinite(array).all(), name
        assert archive['encoder'].dtype == archive['cross_input'].dtype == np.float16
        assert archive['encoder'].shape == archive['cross_input'].shape == (1, 2048, 768)
        assert np.array_equal(archive['encoder'], archive['cross_input'])
        assert np.array_equal(archive['encoder_mask'], mask)
        for axis, index in [('row', 0), ('column', 1)]:
            assert archive[axis + '_positions'].dtype == np.int64
            assert np.array_equal(archive[axis + '_positions'], patches[:, :, index].astype(np.int64))
        assert np.array_equal(archive['vision_mask_bias_row'], 1.0 - mask[:, None, None, :])
        assert np.array_equal(archive['cross_bias_first_row'], cross_row)
        if save_layers:
            for name in layers:
                shape = (1, 1 if name.startswith('decoder.') else 4, 2048)
                for prefix in ('act_input.', 'act_output.'):
                    assert archive[prefix + name].shape == shape and archive[prefix + name].dtype == np.float16
        before = archive['raw_head_logits']
        after = archive['native_penalized_logits']
    assert before.dtype == np.float16 and after.dtype == np.float32 and before.shape == after.shape == (count, 77078)
    division_only = reciprocal_only = native_only = 0
    for step, trace in enumerate(item['steps'], 1):
        buckets, causal, positions = expected_self(step)
        assert trace['prefix'] == sequence[:step] and trace['past_key_values_is_none']
        assert trace['head_shape'] == [1, step, 77078] and trace['relative_buckets'] == buckets
        assert trace['causal_mask'] == causal and trace['self_position_bias'] == positions
        assert trace['cross_bias_shape'] == [1, 12, step, 2048] and trace['cross_bias_row'] == cross_row_record
        if trace['cross_mask'] is None:
            assert np.all(mask == 1), 'Native cross mask omitted despite padded patches'
        else:
            assert trace['cross_mask'] == record(np.broadcast_to(cross_row, (1, 1, step, 2048)))
        raw, result = before[step - 1].astype(np.float32), after[step - 1]
        seen = np.unique(sequence[:step])
        unchanged = np.ones(77078, dtype=bool)
        unchanged[seen] = False
        assert np.array_equal(raw[unchanged], result[unchanged])
        original, adjusted = raw[seen], result[seen]
        negative = original < 0
        assert np.array_equal(adjusted[negative], original[negative] * np.float32(1.2))
        positive, actual = original[~negative], adjusted[~negative]
        divides = actual == positive / np.float32(1.2)
        multiplies = actual == positive * np.float32(1.0 / 1.2)
        if penalty_table is None:
            assert device.startswith('cuda:') and (divides | multiplies).all()
        else:
            assert device.startswith('npu:')
            fp16_bits = before[step - 1, seen].view(np.uint16)
            assert np.array_equal(adjusted, penalty_table[fp16_bits]), ('Native F32 scalar mapping differs', device, step)
            native_only += int((~(divides | multiplies)).sum())
        division_only += int((divides & ~multiplies).sum())
        reciprocal_only += int((multiplies & ~divides).sum())
        assert trace['selected_argmax'] == int(result.argmax()) == sequence[step]
    return division_only, reciprocal_only, native_only


def scalar_table(p, device):
    probe = read_json(p['scalar_receipt'])
    assert probe['device'] == device and probe['learned_model_forward_count'] == 0
    assert file_sha(probe['path']) == probe['sha256'] == p['evidence_sha256'][probe['path']]
    half = np.arange(65536, dtype=np.uint16).view(np.float16)
    finite = np.isfinite(half)
    host = half[finite].astype(np.float32)
    with np.load(probe['path'], allow_pickle=False) as loaded:
        table = loaded['signed_native_penalty_table']
        assert table.shape == (65536,) and table.dtype == np.float32
        assert np.array_equal(loaded['finite'], finite) and int(finite.sum()) == probe['finite_FP16_encodings'] == 63488
        product, quotient = loaded['multiply_F32'], loaded['positive_divide_F32']
        assert np.array_equal(product, host * np.float32(1.2))
        difference = np.abs(quotient.view(np.int32).astype(np.int64) - (host / np.float32(1.2)).view(np.int32).astype(np.int64))
        assert int(difference.max()) == probe['max_division_ULP'] <= 1
        assert int(np.count_nonzero(difference)) == probe['nonidentical_division_encodings']
        assert np.array_equal(table[finite].view(np.uint32), np.where(host < 0, product, quotient).view(np.uint32))
    return table, probe


def check_weights(weight, p, lineage, aa, device):
    config = AutoConfig.from_pretrained(weight, local_files_only=True)
    assert not config.tie_word_embeddings and not config.text_config.tie_word_embeddings
    with torch.device('meta'):
        model = Pix2StructForConditionalGeneration(config)
    shapes = {key: list(param.shape) for key, param in model.named_parameters(remove_duplicate=False)}
    assert len(shapes) == 285 and not list(model.buffers()) and model.decoder.config.vocab_size == VOCAB
    model.generation_config = GenerationConfig.from_pretrained(weight, local_files_only=True)
    effective, unused = model._prepare_generation_config(None, **GENERATION)
    assert not unused and effective.to_dict() == p['effective_generation_config']
    manifest = {item['source_key']: item for item in read_json(p['source_cast_manifest'])}
    loaded_rows = read_json(lineage['weight_loading_evidence'])
    loaded = {item['source_key']: item for item in loaded_rows}
    actual_cuda = aa['same_own_A100_actual_CUDA_weights']
    assert len(loaded_rows) == len(loaded) == len(manifest) == len(actual_cuda) == 285
    assert set(shapes) == set(manifest) == set(loaded) == set(actual_cuda)
    checkpoint = weight / 'model.safetensors'
    assert checkpoint.stat().st_size == p['checkpoint']['bytes'] and file_sha(checkpoint) == p['checkpoint']['SHA256']
    bias = None
    with safe_open(checkpoint, framework='pt', device='cpu') as source:
        assert set(source.keys()) == set(shapes)
        for key in source.keys():
            original = source.get_tensor(key)
            rounded = original.half()
            a, b = manifest[key], loaded[key]
            assert all(b[name] == value for name, value in a.items())
            assert list(original.shape) == a['shape'] == shapes[key] and original.dtype == torch.float32
            assert b['target_key'] == key and b['device'] == device
            assert record(original.numpy())['sha256'] == a['source_SHA256']
            assert record(rounded.numpy())['sha256'] == a['archived_FP16_SHA256'] == actual_cuda[key] == b['actual_NPU_SHA256']
            assert int((rounded.float() != original).sum()) == a['source_cast_roundtrip_changed']
            assert int(((original != 0) & (rounded == 0)).sum()) == a['nonzero_underflow']
            if key == RELATIVE:
                bias = rounded.numpy().copy()
    assert bias is not None and bias.shape == (32, 12)
    assert lineage['source_cast_changed'] == sum(row['source_cast_roundtrip_changed'] for row in loaded_rows)
    assert lineage['source_cast_underflow'] == sum(row['nonzero_underflow'] for row in loaded_rows)
    return bias


def disclosed_evidence(p, lineage, count):
    if 'numerical_disclosure' not in p:
        assert count == 3 and 'numerical_disclosure' not in lineage
        return {}
    path = p['numerical_disclosure']
    assert file_sha(path) == p['numerical_disclosure_sha256']
    audit = read_json(path)
    assert lineage['numerical_disclosure'] == audit and audit['model_id'] == MODEL and audit['numerical_review_verified']
    assert audit['same_rounded_FP32_vision_parameter_count'] == 113 and audit['CPU_FP32_learned_generation_steps'] == 0
    assert audit['all_real_prefix_scalar_values_bit_equal'] and len(audit['evidence_sha256']) >= 10
    for artifact, digest in audit['evidence_sha256'].items():
        assert file_sha(artifact) == digest, artifact
    return {**audit['evidence_sha256'], path: p['numerical_disclosure_sha256']}


def verify(folder, count):
    assert count in (3, 200) and {path.name for path in folder.iterdir()} == REPORT_NAMES
    assert folder.parent == R / 'workspace/reports/InternScience_StructTable-base/image-to-text'
    assert re.fullmatch(r'\d{8}_\d{6}', folder.name)
    report = read_json(folder / 'eval_report.json')
    config = report['pipeline_config']
    model = config['model']
    p = model['kwargs']['protocol']
    device = 'npu:' + str(model['kwargs']['npu_devices'][0])
    assert config['task_type'] == 'image-to-text' and model['name'] == p['model_id'] == MODEL
    assert config['sample_count'] == count and config['sample_range'] == f'0:{count - 1}'
    assert config['output'] == OUTPUT and not model['download'] and not config['data']['download']
    assert config['data']['dataset_path'] == p['dataset_path'] and not config['data']['filter_empty']
    assert config['algorithm'] == {'name': 'greedy_inference', 'batch_size': 1,
                                  'kwargs': {'extract_answer': False, 'truncate_first_sentence': False}}
    assert config['metrics']['override'] == ['ROUGE-L'] and config['metrics']['include_failed_predictions']
    assert Decimal(str(config['baseline']['score'])) == Decimal(p['A100_D']) == Decimal('.0115')
    assert Decimal(p['preserved_original_D']) == Decimal('.01149')
    assert config['baseline']['gap_mode'] == 'relative' and Decimal(str(config['baseline']['threshold'])) == Decimal('.05')
    assert p['source_dtype'] == 'F32' and p['execution_dtype'] == 'FP16'
    assert p['original_requested_max_new_tokens'] == 2048 and p['generation_kwargs'] == GENERATION
    lineage = read_json(folder / 'operator_lineage.json')['actual_execution']
    assert lineage['protocol_review'] == p and lineage['hardware'] == 'Ascend 910B'
    assert lineage['source_dtype'] == 'F32' and lineage['execution_dtype'] == 'FP16'
    assert lineage['all_expected_parameters_loaded'] and lineage['untied_parameter_count'] == lineage['weight_tensor_count'] == 285
    assert lineage['HF32_enabled'] == {'conv': False, 'matmul': False} and set(lineage['module_placements'].values()) == {device}
    assert lineage['native_cache_disabled'] and lineage['native_logit_observation_returns_original_scores']
    assert lineage['actual_attention'] == p['actual_A100_attention'] == ATTENTION
    generation = lineage['effective_generation_config']
    assert generation == p['effective_generation_config'] and generation['use_cache'] is False and generation['num_beams'] == 1
    assert generation['eos_token_id'] == 1 and generation['decoder_start_token_id'] == generation['pad_token_id'] == 0
    assert all(generation[key] == value for key, value in GENERATION.items())
    diagnostics = disclosed_evidence(p, lineage, count)
    external = dict(p['evidence_sha256'])
    for path, digest in {**p['runtime_sha256'], **external}.items():
        assert file_sha(path) == digest, path
    table, probe = scalar_table(p, device)
    weight = Path(model['model_path']).resolve()
    assert str(weight) == p['weight_path'] and '/'.join(weight.parts[-2:]) == MODEL
    for name, digest in p['own_resource_sha256'].items():
        assert Path(name).name == name and file_sha(weight / name) == digest, name
    assert read_json(weight / 'tokenizer_config.json')['clean_up_tokenization_spaces'] is True
    tokenizer = Tokenizer.from_file(str(weight / 'tokenizer.json'))
    vocab = tokenizer.get_vocab()
    assert len(vocab) == TOKENIZER_IDS and set(vocab.values()) == set(range(TOKENIZER_IDS))
    assert hashlib.sha256(json.dumps(vocab, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == p['own_vocab_SHA256']
    aa = read_json(p['A100_controls_review'])
    assert aa['all_actual_A100_arrays_and_full_vocab_recomputed'] and aa['first3_steps'] == 1334 and aa['model_id'] == MODEL
    assert aa['installed_effective_generation'] == generation and aa['actual_A100_attention'] == ATTENTION
    controls = {row['index']: row for row in aa['controls']}
    assert sorted(controls) == [0, 1, 2, 124]
    bias = check_weights(weight, p, lineage, aa, device)
    masks = self_masks(bias)
    layers = set(lineage['activation_paths'])
    assert layers == {*(f'encoder.encoder.layer.{i}.mlp.act' for i in range(12)),
                      *(f'decoder.layer.{i}.mlp.DenseReluDense.act' for i in range(12))}
    data, cpu = read_lines(p['dataset_path']), read_json(p['CPU200_reference'])
    originals = read_json(p['reference_report'])['detailed_results']
    assert len(data) == 1459 and len(cpu) == len(originals) == 200
    assert [int(row['index']) for row in originals] == list(range(200))
    original_score = math.fsum(rouge_l(row['prediction'], row['reference']) for row in originals) / 200
    assert math.isclose(original_score, .011497206091839433, rel_tol=1e-14)
    trace_path = Path(lineage['per_sample_evidence'])
    root = trace_path.parent
    assert root.parent == R / OUTPUT['evidence_dir'] and root.name.startswith('StructTableBase_A100_FP16_2026_')
    trace = read_lines(trace_path)
    inputs, outputs = (read_lines(folder / name) for name in ('sample_inputs.jsonl', 'sample_outputs.jsonl'))
    details = report['detailed_results']
    assert [row['original_index'] for row in trace] == list(range(count))
    for items in (inputs, outputs, details):
        assert [item['index'] for item in items] == list(range(count))

    media_seen = set()
    scores, empty, invalid, truncated, differences, omitted_indices = [], [], [], [], [], []
    totals = np.zeros(3, dtype=np.int64)
    aa_steps = omitted_steps = 0
    cleaned_caption_indices = []
    for i, (inp, out, detail, observed) in enumerate(zip(inputs, outputs, details, trace)):
        row, old, ref = data[i], originals[i], cpu[i]
        assert i == ref['index'] and observed['error'] is None and out['error'] is None
        assert inp['prompt'] == detail['prompt'] == observed['prompt'] == row['prompt'] == old['prompt']
        assert inp['reference'] == out['reference'] == detail['reference'] == row['reference'] == old['reference']
        assert out['extra']['execution'] == detail['extra']['execution'] == observed
        media = (Path(p['dataset_path']).parent / row['media_path']).resolve()
        assert media == Path(inp['media_path']).resolve() == Path(observed['media_path']).resolve()
        assert media not in media_seen and file_sha(media) == ref['media_SHA256'] == observed['source_image_sha256']
        media_seen.add(media)
        with Image.open(media) as image:
            flat, mask = independent_patches(image.convert('RGB'))
        patches, mask = flat.half().numpy(), mask.numpy()
        assert record(flat.numpy())['sha256'] == ref['flat_F32_SHA256']
        expected_input = {'flattened_patches': record(patches), 'attention_mask': record(mask)}
        assert observed['native_inputs'] == expected_input and observed['image_only']
        assert expected_input['flattened_patches']['sha256'] == ref['original_FP16_SHA256']
        assert expected_input['attention_mask']['sha256'] == ref['mask_F32_SHA256']
        artifact = root / 'native_samples' / f'{i:06d}.npz'
        assert observed['native_arrays_path'] == str(artifact) and file_sha(artifact) == observed['native_arrays_sha256']
        external[str(artifact)] = observed['native_arrays_sha256']
        totals += check_trace(artifact, observed, patches, mask, device, layers, masks, i < 3, table)
        omitted = sum(step['cross_mask'] is None for step in observed['steps'])
        if omitted:
            omitted_indices.append(i)
            omitted_steps += omitted
            assert omitted == observed['generation_token_count'] and i in ALL_VALID
        rust_raw = tokenizer.decode(observed['generation_sequence_ids'], skip_special_tokens=True)
        raw = original_cleanup(rust_raw)
        if raw != rust_raw.strip():
            cleaned_caption_indices.append(i)
        assert raw == observed['raw_output'] == out['raw_output'] == detail['raw_output'] == out['prediction'] == detail['prediction']
        if i in controls:
            control = controls[i]
            native = control['capture']
            path = control['arrays_npz_path']
            assert external[path] == control['arrays_npz_SHA256'] and native['inputs'] == expected_input
            assert control['old_raw'] == old['raw_output']
            assert original_cleanup(tokenizer.decode(native['generation_sequence_ids'], skip_special_tokens=True)) == control['actual_raw']
            check_trace(path, native, patches, mask, aa['device'], layers, masks, True)
            if i < 3:
                assert control['same_archived_raw']
            else:
                assert np.all(mask == 1) and all(step['cross_mask'] is None for step in native['steps'])
            aa_steps += native['generation_token_count']
        if raw != old['raw_output']:
            differences.append(i)
        if not raw:
            empty.append(i)
        elif not any(char.isalnum() for char in raw):
            invalid.append(i)
        if observed['truncated']:
            truncated.append(i)
        scores.append(rouge_l(raw, out['reference']))
        if (i + 1) % 25 == 0:
            print(f'Independent verified original rows {i + 1}/{count}', file=sys.stderr, flush=True)
    assert omitted_indices == (ALL_VALID if count == 200 else []) and aa_steps == sum(controls[i]['steps'] for i in controls if i < count)
    assert report['statistics']['total_samples'] == count and report['statistics']['error_count'] == 0
    assert report['summary']['primary_metric'] == 'ROUGE-L' and set(report['metrics']) == {'ROUGE-L'}
    score = report['metrics']['ROUGE-L']
    independent = math.fsum(scores) / count
    assert math.isclose(independent, score, rel_tol=1e-14, abs_tol=1e-17)
    current, preserved = Decimal(p['A100_D']), Decimal(p['preserved_original_D'])
    relative = abs(Decimal(str(score)) - current) / current
    original_relative = abs(Decimal(str(score)) - preserved) / preserved
    assert report['summary']['passed'] == (relative <= Decimal('.05'))
    chrome = read_json(lineage['profiler_trace'])
    events = chrome if isinstance(chrome, list) else chrome['traceEvents']
    hardware = {event['pid'] for event in events if event.get('name') == 'process_name'
                and event.get('args', {}).get('name') == 'Ascend Hardware'}
    kernels = sum(event.get('pid') in hardware and event.get('ph') == 'X'
                  and str(event.get('args', {}).get('Task Type', '')).startswith('KERNEL') for event in events)
    assert kernels > 100 and lineage['profiler_scope'] == 'First actual native encoder and first decoder prefill of sample 0'
    snapshot = root / 'source_snapshot'
    for name, digest in p['project_source_sha256'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert file_sha(R / name) == file_sha(snapshot / name) == digest
    assert read_json(snapshot / 'protocol.json') == p
    yaml, = (snapshot / 'configs').rglob('*.yaml')
    assert read_json(yaml)['model'] == config['model']
    external.update(p['runtime_sha256'])
    external.update(diagnostics)
    for path in root.rglob('*'):
        if path.is_file() and str(path) not in external:
            external[str(path)] = file_sha(path)
    return {
        'passed': True, 'report_dir': str(folder), 'sample_count': count, 'historical_record_reuse': False,
        'metrics': {'ROUGE-L': score}, 'ROUGE-L': score, 'independent_ROUGE-L': independent,
        'original_A100_independent_ROUGE-L': original_score, 'A100_D': p['A100_D'],
        'relative_error_to_A100_table_D': float(relative), 'relative_error_to_preserved_D': float(original_relative),
        'formal_alignment_within_5pct': count == 200 and max(relative, original_relative) <= Decimal('.05'),
        'report_sha256': {name: file_sha(folder / name) for name in sorted(REPORT_NAMES)},
        'external_evidence_sha256': external, 'StructTableBase_numerical_diagnostic_evidence_sha256': diagnostics,
        'source_dtype': 'F32', 'execution_dtype': 'FP16', 'source_tensor_count': 285, 'untied_parameter_count': 285,
        'source_cast_changed': lineage['source_cast_changed'], 'source_cast_underflow': lineage['source_cast_underflow'],
        'actual_attention': ATTENTION, 'actual_activation_module_count': len(layers), 'native_cache_disabled': True,
        'actual_full_vocabulary_size': VOCAB, 'own_tokenizer_ID_count': TOKENIZER_IDS,
        'full_vocab_greedy_steps': sum(row['generation_token_count'] for row in trace),
        'independent_CPU_image_replays': count, 'actual_A100_native_control_steps': aa_steps,
        'native_fully_valid_image_cross_mask_omitted_indices': omitted_indices,
        'native_fully_valid_image_cross_mask_omitted_steps': omitted_steps,
        'actual_A100_all_valid_mask_controls': [124],
        'native_F32_repetition_penalty_rounding': {'NPU_division_only_values': int(totals[0]),
            'NPU_reciprocal_only_values': int(totals[1]), 'NPU_exact_measured_scalar_only_values': int(totals[2]),
            'NPU_original_scalar_probe': probe},
        'original_tokenizer_cleanup_enabled': True, 'native_cleanup_changed_literal_decode_indices': cleaned_caption_indices,
        'error_indices': [], 'empty_indices': empty, 'invalid_indices': invalid, 'truncated_indices': truncated,
        'budget_hit_indices': [i for i, row in enumerate(trace) if row['budget_hit']],
        'raw_difference_indices_vs_selected_original_A100': differences, 'sample_ROUGE_L_F1': scores,
        'NPU_profiler_kernel_events': kernels, 'profiler_scope': lineage['profiler_scope'],
        'full_denominator_preserves_raw_markup_empty_invalid_and_truncation': True,
        'protocol': 'Own archived FP16 StructTable Base, original image-only eager no-cache full-prefix, rep1.2 and internal 512, native complete raw strip and ROUGE-L.',
    }


if __name__ == '__main__':
    torch.set_num_threads(4)
    print(json.dumps(verify(Path(sys.argv[1]).resolve(), int(sys.argv[2])), ensure_ascii=False, indent=2))
