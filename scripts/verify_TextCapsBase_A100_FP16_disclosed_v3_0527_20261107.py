"""Verify native TextCaps Base reports using independent CPU inputs and scoring."""
from decimal import Decimal
from functools import lru_cache
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
from transformers import AutoConfig, Pix2StructForConditionalGeneration

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from universal_eval.metric.rouge_910b_porter import PorterStemmer

MODEL = 'google/pix2struct-textcaps-base'
RELATIVE = 'decoder.layer.0.self_attention.attention.relative_attention_bias.weight'
REPORT_NAMES = {'eval_report.json', 'sample_inputs.jsonl', 'sample_outputs.jsonl', 'run.log', 'operator_lineage.json'}
GENERATION = {'max_new_tokens': 512, 'do_sample': False, 'repetition_penalty': 1.2}
OUTPUT = {'reports_dir': 'workspace/reports', 'layout': 'model_task_timestamp',
          'evidence_dir': 'workspace/diagnostics/model_runs', 'timezone': 'Asia/Hong_Kong', 'verbose': True}
ATTENTION = {'root': 'eager', 'vision': 'eager', 'text': 'eager'}
STEM = PorterStemmer()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def record(array):
    return {'shape': list(array.shape), 'dtype': array.dtype.name,
            'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def independent_patches(image):
    x = torch.from_numpy(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy())[None].float()
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True, correction=1)
    _, channels, height, width = x.shape
    denominator = torch.maximum(std, torch.tensor(1.0 / (channels * height * width) ** .5))
    x = (x - mean) / denominator
    scale = (2048 * (16 / height) * (16 / width)) ** .5
    rows = max(1, min(int(scale * height / 16), 2048))
    columns = max(1, min(int(scale * width / 16), 2048))
    x = torch.nn.functional.interpolate(x, size=(rows * 16, columns * 16), mode='bilinear',
                                         align_corners=False, antialias=True)[0]
    patches = x.reshape(3, rows, 16, columns, 16).permute(1, 3, 2, 4, 0).reshape(rows * columns, 768)
    coordinates = torch.tensor([(row + 1, col + 1) for row in range(rows) for col in range(columns)],
                               dtype=torch.float32)
    occupied = torch.cat((coordinates, patches), dim=-1)
    full = torch.cat((occupied, torch.zeros(2048 - rows * columns, 770)), dim=0)[None]
    return full, (full.sum(dim=-1) != 0).float()


def words(value):
    return [STEM.stem(word) if len(word) > 3 else word for word in re.findall('[a-z0-9]+', value.lower())]


def rouge_l(raw, reference):
    prediction = words(raw)
    reference = words(' '.join(reference) if isinstance(reference, list) else reference)
    if not prediction or not reference:
        return 0.0
    lcs = [0] * (len(reference) + 1)
    for token in prediction:
        diagonal = 0
        for position, other in enumerate(reference, 1):
            saved = lcs[position]
            lcs[position] = diagonal + 1 if token == other else max(lcs[position], lcs[position - 1])
            diagonal = saved
    precision, recall = lcs[-1] / len(prediction), lcs[-1] / len(reference)
    return 2.0 * precision * recall / (precision + recall) if lcs[-1] else 0.0


def check_weights(weight_path, protocol, lineage, aa, device):
    config = AutoConfig.from_pretrained(weight_path, local_files_only=True)
    assert not config.tie_word_embeddings and not config.text_config.tie_word_embeddings
    with torch.device('meta'):
        model = Pix2StructForConditionalGeneration(config)
    shapes = {name: list(weight.shape) for name, weight in model.named_parameters(remove_duplicate=False)}
    assert len(shapes) == 285 and not list(model.buffers())
    resolved, unused = model._prepare_generation_config(None, **GENERATION)
    assert not unused and resolved.to_dict() == lineage['effective_generation_config']
    evidence = read_json(lineage['weight_loading_evidence'])
    loaded = {row['source_key']: row for row in evidence}
    source = {row['source_key']: row for row in read_json(protocol['source_cast_manifest'])}
    cuda = aa['same_own_A100_actual_CUDA_weights']
    assert len(evidence) == len(loaded) == len(source) == 285 and loaded.keys() == source.keys() == cuda.keys() == shapes.keys()
    checkpoint = weight_path / 'model.safetensors'
    assert checkpoint.stat().st_size == protocol['checkpoint']['bytes']
    assert file_sha(checkpoint) == protocol['checkpoint']['SHA256']
    bias = None
    with safe_open(checkpoint, framework='pt', device='cpu') as weights:
        assert set(weights.keys()) == set(shapes)
        for name in weights.keys():
            value = weights.get_tensor(name)
            cast = value.half()
            row, actual = source[name], loaded[name]
            assert value.dtype == torch.float32 and list(value.shape) == row['shape'] == shapes[name]
            assert all(actual[key] == item for key, item in row.items())
            assert row['source_dtype'] == 'torch.float32' and row['archived_execution_dtype'] == 'torch.float16'
            assert actual['target_key'] == name and actual['device'] == device
            assert record(value.numpy())['sha256'] == row['source_SHA256']
            assert record(cast.numpy())['sha256'] == row['archived_FP16_SHA256'] == cuda[name] == actual['actual_NPU_SHA256']
            assert int((cast.float() != value).sum()) == row['source_cast_roundtrip_changed']
            assert int(((value != 0) & (cast == 0)).sum()) == row['nonzero_underflow']
            if name == RELATIVE:
                bias = cast.numpy().copy()
    assert bias is not None and bias.shape == (32, 12)
    assert sum(v['source_cast_roundtrip_changed'] for v in evidence) == lineage['source_cast_changed'] == 282251282
    assert sum(v['nonzero_underflow'] for v in evidence) == lineage['source_cast_underflow'] == 141
    return bias


def self_masks(bias):
    @lru_cache(maxsize=512)
    def expected(length):
        rows = np.arange(length, dtype=np.int64)
        difference = rows[:, None] - rows[None, :]
        distance = np.maximum(difference, 0)
        large = 16 + np.floor(np.log(np.maximum(distance, 16) / 16) / math.log(8) * 16).astype(np.int64)
        buckets = np.where(distance < 16, distance, np.minimum(31, large))
        mask = np.where(difference < 0, -65504.0, 0.0).astype(np.float16)[None, None]
        positions = bias[buckets].transpose(2, 0, 1)[None]
        return record(buckets), record(mask), record(positions + mask)
    return expected


def check_trace(path, item, patches, mask, device, layers, expected_self, save_layers, penalty_table=None):
    sequence = item['generation_sequence_ids']
    count = item['generation_token_count']
    assert 1 <= count <= 512 and len(sequence) == count + 1 and sequence[0] == 0
    assert all(type(v) is int and 0 <= v < 50244 for v in sequence)
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
    assert before.dtype == np.float16 and after.dtype == np.float32 and before.shape == after.shape == (count, 50244)
    division_only = reciprocal_only = native_only = 0
    for step, trace in enumerate(item['steps'], 1):
        buckets, causal, positions = expected_self(step)
        assert trace['prefix'] == sequence[:step] and trace['past_key_values_is_none']
        assert trace['head_shape'] == [1, step, 50244] and trace['relative_buckets'] == buckets
        assert trace['causal_mask'] == causal and trace['self_position_bias'] == positions
        assert trace['cross_bias_shape'] == [1, 12, step, 2048] and trace['cross_bias_row'] == cross_row_record
        if trace['cross_mask'] is None:
            assert np.all(mask == 1), 'Native cross mask omitted despite padded patches'
        else:
            assert trace['cross_mask'] == record(np.broadcast_to(cross_row, (1, 1, step, 2048)))
        raw, result = before[step - 1].astype(np.float32), after[step - 1]
        seen = np.unique(sequence[:step])
        unchanged = np.ones(50244, dtype=bool)
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


def numerical_controls(p, lineage, device):
    disclosure_path = p['numerical_disclosure']
    assert file_sha(disclosure_path) == p['numerical_disclosure_sha256']
    disclosure = read_json(disclosure_path)
    assert lineage['numerical_disclosure'] == disclosure and disclosure['diagnostic_evidence_verified']
    assert disclosure['model_id'] == MODEL and disclosure['CPU_FP32_is_diagnostic_only']
    assert disclosure['no_language_model_CPU_or_FP32_answers'] and disclosure['device'] == device
    assert disclosure['original_precision_attention_cache_budget_and_model_math_preserved']
    assert disclosure['same_F32_to_FP16_rounded_vision_parameters_used_in_FP32_control'] == 113
    assert disclosure['source_cast_changed'] == lineage['source_cast_changed']
    assert disclosure['source_nonzero_underflow'] == lineage['source_cast_underflow']
    diagnostics = dict(disclosure['all_static_diagnostic_evidence_sha256'])
    diagnostics[disclosure_path] = p['numerical_disclosure_sha256']
    for path, digest in diagnostics.items():
        assert file_sha(path) == digest, path
    probe = disclosure['native_repetition_penalty_scalar_control']
    assert diagnostics[probe['path']] == probe['sha256'] and probe['device'] == device
    assert probe['no_tolerance_on_native_table_values'] and probe['multiply_matches_F32']
    all_half = np.arange(65536, dtype=np.uint16).view(np.float16)
    finite = np.isfinite(all_half)
    fp32 = all_half[finite].astype(np.float32)
    with np.load(probe['path'], allow_pickle=False) as measured:
        assert np.array_equal(measured['finite'], finite) and int(finite.sum()) == probe['finite_FP16_values'] == 63488
        assert np.array_equal(measured['multiply_F32'], fp32 * np.float32(1.2))
        actual = measured['positive_divide_F32']
        cpu = fp32 / np.float32(1.2)
        delta = np.abs(actual.view(np.int32).astype(np.int64) - cpu.view(np.int32).astype(np.int64))
        assert int(delta.max()) == probe['max_abs_ULP_vs_CPU_FP32_division'] == 1
        assert int(np.count_nonzero(delta)) == probe['divided_values_not_CPU_FP32_correctly_rounded']
        table = measured['signed_native_penalty_table']
        assert table.shape == (65536,) and table.dtype == np.float32
        assert np.array_equal(table[finite], np.where(fp32 < 0, fp32 * np.float32(1.2), actual))
        assert float(table[np.float16(0.8759765625).view(np.uint16)]) == probe['original_observed_native_result']
    for index, sample in enumerate(disclosure['rows']):
        assert sample['index'] == index and sample['FP32_diagnostic_arrays'] in diagnostics
        with np.load(sample['FP32_diagnostic_arrays'], allow_pickle=False) as diagnostic:
            cpu, npu = diagnostic['CPU.encoder'], diagnostic['NPU.encoder']
            assert cpu.dtype == npu.dtype == np.float32 and cpu.shape == npu.shape == (1, 2048, 768)
            delta = np.abs(cpu.astype(np.float64) - npu.astype(np.float64))
            assert np.isfinite(delta).all()
            assert float(delta.max()) == sample['independent_same_rounded_FP32_CPU_vs_NPU4']['encoder']['max_abs']
    assert [v['index'] for v in disclosure['rows']] == [0, 1, 2]
    return diagnostics, table, probe


def verify(folder, count):
    assert count in (3, 200) and {path.name for path in folder.iterdir()} == REPORT_NAMES
    assert folder.parent.parent == R / 'workspace/reports/google_pix2struct-textcaps-base'
    assert folder.parent.name == 'image-to-text' and re.fullmatch(r'\d{8}_\d{6}', folder.name)
    report = read_json(folder / 'eval_report.json')
    config = report['pipeline_config']
    model = config['model']
    p = model['kwargs']['protocol']
    device = 'npu:' + str(model['kwargs']['npu_devices'][0])
    assert model['name'] == p['model_id'] == MODEL and config['task_type'] == 'image-to-text'
    assert config['sample_count'] == count and config['sample_range'] == f'0:{count - 1}'
    assert config['output'] == OUTPUT and not model['download'] and not config['data']['download']
    assert not config['data']['filter_empty'] and config['metrics']['include_failed_predictions'] is True
    assert config['metrics']['override'] == ['ROUGE-L']
    assert config['algorithm'] == {'name': 'greedy_inference', 'batch_size': 1,
                                  'kwargs': {'extract_answer': False, 'truncate_first_sentence': False}}
    assert p['source_dtype'] == 'F32' and p['execution_dtype'] == 'FP16'
    assert p['original_requested_max_new_tokens'] == 2048 and p['generation_kwargs'] == GENERATION
    assert config['baseline']['primary_metric'] == 'ROUGE-L' and config['baseline']['gap_mode'] == 'relative'
    assert Decimal(str(config['baseline']['score'])) == Decimal(p['A100_D']) == Decimal('.1681')
    assert Decimal(str(config['baseline']['threshold'])) == Decimal('.05')
    lineage = read_json(folder / 'operator_lineage.json')['actual_execution']
    assert lineage['protocol_review'] == p and lineage['hardware'] == 'Ascend 910B'
    assert lineage['execution_dtype'] == 'FP16' and lineage['source_dtype'] == 'F32'
    assert lineage['all_expected_parameters_loaded'] and lineage['untied_parameter_count'] == lineage['weight_tensor_count'] == 285
    assert lineage['HF32_enabled'] == {'conv': False, 'matmul': False} and set(lineage['module_placements'].values()) == {device}
    assert lineage['native_cache_disabled'] and lineage['native_logit_observation_returns_original_scores']
    assert lineage['actual_attention'] == p['actual_A100_attention'] == ATTENTION
    assert lineage['effective_generation_config'] == p['effective_generation_config']
    diagnostics, penalty_table, scalar_probe = numerical_controls(p, lineage, device)
    generation = lineage['effective_generation_config']
    assert generation['use_cache'] is False and generation['num_beams'] == 1 and generation['eos_token_id'] == 1
    assert generation['decoder_start_token_id'] == generation['pad_token_id'] == 0
    assert all(generation[key] == value for key, value in GENERATION.items())
    external = dict(p['evidence_sha256'])
    for path, digest in {**p['runtime_sha256'], **external}.items():
        assert file_sha(path) == digest, path
    weight_path = Path(model['model_path']).resolve()
    assert str(weight_path) == p['weight_path'] and '/'.join(weight_path.parts[-2:]) == MODEL
    for name, digest in p['own_resource_sha256'].items():
        assert Path(name).name == name and file_sha(weight_path / name) == digest, name
    aa = read_json(p['A100_controls_review'])
    assert aa['A100_control_actual_exit'] == 0 and aa['A100_native3_arrays_sha_verified']
    assert aa['model_id'] == MODEL and aa['native_control_header']['attention'] == ATTENTION
    assert aa['native_control_header']['generation'] == generation
    boundary = read_json(p['A100_all_valid_boundary_review'])
    assert boundary['actual_returncode'] == 0 and boundary['native_all_valid_cross_mask_omission_verified']
    assert boundary['all_boundary_arrays_sha256_verified']
    boundary_by_index = {item['index']: item for item in boundary['native_controls']}
    assert sorted(boundary_by_index) == [31, 122, 124, 180]
    bias = check_weights(weight_path, p, lineage, aa, device)
    expected_self = self_masks(bias)
    expected_activations = {
        *(f'encoder.encoder.layer.{i}.mlp.act' for i in range(12)),
        *(f'decoder.layer.{i}.mlp.DenseReluDense.act' for i in range(12)),
    }
    layers = set(lineage['activation_paths'])
    assert layers == expected_activations
    tokenizer = Tokenizer.from_file(str(weight_path / 'tokenizer.json'))
    data = read_lines(p['dataset_path'])
    prior_cpu = read_json(p['CPU200_reference'])
    assert len(data) == 1459 and len(prior_cpu) == 200
    original = read_json(p['reference_report'])
    original_rows = original['detailed_results']
    assert len(original_rows) == 200 and all(v['raw_output'] == v['prediction'] for v in original_rows)
    assert [int(v['index']) for v in original_rows] == list(range(200))
    original_independent = math.fsum(rouge_l(v['prediction'], v['reference']) for v in original_rows) / 200
    assert math.isclose(original_independent, original['metrics']['ROUGE-L'], rel_tol=1e-14)
    trace_path = Path(lineage['per_sample_evidence'])
    evidence_root = trace_path.parent
    assert evidence_root.parent == R / OUTPUT['evidence_dir'] and evidence_root.name.startswith('TextCapsBase_A100_FP16_0527_')
    inputs = read_lines(folder / 'sample_inputs.jsonl')
    outputs = read_lines(folder / 'sample_outputs.jsonl')
    trace = read_lines(trace_path)
    details = report['detailed_results']
    assert len(trace) == count and [v['original_index'] for v in trace] == list(range(count))
    for rows in (inputs, outputs, details):
        assert len(rows) == count and [v['index'] for v in rows] == list(range(count))
    raw_media = set()
    errors, empty, invalid, truncated, raw_differences, per_sample = [], [], [], [], [], []
    division = reciprocal = native_rounding = aa_division = aa_reciprocal = 0
    omitted_mask_samples = []
    omitted_mask_steps = 0
    for i, (entry, output, detail, item) in enumerate(zip(inputs, outputs, details, trace)):
        row, ref, old = data[i], prior_cpu[i], original_rows[i]
        assert entry['prompt'] == detail['prompt'] == row['prompt'] == item['prompt'] == old['prompt']
        assert entry['reference'] == output['reference'] == detail['reference'] == row['reference'] == old['reference']
        assert output['extra']['execution'] == item and detail['extra']['execution'] == item
        if item['error'] is not None or output['error'] is not None:
            errors.append(i)
        assert i == ref['index'] and item['image_only']
        media = Path(entry['media_path']).resolve()
        assert media == (Path(p['dataset_path']).parent / row['media_path']).resolve() == Path(item['media_path']).resolve()
        assert media not in raw_media and file_sha(media) == ref['media_SHA256'] == item['source_image_sha256']
        raw_media.add(media)
        with Image.open(media) as image:
            flat, mask = independent_patches(image.convert('RGB'))
        assert record(flat.numpy())['sha256'] == ref['flat_F32_SHA256']
        patches, mask = flat.half().numpy(), mask.numpy()
        assert item['native_inputs']['flattened_patches'] == record(patches)
        assert item['native_inputs']['attention_mask'] == record(mask)
        assert ref['original_FP16_SHA256'] == record(patches)['sha256'] and ref['mask_F32_SHA256'] == record(mask)['sha256']
        archive = Path(item['native_arrays_path'])
        assert archive == evidence_root / 'native_samples' / f'{i:06d}.npz'
        assert file_sha(archive) == item['native_arrays_sha256']
        external[str(archive)] = item['native_arrays_sha256']
        div, mul, native_only = check_trace(archive, item, patches, mask, device, layers, expected_self, i < 3, penalty_table)
        division += div
        reciprocal += mul
        native_rounding += native_only
        omitted = sum(step['cross_mask'] is None for step in item['steps'])
        if omitted:
            omitted_mask_samples.append(i)
            omitted_mask_steps += omitted
        raw = tokenizer.decode(item['generation_sequence_ids'], skip_special_tokens=True).strip()
        assert raw == output['prediction'] == detail['prediction'] == output['raw_output'] == detail['raw_output'] == item['raw_output']
        if i < 3:
            control = aa['native_control_rows'][i]
            aa_npz = Path(control['arrays_npz_path'])
            assert control['index'] == i and control['raw'] == control['old_raw'] == old['raw_output']
            assert control['capture']['inputs'] == item['native_inputs']
            assert tokenizer.decode(control['capture']['generation_sequence_ids'], skip_special_tokens=True).strip() == old['raw_output']
            assert external[str(aa_npz)] == control['arrays_npz_sha256']
            div, mul, _ = check_trace(aa_npz, control['capture'], patches, mask, aa['native_control_header']['device'],
                                      layers, expected_self, True)
            aa_division += div
            aa_reciprocal += mul
        if i in boundary_by_index:
            control = boundary_by_index[i]
            assert control['raw'] == old['raw_output'] and control['capture']['inputs'] == item['native_inputs']
            assert external[control['arrays_npz_path']] == control['arrays_npz_sha256'] and np.all(mask == 1)
            check_trace(control['arrays_npz_path'], control['capture'], patches, mask, boundary['device'],
                        layers, expected_self, True)
        if raw != old['raw_output']:
            raw_differences.append(i)
        if not raw:
            empty.append(i)
        elif not any(char.isalnum() for char in raw):
            invalid.append(i)
        if item['truncated']:
            truncated.append(i)
        per_sample.append(rouge_l(raw, output['reference']))
    assert not errors and report['statistics']['total_samples'] == count and report['statistics']['error_count'] == 0
    assert omitted_mask_samples == ([31, 122, 124, 180] if count == 200 else [])
    assert report['summary']['primary_metric'] == 'ROUGE-L' and set(report['metrics']) == {'ROUGE-L'}
    score = report['metrics']['ROUGE-L']
    independently_scored = math.fsum(per_sample) / count
    assert math.isclose(independently_scored, score, rel_tol=1e-14)
    baseline = Decimal(p['A100_D'])
    relative = abs(Decimal(str(score)) - baseline) / abs(baseline)
    assert report['summary']['passed'] == (relative <= Decimal('.05'))
    profiler = read_json(lineage['profiler_trace'])
    events = profiler if isinstance(profiler, list) else profiler['traceEvents']
    hardware = {event['pid'] for event in events if event.get('name') == 'process_name'
                and event.get('args', {}).get('name') == 'Ascend Hardware'}
    kernels = [event for event in events if event.get('pid') in hardware and event.get('ph') == 'X'
               and str(event.get('args', {}).get('Task Type', '')).startswith('KERNEL')]
    assert len(kernels) > 100 and lineage['profiler_scope'] == 'First actual native encoder and first decoder prefill of sample 0'
    snapshot = evidence_root / 'source_snapshot'
    for name, digest in p['project_source_sha256'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert file_sha(R / name) == file_sha(snapshot / name) == digest
    assert read_json(snapshot / 'protocol.json') == p
    config_snapshots = list((snapshot / 'configs').rglob('*.yaml'))
    assert len(config_snapshots) == 1 and read_json(config_snapshots[0])['model'] == config['model']
    external.update(diagnostics)
    for path in sorted(evidence_root.rglob('*')):
        if path.is_file() and str(path) not in external:
            external[str(path)] = file_sha(path)
    return {
        'passed': True, 'report_dir': str(folder), 'sample_count': count, 'historical_record_reuse': False,
        'metrics': {'ROUGE-L': score}, 'ROUGE-L': score, 'independent_ROUGE-L': independently_scored,
        'original_A100_independent_ROUGE-L': original_independent, 'A100_D': p['A100_D'],
        'relative_error_to_A100_table_D': float(relative),
        'formal_alignment_within_5pct': count == 200 and relative <= Decimal('.05'),
        'report_sha256': {name: file_sha(folder / name) for name in sorted(REPORT_NAMES)},
        'external_evidence_sha256': external, 'TextCapsBase_numerical_diagnostic_evidence_sha256': diagnostics,
        'source_dtype': 'F32', 'execution_dtype': 'FP16', 'source_tensor_count': 285, 'untied_parameter_count': 285,
        'source_cast_changed': lineage['source_cast_changed'], 'source_cast_underflow': lineage['source_cast_underflow'],
        'actual_activation_module_count': len(layers), 'actual_attention': ATTENTION,
        'native_cache_disabled': True, 'actual_full_vocabulary_size': 50244,
        'full_vocab_greedy_steps': sum(item['generation_token_count'] for item in trace),
        'independent_CPU_image_replays': count, 'actual_A100_native_control_steps': 50,
        'native_fully_valid_image_cross_mask_omitted_indices': omitted_mask_samples,
        'native_fully_valid_image_cross_mask_omitted_steps': omitted_mask_steps,
        'actual_A100_all_valid_mask_controls': sorted(boundary_by_index),
        'native_F32_repetition_penalty_rounding': {
            'NPU_division_only_values': division, 'NPU_reciprocal_only_values': reciprocal,
            'NPU_exact_measured_scalar_only_values': native_rounding,
            'A100_division_only_values': aa_division, 'A100_reciprocal_only_values': aa_reciprocal,
            'NPU_original_scalar_probe': scalar_probe,
        },
        'error_indices': errors, 'empty_indices': empty, 'invalid_indices': invalid, 'truncated_indices': truncated,
        'raw_difference_indices_vs_selected_original_A100': raw_differences,
        'sample_ROUGE_L_F1': per_sample, 'NPU_profiler_kernel_events': len(kernels),
        'profiler_scope': lineage['profiler_scope'],
        'protocol': 'Archived own A100 auto FP16, image-only TextCaps, native eager, serial one, original declared 2048 and effective 512, repetition penalty 1.2, full-prefix no cache, EOS 1, full raw caption scoring.',
    }


if __name__ == '__main__':
    torch.set_num_threads(4)
    print(json.dumps(verify(Path(sys.argv[1]).resolve(), int(sys.argv[2])), ensure_ascii=False, indent=2))
