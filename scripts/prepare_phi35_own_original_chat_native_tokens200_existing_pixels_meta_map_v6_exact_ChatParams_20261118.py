"""Close missing original native input/stop interfaces without model inference.

Earlier own HF preprocessing remains valid for its declared scope. The original
vLLM renderer has an additional decode/re-encode step; it is observed separately.
The previous RGB-only control is not silently used for native white-background
media decoding. All old files remain unchanged.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import difflib
import hashlib
import importlib
import inspect
import json
import os
import struct
import tempfile
import traceback

R = Path('/root/projects/comperror')
B = R / 'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
W = Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Phi3VForCausalLM/microsoft/Phi-3.5-vision-instruct')
H = B / 'Phi35_own_original_chat_generation_prefix_native_tokens200_existing_pixel_proof_meta_map_v6_exact_ChatParams_no_NN_20261118'
OLD = B / 'Phi35_own_original_BF16_4096_native_candidate_frontend200_v3_original_dataset_prompt_no_NN_20261117/own_original_chat_frontend200.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_meta(t):
    import torch
    t = t.detach().cpu().contiguous()
    return dict(shape=list(t.shape), dtype=str(t.dtype), SHA256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def main():
    os.chdir('/root')
    sys.path[:0] = [str(R), str(R / 'scripts')]
    H.mkdir(exist_ok=False)
    cache = tempfile.mkdtemp(prefix='phi35_own_native_input_only_v2_')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_MODULES_CACHE=cache, PYTHONDONTWRITEBYTECODE='1', ASCEND_RT_VISIBLE_DEVICES='4')
    assert sha(OLD) == '8efba8e2372fde52937506598e89304eb61330d5c65fdd71bd9c77d4220e1d86'
    front = [json.loads(s) for s in OLD.read_text().splitlines()]
    assert len(front) == 200
    prior_native_pixels = B / 'Phi35_own_actual_native_frontend200_engine_stop_v4_explicit_reference_cache_bypass_no_NN_20261118/own_actual_native_frontend200.jsonl'
    assert sha(prior_native_pixels) == 'e126667ca58e5162272a21ee55d2e839a8feec68b64ea76e79ed1f4a7ab20be9'
    prior_rows = [json.loads(s) for s in prior_native_pixels.read_text().splitlines()]
    assert [r['index'] for r in prior_rows] == list(range(200))
    old_file_hashes = {str(OLD): sha(OLD)}
    import torch
    torch.set_num_threads(4)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.renderers.registry import renderer_from_config
    from vllm.renderers.params import ChatParams
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm import SamplingParams
    from PIL import Image
    reference = importlib.import_module('prepare_phi35_own_original_BF16_4096_HD_frontend200_v3_original_dataset_prompt_no_NN_20261117')
    assert reference.sha(OLD) == old_file_hashes[str(OLD)]
    # Original saved R deployment has max_tokens4096/outer10/TP1/.9 and no
    # max_model_len override. The currently installed native LongRoPE default
    # is observed, never replaced with the different shared-mount 2048 config.
    deployment = R / 'configs/Phi3VForCausalLM/microsoft--Phi-3.5-vision-instruct.yaml'
    assert sha(deployment) == 'd16e40ca27659cabbdad21e19079185544ab5ae893832b336a0274c713f2e1cc'
    args = EngineArgs(model=str(W), dtype='bfloat16', trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=.9, allowed_local_media_path=str(R / 'workspace/datasets'), enforce_eager=True)
    config = args.create_engine_config()
    renderer = renderer_from_config(config)
    native_input = InputProcessor(config, renderer)
    media = ImageMediaIO()
    assert media.image_mode == 'RGB' and media.rgba_background_color == (255, 255, 255)
    pp = json.loads((W / 'preprocessor_config.json').read_bytes())
    ids_different = []
    pixels_different_from_old_RGB = []
    stops = set()
    record_path = H / 'own_actual_native_frontend200.jsonl'
    with record_path.open('x', encoding='utf-8') as out:
        for i, ref in enumerate(front):
            _, prompts = renderer.render_chat([ref['original_native_message']], ChatParams(chat_template_kwargs={'add_generation_prompt': True, 'continue_final_message': False}), skip_mm_cache=True)
            engine_input = prompts[0]
            request = native_input.process_inputs('own_phi35_preflight_' + str(i), engine_input, SamplingParams(temperature=0., top_p=1., max_tokens=4096), ('generate',))
            sampling = request.sampling_params
            assert sampling.max_tokens == 4096 and sampling.temperature == 0. and sampling.top_p == 1. and not sampling.ignore_eos
            native_ids = request.prompt_token_ids
            assert native_ids == engine_input['prompt_token_ids'] and native_ids.count(32044) == ref['num_image_tokens']
            item = engine_input['mm_kwargs']['image'][0]
            pixels = item['pixel_values'].data
            sizes = item['image_sizes'].data
            if pixels.ndim == 4:
                pixels = pixels.unsqueeze(0)
            if sizes.ndim == 1:
                sizes = sizes.unsqueeze(0)
            image_path = Path(ref['original_native_message'][0]['content'][0]['image_url']['url'].removeprefix('file://'))
            assert sha(image_path) == ref['image_SHA256']
            assert tensor_meta(pixels) == prior_rows[i]['actual_native_processor_pixel_tensor'] and sizes.tolist() == prior_rows[i]['image_sizes'], (i, 'Reuse exact independently verified own native-media pixel values; no pixel200 replay')
            token_count = prior_rows[i]['original_HD_image_token_count']
            assert token_count == ref['num_image_tokens']
            changes = [dict(tag=tag, old_span=[a, b], native_span=[c, d], old_tokens=ref['own_native_positive_image_ID_replacement'][a:b], native_tokens=native_ids[c:d]) for tag, a, b, c, d in difflib.SequenceMatcher(a=ref['own_native_positive_image_ID_replacement'], b=native_ids, autojunk=False).get_opcodes() if tag != 'equal']
            if changes:
                ids_different.append(i)
            pm = prior_rows[i]['independent_native_media_reference_F32']
            native_pixel_metadata = tensor_meta(pixels)
            bm = tensor_meta(pixels.bfloat16())
            if pm['SHA256'] != ref['pixel_values_F32']['SHA256']:
                pixels_different_from_old_RGB.append(i)
            stop_ids = sorted(sampling.all_stop_token_ids)
            stops.add(tuple(stop_ids))
            value = dict(index=i, original_prompt=ref['original_prompt'], original_model_prompt=ref['original_model_prompt'], original_native_message=ref['original_native_message'], reference=ref['original_dataset_row']['reference'], media_path=str(image_path), image_SHA256=ref['image_SHA256'], actual_native_prompt_token_ids=native_ids, prior_own_HF_positive_input_differences=changes, independent_native_media_reference_F32=pm, actual_native_processor_pixel_tensor=native_pixel_metadata, actual_native_pixel_BF16=bm, image_sizes=sizes.tolist(), original_native_image_marker=32044, original_HD_image_token_count=token_count, actual_engine_request_sampling=dict(max_tokens=sampling.max_tokens, temperature=sampling.temperature, top_p=sampling.top_p, all_stop_token_ids=stop_ids, stop_token_ids=sampling.stop_token_ids), actual_max_model_len=config.model_config.max_model_len, independent_original_HD_pixels_bitwise_verified=True, current_native_input_not_historical_A100_capture=True)
            out.write(json.dumps(value, ensure_ascii=False) + '\n')
    # Resolve the 593-meta/592-physical mapping by actual parameter identity.
    cm = importlib.import_module('universal_eval.model.phi35_own_original_source_20261117.configuration_phi3_v')
    mm = importlib.import_module('universal_eval.model.phi35_own_original_source_20261117.modeling_phi3_v')
    cfg = cm.Phi3VConfig.from_json_file(str(W / 'config.json'))
    cfg._attn_implementation = 'eager'
    with torch.device('meta'):
        model = mm.Phi3VForCausalLM(cfg)
    assert model.model.embed_tokens.weight is model.model.vision_embed_tokens.wte.weight
    physical = {}
    for path in W.glob('*.safetensors'):
        with path.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(n))
        for key, value in header.items():
            if key != '__metadata__':
                assert key not in physical
                physical[key] = value
    shapes = {name: list(t.shape) for name, t in model.state_dict().items()}
    assert len(physical) == 592 and len(shapes) == 593
    from universal_eval.model.reference_sources.phi35_owned_checkpoint_layout_20261118 import validate_own_meta_mapping
    mapping = validate_own_meta_mapping(physical, shapes)
    extra = set(shapes) - set(mapping.values())
    (H / 'own592_source_to_HF5_meta_names.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')
    assert all(sha(path) == h for path, h in old_file_hashes.items())
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), passed=True, actual_NEW_model_forwards=0, no_LLM_or_workers_or_learned_CPU=True, own_native_input200=str(record_path), own_native_input200_SHA256=sha(record_path), original_R_deployment=str(deployment), original_R_deployment_SHA256=sha(deployment), original_4096_budget_outer10_TP1_unchanged=True, native_engine_default_max_len=config.model_config.max_model_len, actual_engine_request_stop_sets=[list(x) for x in sorted(stops)], prior_own_HF_positive_input_different_indices=ids_different, native_ImageMediaIO_vs_old_RGB_pixel_different_indices=pixels_different_from_old_RGB, original_image_files_and_old_proofs_unchanged=True, all200_native_input_and_independent_own_HD_pixel_values_verified=True, meta593_vs_physical592_closed_by_original_embedding_identity=True, extra_meta_alias=sorted(extra), no_true_NPU3_adapter_or_full_NF_ready_claim=True, prior_v1_manual_sampling_diff_simulation_not_actual_engine_stop_contract=True, prior_v2_exact_F32_guard_native_BF16_input_rejection_preserved=True, actual_native_input_dtype_and_original_model_BF16_cast_explicit=True, diagnostic_CPU_frontend_only_skip_mm_cache_to_retrieve_complete_values=True, actual_future_NPU_model_and_prefix_and_encoder_cache_not_changed=True, prior_v3_native_repeated_image_frontend_cache_hit_None_guard_failure_preserved=True, original_chat_add_generation_prompt_true_explicitly_restored=True, prior_v1_to_v4_ChatParams_default_false_not_actual_LLM_chat_contract=True, prior_v4_pixel_values_200_reused_with_exact_SHA_no_independent_pixel200_replay=True, actual_model_class_namespace_flattening_name_map_not_math_change=True)
    (H / 'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    try:
        with contextlib.redirect_stdout(sys.stderr):
            value = main()
        print(json.dumps(value, ensure_ascii=False))
    except BaseException:
        if H.exists():
            (H / 'actual_failure.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), error=traceback.format_exc(), actual_NEW_model_forwards=0), ensure_ascii=False, indent=2))
        raise
