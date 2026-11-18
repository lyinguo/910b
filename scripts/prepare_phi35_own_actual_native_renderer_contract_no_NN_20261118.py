"""Inspect the missing native renderer/stop contract, reusing prior own inputs.

This constructs configuration and a CPU renderer only. It never creates LLM,
workers, learned modules, changes the original checkpoint, or runs inference.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import hashlib
import inspect
import json
import os
import tempfile
import traceback

R = Path('/root/projects/comperror')
D = R / 'workspace/diagnostics/batch_910b'
B = D / 'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
W = Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Phi3VForCausalLM/microsoft/Phi-3.5-vision-instruct')
H = B / 'Phi35_own_actual_native_renderer_stop_contract_no_NN_v1_20261118'
OLD = B / 'Phi35_own_original_BF16_4096_native_candidate_frontend200_v3_original_dataset_prompt_no_NN_20261117/own_original_chat_frontend200.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.chdir('/root')
    sys.path[:0] = [str(R), str(R / 'scripts')]
    H.mkdir(exist_ok=False)
    cache = tempfile.mkdtemp(prefix='phi35_own_native_renderer_only_')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_MODULES_CACHE=cache, PYTHONDONTWRITEBYTECODE='1', ASCEND_RT_VISIBLE_DEVICES='4')
    assert sha(OLD) == '8efba8e2372fde52937506598e89304eb61330d5c65fdd71bd9c77d4220e1d86'
    front = [json.loads(s) for s in OLD.read_text().splitlines()]
    assert len(front) == 200
    import torch
    torch.set_num_threads(4)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.renderers.registry import renderer_from_config
    from vllm.renderers.params import ChatParams
    from vllm import SamplingParams
    own_config_paths = [R / 'configs/Phi3VForCausalLM/microsoft--Phi-3.5-vision-instruct.yaml', R / 'configs/microsoft--Phi-3.5-vision-instruct.yaml', Path('/mnt/afs/comp_error/configs/Phi3VForCausalLM/microsoft--Phi-3.5-vision-instruct.yaml')]
    saved = {str(p): dict(SHA256=sha(p), text=p.read_text()) for p in own_config_paths if p.is_file()}
    (H / 'saved_own_deployment_files.json').write_text(json.dumps(saved, ensure_ascii=False, indent=2))
    args = EngineArgs(model=str(W), dtype='bfloat16', trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=.9, allowed_local_media_path=str(R / 'workspace/datasets'), enforce_eager=True)
    config = args.create_engine_config()
    renderer = renderer_from_config(config)
    sampling = SamplingParams(temperature=0., top_p=1., max_tokens=4096)
    sampling.update_from_generation_config(config.model_config.get_diff_sampling_param(), renderer.get_tokenizer().eos_token_id)
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), own_model='microsoft/Phi-3.5-vision-instruct', no_LLM_engine_worker_or_learned_model_constructed=True, no_NEW_NPU_or_learned_CPU=True,
                  own_generation_config=json.loads((W / 'generation_config.json').read_bytes()), actual_tokenizer_EOS=renderer.get_tokenizer().eos_token_id,
                  actual_sampling_all_stop_ids=sorted(sampling.all_stop_token_ids), actual_sampling_stop_token_ids=sampling.stop_token_ids,
                  native_model_max_len=config.model_config.max_model_len, runtime_dtype=str(config.model_config.dtype), saved_own_deployment_paths=list(saved))
    # Native renderer is the missing input interface. Prior independent image
    # preprocessing proof is reused, not rerun as a separate CPU200 model test.
    records = []
    for i in range(3):
        conv, prompts = renderer.render_chat([front[i]['original_native_message']], ChatParams())
        value = prompts[0]
        ids = value['prompt_token_ids']
        record = dict(index=i, native_prompt_token_ids=ids, own_HF_positive_replacement_equal=ids == front[i]['own_native_positive_image_ID_replacement'], engine_input_keys=list(value), native_image_placeholder_count=ids.count(32044))
        records.append(record)
    result.update(native_first3_renderer_records=records, native_input_contract_first3_checked=True, own_prior_CPU200_not_replayed=True, no_adapter_or_full_NPU_verifier_readiness_claim=True,
                  renderer_source=str(inspect.getfile(type(renderer))), renderer_source_SHA256=sha(inspect.getfile(type(renderer))))
    (H / 'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'native_first3_renderer_records'}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        with contextlib.redirect_stdout(sys.stderr):
            main()
    except BaseException:
        if H.exists():
            (H / 'actual_failure.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), error=traceback.format_exc(), no_NEW_NPU=True), ensure_ascii=False, indent=2))
        raise
