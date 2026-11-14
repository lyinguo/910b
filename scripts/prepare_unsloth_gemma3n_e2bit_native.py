"""Bind own instruction template, source/configuration and workbook D before NN.

No A100 execution, learned CPU forward, canonical score or workbook writes.
The previous independent pixel200 proof is reused; only changed chat inputs are
prepared here. The shared vision graph is weight-free architecture code.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import copy, hashlib, inspect, json, os, subprocess, traceback

R = Path('/root/projects/comperror')
sys.path[:0] = [str(R), str(R / 'scripts')]
D = R / 'workspace/diagnostics/batch_910b'
B = D / 'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H = B / 'UnslothE2Bit_own_chat_template_native_candidate_0105_20261115'
OLD = B / 'next_UnslothGemma3nE2Bit_own_frontend_assets_no_AA_claim_1207_20261114'
MID = 'unsloth/gemma-3n-E2B-it'
W = Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma3nForConditionalGeneration/unsloth/gemma-3n-E2B-it')
CFG = 'configs/Gemma3nForConditionalGeneration/unsloth--gemma-3n-E2B-it-native-BF16-chat-2048-20261115.yaml'
PROTOCOL = 'configs/protocols/unsloth_gemma3n_e2bit_native.json'
LAYOUT = 'configs/protocols/unsloth_gemma3n_e2bit_vision_layout.json'
ADAPTER = 'universal_eval/model/adapters/unsloth_gemma3n_e2bit_native.py'
VERIFIER = 'scripts/verify_unsloth_gemma3n_e2bit_native.py'
PARENT = B / 'E4B_capacity_recovered_complete_failure_records_approved_fresh3_fixed200_v10_real_FLOCK_1605_20261113'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def js(p):
    return json.loads(Path(p).read_bytes())
def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    import batch_910b as batch
    from A100_alignment_evidence_runner_20261103 import scope
    from universal_eval.algorithm.inference import GreedyInference
    torch.set_num_threads(4)
    H.mkdir(exist_ok=False)
    protected = {str(p): sha(p) for p in [D / 'state.json', D / 'registry.json', D / 'PAUSE', D / 'POLICY.json', D / 'A100_ALIGNMENT_PUBLICATION_POLICY.json', D / 'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json', D / 'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json', Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]}
    review = js(OLD / 'review.json')
    assert review['passed'] and review['model_id'] == MID and review['samples'] == 200
    assert sha(OLD / 'frontend200.jsonl') == review['frontend_jsonl_SHA256']
    prior = [json.loads(s) for s in (OLD / 'frontend200.jsonl').read_text().splitlines()]
    source = js(OLD / 'source_tensor_headers.json')
    parent_cfg = js(R / 'configs/Gemma3nForConditionalGeneration/google--gemma-3n-E4B-original-Greedy-HighTen-BF16-2048-v9-capacity-recovered-complete-failure-records-1605-20261113.yaml')
    parent_protocol = js(parent_cfg['model']['kwargs']['protocol_file'])
    old_layout = js(parent_protocol['vision_graph_layout'])
    old_vision_names = js(Path(parent_protocol['actual_AA_control_dir']) / 'metadata_before_NN.json')['actual_parameter_aliases']
    prefix = 'model.vision_tower.timm_model.'
    aliases = [group for group in old_vision_names if all(name.startswith(prefix) for name in group)]
    assert aliases and all(name in source for group in aliases for name in group)
    state = {}
    for key, rec in old_layout['own_state'].items():
        assert prefix + key in source
        own = source[prefix + key]
        assert own['shape'] == rec['shape'] and own['dtype'] == 'BF16'
        state[key] = dict(shape=own['shape'], dtype='torch.bfloat16')
    layout = {key: old_layout[key] for key in ['checkpoint_values_exported', 'checkpoint_tensor_constants_in_graph', 'own_original_graph_SHA256', 'original_literal_scalar_tensor_arguments', 'graph_device_argument', 'stage_outputs']}
    layout.update(passed=True, model_id=MID, own_state=state, actual_parameter_aliases=aliases,
                  lowered_parameter_inputs=[dict(own_state_key=rec['own_state_key']) for rec in old_layout['lowered_parameter_inputs']],
                  scope='Shared architecture operator graph; own checkpoint shape/parameter declarations only. No AA values or learned constants.')
    write(R / LAYOUT, layout)
    processor = AutoProcessor.from_pretrained(str(W), local_files_only=True, trust_remote_code=True)
    assert processor.chat_template
    front = []
    for i, old in enumerate(prior):
        prompt = GreedyInference._add_answer_instruction(old['prompt'])
        assert prompt == old['proposed_model_prompt']
        image = Image.open(old['media_path']).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        assert '<start_of_turn>user\n' in text and text.endswith('<start_of_turn>model\n')
        inputs = processor(text=text, images=[image], return_tensors='pt', padding=True)
        assert hashlib.sha256(inputs['pixel_values'].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == old['inputs']['pixel_values']['SHA256']
        assert int((inputs['input_ids'] == 262145).sum()) == 256
        meta = {}
        for key, tensor in inputs.items():
            if tensor.is_floating_point():
                tensor = tensor.to(torch.bfloat16)
            tensor = tensor.contiguous()
            meta[key] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), SHA256=hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest())
        front.append(dict(index=i, prompt=old['prompt'], reference=old['reference'], media_path=old['media_path'],
            image_SHA256=old['media_SHA256'], original_model_prompt=prompt, formatted_text=text, inputs=meta,
            generation={'max_new_tokens':2048, 'do_sample':False}, own_template_used=True,
            previous_independent_F32_pixel200_proof_reused=True, historical_A100_tensor_capture_claimed=False))
    frontend = H / 'own_chat_frontend200.jsonl'
    with frontend.open('x', encoding='utf-8') as f:
        for row in front:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    frozen = js(parent_protocol['three_D_frozen_binding_review'])
    original_snapshot = Path(frozen['preserved_original_snapshot'])
    assert sha(original_snapshot) == '700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba'
    original_D = js(original_snapshot)['models'][MID]['A100_D']
    assert original_D == '0.805'
    decision = D / 'user_decisions/workbook_A100_D_is_comparison_reference_2130_20261114/decision.json'
    assert sha(decision) == '22cbb4581a69875f643902001babfeffa694a2275e0f894a447866e356bccf3d'
    assert js(decision)['workbook_SHA256'] == protected['/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx']
    old_yaml = R / 'configs/Gemma3nForConditionalGeneration/unsloth--gemma-3n-E2B-it.yaml'
    code = [ADAPTER, VERIFIER, 'scripts/prepare_unsloth_gemma3n_e2bit_native.py',
        'universal_eval/model/adapters/gemma3n_E4B_actual_original_HighTen_BF16_native_v5_complete_failure_records_1605_20261113_adapter.py',
        'universal_eval/model/adapters/gemma3n_E4B_HighTen_own_parameterized_BF16_vision_native_v1_after_reboot_20261113.py',
        'universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py',
        'universal_eval/algorithm/native_explicit_mc_v2.py', 'universal_eval/algorithm/inference.py',
        'universal_eval/utils/readonly_npz_indexed.py',
        'scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py',
        'run.py', 'universal_eval/orchestrator/pipeline.py', LAYOUT,
        str(Path(parent_protocol['vision_graph']).relative_to(R))]
    protocol = dict(model_id=MID, weight_path=str(W), report_task='image-text-to-text',
        dataset_path=review['dataset_SHA256'] and str(R / 'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'),
        dataset_SHA256=review['dataset_SHA256'], implementation_source_SHA256={name:sha(R / name) for name in code},
        static_evidence_SHA256={str(path):sha(path) for path in [OLD / 'review.json', OLD / 'frontend200.jsonl', OLD / 'source_tensor_headers.json', frontend, decision, original_snapshot]},
        own_runtime_assets_SHA256=review['own_runtime_assets_SHA256'], own_weight_files=review['own_weight_shards'],
        own_frontend200=str(frontend), vision_graph=parent_protocol['vision_graph'], vision_graph_layout=str(R / LAYOUT),
        hard_reserve_bytes=3*1024**3, log_reserve_bytes=1024**3, per_sample_lossless_space_estimate_bytes=2*1024**3,
        current_A100_D_values={'server':'0.8050','local':'0.8050','preserved_original':original_D},
        workbook_D_reference_authorized=True, saved_original_configuration_used=True,
        saved_original_YAML=str(old_yaml), saved_original_YAML_SHA256=sha(old_yaml),
        preserved_original_snapshot=str(original_snapshot), preserved_original_snapshot_SHA256=sha(original_snapshot),
        reference_workbook='/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx', reference_workbook_SHA256=protected['/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'],
        historical_source_boundary='Saved project vLLM/Greedy 2048/outer10 configuration and own instruction template reconstructed as native HF SDPA BF16/inner serial1 on 910B. A100 metric from user-confirmed workbook D; no claim of captured historical A100 engine bytes or tensors.')
    write(R / PROTOCOL, protocol)
    cfg = dict(name=Path(CFG).stem, description=protocol['historical_source_boundary'], task_type='image-text-to-text',sample_count=200,sample_range='0:199',
        data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=protocol['dataset_path']),
        model=dict(name=MID,adapter=ADAPTER+':UnslothGemma3nE2Bit',model_path=str(W),download=False,
            kwargs=dict(protocol_file=str(R / PROTOCOL),protocol_sha256=sha(R / PROTOCOL),protocol=dict(model_id=MID),npu_devices=[4])),
        algorithm=dict(name='own_native_greedy_explicit_MC',adapter='universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py:E4BActualOriginalHighTenMC',batch_size=10,kwargs=dict(extract_answer=True)),
        metrics=dict(override=['Strict_MC_Accuracy'],adapters=['universal_eval/algorithm/tinyllava_strict_mc.py:StrictMCAccuracy'],include_failed_predictions=True),
        baseline=dict(score=.805,threshold=.05,primary_metric='Strict_MC_Accuracy'),
        output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True))
    write(R / CFG, cfg)
    template = copy.deepcopy(js(PARENT / 'job.json'))
    spec = copy.deepcopy(template['spec'])
    spec.update(config=CFG,verifier=VERIFIER,npu_devices=[4],approved=False,smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600)
    spec.pop('alignment_approval_review',None);spec.pop('alignment_approval_review_sha256',None)
    spec['source_sha256']={name:sha(R / name) for name in code+[PROTOCOL,CFG,'universal_eval/algorithm/tinyllava_strict_mc.py']}
    for key in ['model_id','name']:
        if key in spec:spec[key]=MID
    template.update(model_id=MID,spec=spec,stages=[3],write_workbook=False,protected_canonical_hashes=protected)
    jobhome=B/'UnslothE2Bit_own_chat_native_true3_real_FLOCK_0105_20261115'
    jobhome.mkdir(exist_ok=False);write(jobhome/'job.json',template)
    scope(spec,[3],False);batch.validate_config(cfg,MID);batch.check_sources(spec)
    entry=js(D/'state.json')['models'][MID]
    c3,e3=batch.config_for(spec,entry,3);c200,e200=batch.config_for(spec,entry,200)
    assert e3['sample_range']=='0:2' and e200['sample_range']=='0:199'
    assert protected=={path:sha(path) for path in protected}
    result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,own_chat_inputs=200,
        prior_plain_frontend_not_used_for_NPU=True,previous_independent_pixel200_reused=True,source_path=str(R / ADAPTER),
        YAML=CFG,YAML_SHA256=sha(R/CFG),smoke_YAML=str(c3.relative_to(R)),protocol=PROTOCOL,protocol_SHA256=sha(R/PROTOCOL),
        true3_job=str(jobhome),job_SHA256=sha(jobhome/'job.json'),ready_for_real_runner_after_old_lock_release=True,
        source_SHA256={name:sha(R/name) for name in spec['source_sha256']},no_NPU_or_G_claimed=True)
    write(H/'readiness.json',result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    try: main()
    except BaseException:
        if H.exists() and not (H/'failure.json').exists():
            write(H/'failure.json',dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc(),no_NPU=True))
        raise
