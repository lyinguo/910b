"""Keep every v1 numeric/report guard and add genuine worker isolation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,contextlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_gemma4_unified_12bit_own_BF16_native_v1_20261117 import verify,sha,js,check,EXTERNAL,R

if __name__=='__main__':
    folder=Path(sys.argv[1]).resolve()
    with contextlib.redirect_stdout(sys.stderr):
        proof=verify(folder,int(sys.argv[2]));line=js(folder/'operator_lineage.json')['actual_execution'];assert line['genuine_clean_learned_worker_required'] and line['parent_CLI_legacy_compatibility_not_used_by_learned_worker']
        load=js(check(line['clean_worker_loading_attestation'],line['clean_worker_loading_attestation_SHA256']));assert load['no_CLI_or_original_adapter_initializer_imported'] and load['cwd']==str(R)
        check(load['worker_source'],load['worker_source_SHA256'],R/'scripts');check(load['actual_native_implementation_source'],load['actual_native_implementation_SHA256'],R/'universal_eval')
        for field in ['legacy_mock_modules_before_learned_load','legacy_mock_modules_after_learned_load']:
            for rec in load[field].values():check(rec['path'],rec['SHA256'])
        home=Path(line['evidence_root']);after=js(check(home/'clean_worker_after_actual_inference_attestation.json'));assert after['no_mock_or_stub_inside_learned_worker'] and after['actual_sample_count']==int(sys.argv[2]) and after['PID']==load['PID']
        for rec in after['genuine_loaded_optional_modules'].values():check(rec['path'],rec['SHA256'])
        weights=js(check(line['own_source_loading'],line['own_source_loading_SHA256']));expected={f'model.language_model.layers.{i}.layer_scalar' for i in range(48)};assert set(weights['original_persistent_source_buffers'])==expected and weights['actual_parameter_names_including_own_tied_head']==630;assert {x['target_name'] for x in weights['records'] if x['source_kind']=='original_persistent_source_buffer'}==expected
        proof.update(original48_persistent_layer_scalars_included_in_all677_source_SHA=True,genuine_fresh_learned_worker_no_stub_or_mock_verified=True,base_independent_verifier_SHA256=proof['independent_verifier_SHA256'],independent_verifier_SHA256=sha(__file__),external_evidence_sha256=EXTERNAL)
    target=home/'independent_report_verification.json';assert not target.exists();target.write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps(proof,ensure_ascii=False,allow_nan=False))
