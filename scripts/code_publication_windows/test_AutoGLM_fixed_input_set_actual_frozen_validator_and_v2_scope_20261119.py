"""Use the real frozen configuration validator, no model instantiation."""
from pathlib import Path
import ast,copy,importlib.util,json
P=Path(__file__).resolve().parent;module=Path('D:/code-lv/910b/scripts/batch_910b_support.py');spec=importlib.util.spec_from_file_location('actual_fixed_validator',module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=json.loads((P/'R_AutoGLMPhone_own_original_config_source_SF_headers_only_serial_next_20261119.json').read_bytes());cfg=copy.deepcopy(x['full_saved_config']);cfg['model']['download']=False;cfg['data']['download']=False;cfg['output']=copy.deepcopy(m.OUTPUT);cfg['sample_count']=200
try:m.validate_config(cfg,x['model_id'])
except ValueError as e:assert str(e)=='Do not silently filter the fixed sample set'
else:raise AssertionError('v1 must fail closed before NN')
cfg['data']['filter_empty']=False;cfg['data']['dataset_path']='/root/projects/comperror/workspace/datasets/image-text-to-text-mmbench/eval.jsonl';m.validate_config(cfg,x['model_id'])
nf=(P/'verify_autoglm_phone9b_own_original_BF16_2048_native_v2_fixed_input_set_20261119.py').read_text();oldnf=(P/'verify_autoglm_phone9b_own_original_BF16_2048_20261119_native.py').read_text();assert oldnf.replace("and cfg['data']['filter_empty']","and not cfg['data']['filter_empty']")==nf
prep=(P/'prepare_autoglm_phone9b_own_original_BF16_2048_true3_v2_fixed_input_set_20261119.py').read_text();assert "dataset_path=alias['actual_target_dataset']" in prep and 'filter_empty=False,download=False' in prep and 'own704_active_source_manifest.json' in prep and 'actual_NEW_model_forwards' in prep
watch=(P/'watch_GLM46_then_AutoGLM_preNN_fixed_input_set_recovery_serial_v2_20261119.py').read_text();assert 'Do not silently filter the fixed sample set' in watch and "no_v1_model_job_or_NN=True" in watch and "assert not oldj.exists()" in watch and 'os.kill' not in watch and 'signal.' not in watch
for s in [nf,prep,watch]:ast.parse(s)
alias=json.loads((P/'R_AutoGLM_own_AFS_R_identical_fixed200_input_alias_proof_no_NN_20261119.json').read_bytes());assert alias['passed'] and alias['all200_original_inputs_unchanged'] and alias['actual_NEW_model_forwards']==0
value=dict(passed=True,real_frozen_validator_rejects_v1_filter_before_NN=True,real_frozen_validator_accepts_v2_fixed_set_flags=True,own200_input_alias_proof_passed=True,only_fullNF_change_is_fixed_set_configuration_binding_not_numerical_math_or_threshold=True,recovery_requires_old_natural_exact_preNN_refusal_no_v1_job_or_inference=True,no_signal_hot_edit_or_current_GLM_change=True,actual_NEW_NN=0)
(P/'AutoGLM_fixed_input_set_v2_actual_validator_and_scope_tests_20261119.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');print(json.dumps(value))
