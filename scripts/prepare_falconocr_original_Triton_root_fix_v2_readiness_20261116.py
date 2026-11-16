NEW_SOURCES={'universal_eval/model/falconocr_own_npu_v2_original_triton_20261116/attention.py': '525fc4a209d3c738b02056ca10e33d1b9ce1891a0b59ca80639aa9e77abdc36e', 'universal_eval/model/falconocr_own_npu_v2_original_triton_20261116/configuration_falcon_ocr.py': '452656de36620c58a1ec7ce84c53f2360aae550ef32beb5b7486e2419c7932e9', 'universal_eval/model/falconocr_own_npu_v2_original_triton_20261116/modeling_falcon_ocr.py': '2404ffa753b80e6a2db8a68104712a6a7062d3c371208c195b941bde77c360a3', 'universal_eval/model/falconocr_own_npu_v2_original_triton_20261116/processing_falcon_ocr.py': '80ba44e6aa6eaa5f11b335b10fbe62bfcb63bc2b73c7a8fdbe049304b75e5d8d', 'universal_eval/model/falconocr_own_npu_v2_original_triton_20261116/rope.py': 'c61f28e2684fab2da733aa00c5e10d5985e7724ccf1a93b922bcbae93e05a6b1', 'universal_eval/model/falconocr_own_npu_v2_original_triton_20261116/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'universal_eval/model/falconocr_own_npu_v2_original_triton_20261116/npu_attention_backend.py': 'cd3917fcb7dbc8c70c75b1d5ca9d45b623922ab2b353343d26affcc756019628', 'universal_eval/model/adapters/falconocr_own_original_bf16_native_512_v2_original_triton_20261116.py': '59eb9bbbb8ab34e6bcd9b3b4e2ab0610fec1fa050b983b983d8421698b90f959', 'scripts/verify_falconocr_own_original_bf16_512_v2_original_triton_20261116.py': 'bb68aec03a5ebd02dcfa0466e2074c8ae53f362a59124e49ba7e1b1fa0976e03', 'scripts/continue_falconocr_own_true3_capacity_fresh3_fixed200_v2_original_triton_20261116.py': '7f503c0fea29537c3ec2812a0e87a0419b3a9409ea0c47f8245654bd2d7ae079'}
"""Prospective own original Triton semantic fix; never edit old runs or G."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,ast,importlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 diagnostic=B/'FalconOCR_original_own_Triton_FFN_intrinsic_precision_diagnostic_20261116/review.json';diag=js(diagnostic);assert diag['passed'] and not diag['original_own_kernel_current_F32_port_equivalent'] and all(x['source_kernel_actual_NPU'] and x['eager_BF16_intermediate_equal'] for x in diag['cases'])
 old=B/'FalconOCR_own_original_BF16_512_approved_fresh3_fixed200_v1_real_FLOCK_20261116';done=js(old/'completion.json');assert done['actual_returncode']==0 and done['accepted'][-1]['sample_count']==200 and not done['accepted'][-1]['formal_alignment_within_5pct']
 H=B/'FalconOCR_original_Triton_FFN_semantics_root_cause_new_source_v2_readiness_20261116';H.mkdir(exist_ok=False)
 root=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,root_cause='Old port wrongly assumed original Triton BF16 gate promotes both multiplies to F32. Direct original own Triton kernel on NPU instead exactly matches BF16 intermediate rounding; original kernel is restored literally.',diagnostic=str(diagnostic),diagnostic_SHA256=sha(diagnostic),old_model200_completion=str(old/'completion.json'),old_model200_completion_SHA256=sha(old/'completion.json'),old_model200_numeric_proof=done['accepted'][-1]['proof'],old_model200_numeric_proof_SHA256=done['accepted'][-1]['proof_sha256'],old_200_not_original_protocol_qualified_due_to_proven_FFN_semantic_difference=True,no_old_bytes_edited=True,no_G_write=True,only_original_semantics_restored=True,no_task_data_checkpoint_dtype_budget_stop_or_scoring_change=True)
 create(H/'root_cause_review.json',root)
 parent=js(B/'FalconOCR_own_original_BF16_512_true3_v1_real_FLOCK_20261116/job.json');cfg=js(R/parent['spec']['config']);before=copy.deepcopy(cfg);proto=js(cfg['model']['kwargs']['protocol_file'])
 new_sources=NEW_SOURCES.copy();new_sources['scripts/prepare_falconocr_original_Triton_root_fix_v2_readiness_20261116.py']=sha(__file__)
 for n,h in new_sources.items():assert sha(R/n)==h;ast.parse((R/n).read_bytes())
 proto['implementation_source_SHA256'].update(new_sources);proto['static_evidence_SHA256'].update({str(H/'root_cause_review.json'):sha(H/'root_cause_review.json'),str(diagnostic):sha(diagnostic)});proto['original_Triton_BF16_intermediate_FFN_direct_kernel_restored']=True
 proto['historical_source_boundary']=proto['historical_source_boundary'].replace('original Triton gate uses F32 internal arithmetic with one BF16 store','original Triton gate is called unchanged and preserves BF16 intermediate rounding; old F32-port claim is disproved in the retained root-cause diagnostic')
 pp=R/'configs/protocols/falconocr_own_original_BF16_512_native_v2_original_triton_20261116.json';create(pp,proto)
 config=R/'configs/FalconOCRForCausalLM/tiiuae--Falcon-OCR-own-original-BF16-512-native-v2-original-triton-20261116.yaml';cfg['name']=config.stem;cfg['description']=proto['historical_source_boundary'];cfg['model']['adapter']='universal_eval/model/adapters/falconocr_own_original_bf16_native_512_v2_original_triton_20261116.py:FalconOCROwnOriginal512V2OriginalTriton';cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));assert all(cfg[k]==before[k] for k in cfg if k not in ['name','description','model']);assert cfg['model']['name']==before['model']['name'] and cfg['model']['model_path']==before['model']['model_path']
 create(config,cfg);spec=parent['spec'];spec.update(config=str(config.relative_to(R)),verifier='scripts/verify_falconocr_own_original_bf16_512_v2_original_triton_20261116.py',review_revision='Prospective own original Triton gate semantic restoration with exact22layer operand audit');spec['source_sha256'].update(new_sources);spec['source_sha256'].update({str(pp.relative_to(R)):sha(pp),str(config.relative_to(R)):sha(config)});parent['created_at']=datetime.now(timezone.utc).isoformat()
 batch.check_sources(spec);entry=js(D/'state.json')['models'][parent['model_id']]
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);validate_config(v,parent['model_id'])
 scope(spec,[3],False);unchanged(parent['protected_canonical_hashes']);mod=importlib.import_module('universal_eval.model.adapters.falconocr_own_original_bf16_native_512_v2_original_triton_20261116');assert not mod.FalconOCROwnOriginal512V2OriginalTriton.__abstractmethods__
 J=B/'FalconOCR_own_original_BF16_512_true3_v2_original_triton_real_FLOCK_20261116';create(J/'job.json',parent);create(H/'readiness.json',dict(passed=True,job=str(J),job_SHA256=sha(J/'job.json'),root_cause_review=str(H/'root_cause_review.json'),all_old_reports_and_arrays_preserved=True,no_CPU200_repeated=True,no_new_model_NN_yet=True));print(json.dumps(js(H/'readiness.json'),ensure_ascii=False))
if __name__=='__main__':main()
