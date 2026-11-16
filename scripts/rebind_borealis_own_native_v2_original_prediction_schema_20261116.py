"""Prospective own Borealis rebind after exact original prediction-schema fix.
No previous report, failed job, model math, token budget or scores are changed.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,os,inspect,importlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='NbAiLab/borealis-4b-instruct-preview';OLD=B/'Borealis4B_own_original_BF16_true3_v1_real_FLOCK_20261116';J=B/'Borealis4B_own_original_BF16_true3_v2_original_prediction_schema_real_FLOCK_20261116';H=B/'Borealis4B_own_original_BF16_native_v2_original_prediction_schema_readiness_20261116';DIAG=B/'Borealis4B_v1_complete3_readonly_full_numeric_diagnostic_v2_prediction_schema_20261116'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
 assert not J.exists() and not H.exists() and not runner_lock_owners() and not [x for x in project_processes() if x['role'] in ['model','CPU_verifier_or_audit','alignment_runner']]
 assert js(OLD/'completion.json')['actual_returncode']==1 and js(DIAG/'actual_completion.json')['actual_returncode']==0
 diag=js(DIAG/'review.json');assert diag['passed'] and diag['sample_count']==3 and diag['actual_forward_count']==8 and not diag['NEW_real_NPU_inference'] and diag['all_full_head_rows_independently_recomputed'] and diag['all_native_KV_values_losslessly_reconstructed']
 job=copy.deepcopy(js(OLD/'job.json'));unchanged(job['protected_canonical_hashes']);oldcfg=js(R/job['spec']['config']);p=js(oldcfg['model']['kwargs']['protocol_file'])
 for n,h in p['implementation_source_SHA256'].items():assert sha(R/n)==h
 verifier='scripts/verify_borealis_own_original_BF16_native_v2_original_prediction_schema_20261116.py';news=[verifier,'scripts/measure_borealis_own_original_BF16_actual_capacity_v2_original_prediction_schema_20261116.py','scripts/continue_borealis_own_original_BF16_true3_fresh3_fixed200_v2_original_prediction_schema_20261116.py','scripts/rebind_borealis_own_native_v2_original_prediction_schema_20261116.py','scripts/diagnostic_borealis_existing_true3_full_numeric_v2_original_prediction_schema_20261116.py']
 for n in news:p['implementation_source_SHA256'][n]=sha(R/n)
 p['static_evidence_SHA256'].update({str(DIAG/'review.json'):sha(DIAG/'review.json'),str(DIAG/'actual_completion.json'):sha(DIAG/'actual_completion.json')})
 pp=R/'configs/protocols/borealis_own_original_BF16_native_v2_original_prediction_schema_20261116.json';put(pp,p)
 cf=R/'configs/Gemma3ForConditionalGeneration/NbAiLab--borealis-4b-instruct-preview-own-original-BF16-2048-native-v2-original-prediction-schema-20261116.yaml';cfg=copy.deepcopy(oldcfg);cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));put(cf,cfg)
 assert {k:v for k,v in cfg.items() if k not in ['model','name']}=={k:v for k,v in oldcfg.items() if k not in ['model','name']}
 spec=job['spec'];spec.update(config=str(cf.relative_to(R)),verifier=verifier,review_revision='Borealis_v2_original_prediction_schema_only');spec['source_sha256'].update({n:sha(R/n) for n in news+[str(pp.relative_to(R)),str(cf.relative_to(R))]});batch.check_sources(spec)
 entry=js(D/'state.json')['models'][MID]
 for c in [3,200]:_,co=batch.config_for(spec,entry,c);validate_config(co,MID)
 scope(spec,[3],False);job.update(created_at=datetime.now(timezone.utc).isoformat(),protected_canonical_hashes={n:sha(n) for n in job['protected_canonical_hashes']});put(J/'job.json',job)
 put(H/'readiness.json',dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),ready_for_original_runner_true3=True,true3_job=str(J),job_SHA256=sha(J/'job.json'),original_NN_adapter_and_observer_bytes_unchanged=True,only_verifier_prediction_shape_accepts_original_independently_equal_choice=True,old_failed_job_and_full5_preserved=True,diagnostic_full_numeric_review=str(DIAG/'review.json'),diagnostic_full_numeric_review_SHA256=sha(DIAG/'review.json')))
 print(json.dumps(js(H/'readiness.json'),ensure_ascii=False))
if __name__=='__main__':main()
