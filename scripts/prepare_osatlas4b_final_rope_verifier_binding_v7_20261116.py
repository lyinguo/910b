"""New source-bound protocol/YAML/job after actual-config verifier compatibility fix."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,importlib,inspect,os,yaml
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
OLD=B/'OSAtlas4B_own_original_mcscore_BF16_true3_v4_actual_configured_class_real_FLOCK_20261116';J=B/'OSAtlas4B_own_original_mcscore_BF16_true3_v7_final_rope_verifier_binding_real_FLOCK_20261116';H=B/'OSAtlas4B_final_rope_verifier_binding_readiness_v7_20261116'
SELF='scripts/prepare_osatlas4b_final_rope_verifier_binding_v7_20261116.py';V='scripts/verify_osatlas4b_own_original_mcscore_bf16_v2_actual_rope_config_20261116.py';CONT='scripts/continue_osatlas4b_original_probability_final_rope_binding_v7_20261116.py';CAP='scripts/measure_osatlas4b_original_probability_capacity_v7_final_rope_binding_20261116.py';CFG='configs/InternVLChatModel/OS-Copilot--OS-Atlas-Base-4B-own-original-MCScore-BF16-native-v3-final-RoPE-verifier-20261116.yaml';PROTO='configs/protocols/osatlas4b_own_original_mcscore_bf16_v3_final_rope_verifier_20261116.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 observed=project_processes();assert not runner_lock_owners() and all(x['role']=='other_project_python' and x['PID']==os.getppid() and x['cwd']==str(R) and x['argv'][1:] in [['-'],['-B','-']] for x in observed),observed
 completion=js(OLD/'completion.json');assert completion['actual_returncode']==1 and completion['verifier_returncode']==1
 diag=B/'OSAtlas4B_existing_completed_true3_actual_rope_configuration_full_value_diagnostic_v2_20261116';assert js(diag/'actual_completion.json')['actual_returncode']==0
 proof=js(diag/'review.json');assert proof['passed'] and proof['sample_count']==3 and proof['diagnostic_existing_completed_NPU3_only'] and proof['all_original_probability_scores_and_full_head_verified'] and proof['independent_verifier_SHA256']==sha(R/V)
 oldjob=js(OLD/'job.json');unchanged(oldjob['protected_canonical_hashes']);oldcfg=yaml.safe_load((R/oldjob['spec']['config']).read_bytes());oldp=js(oldcfg['model']['kwargs']['protocol_file']);p=copy.deepcopy(oldp)
 added={n:sha(R/n) for n in [SELF,V,CONT,CAP]};p['implementation_source_SHA256'].update(added)
 p['static_evidence_SHA256'].update({str(x):sha(x) for x in [diag/'review.json',diag/'actual_completion.json',OLD/'completion.json',B/'OSAtlas4B_independent_actual_rope_config_only_v2_prospective_review_20261116/review.json']})
 assert {k:v for k,v in oldp.items() if k not in ['implementation_source_SHA256','static_evidence_SHA256']}=={k:v for k,v in p.items() if k not in ['implementation_source_SHA256','static_evidence_SHA256']}
 H.mkdir(exist_ok=False);put(H/'prospective_binding_only_review.json',dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,native_model_source_and_all_runtime_fields_identical=True,all56_verifier_assertions_and_thresholds_preserved=True,only_CPU_analytic_reference_config_compatibility_changed=True,old_genuine3_report=str(completion['report_dir']),old_runner_actual_returncode=1,old_verifier_actual_returncode=1,existing_full_value_diagnostic=str(diag/'review.json'),existing_full_value_diagnostic_SHA256=sha(diag/'review.json'),not_automatically_accepting_old_runner_failure=True,new_unapproved_source_bound_true3_required_before_formal_measurement=True,no_CPU200_replay_no_original_report_or_proof_edit=True))
 p['static_evidence_SHA256'][str(H/'prospective_binding_only_review.json')]=sha(H/'prospective_binding_only_review.json');put(R/PROTO,p)
 cfg=copy.deepcopy(oldcfg);cfg['name']=Path(CFG).stem;cfg['model']['kwargs'].update(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO));put(R/CFG,cfg)
 oldkw=copy.deepcopy(oldcfg['model']['kwargs']);newkw=copy.deepcopy(cfg['model']['kwargs'])
 for key in ['protocol_file','protocol_sha256']:oldkw.pop(key);newkw.pop(key)
 assert oldkw==newkw and {k:v for k,v in cfg.items() if k not in ['name','model']}=={k:v for k,v in oldcfg.items() if k not in ['name','model']}
 job=copy.deepcopy(oldjob);spec=job['spec'];spec.update(config=CFG,verifier=V,approved=False,review_revision='Own unchanged original probability model, final actual-config independent verifier binding v7')
 for key in ['alignment_approval_review','alignment_approval_review_sha256']:spec.pop(key,None)
 spec['source_sha256'].update(added);spec['source_sha256'].update({CFG:sha(R/CFG),PROTO:sha(R/PROTO)})
 job.update(created_at=datetime.now(timezone.utc).isoformat(),stages=[3]);scope(spec,[3],False);batch.validate_config(cfg,job['model_id']);batch.check_sources(spec);entry=js(D/'state.json')['models'][job['model_id']];c3,e3=batch.config_for(spec,entry,3);c200,e200=batch.config_for(spec,entry,200);assert e3['sample_range']=='0:2' and e200['sample_range']=='0:199'
 path,clsname=cfg['model']['adapter'].rsplit(':',1);cls=getattr(importlib.import_module(path[:-3].replace('/','.')),clsname);assert not inspect.isabstract(cls);instance=cls(**cfg['model']['kwargs']);assert instance.model_name==job['model_id']
 J.mkdir(exist_ok=False);put(J/'job.json',job);unchanged(job['protected_canonical_hashes'])
 put(H/'readiness.json',dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=job['model_id'],ready_for_original_runner_true3=True,true3_job=str(J),job_SHA256=sha(J/'job.json'),YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),no_model_math_input_budget_dtype_or_score_change=True,no_G_write=True,source_SHA256=spec['source_sha256']));print(json.dumps(dict(passed=True,H=str(H),J=str(J),job_SHA256=sha(J/'job.json')),ensure_ascii=False))
if __name__=='__main__':main()
