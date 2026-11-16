"""Read-only configured concrete-class audit; new private job after old pre-NN guard failure."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,importlib,inspect,yaml,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 old=B/'OSAtlas4B_own_original_mcscore_BF16_true3_v2_concrete_interface_real_FLOCK_20261116';J=B/'OSAtlas4B_own_original_mcscore_BF16_true3_v4_actual_configured_class_real_FLOCK_20261116';H=B/'OSAtlas4B_actual_configured_adapter_readiness_v4_20261116'
 assert (old/'job.json').exists() and not (old/'state.json').exists() and not (old/'completion.json').exists() and not J.exists()
 observed=project_processes();assert not runner_lock_owners()
 assert all(x['role']=='other_project_python' and x['PID']==os.getppid() and x['cwd']==str(R) and x['argv'][1:] in [['-'],['-B','-']] for x in observed),observed
 job=json.loads((old/'job.json').read_bytes());unchanged(job['protected_canonical_hashes']);cfg=yaml.safe_load((R/job['spec']['config']).read_bytes());batch.validate_config(cfg,job['model_id']);batch.check_sources(job['spec'])
 adapter=cfg['model']['adapter'];path,clsname=adapter.rsplit(':',1);assert path=='universal_eval/model/adapters/osatlas4b_own_original_mcscore_bf16_v2_concrete_interface_20261116.py'
 module=importlib.import_module(path[:-3].replace('/','.'));cls=getattr(module,clsname);assert not inspect.isabstract(cls)
 instance=cls(**cfg['model']['kwargs']);assert instance.model_name==job['model_id'] and not instance.__abstractmethods__
 added=['scripts/prepare_osatlas4b_actual_configured_adapter_readiness_v6_actual_stdin_parent_20261116.py','scripts/continue_osatlas4b_original_probability_actual_class_v4_20261116.py','scripts/measure_osatlas4b_original_probability_capacity_v4_actual_class_20261116.py']
 job['spec']['source_sha256'].update({n:sha(R/n) for n in added});job['spec']['review_revision']='Own unchanged probability native v2; exact configured concrete class audited; new private source-bound v4 dispatch'
 job['created_at']=datetime.now(timezone.utc).isoformat();scope(job['spec'],[3],False);batch.check_sources(job['spec']);entry=json.loads((D/'state.json').read_bytes())['models'][job['model_id']]
 c3,e3=batch.config_for(job['spec'],entry,3);c200,e200=batch.config_for(job['spec'],entry,200);assert e3['sample_range']=='0:2' and e200['sample_range']=='0:199'
 H.mkdir(exist_ok=False);J.mkdir(exist_ok=False);(J/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 review=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),ready_for_original_runner_true3=True,model_id=job['model_id'],true3_job=str(J),job_SHA256=sha(J/'job.json'),configured_adapter=adapter,configured_concrete_class_instantiated_without_loading=True,observed_non_NN_processes=observed,only_exact_waiting_stdin_parent_permitted=True,old_prepare_wrong_hardcoded_class_guard_failure_retained=True,old_v2_job_never_run=True,no_true3_or_frontend_replay=True,own_source_tensors=541,source_SHA256=job['spec']['source_sha256'],no_NPU_claimed_by_this_readiness=True,no_canonical_write=True)
 (H/'readiness.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(review,ensure_ascii=False))
if __name__=='__main__':main()
