"""Exact standard algorithm/metric registration; original NN never reached first forward."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,subprocess,copy,importlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='openbmb/MiniCPM-V-4';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMV4_own_native_registry_preflight_recovered_v4_alias_user_continue_20261115'
J=B/'MiniCPMV4_own_native_true3_v2_registry_real_FLOCK_user_continue_20261115'
CONF='configs/MiniCPMV/openbmb--MiniCPM-V-4-own-native-BF16-2048-v2-exact-registry-20261115.yaml'
CONTROL='scripts/continue_minicpmv4_true3_fresh3_200_v2_registry.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config
 from A100_alignment_evidence_runner_20261103 import scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=False);old=B/'MiniCPMV4_own_native_true3_v1_real_FLOCK_user_continue_20261115';done=js(old/'completion.json');assert done['actual_returncode']==1 and done['accepted']==[]
 log=(old/'fresh3/console.log').read_text();assert "未找到算法 'native_explicit_mc_v2'" in log
 cfg=js(R/'configs/MiniCPMV/openbmb--MiniCPM-V-4-own-native-BF16-2048-v1-20261115.yaml');parent=js(R/'configs/Eagle2_5_VLForConditionalGeneration/nvidia--Eagle2-2B-own-native-BF16-256-v1-20261115.yaml')
 cfg['name']=Path(CONF).stem;cfg['algorithm']=copy.deepcopy(parent['algorithm']);cfg['metrics']=copy.deepcopy(parent['metrics']);assert json.loads((R/CONF).read_bytes())==cfg
 entry=js(D/'state.json')['models'][MID];job=js(old/'job.json');spec=job['spec'];spec['config']=CONF;spec['review_revision']='MiniCPMV4_own_source741_BF16_exact_algorithm_registry_v2_20261115'
 for n in [CONF,CONTROL,str(Path(__file__).relative_to(R))]:spec['source_sha256'][n]=sha(R/n)
 validate_config(cfg,MID);batch.check_sources(spec)
 for n in [3,200]:_,c=batch.config_for(spec,entry,n);validate_config(c,MID)
 scope(spec,[3],False)
 # Exact already-used standard classes are real imports, without generic initializers.
 from universal_eval.algorithm.gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113 import E4BActualOriginalHighTenMC
 from universal_eval.algorithm.tinyllava_strict_mc import StrictMCAccuracy
 # The standard pipeline registers the explicit YAML alias; the class property is not that alias.
 assert not J.exists() and not runner_lock_owners() and batch.device_idle(spec) and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 for n,h in job['protected_canonical_hashes'].items():assert sha(n)==h
 job.update(spec=spec,created_at=datetime.now(timezone.utc).isoformat());J.mkdir();put(J/'job.json',job)
 put(H/'readiness.json',dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,own741_source_and_200_frontend_evidence_reused=True,original_failed_job=str(old),original_model_before_first_NN_forward=True,original_failed_exit_and_report_files_preserved=True,standard_explicit_adapter_paths_import_verified=True,no_model_math_or_data_dtype_budget_EOS_change=True,job=str(J),job_SHA256=sha(J/'job.json')))
 C=B/'MiniCPMV4_true3_then_measured_capacity_approved_fresh3_fixed200_v2_registry_user_continue_20261115';C.mkdir();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (C/'stdout.log').open('xb') as out,(C/'stderr.log').open('xb') as err:c=subprocess.Popen([sys.executable,'-B',str(R/CONTROL)],cwd=R,stdout=out,stderr=err,env=env,start_new_session=True)
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=c.pid,home=str(C),job=str(J),controller_source_SHA256=sha(R/CONTROL),no_G_or_canonical_write=True);put(C/'launch.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
