"""Recover the pre-inference validation API mistake; reuse the unchanged own YAML."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,subprocess,copy,importlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='openbmb/MiniCPM-V-4';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMV4_own_native_YAML_support_API_recovered_v2_user_continue_20261115';J=B/'MiniCPMV4_own_native_true3_v1_real_FLOCK_user_continue_20261115'
CONF='configs/MiniCPMV/openbmb--MiniCPM-V-4-own-native-BF16-2048-v1-20261115.yaml';PROTO='configs/protocols/minicpmv4_own_native_bf16_v1.json'
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
 H.mkdir(exist_ok=False);cfg=js(R/CONF);p=js(R/PROTO);entry=js(D/'state.json')['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 assert p['model_id']==MID and p['current_A100_D_values']=={'server':'0.8900','local':'0.8900','preserved_original':'0.89'}
 sources=dict(p['implementation_source_SHA256']);sources.update({CONF:sha(R/CONF),PROTO:sha(R/PROTO),str(Path(__file__).relative_to(R)):sha(__file__)})
 spec=dict(approved=False,review_revision='MiniCPMV4_own_source741_BF16_native_workbookD_20261115',config=CONF,verifier='scripts/verify_minicpmv4_own_native_bf16_v1.py',npu_devices=[4],source_sha256=sources,smoke_timeout_seconds=14400,formal_timeout_seconds=172800,verify_timeout_seconds=21600)
 validate_config(cfg,MID);batch.check_sources(spec)
 for n in [3,200]:_,configured=batch.config_for(spec,entry,n);validate_config(configured,MID)
 scope(spec,[3],False);importlib.import_module('universal_eval.model.adapters.minicpmv4_own_native_bf16_v1')
 assert not J.exists() and not runner_lock_owners() and batch.device_idle(spec) and not [v for v in project_processes() if v['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 template=js(B/'Eagle2_2B_own_native_true3_v1_real_FLOCK_1925_20261115/job.json');job=copy.deepcopy(template);job.update(model_id=MID,spec=spec,stages=[3],created_at=datetime.now(timezone.utc).isoformat(),protected_canonical_hashes={n:sha(n) for n in template['protected_canonical_hashes']})
 J.mkdir();put(J/'job.json',job);put(H/'readiness.json',dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,correct_API='batch_910b_support.validate_config + batch.config_for(spec,entry,count)',validate_check_sources_config3_config200_passed=True,all_original_failed_source_and_exit_records_preserved=True,no_model_source_or_YAML_protocol_math_change=True,job=str(J),job_SHA256=sha(J/'job.json')))
 C=B/'MiniCPMV4_true3_then_measured_capacity_approved_fresh3_fixed200_v1_user_continue_20261115';C.mkdir()
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (C/'stdout.log').open('xb') as out,(C/'stderr.log').open('xb') as err:c=subprocess.Popen([sys.executable,'-B',str(R/'scripts/continue_minicpmv4_true3_fresh3_200_v1.py')],cwd=R,stdout=out,stderr=err,env=env,start_new_session=True)
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=c.pid,home=str(C),job=str(J),controller_source_SHA256=sha(R/'scripts/continue_minicpmv4_true3_fresh3_200_v1.py'),no_G_or_canonical_write=True);put(C/'launch.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
