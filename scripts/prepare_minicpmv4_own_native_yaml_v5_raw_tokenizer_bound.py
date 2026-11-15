"""Bind exact own raw tokenizer restoration; old image-unused real3 is rejected."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,subprocess,copy,importlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='openbmb/MiniCPM-V-4';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMV4_own_raw_tokenizer_source_bound_readiness_v5_user_continue_20261115'
F=B/'MiniCPMV4_own_frontend200_v3_exact_raw_tokenizer_special_metadata_user_continue_20261115'
J=B/'MiniCPMV4_own_native_true3_v3_raw_tokenizer_real_FLOCK_user_continue_20261115'
CONF='configs/MiniCPMV/openbmb--MiniCPM-V-4-own-native-BF16-2048-v3-raw-tokenizer-20261115.yaml';PROTO='configs/protocols/minicpmv4_own_native_bf16_v2_raw_tokenizer.json';CONTROL='scripts/continue_minicpmv4_true3_fresh3_200_v3_raw_tokenizer.py';VERIFY='scripts/verify_minicpmv4_own_native_bf16_v2_raw_tokenizer.py'
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
 ready=js(F/'readiness.json');assert ready['passed'] and ready['all200_image_bounds_nonempty_and_each_exact64'] and ready['own_raw_tokenizer_vocabulary_all73448_ids_unchanged'] and js(F/'actual_completion.json')['actual_returncode']==0
 old=B/'MiniCPMV4_own_native_true3_v2_registry_real_FLOCK_user_continue_20261115';done=js(old/'completion.json');assert done['actual_returncode']==1 and done['model_returncode']==1 and done['verifier_returncode']==1 and done['accepted']==[]
 H.mkdir(exist_ok=False);p=js(R/'configs/protocols/minicpmv4_own_native_bf16_v1.json')
 p.update(own_frontend200=ready['own_frontend200'],own_frontend200_SHA256=ready['own_frontend200_SHA256'])
 runtime=['universal_eval/model/adapters/minicpmv4_own_native_bf16_v2_raw_tokenizer.py','universal_eval/model/adapters/minicpmv4_own_frontend_v2_raw_tokenizer.py','universal_eval/model/adapters/minicpmv4_own_frontend_v1.py','universal_eval/model/adapters/minicpmv4_original_pixel_check_v1.py',VERIFY,CONTROL,'scripts/prepare_minicpmv4_own_native_yaml_v5_raw_tokenizer_bound.py','universal_eval/utils/readonly_npz_indexed.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','universal_eval/algorithm/inference.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','run.py','universal_eval/orchestrator/pipeline.py']
 runtime.extend(str(v.relative_to(R)) for v in sorted((R/'universal_eval/model/minicpmv4_own_native_20261115').glob('*.py')))
 p['implementation_source_SHA256']={n:sha(R/n) for n in runtime};p['static_evidence_SHA256'].update({str(F/'readiness.json'):sha(F/'readiness.json'),str(F/'actual_completion.json'):sha(F/'actual_completion.json')})
 p['historical_source_boundary']+=' Exact original own tokenizer.json backend is restored after current HF constructor lost embedded image-special encoding. All 73448 vocabulary ids, original normalizer/pretokenizer/postprocessor/decoder and own JSON bytes remain identical. No added or replaced token ids, borrowed tokenizer, embedding resize or changed model math. Old image-unused 3/91 result is rejected and preserved, not reused as visual evidence.'
 put(R/PROTO,p);cfg=js(R/'configs/MiniCPMV/openbmb--MiniCPM-V-4-own-native-BF16-2048-v2-exact-registry-20261115.yaml');cfg['name']=Path(CONF).stem;cfg['model']['adapter']='universal_eval/model/adapters/minicpmv4_own_native_bf16_v2_raw_tokenizer.py:MiniCPMV4OwnNativeBF16V2';cfg['model']['kwargs'].update(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO));put(R/CONF,cfg)
 job=js(old/'job.json');spec=job['spec'];spec.update(config=CONF,verifier=VERIFY,review_revision='MiniCPMV4_own741_BF16_raw_tokenizer_image_injection_v3_20261115',source_sha256={n:sha(R/n) for n in runtime+[CONF,PROTO]})
 validate_config(cfg,MID);batch.check_sources(spec);entry=js(D/'state.json')['models'][MID]
 for n in [3,200]:_,c=batch.config_for(spec,entry,n);validate_config(c,MID)
 scope(spec,[3],False);importlib.import_module('universal_eval.model.adapters.minicpmv4_own_native_bf16_v2_raw_tokenizer')
 assert not J.exists() and not runner_lock_owners() and batch.device_idle(spec) and not [v for v in project_processes() if v['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 for n,h in job['protected_canonical_hashes'].items():assert sha(n)==h
 job.update(spec=spec,created_at=datetime.now(timezone.utc).isoformat());J.mkdir();put(J/'job.json',job);put(H/'readiness.json',dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,own_original_raw_tokenizer_restored_without_vocabulary_change=True,all200_independent_pixels_and_nonempty_exact64_image_bounds_verified=True,old_failed_real3_is_not_qualified_or_reused=True,all_completed_source_and_old_failure_records_preserved=True,job=str(J),job_SHA256=sha(J/'job.json')))
 C=B/'MiniCPMV4_true3_then_measured_capacity_approved_fresh3_fixed200_v3_raw_tokenizer_user_continue_20261115';C.mkdir();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (C/'stdout.log').open('xb') as out,(C/'stderr.log').open('xb') as err:c=subprocess.Popen([sys.executable,'-B',str(R/CONTROL)],cwd=R,stdout=out,stderr=err,env=env,start_new_session=True)
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=c.pid,home=str(C),job=str(J),controller_source_SHA256=sha(R/CONTROL),no_G_or_canonical_write=True);put(C/'launch.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
