"""Bind the own original resources to independent YAML and a private real-lock job."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,ast,subprocess,copy,os,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/MiniCPMV/openbmb/MiniCPM-V-4');MID='openbmb/MiniCPM-V-4'
F=B/'MiniCPMV4_own_native_frontend200_v2_tensor_metadata_recovery_user_continue_20261115';H=B/'MiniCPMV4_own_native_YAML_resource_source_readiness_v1_user_continue_20261115'
J3=B/'MiniCPMV4_own_native_true3_v1_real_FLOCK_user_continue_20261115'
CONF='configs/MiniCPMV/openbmb--MiniCPM-V-4-own-native-BF16-2048-v1-20261115.yaml';PROTO='configs/protocols/minicpmv4_own_native_bf16_v1.json';VERIFIER='scripts/verify_minicpmv4_own_native_bf16_v1.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def js(p):return json.loads(Path(p).read_bytes())
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
 H.mkdir(exist_ok=False);front=js(F/'readiness.json');assert front['passed'] and js(F/'actual_completion.json')['actual_returncode']==0
 original=B/'local_workbook_delivery_20261103/local_A100_reference_snapshot.json';current=B/'local_reference_external_difference_0910_20261104_120236_794022/current_user_local_D_snapshot.json'
 assert sha(original)=='700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba' and sha(current)=='14ab07ef221d0d9deb6da157bcb3e7cadd9577b6a7d0110eb333235c4728c49b'
 assert js(original)['models'][MID]['A100_D']=='0.89'
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)=='960d16530a4d8f011297efcb4affec5e6a41867c4f7e6665683346509a9c4a11'
 frozen=H/'reference_workbook_original_bytes.xlsx';shutil.copyfile(book,frozen);assert sha(frozen)==sha(book)
 runtime=['universal_eval/model/adapters/minicpmv4_own_native_bf16_v1.py','universal_eval/model/adapters/minicpmv4_own_frontend_v1.py','universal_eval/model/adapters/minicpmv4_original_pixel_check_v1.py',VERIFIER,'scripts/prepare_minicpmv4_own_native_yaml_v1.py','scripts/continue_minicpmv4_true3_fresh3_200_v1.py','universal_eval/utils/readonly_npz_indexed.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','run.py','universal_eval/orchestrator/pipeline.py']
 runtime.extend(str(p.relative_to(R)) for p in sorted((R/'universal_eval/model/minicpmv4_own_native_20261115').glob('*.py')))
 own_assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and (p.suffix in ['.json','.model','.py'] or p.name=='tokenizer.json')}
 p=dict(model_id=MID,weight_path=str(W),own_weight_files=front['own_weight_files'],own_assets_SHA256=own_assets,own_source_manifest=front['own_source_manifest'],own_source_manifest_SHA256=front['own_source_manifest_SHA256'],own_frontend200=front['own_frontend200'],own_frontend200_SHA256=front['own_frontend200_SHA256'],workbook_D_reference_authorized=True,dataset_path=str(R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'),dataset_SHA256=sha(R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'),reference_workbook=str(frozen),reference_workbook_SHA256=sha(frozen),current_A100_D_values={'server':'0.8900','local':'0.8900','preserved_original':'0.89'},implementation_source_SHA256={n:sha(R/n) for n in runtime},static_evidence_SHA256={str(v):sha(v) for v in [F/'readiness.json',F/'actual_completion.json',original,current,Path(front['original_source_copy_review'])]},historical_source_boundary='Own original MiniCPM-V-4 saved VLLMModel: own BF16, Greedy outer10, 2048 budget, own ChatML, 64 soft tokens per original max9 slice image, EOS2/73440, native Llama longrope. Current native own HF eager vision and Llama/resampler, serial1 within original outer10, complete learned weights on NPU. Original A100 engine/backend bytes and tensors were not captured; no bitwise historical runtime claim. Workbook original D remains the authorized comparison.',hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3)
 (R/PROTO).parent.mkdir(exist_ok=True);put(R/PROTO,p)
 cfg=dict(name=Path(CONF).stem,description='Own original MiniCPM-V-4 full source native BF16, complete five reports and full head/KV/input evidence',task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=p['dataset_path']),model=dict(name=MID,adapter='universal_eval/model/adapters/minicpmv4_own_native_bf16_v1.py:MiniCPMV4OwnNativeBF16',model_path=str(W),download=False,kwargs=dict(npu_devices=[4],protocol={'model_id':MID},protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO))),algorithm=dict(name='native_explicit_mc_v2',batch_size=10,kwargs=dict(extract_answer=True)),metrics=dict(override=['Strict_MC_Accuracy'],include_failed_predictions=True),baseline=dict(score=.89,threshold=.05,primary_metric='Strict_MC_Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True))
 put(R/CONF,cfg)
 template=js(B/'Eagle2_2B_own_native_true3_v1_real_FLOCK_1925_20261115/job.json');spec=dict(approved=False,review_revision='MiniCPMV4_own_source741_BF16_native_workbookD_20261115',config=CONF,verifier=VERIFIER,npu_devices=[4],source_sha256={n:sha(R/n) for n in runtime+[CONF,PROTO]},smoke_timeout_seconds=14400,formal_timeout_seconds=172800,verify_timeout_seconds=21600)
 for n in spec['source_sha256']:assert not Path(n).is_absolute() and '..' not in Path(n).parts
 batch.validate(spec);batch.check_sources(spec);batch.config_for(spec,3);batch.config_for(spec,200);scope(spec,[3],False)
 job=copy.deepcopy(template);job.update(model_id=MID,spec=spec,stages=[3],created_at=datetime.now(timezone.utc).isoformat());job['protected_canonical_hashes']={n:sha(n) for n in job['protected_canonical_hashes']}
 assert not runner_lock_owners() and batch.device_idle(spec) and not [q for q in project_processes() if q['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 J3.mkdir();put(J3/'job.json',job);put(H/'readiness.json',dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,own_source_tensor_count=741,validate_check_sources_config3_config200_passed=True,independent_single_YAML=CONF,job=str(J3),job_SHA256=sha(J3/'job.json'),all_original_completed_NN_and_protected_records_preserved=True))
 C=B/'MiniCPMV4_true3_then_measured_capacity_approved_fresh3_fixed200_v1_user_continue_20261115';C.mkdir()
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (C/'stdout.log').open('xb') as out,(C/'stderr.log').open('xb') as err:c=subprocess.Popen([sys.executable,'-B',str(R/'scripts/continue_minicpmv4_true3_fresh3_200_v1.py')],cwd=R,stdout=out,stderr=err,env=env,start_new_session=True)
 launch=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=c.pid,home=str(C),controller_source_SHA256=sha(R/'scripts/continue_minicpmv4_true3_fresh3_200_v1.py'),no_G_or_canonical_write=True);put(C/'launch.json',launch);print(json.dumps(launch,ensure_ascii=False))
if __name__=='__main__':main()
