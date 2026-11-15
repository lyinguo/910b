"""Restore the original standard preprocessing in frozen input evidence, not model math."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,fcntl,subprocess,traceback,copy,shutil,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMV4_exact_original_preprocessing_frozen_input_recovery_v5_user_continue_20261115'
J3=B/'MiniCPMV4_own_native_true3_v3_raw_tokenizer_real_FLOCK_user_continue_20261115';OLD=B/'MiniCPMV4_own_native_approved_fresh3_fixed200_v4_standard_approval_real_FLOCK_user_continue_20261115'
J=B/'MiniCPMV4_original_preprocessing_approved_fresh3_fixed200_v5_real_FLOCK_user_continue_20261115'
CONF='configs/MiniCPMV/openbmb--MiniCPM-V-4-own-native-BF16-2048-v4-original-preprocessing-20261115.yaml';PROTO='configs/protocols/minicpmv4_own_native_bf16_v3_original_preprocessing.json'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def create(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report,validate_config
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists() and not J.exists()
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 old_done=js(OLD/'completion.json');assert old_done['actual_returncode']==1 and old_done['sample_count']==200 and len(old_done['accepted'])==1
 assert {p.name for p in Path(old_done['report_dir']).iterdir()}=={'run.log','operator_lineage.json'}
 old_job=js(OLD/'job.json');unchanged(old_job['protected_canonical_hashes']);batch.check_sources(old_job['spec'])
 passed=js(J3/'completion.json');assert passed['actual_returncode']==0;small=passed['accepted'][0];assert small['sample_count']==3 and sha(small['proof'])==small['proof_sha256']
 proof=js(small['proof']);validate_report(R,small['report_dir'],old_job['model_id'],'image-text-to-text',3,proof)
 for path,h in proof['external_evidence_sha256'].items():assert sha(path)==h
 cfg=js(R/old_job['spec']['config']);old_p=js(cfg['model']['kwargs']['protocol_file']);old_front=js(old_p['own_frontend200'])
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='same_original_DataLoader_DataPreprocessor_Greedy_all200_input_binding_recovery_no_NPU_rerun'))
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 from universal_eval.model.adapters.minicpmv4_own_frontend_v2_raw_tokenizer import OwnMiniCPMV4InputsV2,tensor_sha,flatten_tensors
 from universal_eval.model.adapters.minicpmv4_original_pixel_check_v1 import check_pixels
 import torch
 torch.set_num_threads(4);dataset=Path(old_p['dataset_path']);rows=DatasetLoader().load(str(dataset));rows=TaskDataRegistry._resolve_media_paths(rows,str(dataset.parent));rows=DataPreprocessor().preprocess(rows,task_type=cfg['task_type'],apply_template=False,filter_empty=False);assert len(rows)==4329
 inputs=OwnMiniCPMV4InputsV2(old_p['weight_path']);front=[];changed=[];length_ratio=1.
 for i,(sample,oldref) in enumerate(zip(rows[:200],old_front)):
  path=str(Path(sample['media_path']).resolve());prompt=GreedyInference._add_answer_instruction(sample['prompt']);text,v=inputs.prepare(prompt,path);assert sha(path)==oldref['media_file_SHA256'] and sample['reference']==oldref['reference'];check_pixels(inputs.image_processor,path,v['pixel_values'][0])
  tensors=flatten_tensors(v)
  for key,t in tensors.items():
   if key.startswith(('pixel_values','tgt_sizes','image_sizes')):assert tensor_sha(t)==oldref['tensor_metadata'][key]['SHA256']
  ref=dict(oldref);ref.update(raw_dataset_prompt=oldref['source_prompt'],prompt=sample['prompt'],actual_model_prompt=prompt,text=text,input_ids=v['input_ids'][0].tolist(),image_bound=[t.tolist() for t in v['image_bound']],tensor_metadata={k:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t)) for k,t in tensors.items()},original_project_preprocessing_exactly_replayed=True)
  if i<3:
   for k in ['actual_model_prompt','text','input_ids','image_bound','tensor_metadata','reference','media_file_SHA256']:assert ref[k]==oldref[k],(i,k,'changed passed small input')
  if prompt!=oldref['actual_model_prompt']:changed.append(i)
  length_ratio=max(length_ratio,(len(ref['input_ids'])/len(oldref['input_ids']))**2);front.append(ref)
 put('own_frontend200_original_project_preprocessing.json',front)
 review=dict(UTC=now(),passed=True,standard_project_preprocess_true_apply_template_false_filter_empty_false=True,all4329_rows_preserved=True,fixed200_indices=list(range(200)),preprocessing_changed_frozen_prompt_indices=changed,all_original_image_pixels_and_refs_unchanged=True,all_first3_consumed_input_tensors_tokens_and_text_exactly_unchanged=True,no_completed_NPU_true3_rerun=True,old_failed_formal3_of200_284_forwards_kept_not_reused_or_spliced=True,model_math_dtype_budget_EOS_and_scoring_unchanged=True)
 put('all200_original_input_preprocessing_recovery_review.json',review)
 newp=copy.deepcopy(old_p);newp.update(own_frontend200=str(H/'own_frontend200_original_project_preprocessing.json'),own_frontend200_SHA256=sha(H/'own_frontend200_original_project_preprocessing.json'))
 additions={str(Path(__file__).relative_to(R)):sha(__file__)}
 for cls in [DatasetLoader,TaskDataRegistry,DataPreprocessor]:
  path=Path(inspect.getfile(cls)).resolve();assert path.is_relative_to(R);additions[str(path.relative_to(R))]=sha(path)
 newp['implementation_source_SHA256'].update(additions);newp['static_evidence_SHA256'][str(H/'all200_original_input_preprocessing_recovery_review.json')]=sha(H/'all200_original_input_preprocessing_recovery_review.json')
 newp['historical_source_boundary']+=' The original project preprocess=True text cleaning is replayed for the frozen200 guard (including repeated spaces), not newly applied to model input. Original loader/preprocessor/source/data remain unchanged; first3 exact inputs match the passed full numeric proof. Failed formal2003/284 is retained and not spliced.'
 for key in old_p:
  if key not in ['own_frontend200','own_frontend200_SHA256','implementation_source_SHA256','static_evidence_SHA256','historical_source_boundary']:assert old_p[key]==newp[key]
 create(R/PROTO,newp);cfg['name']=Path(CONF).stem;cfg['model']['kwargs'].update(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO));create(R/CONF,cfg)
 capacity=js(B/'MiniCPMV4_true3_then_measured_capacity_approved_fresh3_fixed200_v3_raw_tokenizer_user_continue_20261115/capacity_review.json');capacity['recovered_input_length_squared_max_factor']=length_ratio;capacity['empirical_estimate_bytes']=int(capacity['empirical_estimate_bytes']*length_ratio);capacity['actual_free_bytes']=shutil.disk_usage(D).free;capacity['capacity_passed']=capacity['actual_free_bytes']>=capacity['empirical_estimate_bytes'];put('capacity_review.json',capacity);assert capacity['capacity_passed']
 approval=dict(UTC=now(),approved=True,model_id=old_job['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=small['proof'],true3_proof_SHA256=small['proof_sha256'],exact_first3_input_identity_review=str(H/'all200_original_input_preprocessing_recovery_review.json'),exact_first3_input_identity_review_SHA256=sha(H/'all200_original_input_preprocessing_recovery_review.json'),capacity_review=str(H/'capacity_review.json'),capacity_review_SHA256=sha(H/'capacity_review.json'),first200_is_measurement_not_predetermined_5pct_qualification=True,no_model_math_or_actual_pipeline_input_change=True)
 put('first200_measurement_standard_approval.json',approval)
 job=copy.deepcopy(old_job);job['spec'].update(config=CONF,approved=True,alignment_approval_review=str(H/'first200_measurement_standard_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_standard_approval.json'));job['spec']['source_sha256'].update(additions);job['spec']['source_sha256'].update({CONF:sha(R/CONF),PROTO:sha(R/PROTO)})
 validate_config(cfg,job['model_id']);batch.check_sources(job['spec']);entry=js(D/'state.json')['models'][job['model_id']]
 for n in [3,200]:_,v=batch.config_for(job['spec'],entry,n);validate_config(v,job['model_id'])
 scope(job['spec'],[3,200],False);unchanged(job['protected_canonical_hashes']);assert not runner_lock_owners() and batch.device_idle(job['spec']);J.mkdir();create(J/'job.json',job)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (H/'runner_stdout.log').open('xb') as out,(H/'runner_stderr.log').open('xb') as err:
  c=subprocess.Popen([sys.executable,'-B',str(RUNNER),str(J)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=env);put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_then_fixed200_exact_original_preprocessing',job=str(J),job_SHA256=sha(J/'job.json'),child_PID=c.pid,no_G_or_canonical_write=True));rc=c.wait()
 final=js(J/'completion.json');assert rc==final['actual_returncode']==0;put('actual_completion.json',dict(UTC=now(),actual_returncode=0,formal_job_completion=final,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True,disk_alert_is_not_chat_delivery=True))
  raise
