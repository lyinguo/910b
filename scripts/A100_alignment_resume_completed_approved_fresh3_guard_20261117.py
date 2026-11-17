"""Continue only a completed approved fresh3 whose unstarted200 was capacity-refused."""
from pathlib import Path
import json,hashlib,copy
FIVE={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def validate_relationship(old,completion,proof,new):
 assert old['model_id']==new['model_id']=='merve/paligemma_vqav2' and old['stages']==new['stages']==[3,200]
 assert old['spec']['approved'] is new['spec']['approved'] is True
 assert old['spec']['config']==new['spec']['config'] and old['spec']['npu_devices']==new['spec']['npu_devices']==[4]
 for name in ['alignment_approval_review','alignment_approval_review_sha256','smoke_timeout_seconds','formal_timeout_seconds','verify_timeout_seconds','protocol_file','protocol_sha256']:
  assert old['spec'][name]==new['spec'][name],name
 assert completion['actual_returncode']==1 and completion['pending_sample_count']==200 and completion['verifier_returncode']==0 and 'Independent stage capacity admission refused' in completion['error']
 assert len(completion['accepted'])==1 and completion['accepted'][0]['sample_count']==3 and completion['accepted'][0]['new_real_NPU_inference']
 assert proof['passed'] and proof['sample_count']==3 and proof['numeric_review_passed'] and proof['historical_record_reuse'] is False
 assert set(proof['report_sha256'])==FIVE and proof['no_original_NPU_capture_occurrence_or_numeric_check_dropped']
 for name,h in old['spec']['source_sha256'].items():assert new['spec']['source_sha256'][name]==h,name
 return copy.deepcopy(completion['accepted'][0])
def validate_resume(job,R):
 ref=job['resume_completed_approved_fresh3'];oldpath=Path(ref['original_job']);old=js(oldpath);cp=Path(ref['original_completion']);assert sha(oldpath)==ref['original_job_SHA256'] and sha(cp)==ref['original_completion_SHA256'];completion=js(cp);a=completion['accepted'][0];assert sha(a['proof'])==a['proof_sha256']==ref['approved_fresh3_proof_SHA256'];proof=js(a['proof']);accepted=validate_relationship(old,completion,proof,job)
 folder=Path(accepted['report_dir']);assert folder==Path(proof['report_dir']) and {p.name for p in folder.iterdir()}==FIVE
 for n,h in proof['report_sha256'].items():assert sha(folder/n)==h,n
 for n,h in old['spec']['source_sha256'].items():assert sha(R/n)==h,n
 assert sha(R/job['spec']['config'])==old['spec']['source_sha256'][job['spec']['config']]
 assert not (cp.parent/'formal200').exists(),'Original200 must genuinely never have started'
 assert proof['new_storage_is_exact_value_sharing_not_pruned_numeric_evidence'] and proof['all_original_numeric_checks_retained']
 accepted.update(prior_same_approved_sequence_fresh3_continued_without_repeating_completed_NPU_stage=True,original_private_job=str(oldpath),original_private_job_SHA256=sha(oldpath),fresh3_is_not_claimed_new_in_this_continuation=True)
 return accepted
