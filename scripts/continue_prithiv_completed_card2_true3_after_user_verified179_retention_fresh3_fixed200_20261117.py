"""Continue accepted own card2 real3 after exact authorized storage recovery.

Never replay completed true3, alter inference, prune active values or write G.
Original sole runner performs NEWfresh3 and then fixed0:199 independently.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import os,json,hashlib,copy,fcntl,subprocess,shutil,traceback,zipfile,math
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='prithivMLmods/gemma-4-E4B-it-Uncensored-MAX'
J3=B/'PrithivGemma4E4B_existing_model_first_independent_card2_true3_parallel_v1_20261117'
H=B/'Prithiv_completed_card2_true3_user_verified179_retention_capacity_recovered_continuation_20261117'
J=B/'Prithiv_completed_card2_true3_verified179_capacity_approved_NEWfresh3_fixed200_original_real_FLOCK_20261117'
RET=B/'authorized_verified179_accepted_current_nonexample_retention_user_continue_Prithiv_20261117'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';BOOK=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as out:out.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def capacity(accepted):
 line=js(Path(accepted['report_dir'])/'operator_lineage.json')['actual_execution'];proto=js(line['protocol_file']);trace=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()];assert [r['index'] for r in trace]==[0,1,2];entries=[]
 for row in trace:
  sample=js(row['actual_execution_file']);archive=Path(sample['archive']);assert archive.stat().st_size==sample['archive_bytes'] and sha(archive)==sample['archive_SHA256']
  with zipfile.ZipFile(archive) as z:
   members={x.filename:x for x in z.infolist()};prefill={n.replace('.','_')+'.npy' for n in sample['activation_prefill']};assert prefill.issubset(members);prefill_bytes=sum(members[n].compress_size for n in prefill)
  entries.append(dict(index=sample['index'],archive=str(archive),archive_SHA256=sha(archive),archive_bytes=sample['archive_bytes'],prompt_length=sample['prompt_length'],forwards=sample['actual_native_forwards'],common_bytes=sample['archive_bytes']-prefill_bytes,all_ZIP_members=len(members)))
 front=[json.loads(x) for x in Path(proto['own_frontend200']).read_text().splitlines()];assert len(front)==200
 lengths=[x['inputs']['input_ids']['shape'][1] for x in front];base=max(e['common_bytes'] for e in entries);min_n=min(e['prompt_length'] for e in entries);full3=sum(e['archive_bytes'] for e in entries);forecast=[math.ceil(base*max(1.,(n/min_n)**2)) for n in lengths[3:]];estimate=4*(2*full3+sum(forecast))+6*1024**3
 assert proto['hard_reserve_bytes']==3*1024**3 and proto['log_reserve_bytes']==1024**3 and proto['per_sample_lossless_space_estimate_bytes']==2*1024**3
 return dict(passed=True,UTC=now(),model_id=MID,true3_proof_SHA256=accepted['proof_sha256'],entries=entries,all200_own_original_input_shapes_checked=True,own_original_prompt_lengths=lengths,forecast197=forecast,empirical_estimate_bytes=estimate,actual_free_bytes=shutil.disk_usage(D).free,capacity_passed=shutil.disk_usage(D).free>=estimate,formula='4*(2*actual_full3_bytes+sum197(max_common*max(1,(own_prompt/min_observed_prompt)**2)))+6GiB',original_factor4_reserve6GiB_and_all_runtime_guards_preserved=True,empirical_not_worst_case_guarantee=True,no_storage_format_change=True,no_model_math_dtype_input_budget_or_retention_change=True,no_NN_or_completed3_replay=True)
def prepare():
 import batch_910b as batch
 from batch_910b_support import validate_config,validate_report,workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 assert not H.exists() and not J.exists();H.mkdir();assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 assert sha(RET/'actual_completion.json')=='af9e67591cfeac9a6ac3a220dcb89d8e0b775706019171e83eec0982613d036e';ret=js(RET/'actual_completion.json');assert ret['actual_returncode']==0 and ret['all_protected_SHA256_unchanged'] and ret['removed_files']==1138
 assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5'
 old=js(J3/'job.json');assert sha(J3/'job.json')=='0877bd7344fea19f82f73c83cb8228eeb031a4b4e632fd5ed0629b808a27b21e';done=js(J3/'completion.json');assert done['actual_returncode']==0;accepted,=done['accepted'];assert accepted['sample_count']==3 and sha(accepted['proof'])==accepted['proof_sha256']=='94f5649a4117467baae93dee177a5b794fc4d171b4ae8d962c9067bfc251f645'
 proof=js(accepted['proof']);assert proof['passed'] and proof['actual_forward_count']==475 and all(proof[k] for k in ['all2130_own_source_actual_NPU_and_unused_SHA256_verified','all54_unused_shared_KV_source_preserved','all_full_head_rows_independently_recomputed','all_native_KV_values_losslessly_reconstructed'])
 cfg=js(R/old['spec']['config']);validate_report(R,accepted['report_dir'],MID,cfg['task_type'],3,proof);unchanged({str(Path(accepted['report_dir'])/n):h for n,h in proof['report_sha256'].items()});batch.check_sources(old['spec']);assert batch.device_idle(old['spec'])
 expected={D/'state.json':'03320e9cc3ff5a912421d01a3dd73f9e354e61762e850619d46a1afb3a31bd9b',D/'registry.json':'3371e7fa50a09142cf0ecbf26d2637d029aa22ed2e310df51f265a09aa0177f2',BOOK:'9bd689a4a763bb716c6f9d4c3869aed95bb8d4b1ea9ede4b418591e3226a859c',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json':'3534a1dc1762fcb276d9c074d52bc47b04cf82459bb58424c2893804dfd301d1'}
 for p,h in expected.items():assert sha(p)==h,(str(p),sha(p),h)
 rows,sheet,bh=workbook_rows(BOOK);record,=[x for x in rows if x['model_id']==MID];assert record['score'] is None and record['task']==cfg['task_type']
 import xml.etree.ElementTree as ET
 with zipfile.ZipFile(BOOK) as z:
  root=ET.fromstring(z.read(sheet));ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};c=root.find('.//s:c[@r="D'+str(record['row'])+'"]',ns);assert Decimal(c.find('s:v',ns).text)==Decimal('.87')
 p=js(cfg['model']['kwargs']['protocol_file']);assert p['current_A100_D_values']=={'server':'0.8700','local':'0.8700','preserved_original':'0.87'}
 cap=capacity(accepted);put('capacity_review.json',cap);assert cap['capacity_passed'],cap
 frozen=H/'reference_workbook_verified179_before_Prithiv.xlsx';frozen.write_bytes(BOOK.read_bytes());assert sha(frozen)==bh
 put('authorized_static_transition_review.json',dict(UTC=now(),model_id=MID,old_true3_proof=accepted['proof'],old_true3_proof_SHA256=accepted['proof_sha256'],old_completed3_not_replayed=True,authorized_retention_receipt=str(RET/'actual_completion.json'),authorized_retention_receipt_SHA256=sha(RET/'actual_completion.json'),current_verified179_book_SHA256=bh,current_server_and_verified_local_D='.8700',preserved_original_D='.87',current_local_book_verified_SHA256=bh,original_job_original_proof_and_all_five_unchanged=True,only_new_static_reference_capacity_and_approval_bindings=True,no_model_math_dtype_prompt_budget_EOS_cache_or_device_change=True))
 approval=dict(approved=True,UTC=now(),model_id=MID,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'capacity_review.json'),capacity_review_SHA256=sha(H/'capacity_review.json'),actual_free_bytes=shutil.disk_usage(D).free,empirical_space_estimate_bytes=cap['empirical_estimate_bytes'],empirical_not_worst_case_guarantee=True,original_hard_log_and_per_sample_guards_preserved=True,small3_relative_error_to_complete200_D=proof['relative_error_to_A100_table_D'],first200_is_measurement_not_predetermined_5pct_qualification=True,no_new_small3_or10_5pct_gate=True)
 put('first200_measurement_approval.json',approval)
 before=copy.deepcopy(p);p['static_evidence_SHA256'].update({str(H/n):sha(H/n) for n in ['capacity_review.json','authorized_static_transition_review.json','first200_measurement_approval.json']});p['reference_workbook']=str(frozen);p['reference_workbook_SHA256']=sha(frozen);p['implementation_source_SHA256'][str(Path(__file__).resolve().relative_to(R))]=sha(__file__)
 runtime=lambda x:{k:v for k,v in x.items() if k not in ['static_evidence_SHA256','reference_workbook','reference_workbook_SHA256','implementation_source_SHA256']}
 assert runtime(p)==runtime(before)
 pp=R/'configs/protocols/Prithiv_own_original_BF16_card2_approved_first200_after_verified179_user_retention_20261117.json';create(pp,p)
 cf=R/'configs/Gemma4ForConditionalGeneration/prithivMLmods--gemma-4-E4B-it-Uncensored-MAX-original-BF16-2048-card2-approved-first200-user-retention-20261117.yaml';cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cf,cfg);validate_config(cfg,MID)
 new=copy.deepcopy(old);new.pop('parallel_stage_type',None);new['stages']=[3,200];new['created_at']=now();new['private_stage_type']='Original sole real runner EX; accepted card2 true3 then new fresh3/fixed200 after user authorized storage recovery'
 spec=new['spec'];spec.update(approved=True,review_revision='Accepted own card2 true3; immutable current179 reference and original format capacity recovery only',config=str(cf.relative_to(R)),protocol_file=str(pp),protocol_sha256=sha(pp),alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'))
 spec['source_sha256'].update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp),str(Path(__file__).resolve().relative_to(R)):sha(__file__)})
 spec.pop('parallel_storage_budget',None);state=js(D/'state.json');assert state['current'] is None
 for n in [3,200]:path,effective=batch.config_for(spec,copy.deepcopy(state['models'][MID]),n);validate_config(effective,MID);spec['source_sha256'][str(path.relative_to(R))]=sha(path)
 baseline={str(D/n):sha(D/n) for n in ['state.json','registry.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json']};baseline[str(BOOK)]=sha(BOOK);new['protected_canonical_hashes']=baseline;new['runner_source_sha256'][str(Path(__file__).resolve().relative_to(R))]=sha(__file__)
 batch.check_sources(spec);scope(spec,[3,200],False);unchanged(baseline);assert batch.device_idle(spec) and shutil.disk_usage(D).free>=cap['empirical_estimate_bytes'];J.mkdir();create(J/'job.json',new)
 put('prepared.json',dict(UTC=now(),passed=True,job=str(J),job_SHA256=sha(J/'job.json'),approved_fresh3_then_fixed200=True,completed_original3_not_replayed=True,card=2,capacity_review=cap,new_config=str(cf.relative_to(R)),new_protocol=str(pp.relative_to(R)),source_SHA256=sha(__file__),no_G_or_canonical_write=True));print(json.dumps(js(H/'prepared.json')),flush=True)
def run():
 from A100_alignment_evidence_runner_20261103 import unchanged
 assert js(H/'prepared.json')['passed'] and not (J/'state.json').exists() and not (H/'actual_completion.json').exists()
 guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);job=js(J/'job.json');unchanged(job['protected_canonical_hashes'])
 with (H/'original_runner_stdout.log').open('xb') as out,(H/'original_runner_stderr.log').open('xb') as err:
  child=subprocess.Popen([sys.executable,'-B',str(RUNNER),str(J)],cwd=R,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_then_fixed200_original_runner',child_PID=child.pid,job=str(J),job_SHA256=sha(J/'job.json'),card=2,no_G_or_canonical_write=True));rc=child.wait()
 completion=js(J/'completion.json');put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_formal_job_completion=completion,no_G_or_canonical_write=True));assert rc==completion['actual_returncode']==0
 accepted=completion['accepted'][-1];assert accepted['sample_count']==200;proof=js(accepted['proof']);put('completed200_pending_dual_publication_or_outside_review.json',dict(UTC=now(),model_id=MID,actual_NPU200_completion=accepted,independent_full_value_passed=proof['passed'],qualified_for_publication=proof['formal_alignment_within_5pct'] and proof['original_protocol_final_qualification_complete'],no_G_write_by_this_controller=True,next_action='Publish under original global EX/CAS after all NN/NF/locks released if all three D pass, otherwise preserve blank G and continue other eligible private stages'))
if __name__=='__main__':
 try:
  assert len(sys.argv)==2 and sys.argv[1] in ['prepare','run'];prepare() if sys.argv[1]=='prepare' else run()
 except BaseException:
  if H.exists():put('preparation_or_continuation_failure.json',dict(UTC=now(),error=traceback.format_exc(),all_failure_evidence_retained=True,no_G_or_canonical_write=True))
  raise
