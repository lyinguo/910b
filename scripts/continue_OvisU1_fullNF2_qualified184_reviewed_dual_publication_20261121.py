"""Wait for actual fullNF2; test and publish only the already reviewed own Ovis score."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,subprocess,time,copy,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
CONFIG='configs/publication/OvisU1_fullNF2_qualified184_reviewed_dual_publication_20261121.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def main():
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 from verified_OvisU1_genuine200_formal_NF2_original_raw_fallback_G_writer_20261121 import verified_evidence,J,NF
 q=js(R/CONFIG);H=Path(q['home']);assert not H.exists();H.mkdir()
 assert q['model_id']=='AIDC-AI/Ovis-U1-3B' and q['row']==144 and q['score']==.895 and q['expected_before_sha256']=='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81'
 for p,h in q['source_sha256'].items():assert sha(R/p)==h,p
 while not (NF/'actual_completion.json').is_file() or not quiet():
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_actual_Ovis_fullNF2_and_original_EX_release_no_NN_no_G',actual_NEW_model_forwards=0));time.sleep(10)
 assert js(NF/'actual_completion.json')['actual_returncode']==0,'Full NF2 failed; no publication permitted'
 a=js(NF/'formal_acceptance.json');pf=js(a['proof']);m=dict(job_completion=str(J/'completion.json'),formal_acceptance=str(NF/'formal_acceptance.json'),formal_acceptance_SHA256=sha(NF/'formal_acceptance.json'),proof=a['proof'],proof_sha256=a['proof_SHA256'],report_dir=pf['report_dir'],task='image-text-to-text',new_score=.895)
 verified_evidence(m)
 assert q['artifact_tool_review']['before_and_author_after_actually_viewed'] and q['artifact_tool_review']['only_expected_G144_numeric_value_visible']==.895
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)==q['expected_before_sha256']
 for p,h in q['source_sha256'].items():assert sha(R/p)==h,p
 with (H/'publication_metadata_tests_stdout.json').open('xb') as out,(H/'publication_metadata_tests_stderr.log').open('xb') as err:
  rc=subprocess.run(['python3','-B',str(R/'scripts/tests/test_OvisU1_fullNF2_publication_metadata_rejections_20261121.py')],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')).returncode
 assert rc==0,'Explicit publication metadata guard tests failed before any write'
 tests=js(H/'publication_metadata_tests_stdout.json');assert tests['passed'] and tests['negative_count']==20 and tests['raw_G_write_function_byte_identical']
 review=copy.deepcopy(q['local_plan']);change=review['changes'][0];assert change['cell']=='G144' and change['row']==144 and change['new_value']==.895 and change['model_id']==q['model_id']
 change.update(formal_proof=a['proof'],formal_proof_sha256=a['proof_SHA256'],report_dir=pf['report_dir'])
 review.update(complete_numeric_verifier_passed=True,formal_acceptance=m['formal_acceptance'],formal_acceptance_SHA256=m['formal_acceptance_SHA256'],models={q['model_id']:dict(job=str(J),job_SHA256=sha(J/'job.json'),formal_acceptance=m['formal_acceptance'],formal_acceptance_SHA256=m['formal_acceptance_SHA256'],fresh3_score=js(J/'completion.json')['accepted'][0]['score'],expected_forward_count=102400,expected_kernels=pf['NPU_profiler_kernel_events'],expected_empty=[],expected_invalid=[198,199],expected_truncated=list(range(200)))},test_template='scripts/tests/OvisU1_NF2_verified184_writer_11_cases_template_20261121.py',test_template_SHA256=q['source_sha256']['scripts/tests/OvisU1_NF2_verified184_writer_11_cases_template_20261121.py'])
 put(H/'reviewed_inputs.json',review);put(H/'actual_artifact_tool_before_after_visual_review.json',q['artifact_tool_review'])
 put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='actual_complete_NF2_qualified_and_metadata_tests_passed_starting_single_original_EX_dual_publication',actual_NEW_model_forwards=0,reviewed_inputs_SHA256=sha(H/'reviewed_inputs.json')))
 with (H/'publication_stdout.json').open('xb') as out,(H/'publication_stderr.log').open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(R/q['publisher']),str(H/'reviewed_inputs.json')],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'));put(H/'publisher_launch.json',dict(UTC=now(),PID=child.pid,source=q['publisher'],source_SHA256=sha(R/q['publisher']),reviewed_inputs_SHA256=sha(H/'reviewed_inputs.json')));rc=child.wait()
 put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_NEW_model_forwards=0,publisher=q['publisher'],reviewed_inputs_SHA256=sha(H/'reviewed_inputs.json'),original_Ovis_job_remains_failed=True));return rc
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  try:
   q=js(R/CONFIG);H=Path(q['home'])
   if H.exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_evidence_retained=True,actual_NEW_model_forwards=0))
  finally:raise
