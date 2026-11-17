import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,hashlib,base64
P=Path(__file__).parent
old=(P/'verify_pali10pt448_own_preserved_vllm_full_native_true3_v4_named_worker_RPC_20261117.py').read_text(encoding='utf8')
line="root=Path(p['weight_path']);config=js(root/'config.json');text=config['text_config'];assert text['num_hidden_layers']==42 and text['hidden_size']==3584 and text['num_key_value_heads']==8 and text['head_dim']==256"
assert line in old
replace="""root=Path(p['weight_path']);config=js(root/'config.json');text=dict(config['text_config'])
 from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
 import inspect
 native_config_source=Path(inspect.getfile(Gemma2Config));check(native_config_source,'d81e5abd688436da573ae299a65598d01ea871a8874d0cabeb5149967855ca07')
 native_text=Gemma2Config(**text);default_was_implicit='head_dim' not in text
 text['head_dim']=native_text.head_dim
 assert text['num_hidden_layers']==42 and text['hidden_size']==3584 and text['num_key_value_heads']==8 and text['head_dim']==256
 q_name,=[n for n in headers if n.endswith('layers.0.self_attn.q_proj.weight')];k_name,=[n for n in headers if n.endswith('layers.0.self_attn.k_proj.weight')]
 assert headers[q_name]['shape']==[text['num_attention_heads']*native_text.head_dim,text['hidden_size']] and headers[k_name]['shape']==[text['num_key_value_heads']*native_text.head_dim,text['hidden_size']]
 head_dimension_resolution=dict(raw_JSON_head_dim_was_implicit=default_was_implicit,native_Gemma2Config_head_dim=native_text.head_dim,native_config_source=str(native_config_source),native_config_source_SHA256=sha(native_config_source),own_Q_K_source_shapes_independently_agree=True,no_model_config_file_or_math_parameter_modified=True)"""
s=old.replace(line,replace)
s=s.replace("base.update(numeric_review_passed=True,","base.update(native_default_head_dimension_resolution=head_dimension_resolution,numeric_review_passed=True,")
s=s.replace(" with contextlib.redirect_stdout(sys.stderr):value=verify(folder,count)\n print(json.dumps(value,ensure_ascii=False,allow_nan=False))", """ diagnostic=len(sys.argv)==4 and sys.argv[3]=='--existing-v4-report-diagnostic'
 assert len(sys.argv)==3 or diagnostic
 if diagnostic:assert folder==R/'workspace/reports/google_paligemma2-10b-pt-448/image-text-to-text/20261117_081350' and count==3
 with contextlib.redirect_stdout(sys.stderr):value=verify(folder,count)
 if diagnostic:value.update(passed=False,diagnostic_only=True,NEW_real_NPU_inference=False,historical_record_reuse=True,existing_real_NPU_full_value_review=True,original_failed_v4_job_or_report_not_promoted=True,new_source_bound_NPU_phase_still_required=True)
 print(json.dumps(value,ensure_ascii=False,allow_nan=False))""")
old_numeric=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='numeric');new_numeric=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='numeric');assert ast.dump(old_numeric,include_attributes=False)==ast.dump(new_numeric,include_attributes=False)
nf='scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v5_native_default_head_dimension_20261117.py'
ast.parse(s);(P/Path(nf).name).write_bytes(s.encode());h=hashlib.sha256(s.encode()).hexdigest()
supervisor='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Pali10Pt448_existing_complete_real3_v5_native_default_head_dimension_full_numeric_diagnostic_original_FLOCK_20261117';J=B/'Pali10Pt448_preserved_completed_compat3_B10_new_full_native_value_true3_v4_named_worker_RPC_real_FLOCK_20261117';REPORT=R/'workspace/reports/google_paligemma2-10b-pt-448/image-text-to-text/20261117_081350';NF=R/__NF__
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert sha(NF)==__SHA__;done=json.loads((J/'completion.json').read_bytes());assert done['actual_returncode']==done['verifier_returncode']==1 and done['model_returncode']==0
 assert {x.name for x in REPORT.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
 assert not runner_lock_owners() and not [x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','independent_verifier']]
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert len(runner_lock_owners())==1 and runner_lock_owners()[0]['PID']==os.getpid()
  protected=[D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'),J/'job.json',J/'completion.json']+list(REPORT.iterdir());before={str(p):sha(p) for p in protected};start=datetime.now(timezone.utc).isoformat()
  with (H/'verification_stdout.json').open('xb') as out,(H/'verification_stderr.log').open('xb') as err:
   child=subprocess.Popen(['python3','-B',str(NF),str(REPORT),'3','--existing-v4-report-diagnostic'],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=start,PID=os.getpid(),child_PID=child.pid,phase='independent_full_value_existing_real3_diagnostic',no_new_NPU=True,not_old_failed_job_acceptance=True));t0=time.monotonic();late=False
   while child.poll() is None:
    if time.monotonic()-t0>21600 and not late:late=True;put('timeout_review_required_no_signal.json',dict(UTC=datetime.now(timezone.utc).isoformat(),timeout_seconds=21600,no_signal_sent=True))
    time.sleep(2)
   rc=child.wait()
  assert before=={str(p):sha(p) for p in protected},'Protected old reports, job or canonical changed'
  result=json.loads((H/'verification_stdout.json').read_bytes()) if rc==0 else None
  if rc==0:assert result['numeric_review_passed'] and result['diagnostic_only'] and not result['passed'] and not result['NEW_real_NPU_inference'] and result['original_failed_v4_job_or_report_not_promoted']
  final=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=rc if not late else 1,underlying_verifier_returncode=rc,timeout_review_required=late,protected_original_SHA256=before,all_old_reports_job_and_canonical_SHA_unchanged=True,no_new_NPU_or_score_G=True,elapsed_seconds=time.monotonic()-t0,review_SHA256=sha(H/'verification_stdout.json'))
  put('actual_completion.json',final);print(json.dumps(final,ensure_ascii=False));assert rc==0 and not late
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:
  if not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_new_NPU=True))
  raise
'''.replace('__NF__',repr(nf)).replace('__SHA__',repr(h))
ast.parse(supervisor);sn='scripts/supervise_Pali10Pt448_existing_real3_v5_native_default_full_numeric_diagnostic_original_FLOCK_20261117.py';(P/Path(sn).name).write_bytes(supervisor.encode())
payload={n:dict(data=base64.b64encode(x.encode()).decode(),SHA256=hashlib.sha256(x.encode()).hexdigest()) for n,x in [(nf,s),(sn,supervisor)]}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,ast,json,base64,hashlib,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__;SN=__SN__
for n,x in FILES.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw);p=R/n
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
I=B/'Pali10Pt448_existing3_v5_native_default_dimension_full_numeric_diagnostic_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.log').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',SN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=SN,source_SHA256=FILES[SN]['SHA256'],existing_real_NPU3_diagnostic_only_not_new_NPU=True,original_failed_v4_job_not_promoted=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(payload)).replace('__SN__',repr(sn));ast.parse(remote);(P/'deploy_launch_Pali_v5_native_default_full_numeric_existing3_diagnostic_original_FLOCK_20261117_remote.py').write_bytes(remote.encode())
print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))
