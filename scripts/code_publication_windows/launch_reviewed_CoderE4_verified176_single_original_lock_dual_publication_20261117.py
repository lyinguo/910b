import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast
P=Path(__file__).parent;H=P/'CoderGemma4E4B_local_result_sync_verified176_20261117'
x=json.loads((P/'R_CoderE4_v2_current_full_evidence_for_publication_20261117.json').read_bytes());pf=x['formal_proof'];q=json.loads((H/'plan.json').read_bytes());MID='coder3101/gemma-4-E4B-it-qat-q4_0-unquantized-heretic'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert pf['passed'] and pf['sample_count']==200 and pf['formal_alignment_within_5pct'] and pf['original_protocol_final_qualification_complete'] and x['completion']['actual_returncode']==x['controller_completion']['actual_returncode']==0
assert x['job_SHA256']=='99691595c6659a198df86e33c965768e64378e5da33ea19839f2c59ead3e61f1' and not x['FLOCK']
assert q['changes'][0]['model_id']==MID and q['changes'][0]['cell']=='G212' and q['changes'][0]['formal_proof_sha256']==x['formal_proof_SHA256']
q['models']={MID:dict(job=x['job'],controller=x['controller'],job_SHA256=x['job_SHA256'],true3_score=x['true3_proof']['metrics']['Accuracy'],true3_proof_SHA256=x['approval']['true3_proof_SHA256'],expected_forward_count=pf['actual_forward_count'],expected_kernels=pf['NPU_profiler_kernel_events'],expected_empty=pf['empty_indices'],expected_invalid=pf['invalid_indices'],expected_truncated=pf['truncated_indices'])}
q['test_template']='scripts/tests/verified_Gemma4E2B_manifest_task_writer_11_cases_template_20261116.py';q['test_template_SHA256']='8e595053cedf0ff3fc46b1f1a76fde787ed268ac190bc037e7521882e048e318'
q['artifact_tool_actual_authoring_evidence']={n:dict(path=str(H/n),SHA256=sha(H/n)) for n in ['before_212.png','after_212.png','artifact_authoring_candidate.xlsx','artifact_authored_G_values.json','formula_scan.ndjson']}
f=H/'reviewed_inputs_after_full_numeric.json';assert not f.exists();f.write_bytes(json.dumps(q,ensure_ascii=False,indent=2).encode());V=dict(data=base64.b64encode(f.read_bytes()).decode(),SHA256=sha(f))
pn='scripts/publish_CoderGemma4E4B_verified176_single_lock_dual_transaction_v2_exact_loaded_source_reference_20261117.py';ph=sha(P/Path(pn).name);assert ph=='fbd2e4d61e21df0ab369066486772ff9ecb6197a160ca920c552be0b0d5dcad3'
I='CoderGemma4E4B_verified176_publication_review_inputs_v1_20261117'
sn='scripts/run_CoderGemma4E4B_verified176_dual_publication_actual_completion_v1_exact_LF_20261117.py'
s='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,hashlib,json,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/__I__
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=R/__PN__;review=I/'reviewed_inputs.json';assert sha(source)==__PH__ and sha(review)==__VH__
assert not (I/'publication_actual_completion_v1.json').exists();started=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout_v1.json').open('xb') as out,(I/'publication_stderr_v1.log').open('xb') as err:p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=started,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256=__PH__,reviewed_inputs_SHA256=__VH__,original_five_reports_and_model_sources_untouched=True);(I/'publication_actual_completion_v1.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
'''.replace('__I__',repr(I)).replace('__PN__',repr(pn)).replace('__PH__',repr(ph)).replace('__VH__',repr(V['SHA256']));ast.parse(s);(P/Path(sn).name).write_bytes(s.encode());S=dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,ast,base64,hashlib,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')];V=__V__;S=__S__;SN=__SN__
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
I=B/__I__;I.mkdir(exist_ok=False);b=base64.b64decode(V['data']);assert hashlib.sha256(b).hexdigest()==V['SHA256'];(I/'reviewed_inputs.json').write_bytes(b)
p=R/SN;assert not p.exists();b=base64.b64decode(S['data']);assert hashlib.sha256(b).hexdigest()==S['SHA256'];ast.parse(b);p.write_bytes(b)
with (I/'supervisor_stdout.log').open('xb') as out,(I/'supervisor_stderr.log').open('xb') as err:child=subprocess.Popen(['python3','-B',str(p)],cwd=R,stdout=out,stderr=err,start_new_session=True,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(PID=child.pid,source=SN,SHA256=S['SHA256'],reviewed_inputs=str(I/'reviewed_inputs.json'),reviewed_inputs_SHA256=V['SHA256'],publication_started_not_claimed_completed=True);(I/'background_launch.json').write_bytes(json.dumps(v).encode());print(json.dumps(v))
'''.replace('__V__',repr(V)).replace('__S__',repr(S)).replace('__SN__',repr(sn)).replace('__I__',repr(I));ast.parse(remote);(P/'launch_reviewed_Coder176_publication_exact_LF_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(reviewed_input_SHA256=V['SHA256'],publisher_SHA256=ph,supervisor_SHA256=S['SHA256'])))
