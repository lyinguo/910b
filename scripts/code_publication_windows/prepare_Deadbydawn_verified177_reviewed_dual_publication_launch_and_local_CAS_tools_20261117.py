import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,base64,hashlib,ast
P=Path(__file__).parent
H=P/'DeadbydawnGemma4E2B_local_result_sync_verified177_20261117';Q=json.loads((H/'plan.json').read_bytes());e=json.loads((P/'R_Deadbydawn_v3_full_evidence_for_publication_final_20261117.json').read_bytes());proof=e['formal_proof'];mid=Q['changes'][0]['model_id']
Q.update(models={mid:dict(job=e['job'],controller=e['controller'],job_SHA256=e['job_SHA256'],true3_score=e['true3_proof']['metrics']['Accuracy'],true3_proof_SHA256=e['true3_proof_SHA256'],expected_forward_count=proof['actual_forward_count'],expected_kernels=proof['NPU_profiler_kernel_events'],expected_empty=proof['empty_indices'],expected_invalid=proof['invalid_indices'],expected_truncated=proof['truncated_indices'])},test_template='scripts/tests/verified_Gemma4E2B_manifest_task_writer_11_cases_template_20261116.py',test_template_SHA256='8e595053cedf0ff3fc46b1f1a76fde787ed268ac190bc037e7521882e048e318')
Q['artifact_tool_actual_authoring_evidence']={n:dict(path=str(H/n),SHA256=hashlib.sha256((H/n).read_bytes()).hexdigest()) for n in ['before_239.png','after_239.png','artifact_authoring_candidate.xlsx','artifact_authored_G_values.json','formula_scan.ndjson']}
qraw=json.dumps(Q,ensure_ascii=False,indent=2).encode();qh=hashlib.sha256(qraw).hexdigest();(H/'reviewed_server_publication_inputs.json').write_bytes(qraw)
pub='scripts/publish_DeadbydawnGemma4E2B_verified177_single_original_lock_dual_transaction_v2_finish_reference_scope_20261117.py';ph=hashlib.sha256((P/Path(pub).name).read_bytes()).hexdigest();assert ph=='cda1648ff5cd99a7945678eb31f764ce690f338696ef73b751f9bf6e23fc06c9'
supervisor='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'DeadbydawnGemma4E2B_verified177_publication_review_inputs_v1_20261117'
source=R/__PUB__;review=I/'reviewed_inputs.json';assert hashlib.sha256(source.read_bytes()).hexdigest()==__PH__ and hashlib.sha256(review.read_bytes()).hexdigest()==__QH__;assert not (I/'publication_actual_completion.json').exists();start=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout.json').open('xb') as out,(I/'publication_stderr.log').open('xb') as err:p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256=__PH__,reviewed_inputs_SHA256=__QH__,original_five_reports_and_model_sources_untouched=True);(I/'publication_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
'''.replace('__PUB__',repr(pub)).replace('__PH__',repr(ph)).replace('__QH__',repr(qh))
ast.parse(supervisor);sn='scripts/run_DeadbydawnGemma4E2B_verified177_dual_publication_actual_completion_v2_20261117.py';(P/Path(sn).name).write_bytes(supervisor.encode());sh=hashlib.sha256(supervisor.encode()).hexdigest()
launch='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,ast,json,hashlib,base64,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','independent_verifier']]
I=B/'DeadbydawnGemma4E2B_verified177_publication_review_inputs_v1_20261117';I.mkdir(exist_ok=False);raw=base64.b64decode(__Q__);assert hashlib.sha256(raw).hexdigest()==__QH__; (I/'reviewed_inputs.json').write_bytes(raw)
raw=base64.b64decode(__S__);assert hashlib.sha256(raw).hexdigest()==__SH__;ast.parse(raw);source=R/__SN__;assert not source.exists();source.write_bytes(raw)
with (I/'supervisor_stdout.log').open('xb') as out,(I/'supervisor_stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',str(source)],cwd=R,stdout=out,stderr=err,start_new_session=True,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(PID=p.pid,source=str(source),source_SHA256=__SH__,reviewed_inputs=str(I/'reviewed_inputs.json'),reviewed_inputs_SHA256=__QH__,publication_started_not_claimed_completed=True);(I/'background_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__Q__',repr(base64.b64encode(qraw).decode())).replace('__QH__',repr(qh)).replace('__S__',repr(base64.b64encode(supervisor.encode()).decode())).replace('__SH__',repr(sh)).replace('__SN__',repr(sn))
ast.parse(launch);(P/'launch_reviewed_Deadbydawn_verified177_single_original_lock_dual_publication_v2_20261117_remote.py').write_bytes(launch.encode())
# All local mutations below remain gated on the actual server receipt and exact
# ZIP/raw-sheet comparison. This generator only prepares their source files.
sync=(P/'sync_Coder_verified176_local_exact_zip_CAS_20261117.py').read_text(encoding='utf8').replace('CoderGemma4E4B','DeadbydawnGemma4E2B').replace('Coder_verified176','Deadbydawn_verified177').replace('verified176','verified177').replace('G212','G239').replace("['done']==176","['done']==177").replace('all_prior175_C_B_D_G_preserved','all_prior176_C_B_D_G_preserved')
ast.parse(sync);(P/'sync_Deadbydawn_verified177_local_exact_zip_CAS_20261117.py').write_bytes(sync.encode())
read=(P/'read_Coder_verified176_server_book_for_local_CAS_20261117_remote.py').read_text(encoding='utf8').replace('CoderGemma4E4B','DeadbydawnGemma4E2B').replace('verified176','verified177')
read=read.replace("['done']==176","['done']==177").replace('coder3101/gemma-4-E4B-it-qat-q4_0-unquantized-heretic','deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit').replace('publication_actual_completion_v1.json','publication_actual_completion.json').replace('publication_stderr_v1.log','publication_stderr.log')
ast.parse(read);(P/'read_Deadbydawn_verified177_server_book_for_local_CAS_20261117_remote.py').write_bytes(read.encode())
print(json.dumps(dict(reviewed_inputs_SHA256=qh,publisher_SHA256=ph,supervisor_SHA256=sh,all_tools_prepared_only_no_production_write=True)))
