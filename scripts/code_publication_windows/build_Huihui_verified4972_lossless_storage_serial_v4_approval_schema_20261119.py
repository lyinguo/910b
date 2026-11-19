from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
old=(P/'continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v3_20261119.py').read_text()
changes={
'HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v3_20261119':'HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v4_approval_schema_20261119',
'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v3_real_FLOCK_20261119':'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119',
'continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v3_20261119.py':'continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v4_approval_schema_20261119.py',
'huihui-ai--Huihui-GLM-4.6V-Flash-abliterated-own-original-BF16-2048-lossless-LZMA-serial-approved-first200-v3-20261119.yaml':'huihui-ai--Huihui-GLM-4.6V-Flash-abliterated-own-original-BF16-2048-lossless-LZMA-serial-approved-first200-v4-approval-schema-20261119.yaml',
'huihui_glm46flash_own_original_BF16_2048_lossless_LZMA_serial_first200_v3_20261119.json':'huihui_glm46flash_own_original_BF16_2048_lossless_LZMA_serial_first200_v4_approval_schema_20261119.json',
}
for a,b in changes.items():assert a in old;old=old.replace(a,b)
needle="approval=dict(UTC=now(),approved=True,"
replacement="approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,"
assert old.count(needle)==1;old=old.replace(needle,replacement)
needle="done=run(J200,'approved_fresh3_fixed200');"
replacement="assert approval['approved'] and approval['real_NPU3_independent_proof_verified'] and approval['numerical_review_verified'] and proof['passed'] and proof['NEW_real_NPU_inference'] and proof['sample_count']==3 and sha(proof_path)==accepted['proof_sha256']\n  done=run(J200,'approved_fresh3_fixed200');"
assert old.count(needle)==1;old=old.replace(needle,replacement)
ast.parse(old);name='continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v4_approval_schema_20261119.py';(P/name).write_text(old,encoding='utf8',newline='\n')
before=ast.parse((P/'continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v3_20261119.py').read_text());after=ast.parse(old)
review=dict(change='Only private paths and first200 approval schema backed by immutable accepted true3 proof; no native model or verifier changes',native_SHA256='1d8d860241d9665fc8c595ed16778925cdb8b447e17733e9f45f7f8c286b243f',verifier_unchanged=True,original3_not_replayed=True,failed_v3_had_no_original_runner_lock_stage_or_model_forward=True,required_approval_keys=['approved','real_NPU3_independent_proof_verified','numerical_review_verified'])
(P/'Huihui_v4_approval_schema_change_actual_review_20261119.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
b=(P/name).read_bytes();d=dict(relative='scripts/'+name,SHA256=sha(b),base64=base64.b64encode(b).decode(),home='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Huihui_verified4972_lossless_serial_v4_approval_schema_launch_20261119')
remote='DATA='+repr(d)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,subprocess,os
R=Path('/root/projects/comperror');p=R/DATA['relative'];b=base64.b64decode(DATA['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==DATA['SHA256'];ast.parse(b)
assert not p.exists();p.write_bytes(b);sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
H=Path(DATA['home']);H.mkdir(exist_ok=False)
with (H/'controller_stdout.log').open('xb') as out,(H/'controller_stderr.log').open('xb') as err:
 child=subprocess.Popen(['python3','-B',str(p)],cwd=str(R),stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PIP_NO_CACHE_DIR='1',UV_NO_CACHE='1',PYTHONUNBUFFERED='1'))
result=dict(controller_PID=child.pid,controller=DATA['relative'],controller_SHA256=DATA['SHA256'],private_new_approval_schema=True,no_original_runner_or_model_math_edits=True,not_preclaiming_model_started=True)
(H/'launch.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False))
'''
ast.parse(remote);(P/'deploy_launch_Huihui_verified4972_lossless_serial_v4_approval_schema_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps({k:v for k,v in d.items() if k!='base64'},ensure_ascii=False))
