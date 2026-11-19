import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;a=json.loads((P/'R_Huihui_completed200_full_numerical_gap_target_audit_20261119.json').read_bytes());ret=json.loads((P/'R_Huihui_exact_reviewed_correct_nonexample_retention_actual_20261119.json').read_bytes());actual=ret['actual']
assert a['passed'] and not a['publication_permitted'] and actual['actual_returncode']==0 and actual['removed_files']==105 and actual['all_protected_SHA256_unchanged']
assert actual['all61_incorrect_debug_control_cases_and_full_original_examples_retained'] and actual['canonical_scores_state_registry_binding_and_both_workbooks_unchanged']
name='continue_GLM46_own_passed3_after_Huihui_numerical_gap_verified_cleanup_serial_v2_20261119.py';rel='scripts/'+name;raw=(P/name).read_bytes();ast.parse(raw)
old=json.loads((P/'R_GLM46_serial_next_own_protocol_source_contract_20261119.json').read_bytes())['old_serial_config']
cfg=dict(max_concurrent_models=1,next_original_true3_replay_forbidden=True,parent_not_qualified_no_G_write=True,parent_measurement_audit_SHA256=a['audit_SHA256'],retention_receipt_SHA256=ret['actual_SHA256'],retention_manifest_SHA256=actual['manifest_SHA256'],retention_ledger_after_SHA256=ret['ledger_SHA256'],component_capacity_review_SHA256=old['component_capacity_review_SHA256'],latest_actual_user_authorization=old['latest_actual_user_authorization'],source_SHA256={rel:hashlib.sha256(raw).hexdigest(),'universal_eval/model/adapters/glm46flash_own_original_BF16_thinking4096_standard_LZMA_serial_v2_20261119.py':'1d2291dbbd38c2c3c465f5f21d206f807834e6ab8e954449f936bf5fe790800e','scripts/glm46_complete_shape_component_capacity_v3_actual_one_based_20261119.py':'2a86561bc32852de4dcc04179699dc8609eba4ce3c61217cda92e85ad9909301'})
cfgraw=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();(P/'GLM46_own_passed3_after_Huihui_numerical_gap_verified_cleanup_serial_v2_20261119.json').write_bytes(cfgraw)
payload=dict(source_relative=rel,source_SHA256=hashlib.sha256(raw).hexdigest(),source_BASE64=base64.b64encode(raw).decode(),config_relative='configs/queues/GLM46_own_passed3_after_Huihui_numerical_gap_verified_cleanup_serial_v2_20261119.json',config_SHA256=hashlib.sha256(cfgraw).hexdigest(),config_BASE64=base64.b64encode(cfgraw).decode())
remote='DATA='+repr(payload)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not project_processes() and not runner_lock_owners()
for kind in ['source','config']:
 p=R/DATA[kind+'_relative'];raw=base64.b64decode(DATA[kind+'_BASE64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA[kind+'_SHA256'] and p.resolve().is_relative_to(R) and not p.exists()
 if kind=='source':ast.parse(raw)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
I=B/'GLM46_after_Huihui_gap_verified105_cleanup_serial_v2_actual_launch_20261119';assert not I.exists();I.mkdir()
with (I/'controller_stdout.log').open('xb') as out,(I/'controller_stderr.log').open('xb') as err:
 child=subprocess.Popen(['python3','-B',str(R/DATA['source_relative'])],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,source=DATA['source_relative'],source_SHA256=DATA['source_SHA256'],config=DATA['config_relative'],config_SHA256=DATA['config_SHA256'],serial_only=True,reused_original_true3_not_replayed=True,parent_Huihui_numerically_accepted_but_outside5pct_no_G_write=True)
(I/'actual_launch.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n',encoding='utf8');print(json.dumps(v,ensure_ascii=False))
'''
ast.parse(remote);(P/'launch_GLM46_after_Huihui_gap_verified105_cleanup_serial_v2_20261119_remote.py').write_text(remote,encoding='utf8')
print(json.dumps(dict(source_SHA256=payload['source_SHA256'],config_SHA256=payload['config_SHA256'],ledger_after_SHA256=ret['ledger_SHA256'],ready=True),ensure_ascii=False))
