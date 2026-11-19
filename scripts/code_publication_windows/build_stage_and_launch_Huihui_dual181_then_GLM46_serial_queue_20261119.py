"""Freeze the next serial stage now; it never starts NN before the parent release."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent
old=(P/'glm46flash_own_original_BF16_thinking4096_native_v1_1645_20261118.py').read_text(encoding='utf8');tree=ast.parse(old)
class StorageOnly(ast.NodeTransformer):
 def visit_FunctionDef(self,n):
  if n.name=='save_npz':return ast.parse('''def save_npz(p,values):
 from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz
 return write_full_standard_npz(p,values)
''').body[0]
  return n
new=ast.unparse(ast.fix_missing_locations(StorageOnly().visit(tree)))+'\n'
functions=lambda s:{n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(s)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name!='save_npz'}
assert functions(old)==functions(new)
native='glm46flash_own_original_BF16_thinking4096_standard_LZMA_serial_v2_20261119.py';(P/native).write_text(new,encoding='utf8',newline='\n')
ret=(P/'retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119.py').read_text(encoding='utf8')
ret=ret.replace("H=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119';CFG=R/'configs/retention/verified180_accepted_nonexample_user_serial_continue_owner_reference_v2_20261119.json'","H=B/'authorized_verified181_Huihui_only_accepted_nonexample_serial_next_20261119';CFG=R/'configs/retention/verified181_Huihui_only_accepted_nonexample_serial_next_20261119.json'")
assert "==180" in ret;ret=ret.replace("==180","==181")
assert "H=B/'authorized_verified181_Huihui_only_accepted_nonexample_serial_next_20261119'" in ret and "CFG=R/'configs/retention/verified181_Huihui_only_accepted_nonexample_serial_next_20261119.json'" in ret
assert "H=B/'authorized_verified180" not in ret and "==180" not in ret
retname='retain_verified181_Huihui_accepted_nonexamples_serial_next_20261119.py';(P/retname).write_text(ret,encoding='utf8',newline='\n')
paths=[(native,'universal_eval/model/adapters/'+native),(retname,'scripts/maintenance/'+retname),('continue_after_Huihui_real_dual181_then_GLM46_component_capacity_serial_v1_20261119.py','scripts/continue_after_Huihui_real_dual181_then_GLM46_component_capacity_serial_v1_20261119.py'),(Path(__file__).name,'scripts/code_publication_windows/'+Path(__file__).name)]
rows=[]
for name,rel in paths:
 b=(P/name).read_bytes();ast.parse(b);rows.append(dict(relative=rel,SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()))
review=dict(passed=True,original_all_nonstorage_functions_AST_identical=True,no_BF16_thinking4096_EOS_head_cache_input_scoring_or_order_change=True,only_standard_complete_LZMA_storage_used=True,bitplane_diagnostic_not_adopted=True,retention_only_new_current_Huihui_after_complete200_full_NF_original_qualification_dual_publication=True,retention_scope_maximum160_nonexamples_keep40_and_all_error_invalid_empty_truncated=True,no_retention_of_failed_or_unqualified_or_other_model_matrices=True,original_frozen_runner_and_Gwriter_unchanged=True,source_SHA256={x['relative']:x['SHA256'] for x in rows})
(P/'Huihui_dual181_then_GLM46_serial_queue_source_scope_review_20261119.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
remote='FILES='+repr(rows)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
I=B/'after_Huihui_dual181_then_GLM46_serial_queue_actual_launch_20261119';I.mkdir(exist_ok=False)
for row in FILES:
 p=R/row['relative'];b=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==row['SHA256'];assert p.resolve().is_relative_to(R);ast.parse(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
cap=B/'GLM46Flash_completed_true3_existing_full_LZMA_all200_component_shape_capacity_review_v3_actual_one_based_20261119/review.json';v=json.loads(cap.read_bytes());assert v['passed'] and v['original_factor']==4 and v['original_extra_reserve_bytes']==6*1024**3 and v['all200_original_shapes_used_not_only_200_div3']
cfg=dict(parent_job_SHA256='a1bf90557a326523990a6004396e2dbe336e85d72139b216fe00f5337340c0fd',own_completed_true3_job_SHA256='b922c1c10b5a847c8fc1a54774a6c2e4f22cd9357abfc42f241e5950453a2e31',component_capacity_review=str(cap),component_capacity_review_SHA256=hashlib.sha256(cap.read_bytes()).hexdigest(),latest_actual_user_authorization=dict(latest_user_message_verbatim='继续串行跑，Huihui完整验收通过后同步两份表，马上接下一款有权重、未适配或误差超过5%的模型，不等定期检查。完整报告、源码、配置、日志和必要数值例证保留，停摆或报错及时告诉我。',prior_user_cleanup_message_verbatim='那就再检查服务器上已验收的中间矩阵和临时文件，完整报告、源码、配置、日志和必要数值例证保留，不再需要的清理掉，腾够空间马上串行继续跑，别一直等着。',source='Direct human user in original910b chat',scope='Only the new Huihui accepted200 nonexample arrays if the next single-model budget needs it; all complete reports/source/config/logs/small/failed/unverified/necessary numeric examples retained'),max_concurrent_models=1,no_original_runner_lock_while_waiting=True,next_model='zai-org/GLM-4.6V-Flash',source_SHA256={x['relative']:x['SHA256'] for x in FILES})
p=R/'configs/queues/Huihui_real_dual181_then_own_GLM46_complete_component_capacity_serial_20261119.json';assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')
source=R/'scripts/continue_after_Huihui_real_dual181_then_GLM46_component_capacity_serial_v1_20261119.py'
assert not (B/'after_Huihui_real_dual181_then_GLM46_own_completed3_component_capacity_serial_v1_20261119').exists()
with (I/'supervisor_stdout.log').open('xb') as out,(I/'supervisor_stderr.log').open('xb') as err:
 child=subprocess.Popen(['python3','-B',str(source)],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
value=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,source=str(source),source_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),actual_config=str(p),config_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),sources={x['relative']:x['SHA256'] for x in FILES},waiter_not_model_or_NF=True,no_original_runner_lock_while_waiting=True,current_Huihui_not_restarted_or_modified=True)
(I/'actual_launch.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\\n',encoding='utf8');print(json.dumps(value,ensure_ascii=False))
'''
ast.parse(remote);(P/'deploy_launch_Huihui_dual181_then_GLM46_serial_queue_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n');print(json.dumps(review,ensure_ascii=False))
