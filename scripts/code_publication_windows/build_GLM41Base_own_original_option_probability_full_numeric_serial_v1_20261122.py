"""Stage explicit own-source files, verify real value-artifact tests, then serial dispatch."""
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
names={
 'universal_eval/utils/complete_module_values_tree_20261122.py':'complete_module_values_tree_20261122.py',
 'universal_eval/model/adapters/glm41base_own_original_option_inputs_v1_20261122.py':'glm41base_own_original_option_inputs_v1_20261122.py',
 'universal_eval/model/adapters/glm41base_own_original_option_logprob_BF16_native_v1_20261122.py':'glm41base_own_original_option_logprob_BF16_native_v1_20261122.py',
 'universal_eval/algorithm/glm41base_own_original_option_logprob_MC_v1_20261122.py':'glm41base_own_original_option_logprob_MC_v1_20261122.py',
 'scripts/verify_glm41base_own_original_option_logprob_BF16_complete_values_v1_20261122.py':'verify_glm41base_own_original_option_logprob_BF16_complete_values_v1_20261122.py',
 'scripts/prepare_GLM41Base_own_original_option_probability_true3_v1_20261122.py':'prepare_GLM41Base_own_original_option_probability_true3_v1_20261122.py',
 'scripts/continue_GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122.py':'continue_GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122.py',
 'scripts/tests/test_complete_module_values_tree_20261122.py':'test_complete_module_values_tree_20261122.py',
}
stage={n:(V/f).read_text(encoding='utf8').encode('utf8') for n,f in names.items()}
for n,b in stage.items():ast.parse(b)
qpath='configs/queues/GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122.json'
q=dict(model_id='zai-org/GLM-4.1V-9B-Base',max_concurrent_models=1,exact_source_SHA256={n:sha(b) for n,b in stage.items()},preserve_original_option_text_mean_logprob=True,no_original_math_precision_position_mask_head_or_scoring_change=True,complete_values_and_original_capacity_guards_unchanged=True,no_G_or_canonical_write=True)
stage[qpath]=json.dumps(q,ensure_ascii=False,indent=2).encode()
(V/Path(qpath).name).write_bytes(stage[qpath])
for f in ['read_GLM41Base_native_numeric_architecture_and_cleanup_exact_contracts_20261122_remote.py','read_actual_cleanup_policy_and_GLM41Base_native_layer_interfaces_20261122_remote.py']:
 b=(V/f).read_text(encoding='utf8').encode();ast.parse(b);stage['scripts/maintenance/'+f]=b
stage['scripts/code_publication_windows/'+Path(__file__).name]=Path(__file__).read_text(encoding='utf8').encode()
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');s=book.stat();h=sha(book.read_bytes());assert h=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
lr=dict(UTC=datetime.now(timezone.utc).isoformat(),path=str(book),SHA256=h,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,verified185_unchanged=True,no_write=True)
(V/'GLM41Base_own_original_probability_verified185_fresh_local_20261122.json').write_text(json.dumps(lr,ensure_ascii=False,indent=2),encoding='utf8')
local_binding='workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/GLM41Base_own_original_option_probability_local185_binding_20261122/fresh_local_workbook_SHA_stat.json'
stage[local_binding]=json.dumps(lr,ensure_ascii=False,indent=2).encode()
manifest={n:sha(b) for n,b in stage.items()};(V/'GLM41Base_own_probability_explicit_primary_stage_plan_v1_20261122.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
cont='scripts/continue_GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122.py'
carrier="""import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
P=json.loads(base64.b64decode(PAYLOAD_LITERAL))
for n,e in P.items():
 p=R/n;assert p.is_relative_to(R) and '..' not in Path(n).parts
 b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256']
 if p.suffix=='.py':ast.parse(b)
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b,('Existing source must not be overwritten',n)
 else:p.write_bytes(b)
H=B/'GLM41Base_own_original_probability_full_numeric_preflight_and_serial_launcher_v1_20261122';H.mkdir(exist_ok=False)
test=R/'scripts/tests/test_complete_module_values_tree_20261122.py'
env=dict(os.environ,PYTHONPATH=str(R),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
with (H/'full_value_artifact_tests_stdout.log').open('xb') as out,(H/'full_value_artifact_tests_stderr.log').open('xb') as err:
 c=subprocess.run(['python3','-B',str(test),'-v'],cwd=R,stdout=out,stderr=err,env=env)
x=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=c.returncode,test=str(test),source_SHA256=hashlib.sha256(test.read_bytes()).hexdigest(),no_learned_checkpoint_or_model_grade=True,no_model_NN=True)
(H/'real_artifact_test_completion.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
assert c.returncode==0,x
with (H/'controller_stdout.json').open('x') as out,(H/'controller_stderr.log').open('x') as err:
 p=subprocess.Popen(['python3','-B',str(R/CONTROLLER_LITERAL)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=env,start_new_session=True)
x=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=CONTROLLER_LITERAL,source_SHA256=P[CONTROLLER_LITERAL]['SHA256'],serial_max1=True,tests_actual_returncode=0,stage_SHA256={n:v['SHA256'] for n,v in P.items()},no_G_or_canonical_write=True,no_current_or_completed_Hancom_job_mutation=True)
(H/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(x,ensure_ascii=False))
""".replace('PAYLOAD_LITERAL',repr(base64.b64encode(json.dumps(payload).encode()).decode())).replace('CONTROLLER_LITERAL',repr(cont))
ast.parse(carrier);(V/'stage_test_launch_GLM41Base_own_probability_full_numeric_serial_v1_20261122_remote.py').write_text(carrier,encoding='utf8')
print(json.dumps(dict(explicit_primary_sources=manifest,carrier='stage_test_launch_GLM41Base_own_probability_full_numeric_serial_v1_20261122_remote.py',actual_forward_not_yet_started=True),ensure_ascii=False))
