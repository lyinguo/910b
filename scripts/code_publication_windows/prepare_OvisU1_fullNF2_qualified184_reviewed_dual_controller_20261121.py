"""Freeze the already authored one-cell proposal into an actual conditional publication consumer."""
from pathlib import Path,PurePosixPath
import json,hashlib,base64,ast
V=Path(__file__).resolve().parent;R=PurePosixPath('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sha=lambda b:hashlib.sha256(b).hexdigest()
H=V/'OvisU1_local_result_sync_verified184_20261121';plan=json.loads((H/'plan.json').read_bytes());review=json.loads((H/'actual_PNG_visual_review.json').read_bytes())
names={'continue_OvisU1_fullNF2_qualified184_reviewed_dual_publication_20261121.py':'scripts','continue_OvisU1_reviewed_exact_local_CAS_and_receipt_20261121.py':'scripts/windows_910b_20261121'}
source={folder+'/'+n:sha((V/n).read_bytes()) for n,folder in names.items()}
for n,folder in {'publish_OvisU1_genuine200_fullNF2_verified184_single_original_lock_dual_transaction_20261121.py':'scripts','verified_OvisU1_genuine200_formal_NF2_original_raw_fallback_G_writer_20261121.py':'scripts','test_OvisU1_fullNF2_publication_metadata_rejections_20261121.py':'scripts/tests','OvisU1_NF2_verified184_writer_11_cases_template_20261121.py':'scripts/tests'}.items():source[folder+'/'+n]=sha((V/n).read_bytes())
source.update({'scripts/verify_ovisu1_completed200_original_raw_prediction_fallback_metadata_v2_20261121.py':'60c0d21fd51f9ad8e712890330e9b7c887d72e8154ace6a14b11ab5f6fc20052','run.py':'9f902cea04a501fa535f7b30c3388ee839a9ad8d99621fbcdd77f00ac5215a51','universal_eval/orchestrator/pipeline.py':'0de40c9e1da2b3cce5ba5da79757de9ae3a938746b3370db94bfe8f8f174e08b'})
q=dict(model_id='AIDC-AI/Ovis-U1-3B',row=144,score=.895,expected_before_sha256=plan['expected_before_sha256'],home=str(B/'OvisU1_fullNF2_qualified184_reviewed_dual_controller_20261121'),publisher='scripts/publish_OvisU1_genuine200_fullNF2_verified184_single_original_lock_dual_transaction_20261121.py',source_sha256=source,local_plan=plan,artifact_tool_review=review,no_model_or_original_NF_source_change=True,only_after_actual_fullNF2_original_qualification_and_three_D_5pct=True)
config='configs/publication/OvisU1_fullNF2_qualified184_reviewed_dual_publication_20261121.json';(V/Path(config).name).write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8');names[Path(config).name]='configs/publication'
records=[dict(path=folder+'/'+n,SHA256=sha((V/n).read_bytes()),base64=base64.b64encode((V/n).read_bytes()).decode()) for n,folder in names.items()]
s='DATA='+repr(records)+'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast,os,subprocess,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for v in DATA:
 p=R/v['path'];b=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==v['SHA256'] and not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.py':ast.parse(b)
 with p.open('xb') as f:f.write(b)
q=json.loads((R/'configs/publication/OvisU1_fullNF2_qualified184_reviewed_dual_publication_20261121.json').read_bytes());assert all(sha(R/p)==h for p,h in q['source_sha256'].items());H=Path(q['home']);assert H.is_absolute() and H.is_relative_to(R) and not H.exists()
launch=D/'OvisU1_fullNF2_conditional_dual_controller_launch_20261121';assert not launch.exists();launch.mkdir()
with (launch/'stdout.log').open('xb') as out,(launch/'stderr.log').open('xb') as err:
 c=subprocess.Popen(['python3','-B',str(R/'scripts/continue_OvisU1_fullNF2_qualified184_reviewed_dual_publication_20261121.py')],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
time.sleep(1);assert c.poll() is None
print(json.dumps(dict(passed=True,PID=c.pid,home=str(H),source_and_config_SHA256={v['path']:v['SHA256'] for v in DATA},state=json.loads((H/'state.json').read_bytes()),no_G_before_complete_NF2_and_true_original_EX=True)))
'''
ast.parse(s);(V/'stage_OvisU1_fullNF2_conditional_dual_publication_controller_20261121_remote.py').write_text(s,encoding='utf8',newline='\n')
print(json.dumps(dict(passed=True,config_SHA256=sha((V/Path(config).name).read_bytes()),all_remote_paths_POSIX=True,actual_before_after_PNG_review_bound=True)))
