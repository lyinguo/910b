"""Build an explicit strict-SSH deployment; no source lives in report folders."""
from pathlib import Path
import sys,json,hashlib,base64,ast
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
cfg={
 'latest_actual_user_authorization':{'UTC':'2026-11-17T07:26:18.947Z_or_later_current_user_turn','source':'Current direct user message in original910b conversation','text':'看看还有哪些已验收的中间矩阵和临时文件可以清理，之前要求的完整报告、源码、配置、日志和必要数值例证都要保留。腾出空间后继续跑Prithiv，能多卡并行就并行，不要一直停着。','original_consent_SHA256':'06eba6a16df1559793352231cc9527955f5f13d58eb5e96929a110695afa9ee3'},
 'preserve_all_five_all_records_sources_configs_snapshots_logs':True,
 'only_current_accepted_complete200':True,
 'keep_original_logical_indices':sorted(set(range(10))|set(range(90,110))|set(range(190,200))),
 'allowed_model_ids':['huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated','unsloth/Llama-3.2-11B-Vision-Instruct','OpenGVLab/HoVLE-HD','nvidia/EGM-8B','opendatalab/MinerU2.5-Pro-2604-1.2B','HuggingFaceM4/idefics2-8b','HuggingFaceM4/idefics2-8b-base','HuggingFaceM4/idefics2-8b-chatty','ibm-granite/granite-vision-4.1-4b','merve/paligemma_vqav2'],
 'max_remove_files':2500,'max_remove_bytes':128*1024**3,
 'temporary_files_policy':'No temporary/log/source/report file is approved by this configuration; only exact accepted nonexample NPZ paths can be removed.'}
files={
 'scripts/retain_verified179_accepted_current_nonexample_matrices_user_request_20261117.py':(P/'retain_verified179_accepted_current_nonexample_matrices_user_request_20261117.py').read_bytes(),
 'scripts/tests/test_verified179_retention_exact_group_scope_20261117.py':(P/'test_verified179_retention_exact_group_scope_20261117.py').read_bytes(),
 'configs/retention/verified179_accepted_nonexample_matrices_for_Prithiv_user_request_20261117.json':json.dumps(cfg,ensure_ascii=False,indent=2).encode()}
for n,b in files.items():
 if n.endswith('.py'):ast.parse(b)
bundle={n:dict(SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()) for n,b in files.items()}
(P/'deploy_verified179_retention_source_config_tests_discover_20261117_remote.py').write_text('import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport os,json,base64,hashlib,subprocess\nos.chdir("/root")\nR=Path("/root/projects/comperror")\nbundle='+repr(bundle)+'\nfor n,v in bundle.items():\n p=R/n;b=base64.b64decode(v["base64"]);assert hashlib.sha256(b).hexdigest()==v["SHA256"];p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b,("Existing new path changed",n)\n else:p.write_bytes(b)\n assert hashlib.sha256(p.read_bytes()).hexdigest()==v["SHA256"]\nprint(json.dumps({"source_deployment_SHA256":{n:v["SHA256"] for n,v in bundle.items()}}),flush=True)\nsubprocess.run(["python3","-B",str(R/"scripts/tests/test_verified179_retention_exact_group_scope_20261117.py")],check=True,cwd="/root")\nsubprocess.run(["python3","-B",str(R/"scripts/retain_verified179_accepted_current_nonexample_matrices_user_request_20261117.py"),"discover"],check=True,cwd="/root")\n',encoding='utf8',newline='\n')
p=P/'deploy_verified179_retention_source_config_tests_discover_20261117_remote.py'
s=p.read_text();s=s.replace('if p.exists():assert p.read_bytes()==b,("Existing new path changed",n)','if p.exists() and p.read_bytes()!=b:\n  assert n=="scripts/retain_verified179_accepted_current_nonexample_matrices_user_request_20261117.py" and hashlib.sha256(p.read_bytes()).hexdigest()=="6f715b5afc74b65bbd73b8d7f0c93025885e64092c2f73b01a35ae1f91b8d9a4"\n  backup=R/"workspace/diagnostics/batch_910b/retention_user_requested_20261117_source_preflight_failure";backup.mkdir(exist_ok=True);(backup/p.name).write_bytes(p.read_bytes());p.write_bytes(b)\n elif p.exists():assert p.read_bytes()==b,("Existing new path changed",n)');p.write_text(s,encoding='utf8',newline='\n')
print(json.dumps({n:v['SHA256'] for n,v in bundle.items()}))
