"""Create and scope-review new Bee storage-only implementation and serial consumer."""
from pathlib import Path
import ast,json,hashlib,base64,difflib
V=Path(__file__).resolve().parent;R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sha=lambda b:hashlib.sha256(b).hexdigest()
model='bee8bsft_own_native_bf16_4096_complete_LZMA_20261121.py';nf='verify_bee8bsft_own_native_bf16_4096_original_raw_fallback_v2_20261121.py';controller='continue_BeeSFT_existing_true3_full_LZMA_NF_capacity_after_Ovis_dual_serial_20261121.py'
old=(V/'bee8bsft_own_native_bf16_4096_20261116.py').read_text(encoding='utf8');tree=ast.parse(old);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_npz');lines=old.splitlines(keepends=True)
new=''.join(lines[:f.lineno-1])+"def save_npz(p,values):\n from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz\n return write_full_standard_npz(p,values)\n"+''.join(lines[f.end_lineno:]);ast.parse(new)
functions=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='save_npz'}
assert functions(ast.parse(old))==functions(ast.parse(new));(V/model).write_text(new,encoding='utf8',newline='\n')
oldnf=(V/'verify_bee8bsft_own_native_bf16_4096_20261116.py').read_text(encoding='utf8');assert oldnf.count("(choice or '')==outs[i]['prediction']")==1
newnf=oldnf.replace("(choice or '')==outs[i]['prediction']","(choice or raw)==outs[i]['prediction']");assert newnf.replace("(choice or raw)==outs[i]['prediction']","(choice or '')==outs[i]['prediction']")==oldnf;(V/nf).write_text(newnf,encoding='utf8',newline='\n')
source={
 'universal_eval/model/adapters/'+model:sha((V/model).read_bytes()),
 'scripts/'+nf:sha((V/nf).read_bytes()),
 'scripts/'+controller:sha((V/controller).read_bytes()),
 'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py':'a858f771182c1e8240e8e94d158d6089a4c7ee421b7cae94ab4301c9fb2be485',
 'universal_eval/model/adapters/bee8bsft_own_native_bf16_4096_20261116.py':sha((V/'bee8bsft_own_native_bf16_4096_20261116.py').read_bytes()),
 'scripts/verify_bee8bsft_own_native_bf16_4096_20261116.py':sha((V/'verify_bee8bsft_own_native_bf16_4096_20261116.py').read_bytes())}
config='configs/queues/BeeSFT_completed_true3_complete_LZMA_after_Ovis_dual_serial_20261121.json'
q=dict(model_id='Open-Bee/Bee-8B-SFT',max_concurrent_models=1,config=str(R/config),home=str(B/'BeeSFT_completed_true3_full_LZMA_NF_capacity_after_Ovis_verified184_dual_serial_20261121'),new_job=str(B/'BeeSFT_completed_true3_full_LZMA_approved_NEWfresh3_fixed200_after_Ovis_dual_serial_real_FLOCK_20261121'),previous_dual_publication_home=str(B/'OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121'),new_protocol='configs/protocols/bee8bsft_completed_true3_full_LZMA_after_Ovis_dual_approved_20261121.json',new_config='configs/BeeForConditionalGeneration/Open-Bee--Bee-8B-SFT-own-original-BF16-4096-complete-LZMA-after-Ovis-dual-serial-20261121.yaml',source_sha256=source)
(V/Path(config).name).write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
items={model:'universal_eval/model/adapters/'+model,nf:'scripts/'+nf,controller:'scripts/'+controller,Path(config).name:config}
records=[dict(path=rel,SHA256=sha((V/name).read_bytes()),base64=base64.b64encode((V/name).read_bytes()).decode()) for name,rel in items.items()]
transport='DATA='+repr(records)+'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,hashlib,base64,subprocess,os,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
before={str(p):sha(p) for p in [D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',R/'scripts/verify_ovisu1_completed200_original_raw_prediction_fallback_metadata_v2_20261121.py']}
for x in DATA:
 p=R/x['path'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'];assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.py':ast.parse(b)
 with p.open('xb') as f:f.write(b)
q=json.loads((R/'configs/queues/BeeSFT_completed_true3_complete_LZMA_after_Ovis_dual_serial_20261121.json').read_bytes());assert all(sha(R/p)==h for p,h in q['source_sha256'].items())
H=Path(q['home']);assert not H.exists();launch=D/'BeeSFT_after_Ovis_dual_serial_waiter_launch_20261121';assert not launch.exists();launch.mkdir()
with (launch/'stdout.log').open('xb') as out,(launch/'stderr.log').open('xb') as err:
 c=subprocess.Popen(['python3','-B',str(R/'scripts/continue_BeeSFT_existing_true3_full_LZMA_NF_capacity_after_Ovis_dual_serial_20261121.py')],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
time.sleep(1);assert c.poll() is None;assert all(sha(p)==h for p,h in before.items())
print(json.dumps(dict(passed=True,PID=c.pid,source_and_config_SHA256={x['path']:x['SHA256'] for x in DATA},home=str(H),initial_state=json.loads((H/'state.json').read_bytes()),no_model_NN_or_original_EX_by_waiter=True,only_storage_function_and_original_raw_fallback_expected_metadata_changed=True,no_original_source_or_live_Ovis_job_change=True)))
'''
ast.parse(transport);(V/'stage_BeeSFT_existing3_full_LZMA_after_Ovis_dual_serial_20261121_remote.py').write_text(transport,encoding='utf8',newline='\n')
print(json.dumps(dict(passed=True,model_nonstorage_AST_exact=True,verifier_exact_metadata_expression_only=True,source_SHA256=source,config_SHA256=sha((V/Path(config).name).read_bytes()),no_second_model_started=True)))
