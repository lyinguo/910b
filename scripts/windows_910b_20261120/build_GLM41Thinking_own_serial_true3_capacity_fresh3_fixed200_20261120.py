import ast,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent;P=Path('D:/code-lv/.tmp/batch_910b_20261030')
old=P/'continue_AutoGLMPhone9B_single_completed_true3_verified182_GLM_retention_first200_serial_v3_20261120.py';raw=old.read_text();tree=ast.parse(raw)
parts=[]
for name in ['now','sha','js','put','quiet','actual_full200_qualification','continue_own_passed3_full_shape_capacity_first200']:
 n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);parts.append(ast.get_source_segment(raw,n))
body='\n\n'.join(parts)
body=body.replace('AutoGLMPhone9BSingle_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120','GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120')
body=body.replace('zai-org--AutoGLM-Phone-9B-single-own-original-BF16-2048-LZMA-approved-first200-serial-v1-20261120.yaml','zai-org--GLM-4.1V-9B-Thinking-own-original-VLLM-BF16-8192-LZMA-approved-first200-serial-v1-20261120.yaml').replace('autoglm_phone9b_single_own_original_BF16_2048_LZMA_approved_first200_serial_v1_20261120.json','glm41thinking_own_original_VLLM_BF16_8192_LZMA_approved_first200_serial_v1_20261120.json')
body=body.replace('own_AutoGLM','own_GLM41Thinking').replace('Own AutoGLM genuinely','Own GLM4.1 Thinking genuinely')
header='''"""Original183 delivered, own native inputs, NEW3 then unchanged full-capacity serial200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,os,shutil,subprocess,time,traceback,zipfile
from decimal import Decimal
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
SELF='scripts/continue_GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py'
CFG='configs/queues/GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.json'
H=B/'GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120'
MID='zai-org/GLM-4.1V-9B-Thinking'
FRONT=B/'GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v3_actual_media_wrapper_20261120'
PREPARE='scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_true3_v1_20261120.py'
READY=B/'GLM41Thinking_own_original_VLLM_BF16_8192_native_source_readiness_v1_20261120'
J3=B/'GLM41Thinking_own_original_VLLM_BF16_8192_true3_v1_real_FLOCK_20261120'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
ROLES={'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}
'''
main=r'''
def source_processes(source):
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
   if str(source) in argv:rows.append(dict(PID=int(p.name),argv=argv))
  except FileNotFoundError:continue
 return rows
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import unchanged
 cfg=js(R/CFG);assert cfg['model_id']==MID and cfg['max_concurrent_models']==1 and cfg['original_max_tokens']==8192 and cfg['original_no_thinking_override']
 assert not H.exists() and not READY.exists() and not J3.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(RUNNER)==cfg['frozen_runner_SHA256'];unchanged({str(R/n):h for n,h in cfg['exact_source_SHA256'].items()})
 while not (FRONT/'actual_completion.json').is_file() or source_processes(R/cfg['frontend_prepare_source']):
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_actual_own_native_input_completion_no_model_NN_no_runner_EX',max_concurrent_models=1));time.sleep(5)
 outcome=js(FRONT/'actual_completion.json');assert outcome['actual_returncode']==0,outcome
 front_review=js(FRONT/'review.json');assert front_review['passed'] and front_review['all200_own_original_input_pixels_tokens_refs_verified'] and front_review['actual_original_no_thinking_override_preserved'] and front_review['max_new_tokens']==8192
 assert quiet();book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';assert sha(book)==cfg['verified183_book_SHA256']
 assert sha(R/PREPARE)==cfg['exact_source_SHA256'][PREPARE]
 with (H/'own_source_readiness_stdout.log').open('xb') as out,(H/'own_source_readiness_stderr.log').open('xb') as err:
  prep=subprocess.Popen(['python3','-B',str(R/PREPARE)],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='own704_source_and_independent_YAML_readiness_no_model_NN',source_readiness_PID=prep.pid));prc=prep.wait()
 assert prc==0,dict(prepare_returncode=prc,readiness_failure=js(READY/'actual_failure.json') if (READY/'actual_failure.json').is_file() else None)
 ready=js(READY/'readiness.json');assert ready['passed'] and ready['ready_for_unique_original_runner_true3'];job=js(J3/'job.json');assert sha(J3/'job.json')==ready['job_SHA256'] and job['stages']==[3]
 unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert quiet() and batch.device_idle(job['spec'])
 with (H/'own_true3_runner_stdout.log').open('xb') as out,(H/'own_true3_runner_stderr.log').open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J3)],cwd='/root',stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='own_genuine_NEW3_original_runner_active_not_claimed_passed',runner_PID=child.pid,job=str(J3),job_SHA256=sha(J3/'job.json'),max_concurrent_models=1,no_G_or_canonical_write=True));rc=child.wait()
 done=js(J3/'completion.json');assert rc==done['actual_returncode']==done['verifier_returncode']==0,done
 accepted,=done['accepted'];assert sha(accepted['proof'])==accepted['proof_sha256'] and js(accepted['proof'])['sample_count']==3;assert quiet()
 result=continue_own_passed3_full_shape_capacity_first200(cfg,job,accepted);put(H/'actual_completion.json',result);return result['actual_returncode']
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True,all_original_values_reports_sources_configs_logs_retained=True,chat_notification_not_claimed=True))
  raise
'''
text=header+'\n'+body+'\n'+main;ast.parse(text);p=V/'continue_GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py';p.write_text(text,encoding='utf8',newline='\n')
config=dict(model_id='zai-org/GLM-4.1V-9B-Thinking',max_concurrent_models=1,original_max_tokens=8192,original_no_thinking_override=True,frozen_runner_SHA256='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90',verified183_book_SHA256='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81',frontend_prepare_source='scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_wrapper_20261120.py',exact_source_SHA256={})
names={
 p.name:'scripts/'+p.name,
 'prepare_glm41thinking_own_original_VLLM_BF16_8192_true3_v1_20261120.py':'scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_true3_v1_20261120.py',
 config['frontend_prepare_source'].split('/')[-1]:config['frontend_prepare_source'],
 'glm41thinking_owned_tokenizer_config_20261120.py':'universal_eval/model/reference_sources/glm41thinking_owned_tokenizer_config_20261120.py',
 'glm41thinking_own_original_VLLM_BF16_8192_20261120_native_v2_actual_media.py':'universal_eval/model/adapters/glm41thinking_own_original_VLLM_BF16_8192_20261120_native_v2_actual_media.py',
 'glm41thinking_own_original_VLLM_BF16_8192_20261120_frontend_v2_actual_media.py':'universal_eval/model/adapters/glm41thinking_own_original_VLLM_BF16_8192_20261120_frontend_v2_actual_media.py',
 'glm41thinking_own_original_VLLM_BF16_8192_20261120_MC.py':'universal_eval/algorithm/glm41thinking_own_original_VLLM_BF16_8192_20261120_MC.py',
 'verify_glm41thinking_own_original_VLLM_BF16_8192_native_v2_actual_media_20261120.py':'scripts/verify_glm41thinking_own_original_VLLM_BF16_8192_native_v2_actual_media_20261120.py'}
for n,rel in names.items():config['exact_source_SHA256'][rel]=hashlib.sha256((V/n).read_bytes()).hexdigest()
c=V/'GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.json';c.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(source=str(p),source_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),actual_consumer_config=str(c),config_SHA256=hashlib.sha256(c.read_bytes()).hexdigest(),copied_capacity_axes_and_factor_and_reserves_unchanged=True,first200_not_predetermined_qualified=True)))
