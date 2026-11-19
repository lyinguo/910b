"""New private recovery source only; never edit/signal the live GLM or queue v1."""
from pathlib import Path
import ast,base64,hashlib,json,re
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
PREP='prepare_autoglm_phone9b_own_original_BF16_2048_true3_v2_fixed_input_set_20261119.py'
NF='verify_autoglm_phone9b_own_original_BF16_2048_native_v2_fixed_input_set_20261119.py'
SELF='watch_GLM46_then_AutoGLM_preNN_fixed_input_set_recovery_serial_v2_20261119.py'
CFG='after_GLM46_AutoGLM_v1_preNN_config_refusal_own_fixed_input_set_serial_v2_20261119.json'
OLDREADY='AutoGLMPhone9B_own_original_BF16_2048_native_source_readiness_v1_20261119';READY='AutoGLMPhone9B_own_original_BF16_2048_native_source_readiness_v2_fixed_input_set_20261119'
OLDJ='AutoGLMPhone9B_own_original_BF16_2048_true3_v1_real_FLOCK_20261119';J='AutoGLMPhone9B_own_original_BF16_2048_true3_v2_fixed_input_set_real_FLOCK_20261119'
OLDQUEUE='after_GLM46_actual200_qualified_dual_commit_or_preserved_gap_AutoGLMPhone_serial_20261119';QUEUE='after_GLM46_AutoGLM_v1_preNN_config_refusal_own_fixed_input_set_serial_v2_20261119'
ALIAS='AutoGLMPhone9B_own_original_AFS_R_identical_fixed200_no_filter_input_alias_no_NN_20261119/own_fixed200_no_filter_input_alias_proof.json'
items=[]
def add(name,relative,s):
 raw=s.encode();ast.parse(raw);(P/name).write_bytes(raw);items.append(dict(relative=relative,SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
s=(P/'verify_autoglm_phone9b_own_original_BF16_2048_20261119_native.py').read_text(encoding='utf8');assert "and cfg['data']['filter_empty']" in s;s=s.replace("and cfg['data']['filter_empty']","and not cfg['data']['filter_empty']");add(NF,'scripts/'+NF,s)
s=(P/'prepare_autoglm_phone9b_own_original_BF16_2048_20261119_true3.py').read_text(encoding='utf8');s=s.replace(OLDREADY,READY).replace(OLDJ,J).replace('scripts/verify_autoglm_phone9b_own_original_BF16_2048_20261119_native.py','scripts/'+NF).replace('scripts/prepare_autoglm_phone9b_own_original_BF16_2048_20261119_true3.py','scripts/'+PREP).replace('native-v1-20261119.yaml','native-v2-fixed-input-set-20261119.yaml').replace('2048_20261119_native_v1.json','2048_20261119_native_v2_fixed_input_set.json').replace('filter_empty=True,download=False','filter_empty=False,download=False')
# Complete own source manifest from v1's pre-NN metadata work is reused only
# after its exact validator refusal, all full own SF hashes, and producer SHA.
a=s.index(' headers=js(');b=s.index(" create(H/'own704_active_source_manifest.json'",a)
s=s[:a]+" headers=js(ready['own_source_headers']);oldready=B/'"+OLDREADY+"';failure=js(oldready/'actual_failure.json');assert failure['actual_NEW_model_forwards']==0 and 'Do not silently filter the fixed sample set' in failure['error']\n assert sha(R/'scripts/prepare_autoglm_phone9b_own_original_BF16_2048_20261119_true3.py')=='e67b91da18c30aae5a2df4487055c8a9e2ef233b9ebc4d7c1432d3c039ba9d75'\n manifest=js(oldready/'own704_active_source_manifest.json');assert set(manifest)==set(headers) and len(manifest)==704\n for filename,record in ready['own_weight_files'].items():assert sha(W/filename)==record['SHA256'] and (W/filename).stat().st_size==record['bytes']\n for name,v in manifest.items():assert v['source_key']==v['runtime_name']==name and v['source_file']==headers[name]['source_file'] and v['shape']==headers[name]['shape'] and v['source_dtype']=='torch.bfloat16' and v['source_SHA256']==v['cast_SHA256']\n alias_path=B/'"+ALIAS+"';alias=js(alias_path);assert alias['passed'] and alias['all200_original_fixed_input_order_prompt_ref_and_full_image_bytes_equal'] and alias['original_no_rows_removed_or_added_or_selected'] and alias['existing_own_frontend200_SHA256']==ready['own_frontend200_SHA256']\n"+s[b:]
s=s.replace("media_path=row['media_path'],media_file_SHA256=row['image_SHA256']","media_path=alias['rows'][row['index']]['actual_target_media_path'],own_original_AFS_media_path=row['media_path'],media_file_SHA256=row['image_SHA256']")
s=s.replace("dataset_path=ready['dataset'],dataset_SHA256=ready['dataset_SHA256']","dataset_path=alias['actual_target_dataset'],dataset_SHA256=alias['original_and_target_dataset_SHA256']")
s=s.replace("[OLD/'review.json',Path(ready['own_frontend200']),original,preserved,current]","[OLD/'review.json',Path(ready['own_frontend200']),original,preserved,current,alias_path,oldready/'actual_failure.json',oldready/'own704_active_source_manifest.json']")
s=s.replace("'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py',","'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py','scripts/prepare_autoglm_phone9b_own_AFS_R_identical_fixed200_no_filter_input_alias_20261119.py',")
s=s.replace('Historical A100 engine/tensor bytes were not captured.','Own original AFS and R task files and all200 actual image bytes/prompt/ref order are independently identical; original filter_empty=True removes zero of all4329, runtime filter_empty=False keeps the same fixed0:199 and satisfies the frozen runner. No pixel/token control or model NN replay. Historical A100 engine/tensor bytes were not captured.')
assert 'filter_empty=True,download' not in s and 'alias_path' in s and 'oldready' in s;add(PREP,'scripts/'+PREP,s)
old=(P/'watch_GLM46_actual_completed200_then_qualified_dual_commit_or_preserved_gap_start_AutoGLMPhone_serial_20261119.py').read_text(encoding='utf8');s=old.replace('scripts/watch_GLM46_actual_completed200_then_qualified_dual_commit_or_preserved_gap_start_AutoGLMPhone_serial_20261119.py','scripts/'+SELF).replace('configs/queues/'+OLDQUEUE+'.json','configs/queues/'+CFG).replace(OLDQUEUE,QUEUE).replace(OLDREADY,READY).replace(OLDJ,J).replace("original['data']['filter_empty']","not original['data']['filter_empty']").replace('approved-first200-serial-v1-20261119.yaml','approved-first200-serial-v2-fixed-input-set-20261119.yaml').replace('approved_first200_serial_v1_20261119.json','approved_first200_serial_v2_fixed_input_set_20261119.json').replace('fixed200_serial_v1_real_FLOCK_20261119','fixed200_serial_v2_fixed_input_set_real_FLOCK_20261119')
t=ast.parse(s);main=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='main');main_text=ast.get_source_segment(s,main);tail=main_text[main_text.index(' validate_report(R,formal['):]
newmain="""def main():
 import batch_910b as batch
 from batch_910b_support import validate_report
 from A100_alignment_evidence_runner_20261103 import unchanged
 cfg=js(R/CFG);assert cfg['max_concurrent_models']==1 and cfg['known_v1_fixed_set_validation_failure_before_any_NN_is_only_recovery_scope'];assert sha(cfg['own_input_alias_proof'])==cfg['own_input_alias_proof_SHA256']
 unchanged({str(R/n):h for n,h in cfg['own_next_source_SHA256'].items()});assert not H.exists() and not READY.exists() and not J3.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=B/'OLD_QUEUE';oldready=B/'OLD_READY';oldj=B/'OLD_J'
 while True:
  if (old/'actual_completion.json').is_file():
   done=js(old/'actual_completion.json');assert done['actual_returncode']==1 and 'Own next readiness failed before NN' in done['error']
   failure=js(oldready/'actual_failure.json');assert failure['actual_NEW_model_forwards']==0 and 'Do not silently filter the fixed sample set' in failure['error'] and not oldj.exists()
   contract=js(old/'queue_contract.json');assert contract['source_SHA256']=='4dbd18711209b97e23fdf6d2c0b3ebe1d8f415bc0c2d4653bfd95e8dd75412af'
   prior=js(old/'preceding_GLM_actual_outcome_gate.json');assert prior['passed'] and prior['all_real_NN_NF_and_original_EX_released']
   if quiet():break
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_v1_natural_known_fixed_set_preNN_refusal_no_signal_no_hot_edit_no_NN',previous_queue=str(old),no_runner_EX_in_waiter=True,no_NN_or_NF_in_waiter=True));time.sleep(3)
 done=js(JGLM/'completion.json');formal,=[v for v in done['accepted'] if v['sample_count']==200];assert sha(formal['proof'])==formal['proof_sha256'];pf=js(formal['proof']);passed=actual_full200_qualification(pf)
 assert not oldj.exists() and quiet();put(H/'v1_preNN_refusal_exact_scope_review.json',dict(UTC=now(),passed=True,old_failure=str(oldready/'actual_failure.json'),old_failure_SHA256=sha(oldready/'actual_failure.json'),no_v1_model_job_or_NN=True,no_v1_source_job_config_or_signal_changes=True,own_input_alias_proof=cfg['own_input_alias_proof'],own_input_alias_proof_SHA256=cfg['own_input_alias_proof_SHA256']))
""".replace('OLD_QUEUE',OLDQUEUE).replace('OLD_READY',OLDREADY).replace('OLD_J',OLDJ)+tail
s=s[:main.col_offset+sum(len(x) for x in s.splitlines(keepends=True)[:main.lineno-1])]+newmain+'\n'+s[sum(len(x) for x in s.splitlines(keepends=True)[:main.end_lineno]):]
assert 'filter_empty=True' not in s and 'waiting_v1_natural_known' in s;add(SELF,'scripts/'+SELF,s)
cfg=json.loads((P/(OLDQUEUE+'.json')).read_bytes());cfg['own_next_readiness_preparer']='scripts/'+PREP;cfg['known_v1_fixed_set_validation_failure_before_any_NN_is_only_recovery_scope']=True;cfg['own_input_alias_proof']='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/'+ALIAS
alias_result=json.loads((P/'R_AutoGLM_own_AFS_R_identical_fixed200_input_alias_proof_no_NN_20261119.json').read_bytes());cfg['own_input_alias_proof_SHA256']=alias_result['proof_SHA256'];cfg['own_next_source_SHA256'].pop('scripts/prepare_autoglm_phone9b_own_original_BF16_2048_20261119_true3.py');cfg['own_next_source_SHA256'].pop('scripts/verify_autoglm_phone9b_own_original_BF16_2048_20261119_native.py')
cfg['own_next_source_SHA256'].update({x['relative']:x['SHA256'] for x in items if x['relative']!='scripts/'+SELF});cfg['own_next_source_SHA256']['scripts/prepare_autoglm_phone9b_own_AFS_R_identical_fixed200_no_filter_input_alias_20261119.py']='dd9d09b4de766d219cdf8bcb592c7888999610e8266beed95d2ab36752451e8c'
raw=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();(P/CFG).write_bytes(raw);items.append(dict(relative='configs/queues/'+CFG,SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
data=dict(items=items,source='scripts/'+SELF);loader='DATA='+repr(data)+'\n'+r'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
for v in DATA['items']:
 p=R/v['relative'];b=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==v['SHA256'] and not p.exists()
 if p.suffix=='.py':ast.parse(b)
for v in DATA['items']:
 p=R/v['relative'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v['base64'],validate=True))
L=B/'AutoGLM_known_preNN_fixed_input_set_recovery_serial_v2_launch_20261119';assert not L.exists();L.mkdir()
with (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',str(R/DATA['source'])],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
value=dict(PID=p.pid,source=DATA['source'],source_SHA256=hashlib.sha256((R/DATA['source']).read_bytes()).hexdigest(),no_old_queue_signal_or_hot_edit=True,no_current_GLM_change=True,new_sources=[{k:v for k,v in x.items() if k!='base64'} for x in DATA['items']]);(L/'launch.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');print(json.dumps(value))
'''
ast.parse(loader);(P/'stage_and_launch_AutoGLM_known_preNN_fixed_set_recovery_serial_v2_20261119_remote.py').write_text(loader,encoding='utf8');(P/'AutoGLM_fixed_input_set_recovery_v2_source_scope_20261119.json').write_text(json.dumps(dict(items=[{k:v for k,v in x.items() if k!='base64'} for x in items],no_live_GLM_or_queue_hot_edit_or_signal=True,old_preNN_validation_refusal_required_before_new_dispatch=True,all_own200_inputs_and_original_model_math_unchanged=True),indent=2)+'\n',encoding='utf8');print(json.dumps(dict(sources=[{k:v for k,v in x.items() if k!='base64'} for x in items],no_NN_or_source_change_yet=True)))
