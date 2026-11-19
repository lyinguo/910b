"""Build a new private same-math Huihui job using fully checked standard lossless storage."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
review=json.loads((P/'R_user_Huihui_complete_lossless_storage_capacity_after_cleanup_20261119.json').read_bytes())
assert review['all_complete_original_names_shapes_dtypes_every_raw_bit_equal'] and review['actual_NEW_model_forwards']==0
old_native=(P/'huihui_glm46flash_own_original_BF16_2048_20261118_native.py').read_text()
tree=ast.parse(old_native);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_npz');lines=old_native.splitlines(keepends=True)
native=''.join(lines[:f.lineno-1])+"""def save_npz(p,values):
 from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz
 return write_full_standard_npz(p,values)
"""+''.join(lines[f.end_lineno:])
newtree=ast.parse(native)
old_funcs={n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
new_funcs={n.name:ast.dump(n,include_attributes=False) for n in ast.walk(newtree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
assert old_funcs.keys()==new_funcs.keys() and [n for n in old_funcs if old_funcs[n]!=new_funcs[n]]==['save_npz']
NATIVE='universal_eval/model/adapters/huihui_glm46flash_own_original_BF16_2048_lossless_LZMA_serial_v3_20261119.py'
(P/Path(NATIVE).name).write_text(native,encoding='utf8',newline='\n')
controller=(P/'continue_huihui_completed3_serial_efficiency_wait_capacity_fresh3_fixed200_v2_1155_20261119.py').read_text()
reps={
'HuihuiGLM46_completed_true3_serial_efficiency_original_capacity_resumer_v2_1155_20261119':'HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v3_20261119',
'HuihuiGLM46_completed_true3_single_model_serial_approved_NEWfresh3_fixed200_v2_real_FLOCK_1155_20261119':'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v3_real_FLOCK_20261119',
'continue_huihui_completed3_serial_efficiency_wait_capacity_fresh3_fixed200_v2_1155_20261119.py':'continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v3_20261119.py',
'huihui-ai--Huihui-GLM-4.6V-Flash-abliterated-own-original-BF16-2048-serial-approved-first200-v2-1155-20261119.yaml':'huihui-ai--Huihui-GLM-4.6V-Flash-abliterated-own-original-BF16-2048-lossless-LZMA-serial-approved-first200-v3-20261119.yaml',
'huihui_glm46flash_own_original_BF16_2048_serial_approved_first200_v2_1155_20261119.json':'huihui_glm46flash_own_original_BF16_2048_lossless_LZMA_serial_first200_v3_20261119.json',
}
for a,b in reps.items():assert a in controller;controller=controller.replace(a,b)
needle="parent=js(J3/'job.json');spec=parent['spec'];import yaml"
transition="""parent=js(J3/'job.json');spec=parent['spec'];import yaml
  cleanup=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119';ledger=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
  assert sha(cleanup/'actual_completion.json')=='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3'
  cleanup_actual=js(cleanup/'actual_completion.json');assert cleanup_actual['actual_returncode']==0 and cleanup_actual['all_protected_SHA256_unchanged'] and cleanup_actual['removed_files']==4972
  assert parent['protected_canonical_hashes'][str(ledger)]=='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5' and sha(ledger)=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'
  unchanged({n:h for n,h in parent['protected_canonical_hashes'].items() if n!=str(ledger)})
  parent=copy.deepcopy(parent);parent['protected_canonical_hashes'][str(ledger)]=sha(ledger)
  put(H/'only_authorized_retention_ledger_transition.json',dict(original_true3_job_SHA256=sha(J3/'job.json'),old_ledger_SHA256='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5',new_ledger_SHA256=sha(ledger),cleanup_receipt=str(cleanup/'actual_completion.json'),cleanup_receipt_SHA256=sha(cleanup/'actual_completion.json'),all_other_protected_canonical_SHA_identical=True,original_job_not_modified=True))
"""
assert controller.count(needle)==1;controller=controller.replace(needle,transition.rstrip())
needle="assert estimate==164003454340,'Original own complete storage estimate changed; preserve and review, never weaken guard'"
replacement="""assert estimate==164003454340,'Original own complete storage estimate changed; preserve and review, never weaken guard'
  storage=B/'HuihuiGLM46_completed_true3_complete_standard_Fortran_LZMA_capacity_after_verified4972_cleanup_20261119';storage_review=js(storage/'review.json');storage_done=js(storage/'actual_completion.json')
  assert storage_done['actual_returncode']==0 and storage_review['all_complete_original_names_shapes_dtypes_every_raw_bit_equal'] and storage_review['actual_NEW_model_forwards']==0 and storage_review['original_true3_proof_SHA256']==accepted['proof_sha256']
  assert sha(R/'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py')=='a858f771182c1e8240e8e94d158d6089a4c7ee421b7cae94ab4301c9fb2be485'
  converted=[]
  for name in storage_review['complete_per_array_value_reviews']:
   r=js(name);assert r['all_names_shapes_dtypes_and_every_raw_bit_preserved'] and r['no_alias_approximation_sampling_or_math_change'];assert sha(r['original_path'])==r['original_SHA256'] and sha(r['new_path'])==r['new_SHA256'];converted.append(r)
  assert sorted(r['index'] for r in converted)==[0,1,2] and sum(len(r['all_original_semantic_array_names']) for r in converted)==333
  new_total=sum(r['new_bytes']+Path(trace[r['index']]['actual_execution_file']).stat().st_size for r in converted)
  original_estimate=estimate;estimate=max(math.ceil(new_total*max(ratios.values())*4)+6*1024**3,int(4.5*1024**3)+20*1024**3)
  assert estimate==146869927228==storage_review['capacity']['empirical_estimate_bytes']
  cap.update(empirical_estimate_bytes=estimate,actual_free_bytes=shutil.disk_usage(R).free,passed=shutil.disk_usage(R).free>=estimate,original_format_empirical_estimate_bytes=original_estimate,only_standard_lossless_NPY_order_ZIP_codec_change=True,all333_names_shapes_dtypes_every_raw_bit_equal=True,storage_review=str(storage/'review.json'),storage_review_SHA256=sha(storage/'review.json'),original_factor4_plus6GiB_and_all_sample_guards_unchanged=True)
"""
assert controller.count(needle)==1;controller=controller.replace(needle,replacement.rstrip())
needle="p2=copy.deepcopy(p);p2['implementation_source_SHA256'].update"
replacement="p2=copy.deepcopy(p);p2['lossless_numeric_archive_encoding']=dict(format='standard NPY in standard ZIP-LZMA NPZ',NPY_order='Fortran for ndim>=2; all C-order logical raw bits preserved',complete_storage_review=str(storage/'review.json'),complete_storage_review_SHA256=sha(storage/'review.json'),all_original_names_shapes_dtypes_values_and_NF_checks_preserved=True);p2['implementation_source_SHA256'].update({NEW_MODEL:sha(R/NEW_MODEL),'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py':sha(R/'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py')});p2['static_evidence_SHA256'].update({str(storage/'review.json'):sha(storage/'review.json'),str(storage/'actual_completion.json'):sha(storage/'actual_completion.json'),str(cleanup/'actual_completion.json'):sha(cleanup/'actual_completion.json')});p2['implementation_source_SHA256'].update"
assert controller.count(needle)==1;controller=controller.replace(needle,replacement)
needle="c2=copy.deepcopy(cfg);c2['name']=Path(CFG).stem;"
replacement="c2=copy.deepcopy(cfg);c2['model']['adapter']=NEW_MODEL+':HuihuiGlm46FlashOwnNativeBF16';c2['name']=Path(CFG).stem;"
assert controller.count(needle)==1;controller=controller.replace(needle,replacement)
needle="s2=copy.deepcopy(spec);sources="
replacement="s2=copy.deepcopy(spec);sources="
assert needle in controller
needle="alignment_approval_review=str(H/'first200_measurement_approval.json')"
replacement="alignment_approval_review=str(H/'first200_measurement_approval.json')"
# New actual serializer and model adapter are explicit bound runtime dependencies.
needle="job=copy.deepcopy(parent);job.update"
replacement="s2['source_sha256'].update({NEW_MODEL:sha(R/NEW_MODEL),'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py':sha(R/'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py')})\n  job=copy.deepcopy(parent);job.update"
assert controller.count(needle)==1;controller=controller.replace(needle,replacement)
controller=controller.replace("LEGACY_BUDGET=235720910370","LEGACY_BUDGET=235720910370\nNEW_MODEL="+repr(NATIVE))
controller=controller.replace("env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',","env=dict(os.environ,PIP_NO_CACHE_DIR='1',UV_NO_CACHE='1',PYTHONDONTWRITEBYTECODE='1',")
ast.parse(controller);CTRL='scripts/continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v3_20261119.py';(P/Path(CTRL).name).write_text(controller,encoding='utf8',newline='\n')
(P/'Huihui_lossless_storage_same_model_math_AST_review_20261119.json').write_text(json.dumps(dict(passed=True,only_native_function_changed='save_npz',all_original_model_methods_AST_identical=True,original_weight_input_dtype_budget_thinking_EOS_cache_head_and_scoring_unchanged=True,old_native_SHA256=sha(old_native.encode()),new_native_SHA256=sha(native.encode()),original_completed_true3_not_replayed=True,new_stages=[3,200],first200_measurement_only=True),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sources=[]
for name,relative in [(Path(NATIVE).name,NATIVE),(Path(CTRL).name,CTRL)]:
 b=(P/name).read_bytes();sources.append(dict(relative=relative,SHA256=sha(b),base64=base64.b64encode(b).decode()))
data=dict(sources=sources,controller=CTRL,home='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Huihui_verified4972_lossless_LZMA_serial_v3_launch_20261119')
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,subprocess,os
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
for x in DATA['sources']:
 p=R/x['relative'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'];ast.parse(b);assert p.resolve().is_relative_to(R)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
H=Path(DATA['home']);H.mkdir(exist_ok=False)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PIP_NO_CACHE_DIR='1',UV_NO_CACHE='1',PYTHONUNBUFFERED='1')
with (H/'controller_stdout.log').open('xb') as out,(H/'controller_stderr.log').open('xb') as err:
 child=subprocess.Popen(['python3','-B',str(R/DATA['controller'])],cwd=str(R),stdout=out,stderr=err,start_new_session=True,env=env)
record=dict(controller_PID=child.pid,controller=DATA['controller'],primary_sources=[{k:v for k,v in row.items() if k!='base64'} for row in DATA['sources']],serial_only=True,new_true3_replay=False,new_stage_sequence='NEWfresh3 then fixed0:199200 original runner',not_preclaiming_actual_NN_started=True)
(H/'launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\\n',encoding='utf8');print(json.dumps(record,ensure_ascii=False))
'''
ast.parse(remote);(P/'deploy_launch_Huihui_verified4972_lossless_storage_serial_v3_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(sources=[{k:v for k,v in x.items() if k!='base64'} for x in sources],only_native_save_npz_AST_changed=True,new_jobs_private=True),ensure_ascii=False))
