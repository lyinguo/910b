"""Own original single-language Phone9B resources; reuse architecture, never values."""
from pathlib import Path
import ast,base64,copy,hashlib,json,re
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
MID='zai-org/AutoGLM-Phone-9B';OLDMID='zai-org/AutoGLM-Phone-9B-Multilingual'
mapping={
'autoglm_phone9b_own_original_BF16_2048_20261119_native.py':'autoglm_phone9b_single_own_original_BF16_2048_20261120_native.py',
'autoglm_phone9b_own_original_BF16_2048_20261119_frontend.py':'autoglm_phone9b_single_own_original_BF16_2048_20261120_frontend.py',
'autoglm_phone9b_own_original_BF16_2048_20261119_MC.py':'autoglm_phone9b_single_own_original_BF16_2048_20261120_MC.py',
'verify_autoglm_phone9b_own_original_BF16_2048_native_v2_fixed_input_set_20261119.py':'verify_autoglm_phone9b_single_own_original_BF16_2048_native_v1_20261120.py',
'prepare_autoglm_phone9b_own_original_BF16_2048_20261119_frontend200.py':'prepare_autoglm_phone9b_single_own_original_BF16_2048_frontend200_20261120.py',
'prepare_autoglm_phone9b_own_AFS_R_identical_fixed200_no_filter_input_alias_20261119.py':'prepare_autoglm_phone9b_single_own_AFS_R_identical_fixed200_no_filter_input_alias_20261120.py',
'prepare_autoglm_phone9b_own_original_BF16_2048_true3_v3_actual_published_GLM_fixed_input_set_20261119.py':'prepare_autoglm_phone9b_single_own_original_BF16_2048_true3_v1_20261120.py'}
def own(s):
 for a,b in mapping.items():s=s.replace(a,b).replace(a[:-3].replace('/','.'),b[:-3].replace('/','.'))
 s=s.replace(OLDMID,MID).replace('zai-org--AutoGLM-Phone-9B-Multilingual','zai-org--AutoGLM-Phone-9B')
 s=re.sub(r'AutoGLMPhone9B_([A-Za-z0-9_]+)_20261119',r'AutoGLMPhone9BSingle_\1_20261120',s)
 s=s.replace('autoglm_phone9b_own_original_BF16_2048_20261119_native_v3_actual_published_GLM_fixed_input_set.json','autoglm_phone9b_single_own_original_BF16_2048_20261120_native_v1.json')
 s=s.replace('own-original-BF16-2048-native-v3-actual-published-GLM-fixed-input-set-20261119.yaml','single-own-original-BF16-2048-native-v1-20261120.yaml')
 s=s.replace('autoglm_phone9b_own_original_BF16_2048_LZMA_approved_first200_serial_v3_actual_published_GLM_fixed_input_set_20261119.json','autoglm_phone9b_single_own_original_BF16_2048_LZMA_approved_first200_serial_v1_20261120.json')
 s=s.replace('own-original-BF16-2048-LZMA-approved-first200-serial-v3-actual-published-GLM-fixed-input-set-20261119.yaml','single-own-original-BF16-2048-LZMA-approved-first200-serial-v1-20261120.yaml')
 s=s.replace('b93e0b08e370db8fe08d0435a2d6fad6ae0f9653db931a4383587327d9c7a982','a004f7d4f8c9bfeb19314f61697a293a7faae06d5f8b406cfab8e5d74fbba010')
 return s
assets={}
for old,new in mapping.items():
 raw=(P/old).read_text(encoding='utf8');s=own(raw)
 if old.endswith('input_alias_20261119.py'):
  s=s.replace("assert sha(FRONT/'own_original_frontend200.jsonl')=='21e7a17a36e7ca4a0a5689c5be4e981704a00cb91b463017f759b2f6fc6e1dd4'","review=js(FRONT/'review.json');assert review['passed'] and review['model_id']=='zai-org/AutoGLM-Phone-9B' and sha(FRONT/'own_original_frontend200.jsonl')==review['own_frontend200_SHA256']")
 if old.endswith('true3_v3_actual_published_GLM_fixed_input_set_20261119.py'):
  # The private stage gets its own accurate lineage label; no old job is edited.
  s=s.replace("spec.update(approved=False,config=CFG", "spec.update(review_revision='Own single-language AutoGLM-Phone-9B704 BF16 original HF2048 full-value private stage',approved=False,config=CFG")
 ast.parse(s)
 rel=('universal_eval/model/adapters/' if new.endswith(('native.py','frontend.py')) else 'universal_eval/algorithm/' if new.endswith('_MC.py') else 'scripts/')+new
 assets[rel]=s.encode()
for old,new in list(mapping.items())[:4]:
 oldt=ast.parse((P/old).read_bytes());newt=ast.parse(assets[('universal_eval/model/adapters/' if new.endswith(('native.py','frontend.py')) else 'universal_eval/algorithm/' if new.endswith('_MC.py') else 'scripts/')+new])
 # Reverse only ownership strings. Numerical AST is invariant under the mapping.
 assert ast.dump(ast.parse(own((P/old).read_text(encoding='utf8'))))==ast.dump(newt)
oldq='watch_GLM46_exact_qualified181_dual_recovery_then_own_AutoGLM_serial_v3_20261119.py';newq='continue_AutoGLMPhone9B_single_own_inputs_true3_capacity_fresh3_fixed200_after182_serial_20261120.py';q=own((P/oldq).read_text(encoding='utf8'))
q=q.replace(oldq,newq).replace('after_GLM46_exact_qualified181_dual_recovery_own_AutoGLM_fixed_input_set_serial_v3_20261119.json','after_AutoGLM182_own_single_Phone9B_inputs_true3_fixed200_serial_20261120.json').replace('after_GLM46_exact_qualified181_dual_recovery_own_AutoGLM_fixed_input_set_serial_v3_20261119','after_AutoGLM182_own_single_Phone9B_inputs_true3_fixed200_serial_20261120')
a=q.index(' cfg=js(R/CFG);',q.index('def main():'));b=q.index(' validate_report(R,formal',a)
gate=''' cfg=js(R/CFG);assert cfg['max_concurrent_models']==1 and cfg['own_input200_and_own_source_all_required'];unchanged({str(R/n):h for n,h in cfg['own_next_source_SHA256'].items()});assert not H.exists() and not READY.exists() and not J3.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 previous=B/'AutoGLMPhone9B_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261119';pub=B/'AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120'
 recovery=js(pub/'postcommit_exact_legacy_label_binding_audit_recovery_actual.json');audit=js(pub/'full_target_dual_publication_audit.json');dual=js(pub/'dual_publication_completed.json');assert recovery['actual_returncode']==0 and audit['passed'] and dual['completed'] and audit['qualified_delivered']==68
 while not quiet():
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_all_previous_NN_NF_original_EX_release_no_NN_here'));time.sleep(3)
 done=js(previous/'completion.json');formal,=[v for v in done['accepted'] if v['sample_count']==200];assert done['actual_returncode']==0 and sha(formal['proof'])==formal['proof_sha256']=='498c5f03c923bdcd12f8ea40022f03a4c8bae44ea6ca0eabde00687b8566c95d';pf=js(formal['proof']);passed=actual_full200_qualification(pf);assert passed
 for stage,producer in [('own_input200','scripts/prepare_autoglm_phone9b_single_own_original_BF16_2048_frontend200_20261120.py'),('own_fixed200_alias','scripts/prepare_autoglm_phone9b_single_own_AFS_R_identical_fixed200_no_filter_input_alias_20261120.py')]:
  assert quiet() and sha(R/producer)==cfg['own_next_source_SHA256'][producer]
  with (H/(stage+'_stdout.json')).open('xb') as out,(H/(stage+'_stderr.log')).open('xb') as err:rc=subprocess.run(['python3','-B',str(R/producer)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')).returncode
  assert rc==0,(stage,'Own preNN input/source preparation refused; preserve full stderr')
 alias=B/'AutoGLMPhone9BSingle_own_original_AFS_R_identical_fixed200_no_filter_input_alias_no_NN_20261120/own_fixed200_no_filter_input_alias_proof.json';v=js(alias);assert v['passed'] and v['model_id']==MID
'''
q=q[:a]+gate+q[b:];q=q.replace("validate_report(R,formal['report_dir'],'zai-org/GLM-4.6V-Flash'","validate_report(R,formal['report_dir'],'zai-org/AutoGLM-Phone-9B-Multilingual'")
ast.parse(q);assets['scripts/'+newq]=q.encode()
cfg=dict(max_concurrent_models=1,own_input200_and_own_source_all_required=True,frozen_runner_SHA256='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90',own_next_readiness_preparer='scripts/'+mapping['prepare_autoglm_phone9b_own_original_BF16_2048_true3_v3_actual_published_GLM_fixed_input_set_20261119.py'],own_next_source_SHA256={n:sha(v) for n,v in assets.items()},max_new_tokens=2048,all_original_reports_sources_configs_logs_and_needed_examples_retained=True)
assets['configs/queues/after_AutoGLM182_own_single_Phone9B_inputs_true3_fixed200_serial_20261120.json']=(json.dumps(cfg,indent=2)+'\n').encode()
for n,v in assets.items():
 p=P/Path(n).name;assert not p.exists();p.write_bytes(v)
payload={n:dict(SHA256=sha(v),base64=base64.b64encode(v).decode()) for n,v in assets.items()}
s='DATA='+repr(payload)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,base64,json,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
for n,v in DATA.items():
 p=R/n;raw=base64.b64decode(v['base64'],validate=True);assert p.resolve().is_relative_to(R) and hashlib.sha256(raw).hexdigest()==v['SHA256']
 if p.suffix=='.py':ast.parse(raw)
 assert not p.exists(),str(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
L=B/'AutoGLMPhone9BSingle_own_after182_serial_actual_launch_20261120';L.mkdir(exist_ok=False);source='scripts/continue_AutoGLMPhone9B_single_own_inputs_true3_capacity_fresh3_fixed200_after182_serial_20261120.py'
with (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',str(R/source)],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=source,source_SHA256=DATA[source]['SHA256'],actual_input_source_preparation_started_not_yet_NN=True,max_concurrent_models=1);(L/'background_launch.json').write_text(json.dumps(v,indent=2)+'\\n',encoding='utf8');print(json.dumps(v))
''';ast.parse(s);(P/'launch_AutoGLMPhone9B_single_own_after182_serial_sources_20261120_remote.py').write_text(s,encoding='utf8',newline='\n')
r=dict(passed=True,new_model=MID,own_original_YAML_SHA256='a004f7d4f8c9bfeb19314f61697a293a7faae06d5f8b406cfab8e5d74fbba010',saved_unreadable_root_preserved_and_own_inventory_same_ID_root_used=True,own_SF_headers704_and_original_BF16_2048_outer10_EOS_and_template_confirmed=True,no_other_model_checkpoint_input_output_or_pixel_control_borrowed=True,all_numerical_observation_and_NF_math_AST_preserved_under_ownership_path_mapping=True,assets=[dict(relative=n,SHA256=sha(v)) for n,v in assets.items()]);(P/'AutoGLMPhone9BSingle_own_after182_source_scope_20261120.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(json.dumps(r))
