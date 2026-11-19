from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
oldprep='prepare_autoglm_phone9b_single_own_original_BF16_2048_true3_v1_20261120.py';newprep='prepare_autoglm_phone9b_single_own_original_BF16_2048_true3_v2_exact_frontend_path_20261120.py'
s=(P/oldprep).read_text(encoding='utf8');assert sha(s.encode())=='d18fd3b9da36ccb63e0d31a67e939c1f4caed046070ef4e01fdaedaebcf7b860'
s=s.replace('prepare_autoglm_phone9b_single_own_original_BF16_2048_20261120_frontend200.py','prepare_autoglm_phone9b_single_own_original_BF16_2048_frontend200_20261120.py').replace(oldprep,newprep)
s=s.replace('AutoGLMPhone9BSingle_own_original_BF16_2048_native_source_readiness_v3_actual_published_GLM_fixed_input_set_20261120','AutoGLMPhone9BSingle_own_original_BF16_2048_readiness_v2_exact_frontend_path_20261120').replace('AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120','AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v2_exact_frontend_path_real_FLOCK_20261120')
a=s.index(" headers=js(ready['own_source_headers']);manifest={}");b=s.index(' alias_path=',a)
s=s[:a]+''' headers=js(ready['own_source_headers']);previous=B/'AutoGLMPhone9BSingle_own_original_BF16_2048_native_source_readiness_v3_actual_published_GLM_fixed_input_set_20261120';failure=js(previous/'actual_failure.json');assert failure['actual_NEW_model_forwards']==0 and 'prepare_autoglm_phone9b_single_own_original_BF16_2048_20261120_frontend200.py' in failure['error'] and 'FileNotFoundError' in failure['error']
 assert sha(R/'scripts/prepare_autoglm_phone9b_single_own_original_BF16_2048_true3_v1_20261120.py')=='d18fd3b9da36ccb63e0d31a67e939c1f4caed046070ef4e01fdaedaebcf7b860'
 manifest=js(previous/'own704_active_source_manifest.json');assert set(manifest)==set(headers) and len(manifest)==704
 for filename,record in ready['own_weight_files'].items():assert sha(W/filename)==record['SHA256'] and (W/filename).stat().st_size==record['bytes']
 for name,v in manifest.items():assert v['source_key']==v['runtime_name']==name and v['source_file']==headers[name]['source_file'] and v['shape']==headers[name]['shape'] and v['source_dtype']=='torch.bfloat16' and v['source_SHA256']==v['cast_SHA256']
''' +s[b:]
ast.parse(s)
oldqueue='continue_AutoGLMPhone9B_single_own_inputs_true3_capacity_fresh3_fixed200_after182_serial_20261120.py';newqueue='continue_AutoGLMPhone9B_single_completed_inputs_exact_frontend_path_recovery_true3_first200_serial_v2_20261120.py';q=(P/oldqueue).read_text(encoding='utf8')
q=q.replace(oldqueue,newqueue).replace('after_AutoGLM182_own_single_Phone9B_inputs_true3_fixed200_serial_20261120.json','after_AutoGLM182_own_single_Phone9B_exact_frontend_path_true3_fixed200_serial_v2_20261120.json').replace('after_AutoGLM182_own_single_Phone9B_inputs_true3_fixed200_serial_20261120','after_AutoGLM182_own_single_Phone9B_exact_frontend_path_true3_fixed200_serial_v2_20261120').replace('AutoGLMPhone9BSingle_own_original_BF16_2048_native_source_readiness_v3_actual_published_GLM_fixed_input_set_20261120','AutoGLMPhone9BSingle_own_original_BF16_2048_readiness_v2_exact_frontend_path_20261120').replace('AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120','AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v2_exact_frontend_path_real_FLOCK_20261120')
a=q.index(" for stage,producer in [('own_input200'");b=q.index(' validate_report(R,formal',a)
q=q[:a]+''' old=B/'after_AutoGLM182_own_single_Phone9B_inputs_true3_fixed200_serial_20261120';oldready=B/'AutoGLMPhone9BSingle_own_original_BF16_2048_native_source_readiness_v3_actual_published_GLM_fixed_input_set_20261120';failure=js(old/'actual_completion.json');assert failure['actual_returncode']==1 and 'Own next readiness failed before NN' in failure['error'];rawfail=js(oldready/'actual_failure.json');assert rawfail['actual_NEW_model_forwards']==0 and 'prepare_autoglm_phone9b_single_own_original_BF16_2048_20261120_frontend200.py' in rawfail['error'] and not (B/'AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120').exists()
 front=B/'AutoGLMPhone9BSingle_own_original_BF16_2048_frontend200_meta_v1_20261120';review=js(front/'review.json');assert review['passed'] and review['model_id']==MID and sha(review['own_frontend200'])==review['own_frontend200_SHA256']
 alias=B/'AutoGLMPhone9BSingle_own_original_AFS_R_identical_fixed200_no_filter_input_alias_no_NN_20261120/own_fixed200_no_filter_input_alias_proof.json';v=js(alias);assert v['passed'] and v['model_id']==MID and v['existing_own_frontend200_SHA256']==review['own_frontend200_SHA256']
''' +q[b:]
ast.parse(q)
cfg=json.loads((P/'after_AutoGLM182_own_single_Phone9B_inputs_true3_fixed200_serial_20261120.json').read_bytes());cfg['own_next_readiness_preparer']='scripts/'+newprep
cfg['own_next_source_SHA256'].pop('scripts/'+oldprep);cfg['own_next_source_SHA256'].pop('scripts/'+oldqueue);cfg['own_next_source_SHA256'].update({'scripts/'+newprep:sha(s.encode()),'scripts/'+newqueue:sha(q.encode())})
assets={'scripts/'+newprep:s.encode(),'scripts/'+newqueue:q.encode(),'configs/queues/after_AutoGLM182_own_single_Phone9B_exact_frontend_path_true3_fixed200_serial_v2_20261120.json':(json.dumps(cfg,indent=2)+'\n').encode()}
for n,v in assets.items():ast.parse(v) if n.endswith('.py') else None;p=P/Path(n).name;assert not p.exists();p.write_bytes(v)
data={n:dict(SHA256=sha(v),base64=base64.b64encode(v).decode()) for n,v in assets.items()}
launch='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
for n,v in DATA.items():
 p=R/n;raw=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==v['SHA256'] and not p.exists()
 if n.endswith('.py'):ast.parse(raw)
 p.write_bytes(raw)
L=B/'AutoGLMPhone9BSingle_exact_frontend_path_preNN_recovery_v2_actual_launch_20261120';L.mkdir(exist_ok=False);source='scripts/continue_AutoGLMPhone9B_single_completed_inputs_exact_frontend_path_recovery_true3_first200_serial_v2_20261120.py'
with (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',str(R/source)],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=source,source_SHA256=DATA[source]['SHA256'],no_original_source_hot_edit_no_input200_or_weight_array_hash_replay=True);(L/'background_launch.json').write_text(json.dumps(v,indent=2)+'\\n',encoding='utf8');print(json.dumps(v))
''';ast.parse(launch);(P/'launch_AutoGLMPhone9BSingle_exact_frontend_path_preNN_recovery_v2_20261120_remote.py').write_text(launch,encoding='utf8',newline='\n');result=dict(passed=True,only_readiness_producer_path_and_fresh_private_recovery_binding_changed=True,model_frontend_MC_NF_and_all_math_sources_unchanged=True,already_passed_own_input200_and704_source_manifest_reused_without_replay=True,assets=[dict(relative=n,SHA256=sha(v)) for n,v in assets.items()]);(P/'AutoGLMPhone9BSingle_exact_frontend_path_preNN_recovery_v2_scope_20261120.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
