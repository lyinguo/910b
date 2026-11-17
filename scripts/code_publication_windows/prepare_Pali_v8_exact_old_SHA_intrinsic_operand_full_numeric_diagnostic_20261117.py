import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,hashlib,base64
P=Path(__file__).parent
old=(P/'verify_pali10pt448_own_preserved_vllm_full_native_true3_v7_original_native_request_ID_map_20261117.py').read_text(encoding='utf8')
line="expected_softcap=text['final_logit_softcapping'];theta=float(text.get('rope_theta',10000.));assert theta==10000."
extra="""expected_softcap=text['final_logit_softcapping'];theta=float(text.get('rope_theta',10000.));assert theta==10000.
 recovered_home=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Pali10Pt448_original_intrinsic_RoPE_NPU_table_exact_old_committed_SHA_reconstruction_diagnostic_20261117'
 recovered_review=recovered_home/'review.json';check(recovered_review,'c5f40e798f8727ea6b494d4491e1993f0dbe1a2417b7bfac87df8bf53bd1176a');recovered=js(recovered_review);assert recovered['exact_old_recorded_table_SHA_match'] and recovered['no_learned_model_or_model_score'] and recovered['no_original_report_or_failed_job_rewrite']
 recovered_values=Path(recovered['values_file']);check(recovered_values,'e001ea774f893ee23a05c67d634e2404c0c09810b7a2ae5ba1875e3584f2541e');committed_cache=torch.from_numpy(np.load(recovered_values,allow_pickle=False)).view(torch.bfloat16);assert meta(committed_cache)==recovered['recorded_real_NPU_operand']==recovered['reconstructed_operand']
 check('/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/rope.py','82422c46f93c0962673ad51bc77fdd5caef412f45f2ff8b20bbeb20a4407f64b');check('/vllm-workspace/vllm/vllm/model_executor/layers/rotary_embedding/base.py','c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5')"""
assert line in old;s=old.replace(line,extra)
before="freq=1/(theta**(torch.arange(0,256,2,dtype=torch.float32)/256));angles=positions.float().unsqueeze(-1)*freq.unsqueeze(0);dtype=getattr(torch,r['native_cos_sin_cache']['dtype'].split('.')[-1]);cos=angles.cos().to(dtype).float().unsqueeze(1);sin=angles.sin().to(dtype).float().unsqueeze(1);before=packed[:,4096:6144].reshape(-1,8,256).float();a,b=before.chunk(2,-1);rotated=torch.cat([a*cos-b*sin,b*cos+a*sin],-1)"
after="assert meta(committed_cache)==r['native_cos_sin_cache'];cos,sin=committed_cache.index_select(0,positions).chunk(2,-1);cos=cos.float().unsqueeze(1);sin=sin.float().unsqueeze(1);before=packed[:,4096:6144].reshape(-1,8,256).float();a,b=before.chunk(2,-1);rotated=torch.cat([a*cos-b*sin,b*cos+a*sin],-1)"
assert before in s;s=s.replace(before,after)
s=s.replace('base.update(original_native_internal_to_public_request_ID_map=', 'base.update(original_intrinsic_RoPE_operand_exact_old_recorded_SHA_reconstructed=True,old_CPU_rebuilt_trigonometric_table_not_used_as_if_same_operand=True,intrinsic_operand_reconstruction_review=str(recovered_review),intrinsic_operand_reconstruction_review_SHA256=sha(recovered_review),original_native_internal_to_public_request_ID_map=')
a=next(x for x in ast.parse(old).body if isinstance(x,ast.FunctionDef) and x.name=='numeric');b=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='numeric');assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
nf='scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v8_exact_committed_intrinsic_RoPE_operand_20261117.py';ast.parse(s);(P/Path(nf).name).write_bytes(s.encode());h=hashlib.sha256(s.encode()).hexdigest()
supervisor=(P/'supervise_Pali10Pt448_existing_real3_v7_original_native_request_ID_full_numeric_diagnostic_original_FLOCK_20261117.py').read_text(encoding='utf8').replace('Pali10Pt448_existing_complete_real3_v7_original_native_request_ID_full_numeric_diagnostic_original_FLOCK_20261117','Pali10Pt448_existing_complete_real3_v8_exact_committed_intrinsic_RoPE_full_numeric_diagnostic_original_FLOCK_20261117').replace('scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v7_original_native_request_ID_map_20261117.py',nf).replace('2485558c550db22bb57ce54b3e5390602852d9e6472aeabeea31f36bf0d4973f',h)
sn='scripts/supervise_Pali10Pt448_existing_real3_v8_exact_committed_intrinsic_RoPE_full_numeric_diagnostic_original_FLOCK_20261117.py';ast.parse(supervisor);(P/Path(sn).name).write_bytes(supervisor.encode());files={n:dict(data=base64.b64encode(v.encode()).decode(),SHA256=hashlib.sha256(v.encode()).hexdigest()) for n,v in [(nf,s),(sn,supervisor)]}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__;SN=__SN__
for n,x in FILES.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw);p=R/n
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
I=B/'Pali10Pt448_existing3_v8_exact_committed_intrinsic_RoPE_full_numeric_diagnostic_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.log').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',SN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=SN,source_SHA256=FILES[SN]['SHA256'],existing_real_NPU3_diagnostic_only_not_new_NPU=True,all_numeric_thresholds_and_model_math_unchanged=True,only_old_SHA_identical_original_intrinsic_cache_values_used_as_operand=True,original_failed_jobs_and_reviews_preserved=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(files)).replace('__SN__',repr(sn));ast.parse(remote);(P/'deploy_launch_Pali_v8_exact_old_SHA_intrinsic_operand_full_numeric_diagnostic_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in files.items()}))
