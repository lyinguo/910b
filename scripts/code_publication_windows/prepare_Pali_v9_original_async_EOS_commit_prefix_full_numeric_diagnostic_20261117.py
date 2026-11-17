import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,hashlib,base64
P=Path(__file__).parent
old=(P/'verify_pali10pt448_own_preserved_vllm_full_native_true3_v8_exact_committed_intrinsic_RoPE_operand_20261117.py').read_text(encoding='utf8')
before="public=internal_request_map[rid];assert frame['request_state_before_forward'][rid]['prompt_token_ids']==emitted[public]['prompt_token_ids'];all_selected[public].append(token)"
after="""public=internal_request_map[rid];assert frame['request_state_before_forward'][rid]['prompt_token_ids']==emitted[public]['prompt_token_ids']
      prior=frame['request_state_before_forward'][rid]['output_token_ids_before_forward'];assert len(prior)==len(all_selected[public]) and all(value==-1 or value==all_selected[public][i] for i,value in enumerate(prior)),(rid,prior)
      all_selected[public].append(token)"""
assert before in old;s=old.replace(before,after)
before="for row in trace:assert all_selected[row['request_id']]==row['generated_token_ids'],('Native sampler/request output token mapping mismatch',row['index'])"
after="""check('/vllm-workspace/vllm/vllm/v1/core/sched/utils.py','1ed988044eb8028c77ede875bd1a5fd05ff6998b455d61e3997d6a3d7fc1c3a7');check('/vllm-workspace/vllm/vllm/v1/core/sched/async_scheduler.py','e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508');check('/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py','c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884')
  from transformers import AutoTokenizer
  own_tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True);eos=own_tokenizer.eos_token_id;assert eos==1;native_stop_commit_review=[]
  for row in trace:
   sampled=all_selected[row['request_id']];assert eos in sampled and row['finish_reason']=='stop' and row['stop_reason'] is None
   stop=sampled.index(eos)+1;committed=sampled[:stop];trailing=sampled[stop:]
   assert committed==row['generated_token_ids'] and len(trailing)<=1 and all(token==eos for token in trailing),('Exact original EOS commit prefix or preserved in-flight tail mismatch',row['index'])
   native_stop_commit_review.append(dict(index=row['index'],public_request_id=row['request_id'],all_actual_native_sampled_tokens=sampled,all_public_committed_tokens=row['generated_token_ids'],original_primary_EOS=eos,original_stop_commit_prefix_exact=True,post_EOS_inflight_native_tokens_preserved_and_numerically_verified=trailing,no_runtime_stop_budget_or_outputs_modified=True))"""
assert before in s;s=s.replace(before,after)
s=s.replace('base.update(original_intrinsic_RoPE_operand_exact_old_recorded_SHA_reconstructed=', 'base.update(original_async_native_stop_commit_review=native_stop_commit_review,total_actual_native_sampled_tokens=sum(len(x) for x in all_selected.values()),total_original_public_committed_tokens=sum(len(x[\'generated_token_ids\']) for x in trace),all_native_post_stop_inflight_head_QKV_KV_values_preserved_and_verified=True,original_intrinsic_RoPE_operand_exact_old_recorded_SHA_reconstructed=')
s=s.replace('all_original_sampler_rows_and_request_tokens_exact=True','all_original_sampler_rows_verified_and_public_tokens_follow_original_EOS_commit=True')
a=next(x for x in ast.parse(old).body if isinstance(x,ast.FunctionDef) and x.name=='numeric');b=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='numeric');assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
nf='scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v9_original_async_EOS_commit_20261117.py';ast.parse(s);(P/Path(nf).name).write_bytes(s.encode());h=hashlib.sha256(s.encode()).hexdigest()
supervisor=(P/'supervise_Pali10Pt448_existing_real3_v8_exact_committed_intrinsic_RoPE_full_numeric_diagnostic_original_FLOCK_20261117.py').read_text(encoding='utf8').replace('Pali10Pt448_existing_complete_real3_v8_exact_committed_intrinsic_RoPE_full_numeric_diagnostic_original_FLOCK_20261117','Pali10Pt448_existing_complete_real3_v9_original_async_EOS_commit_full_numeric_diagnostic_original_FLOCK_20261117').replace('scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v8_exact_committed_intrinsic_RoPE_operand_20261117.py',nf).replace('9775b71ad082bfe52669b321a4ce3eb3fe97c8ae3185dc07bdf114282d286c51',h)
sn='scripts/supervise_Pali10Pt448_existing_real3_v9_original_async_EOS_commit_full_numeric_diagnostic_original_FLOCK_20261117.py';ast.parse(supervisor);(P/Path(sn).name).write_bytes(supervisor.encode());files={n:dict(data=base64.b64encode(v.encode()).decode(),SHA256=hashlib.sha256(v.encode()).hexdigest()) for n,v in [(nf,s),(sn,supervisor)]}
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
I=B/'Pali10Pt448_existing3_v9_original_async_EOS_commit_full_numeric_diagnostic_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.log').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',SN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=SN,source_SHA256=FILES[SN]['SHA256'],existing_real_NPU3_diagnostic_only_not_new_NPU=True,all_numeric_thresholds_and_original_sampling_EOS_budget_unchanged=True,all_extra_native_inflight_samples_preserved=True,original_failed_jobs_and_reviews_preserved=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(files)).replace('__SN__',repr(sn));ast.parse(remote);(P/'deploy_launch_Pali_v9_original_async_EOS_commit_full_numeric_diagnostic_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in files.items()}))
