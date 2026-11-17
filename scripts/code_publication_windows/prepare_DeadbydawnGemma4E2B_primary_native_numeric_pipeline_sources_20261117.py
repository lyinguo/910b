import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,ast,base64,difflib
P=Path(__file__).parent
coder=json.loads((P/'R_Gemma4E2B_existing_primary_source_contracts_own_next_candidate_20261117.json').read_text(encoding='utf8'))['sources']
google=json.loads((P/'R_GoogleGemma4E2B_sharedKV_unused_source_code_contracts_20261117.json').read_text(encoding='utf8'))
old_mid='coder3101/gemma-4-E2B-it-qat-q4_0-unquantized-heretic';mid='deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit'
old_stem='coder3101--gemma-4-E2B-it-qat-q4_0-unquantized-heretic';stem='deadbydawn101--gemma-4-E2B-Heretic-Uncensored-mlx-4bit'
roles={
 'native':'universal_eval/model/adapters/coder_gemma4_e2b_own_original_BF16_native_v1_saved_HF_F32_20261116.py',
 'verify':'scripts/verify_coder_gemma4_e2b_own_original_BF16_native_v1_saved_HF_F32_20261116.py',
 'ready':'scripts/prepare_coder_gemma4_e2b_own_original_BF16_native_readiness_v1_saved_HF_F32_20261116.py',
 'measure':'scripts/measure_coder_gemma4_e2b_own_original_BF16_actual_capacity_v1_saved_HF_F32_20261116.py',
 'continue':'scripts/continue_coder_gemma4_e2b_own_original_BF16_true3_fresh3_fixed200_v1_saved_HF_F32_20261116.py',
}
def source(name):
 item=coder['/root/projects/comperror/'+name];s=item['text'];assert hashlib.sha256(s.encode()).hexdigest()==item['SHA256'];return s
def rename(s):
 s=s.replace(old_mid,mid).replace(old_mid.replace('/','_'),mid.replace('/','_')).replace(old_stem,stem).replace('CoderGemma4E2B','DeadbydawnGemma4E2B').replace('CoderGemma4_E2B','DeadbydawnGemma4_E2B').replace('coder_gemma4_e2b','deadbydawn_gemma4_e2b').replace('20261116','20261117')
 s=s.replace('deadbydawn_gemma4_e2b_own_20261117','coder_gemma4_e2b_own_20261116')
 s=s.replace('MiniInternVL2_Medical_own_original_BF16_2048_true3_v1_accuracy_real_FLOCK_20261117','MiniInternVL2_Medical_own_original_BF16_2048_true3_v1_accuracy_real_FLOCK_20261116')
 return s
out={};review=[]
for role,name in roles.items():
 old=source(name);s=rename(old);new_name=rename(name)
 if role=='native':
  item=google['/root/projects/comperror/universal_eval/model/adapters/gemma4_e2b_own_original_BF16_native_v2_loading_info_set_safe_20261116.py'];g=item['text'];assert hashlib.sha256(g.encode()).hexdigest()==item['SHA256'];gfunc=next(x for x in ast.parse(g).body if isinstance(x,ast.FunctionDef) and x.name=='load_own_model');loader=ast.get_source_segment(g,gfunc)
  # The own saved HF loader has no attention override; preserve that branch.
  loader=loader.replace(",attn_implementation={'text_config':'sdpa','vision_config':'sdpa','audio_config':'eager'}",'')
  node=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='load_own_model');s=s[:node.col_offset+sum(len(x) for x in s.splitlines(True)[:node.lineno-1])]+loader+'\n\n'+s[sum(len(x) for x in s.splitlines(True)[:node.end_lineno]):]
  s=s.replace('own1951_all_source_actual_NPU_unused0.json','own2011_source_active1951_NPU_unused60_original_sharedKV.json')
  s=s.replace('own native Gemma4 physical15 plus shared20 dynamic/sliding KV for saved vLLM persistent cache','own native saved HF Gemma4 physical15 plus shared20 dynamic/sliding KV')
  before=next(x for x in ast.parse(old).body if isinstance(x,ast.ClassDef));after=next(x for x in ast.parse(s).body if isinstance(x,ast.ClassDef))
  for method in ['_install','batch_predict','get_sample_execution','_predict_original_compute_and_capture']:
   a=next(x for x in before.body if isinstance(x,ast.FunctionDef) and x.name==method);b=next(x for x in after.body if isinstance(x,ast.FunctionDef) and x.name==method);assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),method
 if role=='verify':
  s=s.replace("len(weights['original_unused_shared_KV_source'])==0","len(weights['original_unused_shared_KV_source'])==60")
  s=s.replace('len(hdr)==1951','len(hdr)==2011')
  s=s.replace("unused=set();assert set(weights['original_unused_shared_KV_source'])==unused","unused={f'model.language_model.layers.{i}.self_attn.{n}' for i in range(15,35) for n in ['k_proj.weight','v_proj.weight','k_norm.weight']};assert set(weights['original_unused_shared_KV_source'])==unused")
  s=s.replace('Independent own-checkpoint/full-head/cache/mask/standard-report audit; no CPU NN.','Independent own-checkpoint, full-head and native-cache audit. CPU same-operand numerical reference is diagnostic evidence and never a model score.')
  s=s.replace('all1951_own_source_actual_NPU_SHA256_verified=True,','all1951_own_source_actual_NPU_SHA256_verified=True,all2011_own_source_SHA_verified_with60_original_unused_sharedKV=True,')
 if role=='ready':
  line=" assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]"
  assert line in s;s=s.replace(line," resource_snapshot=dict(actual_project_processes=project_processes(),actual_FLOCK=runner_lock_owners(),no_NPU_dispatch_during_config_preparation=True)")
  s=s.replace("pre['own_checkpoint_tensor_count']==1951","pre['own_checkpoint_tensor_count']==2011")
  s=s.replace("dict(device='cuda',max_new_tokens=2048,torch_dtype='auto')","dict(device='auto',max_new_tokens=2048,torch_dtype='auto')")
  s=s.replace("own['dtype']==own['text_config']['dtype']==own['vision_config']['dtype']=='bfloat16'","own.get('dtype',own.get('torch_dtype'))==own['text_config'].get('dtype',own['text_config'].get('torch_dtype'))==own['vision_config'].get('dtype',own['vision_config'].get('torch_dtype'))=='bfloat16'")
  s=s.replace('Own coder3101 unquantized BF16 1951 source checkpoint','Own deadbydawn101 checkpoint: all2011 BF16 stored sources,1951 native active parameters plus60 original unused sharedKV projections. The MLX/4bit repository label is not an execution-precision claim; the actual saved local checkpoint and HF branch are BF16. Own')
  s=s.replace("proto=dict(model_id=MID,", "proto=dict(dispatch_resource_snapshot_at_config_preparation=resource_snapshot,dispatch_still_requires_actual_NN_NF_and_original_lock_release=True,own_source_total=2011,own_native_active_source_count=1951,own_original_unused_sharedKV_source_count=60,model_id=MID,")
  s=s.replace('unquantized BF16','BF16')
 ast.parse(s)
 assert 'git add' not in s and old_mid not in s
 out[new_name]=s.encode();review.append(dict(role=role,old_source=name,old_SHA256=hashlib.sha256(old.encode()).hexdigest(),new_source=new_name,new_SHA256=hashlib.sha256(s.encode()).hexdigest(),new_bytes=len(s.encode())))
 (P/Path(new_name).name).write_bytes(s.encode())
 (P/(role+'_DeadbydawnGemma4E2B_primary_source_review_20261117.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=name,tofile=new_name)),encoding='utf8')
front='scripts/prepare_deadbydawn_gemma4_e2b_own_saved_HF_default_template_F32_frontend200_v1_20261117.py';out[front]=(P/Path(front).name).read_bytes()
payload={n:dict(SHA256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()) for n,raw in out.items()}
(P/'DeadbydawnGemma4E2B_primary_native_numeric_sources_reviewed_payload_20261117.json').write_text(json.dumps(dict(sources=payload,review=review),ensure_ascii=False,indent=2),encoding='utf8')
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,base64,hashlib,ast,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';PAYLOAD=__PAYLOAD__
for name,item in PAYLOAD.items():
 raw=base64.b64decode(item['data']);assert hashlib.sha256(raw).hexdigest()==item['SHA256'];ast.parse(raw);f=R/name
 if f.exists():assert f.read_bytes()==raw,name
 else:f.write_bytes(raw)
I=B/'DeadbydawnGemma4E2B_native_source_and_independent_config_readiness_no_NN_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.json').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.run(['python3','-B','scripts/prepare_deadbydawn_gemma4_e2b_own_original_BF16_native_readiness_v1_saved_HF_F32_20261117.py'],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,sources={n:x['SHA256'] for n,x in PAYLOAD.items()},no_NPU_model_forward_or_dispatch=True,stdout=(I/'stdout.json').read_text(),stderr=(I/'stderr.log').read_text());(I/'actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False));raise SystemExit(p.returncode)
'''.replace('__PAYLOAD__',repr(payload))
ast.parse(remote);(P/'deploy_DeadbydawnGemma4E2B_primary_numeric_pipeline_source_and_config_no_NN_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(review,ensure_ascii=False))
