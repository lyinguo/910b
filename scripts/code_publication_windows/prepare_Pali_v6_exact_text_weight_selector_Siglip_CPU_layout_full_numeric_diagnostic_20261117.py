import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,hashlib,base64
P=Path(__file__).parent
old=(P/'verify_pali10pt448_own_preserved_vllm_full_native_true3_v5_native_default_head_dimension_20261117.py').read_text(encoding='utf8')
s=old.replace("[n for n in headers if n.endswith('layers.0.self_attn.q_proj.weight')]","[n for n in headers if 'language_model.' in n and n.endswith('layers.0.self_attn.q_proj.weight')]").replace("[n for n in headers if n.endswith('layers.0.self_attn.k_proj.weight')]","[n for n in headers if 'language_model.' in n and n.endswith('layers.0.self_attn.k_proj.weight')]")
before="state_vision={n.split('vision_tower.',1)[1]:tensor(n) for n in headers if 'vision_tower.' in n};vision_model.load_state_dict(state_vision,strict=True,assign=True)"
after="""state_vision={n.split('vision_tower.',1)[1]:tensor(n) for n in headers if 'vision_tower.' in n};native_vision_state=vision_model.state_dict();vision_key_map={};mapped_vision={}
  for name,value in state_vision.items():
   candidate=name if name in native_vision_state else name.removeprefix('vision_model.')
   assert candidate in native_vision_state and value.shape==native_vision_state[candidate].shape and candidate not in mapped_vision,(name,candidate)
   vision_key_map[name]=candidate;mapped_vision[candidate]=value
  assert set(mapped_vision)==set(native_vision_state)
  vision_model.load_state_dict(mapped_vision,strict=True,assign=True)"""
assert before in s;s=s.replace(before,after)
s=s.replace('base.update(native_default_head_dimension_resolution=', 'base.update(own_CPU_vision_optional_wrapper_key_map=vision_key_map,own_CPU_vision_all_source_values_and_shapes_preserved=True,native_default_head_dimension_resolution=')
a=next(x for x in ast.parse(old).body if isinstance(x,ast.FunctionDef) and x.name=='numeric');b=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='numeric');assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
nf='scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v6_text_namespace_and_CPU_Siglip_layout_20261117.py';ast.parse(s);(P/Path(nf).name).write_bytes(s.encode());h=hashlib.sha256(s.encode()).hexdigest()
supervisor=(P/'supervise_Pali10Pt448_existing_real3_v5_native_default_full_numeric_diagnostic_original_FLOCK_20261117.py').read_text(encoding='utf8').replace('Pali10Pt448_existing_complete_real3_v5_native_default_head_dimension_full_numeric_diagnostic_original_FLOCK_20261117','Pali10Pt448_existing_complete_real3_v6_text_namespace_Siglip_layout_full_numeric_diagnostic_original_FLOCK_20261117').replace('scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v5_native_default_head_dimension_20261117.py',nf).replace('767593bd560c20c1f7df874fd29459593fb3b76b7a9377f59a8d72a41d29c732',h)
sn='scripts/supervise_Pali10Pt448_existing_real3_v6_text_namespace_Siglip_layout_full_numeric_diagnostic_original_FLOCK_20261117.py';ast.parse(supervisor);(P/Path(sn).name).write_bytes(supervisor.encode());files={n:dict(data=base64.b64encode(v.encode()).decode(),SHA256=hashlib.sha256(v.encode()).hexdigest()) for n,v in [(nf,s),(sn,supervisor)]}
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
I=B/'Pali10Pt448_existing3_v6_text_namespace_Siglip_layout_full_numeric_diagnostic_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.log').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',SN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=SN,source_SHA256=FILES[SN]['SHA256'],existing_real_NPU3_diagnostic_only_not_new_NPU=True,all_numeric_thresholds_unchanged=True,original_failed_v4_job_and_v5_diagnostic_preserved=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(files)).replace('__SN__',repr(sn));ast.parse(remote);(P/'deploy_launch_Pali_v6_exact_text_namespace_Siglip_layout_full_numeric_diagnostic_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in files.items()}))
