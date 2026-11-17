import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,hashlib,base64
P=Path(__file__).parent
old=(P/'verify_pali10pt448_own_preserved_vllm_full_native_true3_v6_text_namespace_and_CPU_Siglip_layout_20261117.py').read_text(encoding='utf8')
s=old.replace('import json,hashlib,decimal','import json,hashlib,decimal,re')
before="check(g['actual_worker_source'],g['actual_worker_source_SHA256']);history=defaultdict(dict)"
after="""check(g['actual_worker_source'],g['actual_worker_source_SHA256']);history=defaultdict(dict)
   request_input_source=Path('/vllm-workspace/vllm/vllm/v1/engine/input_processor.py');request_output_source=Path('/vllm-workspace/vllm/vllm/v1/engine/output_processor.py')
   check(request_input_source,'c8abd8e4e14d99929d45a7ff3b2a74898269c5e2a7fa30b826623ee85e4ce813');check(request_output_source,'e47b86cd69ce1c1e3f53dac655eaa75c32a4057d10f4a1db16bb83a397f2a329')
   # The unchanged producer explicitly stores external_req_id then assigns
   # f'{external_req_id}-{random_uuid():.8}'; public RequestOutput restores
   # external_req_id. Resolve only this exact frozen source-proven encoding,
   # never output text, reference answers or ordinal guesses.
   internal_request_map={}
   for internal in {rid for frame in g['frames'] for rid in frame['active_request_ids']}:
    matches=[external for external in emitted if re.fullmatch(re.escape(external)+r'-[0-9a-f]{8}',internal)]
    assert len(matches)==1,(internal,matches)
    internal_request_map[internal]=matches[0]
   assert len(internal_request_map)==len(set(internal_request_map.values()))==len(emitted) and set(internal_request_map.values())==set(emitted)"""
assert before in s;s=s.replace(before,after)
before="for rid,token in zip(frame['active_request_ids'],ids.tolist()):assert rid in emitted;all_selected[rid].append(token)"
after="""for rid,token in zip(frame['active_request_ids'],ids.tolist()):
      public=internal_request_map[rid];assert frame['request_state_before_forward'][rid]['prompt_token_ids']==emitted[public]['prompt_token_ids'];all_selected[public].append(token)"""
assert before in s;s=s.replace(before,after)
s=s.replace('base.update(own_CPU_vision_optional_wrapper_key_map=', 'base.update(original_native_internal_to_public_request_ID_map=internal_request_map,request_ID_mapping_uses_only_frozen_native_UUID_producer_not_outputs_or_answers=True,own_CPU_vision_optional_wrapper_key_map=')
a=next(x for x in ast.parse(old).body if isinstance(x,ast.FunctionDef) and x.name=='numeric');b=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='numeric');assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
nf='scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v7_original_native_request_ID_map_20261117.py';ast.parse(s);(P/Path(nf).name).write_bytes(s.encode());h=hashlib.sha256(s.encode()).hexdigest()
supervisor=(P/'supervise_Pali10Pt448_existing_real3_v6_text_namespace_Siglip_layout_full_numeric_diagnostic_original_FLOCK_20261117.py').read_text(encoding='utf8').replace('Pali10Pt448_existing_complete_real3_v6_text_namespace_Siglip_layout_full_numeric_diagnostic_original_FLOCK_20261117','Pali10Pt448_existing_complete_real3_v7_original_native_request_ID_full_numeric_diagnostic_original_FLOCK_20261117').replace('scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v6_text_namespace_and_CPU_Siglip_layout_20261117.py',nf).replace('41de72c494751267224859d7a120b70ce05d85bff07704a42f0cf2086ef069e4',h)
sn='scripts/supervise_Pali10Pt448_existing_real3_v7_original_native_request_ID_full_numeric_diagnostic_original_FLOCK_20261117.py';ast.parse(supervisor);(P/Path(sn).name).write_bytes(supervisor.encode());files={n:dict(data=base64.b64encode(v.encode()).decode(),SHA256=hashlib.sha256(v.encode()).hexdigest()) for n,v in [(nf,s),(sn,supervisor)]}
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
I=B/'Pali10Pt448_existing3_v7_original_native_request_ID_full_numeric_diagnostic_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.log').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',SN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=SN,source_SHA256=FILES[SN]['SHA256'],existing_real_NPU3_diagnostic_only_not_new_NPU=True,all_numeric_thresholds_and_model_math_unchanged=True,original_failed_v4_v5_v6_evidence_preserved=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(files)).replace('__SN__',repr(sn));ast.parse(remote);(P/'deploy_launch_Pali_v7_native_request_ID_mapping_full_numeric_diagnostic_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in files.items()}))
