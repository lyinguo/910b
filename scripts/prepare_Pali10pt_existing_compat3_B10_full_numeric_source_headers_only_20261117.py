"""Preserve passed compatibility stages; prepare only missing own value capture.

This preflight performs no model load or CPU/NPU forward and creates no runnable
job. The untested observer, independent numeric verifier and actual vLLM schema
still need review before any new numerical collection stage can be dispatched.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct,ast
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'Pali10Pt448_completed_compat3_B10_missing_full_numeric_source_header_preflight_20261117'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def main():
 H.mkdir(exist_ok=False)
 old=B/'Pali10Pt448_A100_original_native_vllm_compat3_1322_v2_20261110';b10=B/'Pali10Pt448_A100_original_native_vllm_B10_1322_v2_20261110'
 c=js(old/'completion.json');assert c['actual_returncode']==0;accepted,=c['accepted'];assert accepted['proof_sha256']==sha(accepted['proof'])=='13f4049cb3e6440948ec628140a72fff27e5eaa5a8e856dc547fa4f581e12917'
 ten=js(b10/'actual_completion.json');assert ten['actual_returncode']==0 and sha(ten['proof'])==ten['proof_SHA256']=='bf0f1c1bb504fa98af0de2f1999a8c8f8405d326f712553ffc7466713b871b1a'
 p=js(old/'protocol.json');root=Path(p['weight_path']);index=js(root/'model.safetensors.index.json')['weight_map'];headers={};files={}
 for shard in sorted(set(index.values())):
  f=root/shard;assert f.is_file() and f.stat().st_size>8
  with f.open('rb') as inp:
   length=struct.unpack('<Q',inp.read(8))[0];assert 0<length<16*1024**2;head=json.loads(inp.read(length))
  entries={k:v for k,v in head.items() if k!='__metadata__'};assert set(entries)=={k for k,v in index.items() if v==shard}
  for name,v in entries.items():assert v['dtype']=='BF16';headers[name]=dict(shard=shard,shape=v['shape'],dtype=v['dtype'],data_offsets=v['data_offsets'])
  files[shard]=dict(bytes=f.stat().st_size,previous_frozen_SHA256=p['resource_sha256'][shard],fresh_full_SHA256_pending_at_new_real_load=True)
 assert set(headers)==set(index) and len(headers)==903
 for n in ['config.json','generation_config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json']:assert sha(root/n)==p['resource_sha256'][n]
 sources=['universal_eval/model/adapters/paligemma2_10b_pt448_A100_native_vllm_20261110_adapter.py','universal_eval/model/adapters/pali10pt448_own_preserved_vllm_full_value_observer_20261117.py','universal_eval/model/pali10pt448_full_native_value_20261117/worker.py']
 source_hash={}
 for n in sources:
  f=R/n;raw=f.read_bytes();ast.parse(raw);dest=H/'source_original_bytes'/(n+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);source_hash[n]=sha(f)
 assert source_hash[sources[0]]=='652112898c2d03e7174102439853a7e820ab0581db319f1037e063d8d84ca8b8'
 (H/'own903_source_headers.json').write_bytes(json.dumps(headers,ensure_ascii=False,indent=2).encode())
 value=dict(UTC=datetime.now(timezone.utc).isoformat(),source_header_preflight_passed=True,model_id='google/paligemma2-10b-pt-448',own_source_count=len(headers),own_shards=files,own_source_headers=str(H/'own903_source_headers.json'),own_source_headers_SHA256=sha(H/'own903_source_headers.json'),old_accepted_compat3_not_rerun_or_edited=True,old_accepted_compat3_completion_SHA256=sha(old/'completion.json'),old_compat3_proof_SHA256=accepted['proof_sha256'],old_accepted_B10_not_rerun_or_edited=True,old_B10_actual_completion_SHA256=sha(b10/'actual_completion.json'),old_B10_proof_SHA256=ten['proof_SHA256'],staged_primary_source_SHA256=source_hash,model_math_and_original_vllm_engine_dtype_batching_prompt_sampling_not_changed=True,CPU_or_NPU_model_forward_performed=False,no_executable_or_snapshot_in_report_directory=True,dispatch_ready=False,missing_before_new_full_numeric_stage=['complete independent own-source full-head/KV/RoPE/input numerical verifier','source-bound standalone YAML and private job validation','actual native worker schema verification and full numerical acceptance'],no_job_or_approval_or_G_created=True)
 (H/'readiness_NOT_dispatch_ready.json').write_bytes(json.dumps(value,ensure_ascii=False,indent=2).encode());print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':main()
