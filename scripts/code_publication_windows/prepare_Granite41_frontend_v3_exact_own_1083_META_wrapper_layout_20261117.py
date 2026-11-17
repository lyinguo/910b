import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).parent
s=(P/'prepare_granite41_own_original_HF_BF16_4096_frontend200_META_v2_actual_SizeDict_tuple_20261117.py').read_text(encoding='utf8')
s=s.replace('Granite41_own_original_HF_BF16_4096_frontend200_source_META_v2_actual_SizeDict_tuple_20261117','Granite41_own_original_HF_BF16_4096_frontend200_source_META_v3_exact_own_wrapper_layout_20261117')
old="native=model.state_dict();extra=set(headers)-set(native);missing=set(native)-set(headers);assert not extra and missing<=set(['lm_head.weight']),(sorted(extra),sorted(missing))\n if 'lm_head.weight' in missing:assert model.lm_head.weight is model.model.language_model.embed_tokens.weight\n for n in headers:assert headers[n]['shape']==list(native[n].shape),n"
new="""native=model.state_dict();mapping={}
 for n,x in headers.items():
  candidate=n if n in native else n.replace('.vision_tower.vision_model.','.vision_tower.',1)
  assert candidate in native and candidate not in mapping.values() and x['shape']==list(native[candidate].shape),(n,candidate)
  mapping[n]=candidate
 extra=set(headers)-set(mapping);missing=set(native)-set(mapping.values());assert not extra and missing==set(['lm_head.weight']),(sorted(extra),sorted(missing))
 assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
 put('own_source_to_native_META_name_map.json',dict(mapping=mapping,only_optional_vision_model_wrapper_prefix_removed=True,own_source_count=len(headers),native_state_count=len(native),all_own_sources_mapped_exactly_once=True,no_tensor_value_dtype_or_model_math_changed=True))"""
assert old in s;s=s.replace(old,new)
s=s.replace("assert len(weight_files)==2", "put('own_checkpoint_files_full_SHA.json',weight_files)\n assert len(weight_files)==2")
name='scripts/prepare_granite41_own_original_HF_BF16_4096_frontend200_META_v3_exact_own_wrapper_layout_20261117.py';ast.parse(s);(P/Path(name).name).write_bytes(s.encode());h=hashlib.sha256(s.encode()).hexdigest()
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';name=__NAME__;raw=base64.b64decode(__DATA__);assert hashlib.sha256(raw).hexdigest()==__SHA__;ast.parse(raw);p=R/name
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
I=B/'Granite41_frontend_META_v3_exact_own_wrapper_layout_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.json').open('xb') as out,(I/'stderr.log').open('xb') as err:child=subprocess.Popen(['python3','-B',name],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,source=name,source_SHA256=__SHA__,own_source_to_current_META_key_prefix_mapping_only=True,no_checkpoint_tensor_or_dtype_or_pixel_or_generation_parameter_changed=True,no_NPU_or_learned_CPU_model_forward=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__NAME__',repr(name)).replace('__DATA__',repr(base64.b64encode(s.encode()).decode())).replace('__SHA__',repr(h));ast.parse(remote);(P/'deploy_launch_Granite41_frontend_v3_exact_own_wrapper_META_no_NN_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(source=name,SHA256=h)))
