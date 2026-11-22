"""Derive own original frontend and source-only constructor; never dispatch NN."""
from pathlib import Path
import ast, base64, hashlib, json
V = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
front_name = 'prepare_HancomHiLlava31_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122.py'
ctor_name = 'inspect_HancomHiLlava31_own_original_auto_FP16_loader_resize_remap_no_forward_v1_20261122.py'
old_front = (V / 'prepare_xtuner_llava3_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_original_tokenizer_fallback_20261122.py').read_bytes()
old_ctor = (V / 'inspect_xtuner_llava3_own_original_FP16_loader_resize_remap_no_forward_v3_full_loading_info_schema_20261122.py').read_bytes()
assert sha(old_front) == 'c9db4b2c4b66c61e9ad86f4581a92491880cb2cca6cd194bbc1b183cf9d90634'
assert sha(old_ctor) == 'fce0cc708f974cfb5f8f715eeee7f38043351fd30659f3a781f62fab73d7cf90'
def identity(s):
    s = s.replace('xtuner/llava-llama-3-8b-v1_1-hf', 'HancomInSpaceAI/HiLLaVA-llama_3_1-8b')
    s = s.replace('xtuner--llava-llama-3-8b-v1_1-hf.yaml', 'HancomInSpaceAI--HiLLaVA-llama_3_1-8b.yaml')
    s = s.replace('XtunerLlava3_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_original_tokenizer_fallback_20261122', 'HancomHiLlava31_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122')
    s = s.replace('XtunerLlava3_own_original_FP16_actual_CPU_loader_resize_remap_no_forward_v3_full_loading_info_schema_20261122', 'HancomHiLlava31_own_original_auto_FP16_actual_CPU_loader_resize_remap_no_forward_v1_20261122')
    s = s.replace('331d6218b97a78c96f5d69d1d8fc708313a3ed4f1a053ffa95a2aaaa7245bc5a', '624d81316384e7bff93ba2fd7024f3e7b47051198eb3deff2313aa400a83f84e')
    return s
s = identity(old_front.decode())
s = s.replace("raw['rope_scaling'] is None", "raw['rope_scaling']==dict(factor=8.0,high_freq_factor=4.0,low_freq_factor=1.0,original_max_position_embeddings=8192,rope_type='llama3')")
start = s.index(' previous=B/')
end = s.index(" put('own_weight_files_and_headers.json'", start)
fresh = ''' for n in sorted(set(index.values())):
  file=W/n
  with file.open('rb') as stream:
   header_length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(header_length))
  h={k:v for k,v in header.items() if k!='__metadata__'};headers[n]=h
  for k,v in h.items():
   assert k not in flat and index[k]==n;flat[k]=v;dtype_counts[v['dtype']]+=1
  files[n]=dict(bytes=file.stat().st_size,SHA256=sha(file))
 assert len(flat)==len(index)==686 and dict(dtype_counts)=={'BF16':686} and sum(v['bytes'] for v in files.values())==16709592504
'''
s = s[:start] + fresh + s[end:]
old = "independent_text=js(W/'tokenizer_config.json')['bos_token']+'<|start_header_id|>user<|end_header_id|>\\n\\n'+messages[0]['content'].strip()+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n'"
new = "independent_text=js(W/'tokenizer_config.json')['bos_token']+'<|start_header_id|>system<|end_header_id|>\\n\\nCutting Knowledge Date: December 2023\\nToday Date: 26 Jul 2024\\n\\n<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n'+messages[0]['content'].strip()+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n'"
assert old in s; s = s.replace(old, new)
s = s.replace('dict(row=686,model_id=MID', 'dict(row=338,model_id=MID').replace('D686', 'D338').replace('current_row=686', 'current_row=338').replace("Decimal('.71')", "Decimal('.715')")
s = s.replace('original_generation_budget2048_outer10_Greedy_preserved=True,', 'original_generation_budget2048_outer10_Greedy_preserved=True,original_default_system_date_26_Jul_2024_preserved=True,original_saved_legacy_reconstruction_omits_raw_llama31_rope_scaling=True,own_BF16_storage_original_auto_nonCPU_FP16_request=True,')
assert 'Xtuner' not in s and 'xtuner' not in s
ast.parse(s); (V / front_name).write_text(s, encoding='utf8')
t = identity(old_ctor.decode())
t = t.replace("dtype_counts=={'torch.float16':682,'torch.float32':4}", "dtype_counts=={'torch.bfloat16':686}")
t = t.replace('assert generation.eos_token_id==128001', 'assert generation.eos_token_id==[128001,128008,128009]')
t = t.replace('original_only_EOS128001_not_borrowed_Mini_EOS128009=True', 'own_original_EOS128001_128008_128009_preserved=True,original_saved_legacy_plain_RoPE500000_context131072_preserved=True')
assert 'Xtuner' not in t and 'xtuner' not in t
ast.parse(t); (V / ctor_name).write_text(t, encoding='utf8')
front_rel = 'scripts/' + front_name; ctor_rel = 'scripts/' + ctor_name
main_name = 'continue_HancomHiLlava31_own_frontend_then_original_CPU_constructor_no_NN_v1_20261122.py'
controller = '''"""Own source/frontend/constructor only; no learned CPU/NPU or global runner lock."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,traceback,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'HancomHiLlava31_own_frontend200_then_original_CPU_constructor_no_NN_v1_20261122'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n',encoding='utf8');os.replace(t,p)
def main():
 H.mkdir(exist_ok=False)
 guard=(H/'private_no_NN_guard.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sources=SOURCE_BINDINGS
 assert all(sha(R/n)==h for n,h in sources.items())
 assert sha(D/'state.json')=='47eb91c93d13fae449ec89fee6e22fd925c68066cb15c20293869737df1b00c3'
 for name,source in [('own_frontend200',FRONT_SOURCE),('own_original_CPU_constructor_no_learned_forward',CTOR_SOURCE)]:
  put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase=name,no_learned_CPU_or_NPU_forward=True,no_original_runner_lock_held=True,no_canonical_write=True))
  with (H/(name+'_stdout.json')).open('x') as out,(H/(name+'_stderr.log')).open('x') as err:
   result=subprocess.run(['python3','-B',str(R/source)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  assert result.returncode==0,(name,result.returncode)
  put(name+'_actual_exit.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=result.returncode,source=source,source_SHA256=sources[source]))
 assert all(sha(R/n)==h for n,h in sources.items())
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,own_frontend_and_original_CPU_constructor_passed=True,no_learned_CPU_or_NPU_forward=True,no_original_runner_lock_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_learned_CPU_or_NPU_forward=True))
  raise
'''
bindings = {front_rel: sha(s.encode()), ctor_rel: sha(t.encode())}
controller = controller.replace('SOURCE_BINDINGS', repr(bindings)).replace('FRONT_SOURCE', repr(front_rel)).replace('CTOR_SOURCE', repr(ctor_rel))
ast.parse(controller); (V / main_name).write_text(controller, encoding='utf8')
stage = {front_rel:s.encode(),ctor_rel:t.encode(),'scripts/'+main_name:controller.encode(),
    'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_bytes()}
manifest={n:sha(b) for n,b in stage.items()};(V/'HancomHiLlava31_own_frontend_ctor_primary_paths_v1_20261122.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];p=R/n\n if p.suffix=='.py':ast.parse(b)\n p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nD=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'HancomHiLlava31_own_frontend_CPU_no_NN_v1_actual_launcher_20261122';H.mkdir(exist_ok=False)\nwith (H/'stdout.json').open('x') as out,(H/'stderr.log').open('x') as err:p=subprocess.Popen(['python3','-B',str(R/"+repr('scripts/'+main_name)+")],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)\nx=dict(PID=p.pid,source="+repr('scripts/'+main_name)+",source_SHA256="+repr(sha(controller.encode()))+",no_learned_CPU_or_NPU_forward=True,no_original_runner_lock_held=True,front_then_constructor_only=True);(H/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(x,ensure_ascii=False))\n"
ast.parse(carrier);(V/'stage_launch_HancomHiLlava31_own_frontend_CPU_no_NN_v1_20261122_remote.py').write_text(carrier,encoding='utf8')
print(json.dumps(dict(primary_paths=manifest,no_model_NN_dispatch=True)))
