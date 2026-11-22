"""Audit only this checkpoint's original 128257-to-128258 source construction."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
s=(V/'inspect_HancomHiLlava31_own_original_auto_FP16_loader_resize_remap_no_forward_v1_20261122.py').read_text(encoding='utf8')
s=s.replace('HancomInSpaceAI/HiLLaVA-llama_3_1-8b','ytu-ce-cosmos/Turkish-LLaVA-v0.1').replace('HancomHiLlava31','TurkishLlava').replace('frontend200_meta_no_NN_v1_20261122','frontend200_meta_no_NN_v2_exact_own_template_20261122')
s=s.replace('128257','128258').replace('128256','128257').replace("generation.eos_token_id==[128001,128008,128009]","generation.eos_token_id==128009").replace('own_original_EOS128001_128008_128009_preserved=True','own_original_single_EOS128009_preserved=True').replace('plain_RoPE500000_context131072','plain_RoPE500000_context8192')
assert all(t not in s for t in ['Hancom','context131072','128256','[128001,128008,128009]']);ast.parse(s)
source='scripts/inspect_TurkishLlava_own_original_auto_FP16_loader_resize_remap_no_forward_v1_20261122.py';(V/Path(source).name).write_text(s,encoding='utf8')
controller='''"""Wait for own input proof, then original CPU source construction only; no learned forward."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time,traceback
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';F=B/'TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_exact_own_template_20261122';H=B/'TurkishLlava_own_frontend_v2_then_original_CPU_resize128258_no_NN_20261122'
def put(name,value):(H/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 H.mkdir(exist_ok=False)
 while not (F/'actual_completion.json').is_file():put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='waiting_own_input_proof_no_NN',no_original_runner_lock_held=True));time.sleep(20)
 done=json.loads((F/'actual_completion.json').read_bytes());assert done['actual_returncode']==0,done
 src=R/CTOR_SOURCE;assert hashlib.sha256(src.read_bytes()).hexdigest()==CTOR_SHA
 with (H/'constructor_stdout.json').open('x') as out,(H/'constructor_stderr.log').open('x') as err:
  child=subprocess.Popen(['python3','-B',str(src)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),constructor_PID=child.pid,phase='own_original_CPU_source_loader_resize_remap_no_forward',no_model_NN_or_original_runner_EX=True));rc=child.wait()
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=rc,no_learned_CPU_or_NPU_forward=True,no_current_GLM_source_job_or_canonical_mutation=True));return rc
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_learned_CPU_or_NPU_forward=True))
  raise
'''.replace('CTOR_SOURCE',repr(source)).replace('CTOR_SHA',repr(sha(s.encode())))
cont='scripts/continue_TurkishLlava_own_frontend_v2_then_original_CPU_constructor_no_forward_v1_20261122.py';ast.parse(controller);(V/Path(cont).name).write_text(controller,encoding='utf8')
stage={source:s.encode(),cont:controller.encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_text(encoding='utf8').encode()};payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];p=R/n;ast.parse(b)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nB=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'TurkishLlava_own_CPU_constructor_source_no_forward_actual_launcher_v1_20261122';H.mkdir(exist_ok=False)\nwith (H/'stdout.json').open('x') as out,(H/'stderr.log').open('x') as err:p=subprocess.Popen(['python3','-B',str(R/"+repr(cont)+")],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)\nx=dict(PID=p.pid,source="+repr(cont)+",source_SHA256="+repr(sha(controller.encode()))+",all_primary_SHA256={n:r['SHA256'] for n,r in P.items()},no_learned_CPU_NPU_forward_or_original_runner_EX=True,no_current_GLM_source_job_deadline_signal_or_canonical_mutation=True);(H/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(x,ensure_ascii=False))\n"
ast.parse(carrier);(V/'stage_launch_TurkishLlava_own_original_CPU_constructor_no_forward_v1_20261122_remote.py').write_text(carrier,encoding='utf8');print(json.dumps(dict(own_constructor_source=source,source_SHA256=sha(s.encode()),controller=cont,controller_SHA256=sha(controller.encode()),CPU_source_construction_only_no_forward=True)))
