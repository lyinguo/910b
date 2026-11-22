"""Keep complete passed own inputs; recover only final identity metadata and original CPU audit."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
meta_name='recover_TurkishLlava_completed_frontend200_current_unique_C690_metadata_only_v3_20261122.py';meta=(V/meta_name).read_text(encoding='utf8');ast.parse(meta)
old=(V/'inspect_TurkishLlava_own_original_auto_FP16_loader_resize_remap_no_forward_v1_20261122.py').read_text(encoding='utf8');assert sha(old.encode())=='50ec430f5dd2fe662f4aa2f5f044ca049032e7a705fc13eab984715f6ba37354'
s=old.replace('TurkishLlava_own_original_auto_FP16_actual_CPU_loader_resize_remap_no_forward_v1_20261122','TurkishLlava_own_original_auto_FP16_actual_CPU_loader_resize_remap_no_forward_v2_current_unique_C690_20261122').replace('TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_exact_own_template_20261122','TurkishLlava_completed_own_frontend200_current_unique_C690_metadata_only_v3_20261122')
assert s.replace('TurkishLlava_own_original_auto_FP16_actual_CPU_loader_resize_remap_no_forward_v2_current_unique_C690_20261122','TurkishLlava_own_original_auto_FP16_actual_CPU_loader_resize_remap_no_forward_v1_20261122').replace('TurkishLlava_completed_own_frontend200_current_unique_C690_metadata_only_v3_20261122','TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_exact_own_template_20261122')==old
source='scripts/inspect_TurkishLlava_own_original_auto_FP16_loader_resize_remap_no_forward_v2_current_unique_C690_20261122.py';ast.parse(s);(V/Path(source).name).write_text(s,encoding='utf8');metasource='scripts/'+meta_name
bindings={metasource:sha(meta.encode()),source:sha(s.encode())}
controller='''"""Completed own frontend200 metadata recovery then original source CPU construction, no NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,traceback
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'TurkishLlava_completed_frontend200_metadata_v3_then_original_CPU_constructor_v2_no_NN_20261122'
SOURCES=SOURCE_BINDINGS
def put(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 H.mkdir(exist_ok=False);prior=B/'TurkishLlava_own_frontend_v2_then_original_CPU_resize128258_no_NN_20261122';assert json.loads((prior/'actual_completion.json').read_bytes())['actual_returncode']==1 and not (B/'TurkishLlava_own_original_auto_FP16_actual_CPU_loader_resize_remap_no_forward_v1_20261122').exists()
 for phase,source in [('passed_frontend200_actual_unique_C_metadata_only_no_replay',META_SOURCE),('original_own_CPU_constructor128258_no_learned_forward',CTOR_SOURCE)]:
  p=R/source;assert hashlib.sha256(p.read_bytes()).hexdigest()==SOURCES[source]
  with (H/(phase+'_stdout.json')).open('x') as out,(H/(phase+'_stderr.log')).open('x') as err:
   child=subprocess.Popen(['python3','-B',str(p)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase=phase,child_PID=child.pid,no_NN_or_original_runner_EX=True));rc=child.wait()
  put(phase+'_actual_exit.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=rc,source=source));assert rc==0,(phase,rc)
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,no_frontend200_replay_or_learned_CPU_NPU_forward=True,no_original_runner_lock_held=True,no_current_GLM_or_canonical_mutation=True));return 0
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_frontend200_replay_or_learned_CPU_NPU_forward=True))
  raise
'''.replace('SOURCE_BINDINGS',repr(bindings)).replace('META_SOURCE',repr(metasource)).replace('CTOR_SOURCE',repr(source))
cont='scripts/continue_TurkishLlava_completed_frontend200_metadata_v3_original_CPU_constructor_v2_no_NN_20261122.py';ast.parse(controller);(V/Path(cont).name).write_text(controller,encoding='utf8');stage={metasource:meta.encode(),source:s.encode(),cont:controller.encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_text(encoding='utf8').encode()};payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];p=R/n;ast.parse(b)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nB=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'TurkishLlava_frontend200_metadata_v3_CPU_constructor_v2_no_NN_actual_launcher_20261122';H.mkdir(exist_ok=False)\nwith (H/'stdout.json').open('x') as out,(H/'stderr.log').open('x') as err:p=subprocess.Popen(['python3','-B',str(R/"+repr(cont)+")],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)\nx=dict(PID=p.pid,source="+repr(cont)+",source_SHA256="+repr(sha(controller.encode()))+",all_primary_SHA256={n:r['SHA256'] for n,r in P.items()},old_frontend_v1_v2_and_controller_errors_preserved=True,no_frontend200_replay_or_learned_CPU_NPU_forward_or_original_EX=True,no_current_GLM_source_job_deadline_signal_or_canonical_mutation=True);(H/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(x,ensure_ascii=False))\n"
ast.parse(carrier);(V/'stage_launch_TurkishLlava_frontend_metadata_v3_CPU_constructor_v2_no_replay_20261122_remote.py').write_text(carrier,encoding='utf8');print(json.dumps(dict(primary_sources={n:sha(b) for n,b in stage.items()},no_frontend200_replay_or_NN=True)))
