"""Next own-weight adaptation stage only; no NN/CPU model or borrowed assets."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct,ast,shutil,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='rednote-hilab/dots.ocr';H=B/'DotsOCR_own_original_resource_source_preflight_after_MiniCPMV4_user_continue_20261115'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 H.mkdir(exist_ok=False);assert shutil.disk_usage(D).free>4*1024**3
 entry=json.loads((D/'state.json').read_bytes())['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 paths=[Path(s) for s in entry['weight_paths'] if Path(s).is_dir()];assert len(paths)==1;W=paths[0]
 put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='next_own_original_weight_and_source_preflight_not_NPU_inference',model_id=MID))
 weights={};seen={}
 for p in sorted(W.glob('*.safetensors')):
  with p.open('rb') as f:n=f.read(8);raw=f.read(struct.unpack('<Q',n)[0]);header=json.loads(raw)
  weights[p.name]=dict(bytes=p.stat().st_size,SHA256=sha(p),header_SHA256=hashlib.sha256(n+raw).hexdigest())
  for key,v in header.items():
   if key!='__metadata__':assert key not in seen;seen[key]=dict(source=p.name,shape=v['shape'],dtype=v['dtype'])
 assert seen
 out=R/'universal_eval/model/dotsocr_own_original_sources_20261115';out.mkdir(exist_ok=True);sources=[]
 for p in W.glob('*.py'):
  raw=p.read_bytes();ast.parse(raw);dest=out/p.name
  if dest.exists():assert sha(dest)==sha(p)
  else:dest.write_bytes(raw)
  sources.append(dict(own_source=str(p),primary_source_copy=str(dest),SHA256=sha(p),all_original_functions_unchanged=True))
 index=W/'model.safetensors.index.json'
 if index.exists():assert set(json.loads(index.read_bytes())['weight_map'])==set(seen)
 saved=[p for p in (R/'configs').rglob(MID.replace('/','--')+'.yaml')];assert len(saved)==1
 put('resource_source_review.json',dict(UTC=datetime.now(timezone.utc).isoformat(),resource_source_preflight_completed=True,model_id=MID,weight_path=str(W),own_weight_files=weights,own_tensor_headers=seen,own_original_source_copies=sources,original_saved_YAML=str(saved[0]),original_saved_YAML_SHA256=sha(saved[0]),canonical_task=entry['task'],own_assets_SHA256={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.model','.jinja','.txt']},no_NPU_or_learned_CPU_forward=True,not_a_real3_or_formal200_or_numeric_qualification=True,not_queued_ready_for_NPU_until_adapter_YAML_inputs_and_independent_numeric_guards_completed=True,next_action='Use own original deployment/assets to build independent native adapter/YAML and full numeric verifier; no online A100 prerequisite, no generic mock/stub initializer.',no_canonical_or_G_write=True))
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,next_model_adaptation_resource_stage_completed_not_NPU_running=True));print(json.dumps(dict(resource_source_preflight_completed=True,model_id=MID,home=str(H),own_source_count=len(sources),own_tensor_header_count=len(seen)),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_NPU_or_canonical_write=True))
  raise
