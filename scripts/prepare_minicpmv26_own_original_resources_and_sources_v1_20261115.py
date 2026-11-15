"""Next own checkpoint preflight only; no NN, borrowed assets or canonical writes."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,struct,ast,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'MiniCPMV26_own_original_resources_source_preflight_v1_after_DotsOCR_user_continue_20261115'
MID='openbmb/MiniCPM-V-2_6'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 H.mkdir(exist_ok=False);e=json.loads((D/'state.json').read_bytes())['models'][MID];assert e['status']=='needs_adaptation' and e['score'] is None
 W,=[Path(p) for p in e['weight_paths'] if Path(p).is_dir()]
 put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),model_id=MID,phase='next_own_checkpoint_full_shard_SHA_and_original_source_preservation_not_NPU_inference'))
 weights={};headers={}
 for p in sorted(W.glob('*.safetensors')):
  with p.open('rb') as f:size=f.read(8);raw=f.read(struct.unpack('<Q',size)[0]);head=json.loads(raw)
  weights[p.name]=dict(bytes=p.stat().st_size,SHA256=sha(p),header_SHA256=hashlib.sha256(size+raw).hexdigest())
  for n,t in head.items():
   if n!='__metadata__':assert n not in headers;headers[n]=dict(source=p.name,shape=t['shape'],dtype=t['dtype'])
 index=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map'];assert set(headers)==set(index)
 assert all(t['dtype']=='BF16' for t in headers.values())
 target=R/'universal_eval/model/minicpmv26_own_original_20261115';target.mkdir(exist_ok=False);sources=[]
 for p in W.glob('*.py'):
  raw=p.read_bytes();ast.parse(raw);dest=target/p.name;dest.write_bytes(raw);sources.append(dict(original_path=str(p),primary_source=str(dest),SHA256=sha(p),all_original_functions_unchanged=True))
 if not (target/'__init__.py').exists():(target/'__init__.py').write_text('"""Own original MiniCPM-V-2_6 sources; no generic mock/stub initializer."""\n',encoding='utf-8')
 put('resource_source_review.json',dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id=MID,weight_path=str(W),source_tensor_headers=headers,own_weight_files=weights,own_assets_SHA256={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.model','.txt','.jinja']},own_config=json.loads((W/'config.json').read_bytes()),original_sources=sources,saved_YAML=str(R/'configs/MiniCPMV/openbmb--MiniCPM-V-2_6.yaml'),saved_YAML_SHA256=sha(R/'configs/MiniCPMV/openbmb--MiniCPM-V-2_6.yaml'),no_NPU_or_learned_CPU_inference=True,not_a_numeric_or_formal200_qualification=True))
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),original_canonical_and_evidence_unchanged=True))
  raise
