import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');canon={n:sha(n) for n in ptr['canonical_SHA256']};assert canon==ptr['canonical_SHA256']
homes=[B/'MetaMllama_completed_true3_complete_shaped_raw_Fortran_LZMA_storage_capacity_diagnostic_only_20261120',B/'MetaMllama_completed_true3_complete_natural_order_byteplanes_LZMA_storage_v2_diagnostic_only_20261120',B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_wait_actual_capacity_first200_serial_20261120',B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'];files={}
for home in homes:
 for n in ['state.json','progress.json','actual_completion.json','review.json','all200_complete_storage_capacity_review.json']:
  p=home/n
  if p.is_file():
   x=js(p);v={k:z for k,z in x.items() if k not in ['all_three_complete_archive_reviews','all_complete_ZIP_members']};files[str(p)]=dict(SHA256=sha(p),bytes=p.stat().st_size,value=v)
procs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
  if argv and Path(argv[0]).name.startswith('python') and any(any(k in s.lower() for k in ['metamllama','meta_mllama','glm41thinking','a100_alignment_evidence_runner','/run.py']) for s in argv[1:]):procs.append(dict(PID=int(p.name),argv=argv,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),io=(p/'io').read_text()))
 except FileNotFoundError:continue
cards=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert cards.returncode==0
v=dict(UTC=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(R).free,canonical_SHA256=canon,all183_canonical_and_ledger_unchanged=True,original_FLOCK=runner_lock_owners(),actual_processes=procs,files=files,npu_smi_complete=cards.stdout,max_concurrent_models=1)
print(json.dumps(v,ensure_ascii=False))
