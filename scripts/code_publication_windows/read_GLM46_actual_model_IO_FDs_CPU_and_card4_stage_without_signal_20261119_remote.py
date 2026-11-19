import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,time
pid=1574589;p=Path('/proc')/str(pid)
def sample():
 stat=(p/'stat').read_text();v=stat.rsplit(')',1)[1].split();io={k:int(n.strip()) for k,n in (x.split(':',1) for x in (p/'io').read_text().splitlines())};return dict(UTC=datetime.now(timezone.utc).isoformat(),ticks=int(v[11])+int(v[12]),io=io,status=(p/'status').read_text(),stat=stat)
if not p.exists():print(json.dumps(dict(actual_process_exists=False,actual_NEW_NN=False)));sys.exit(0)
args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x];assert any(Path(x).name=='run.py' for x in args)
a=sample();fds={}
for f in (p/'fd').iterdir():
 try:target=os.readlink(f)
 except FileNotFoundError:continue
 if '/workspace/' in target and not target.startswith('anon_'):fds[f.name]=target
t=time.monotonic();time.sleep(1);b=sample();elapsed=time.monotonic()-t
result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert result.returncode==0
print(json.dumps(dict(UTC=b['UTC'],PID=pid,actual_argv=args,cwd=os.readlink(p/'cwd'),first=a,second=b,observed_seconds=elapsed,observed_CPU_core_equivalents=(b['ticks']-a['ticks'])/os.sysconf('SC_CLK_TCK')/elapsed,write_bytes_delta=b['io']['write_bytes']-a['io']['write_bytes'],real_workspace_FDs=fds,full_npu_smi=result.stdout,no_signal_or_model_source_job_or_deadline_change=True),ensure_ascii=False))
