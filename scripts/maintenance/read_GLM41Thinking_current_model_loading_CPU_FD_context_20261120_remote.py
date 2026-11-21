import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,os
R=Path('/root/projects/comperror');out=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
  if not argv or not Path(argv[0]).name.startswith('python') or not any('GLM-4.1V-9B-Thinking-own-original' in x for x in argv):continue
  fds={}
  for fd in (p/'fd').iterdir():
   try:fds[fd.name]=os.readlink(fd)
   except FileNotFoundError:continue
  out.append(dict(PID=p.name,argv=argv,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),status=(p/'status').read_text(),io=(p/'io').read_text(),wchan=(p/'wchan').read_text(),FD_targets_only_no_file_contents=fds,thread_stats={t.name:(t/'stat').read_text() for t in (p/'task').iterdir()}))
 except FileNotFoundError:continue
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),read_only_no_signal_or_model_state_change=True,actual_model_loading_context=out),ensure_ascii=False))
