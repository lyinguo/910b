from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
P=Path(__file__).resolve().parent;book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');raw=book.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h=='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34';s=book.stat()
status=subprocess.run(['git','-C','D:/code-lv/910b','status','--porcelain=v1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True);head=subprocess.run(['git','-C','D:/code-lv/910b','rev-parse','HEAD'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
assert not status.stdout and head.stdout.decode().strip()=='e8da8ad87ae3c9d42e6ff73c350d80c0bec7cd4d'
v=dict(UTC=datetime.now(timezone.utc).isoformat(),local_book=str(book),SHA256=h,stat=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns),verified180_bytes_unchanged=True,clean_code_worktree=True,local_HEAD=head.stdout.decode().strip(),no_workbook_write_or_second_daily_commit=True)
(P/'Huihui_gap105_GLM_serial_verified180_local_current_book_code_stat_20261119.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(v,ensure_ascii=False))
