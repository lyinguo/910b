import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,base64,hashlib,ast
P=Path(__file__).parent
names={'universal_eval/utils/lossless_complete_alias_npz_fortran_head_20261117.py':'lossless_complete_alias_npz_fortran_head_20261117.py','scripts/diagnostic_Prithiv_completed_card2_true3_lossless_standard_Fortran_head_no_NN_20261117.py':'diagnostic_Prithiv_completed_card2_true3_lossless_standard_Fortran_head_no_NN_20261117.py','scripts/tests/test_lossless_complete_alias_standard_Fortran_head_20261117.py':'test_lossless_complete_alias_standard_Fortran_head_20261117.py'}
files={}
for n,f in names.items():
 b=(P/f).read_bytes();ast.parse(b);files[n]=dict(data=base64.b64encode(b).decode(),SHA256=hashlib.sha256(b).hexdigest())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,base64,hashlib,ast,subprocess,os
R=Path('/root/projects/comperror');F=__F__;os.chdir('/root')
for n,v in F.items():
 p=R/n;assert not p.exists();raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);p.write_bytes(raw)
r=subprocess.run(['python3','-B',str(R/'scripts/tests/test_lossless_complete_alias_standard_Fortran_head_20261117.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));print(json.dumps(dict(actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,sources={n:v['SHA256'] for n,v in F.items()},no_NN_no_canonical_or_old_files_mutated=True),ensure_ascii=False));raise SystemExit(r.returncode)
'''.replace('__F__',repr(files));ast.parse(remote);(P/'deploy_test_Prithiv_complete_NPY_lossless_storage_no_NN_20261117_remote.py').write_bytes(remote.encode())
launch='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';os.chdir('/root');sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not project_processes() and not runner_lock_owners()
S=R/'scripts/diagnostic_Prithiv_completed_card2_true3_lossless_standard_Fortran_head_no_NN_20261117.py';assert hashlib.sha256(S.read_bytes()).hexdigest()=='__SH__'
H=B/'Prithiv_complete_NPY_lossless_storage_diagnostic_launch_control_20261117';H.mkdir(exist_ok=False)
with (H/'stdout.log').open('xb') as out,(H/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',str(S)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
r=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=str(S),source_SHA256='__SH__',no_NEW_NPU=True,scope='completed own card2 numerical3 full-value lossless storage diagnostic only, no score or new qualification');(H/'background_launch.json').write_bytes(json.dumps(r,ensure_ascii=False,indent=2).encode());print(json.dumps(r,ensure_ascii=False))
'''.replace('__SH__',files['scripts/diagnostic_Prithiv_completed_card2_true3_lossless_standard_Fortran_head_no_NN_20261117.py']['SHA256']);ast.parse(launch);(P/'launch_Prithiv_complete_NPY_lossless_storage_completed3_only_20261117_remote.py').write_bytes(launch.encode());print(json.dumps({n:v['SHA256'] for n,v in files.items()}))
