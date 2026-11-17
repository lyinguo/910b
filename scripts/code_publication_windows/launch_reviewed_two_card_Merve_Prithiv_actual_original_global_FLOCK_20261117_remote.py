import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,subprocess,json,hashlib
R=Path('/root/projects/comperror');os.chdir('/root')
script=R/'scripts/prepare_launch_Merve_lossless_Prithiv_card2_parallel_v2_20261117.py';queue=R/'configs/queues/Merve_lossless_Prithiv_card2_parallel_v2_20261117.json'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='ad49a61aa6faa1137b1692f2b95357f4c4b0af72ee7035db7e7aaa1a1a1e54ee'
assert hashlib.sha256(queue.read_bytes()).hexdigest()=='994cf74746bf47455e893b4fbfa7ab93946468d768d782e0ff762a9140a7456c'
r=subprocess.run(['python3','-B',str(script),str(queue),'launch'],cwd='/root',capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));print(json.dumps(dict(actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,starts_only_prepared_and_source_bound_private_batch=True),ensure_ascii=False));raise SystemExit(r.returncode)
