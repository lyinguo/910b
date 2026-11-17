import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,hashlib,zipfile
os.chdir('/root');R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'OvisU1_own_original_BF16_512_true3_v2_actual_empty_precision_guard_real_FLOCK_20261116';js=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();job=js(J/'job.json');done=js(J/'completion.json');a=done['accepted'][-1];proof=js(a['proof']);line=js(Path(a['report_dir'])/'operator_lineage.json')['actual_execution'];sources={}
for n in set(job['spec']['source_sha256'])|{'universal_eval/utils/lossless_complete_alias_npz_fortran_head_20261117.py','scripts/diagnostic_Prithiv_completed_card2_true3_lossless_standard_Fortran_head_no_NN_20261117.py'}:
 p=R/n
 if p.exists() and p.suffix in ['.py','.json','.yaml'] and p.stat().st_size<150000:sources[n]=dict(SHA256=sha(p),text=p.read_text())
print(json.dumps(dict(job=job,completion=done,proof=proof,actual_execution=line,source_files=sources,no_model_or_old_value_mutation=True),ensure_ascii=False))
