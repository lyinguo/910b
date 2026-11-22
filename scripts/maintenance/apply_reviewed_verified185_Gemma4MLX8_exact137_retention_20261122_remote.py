import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,subprocess,os,json
R=Path('/root/projects/comperror');p=R/'scripts/maintenance/retain_verified185_Gemma4MLX8_accepted_correct_nonexample137_exact_20261122.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='47129787c4974d9067c13d92d8224a6c7159deeb52cb033ab29c11d6eb257c8f'
B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'verified185_Gemma4MLX8_fullNF2_dual_audit_correct_nonexample137_exact_retention_20261122';assert hashlib.sha256((H/'plan.json').read_bytes()).hexdigest()=='1045c51dab0fd58365eaf33955a9dd677ac33638d36551be771a1feb1b70e6f0'
I=B/'verified185_Gemma4MLX8_reviewed_exact_retention_apply_launcher_20261122';assert not I.exists();I.mkdir()
with (I/'stdout.json').open('xb') as out,(I/'stderr.log').open('xb') as err:q=subprocess.Popen(['python3','-B',str(p),'apply','1045c51dab0fd58365eaf33955a9dd677ac33638d36551be771a1feb1b70e6f0'],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(PID=q.pid,exact137_correct_nonexamples_only=True,reviewed_plan_SHA256='1045c51dab0fd58365eaf33955a9dd677ac33638d36551be771a1feb1b70e6f0',actual_cleanup_not_yet_claimed=True);(I/'launch.json').write_text(json.dumps(v),encoding='utf-8');print(json.dumps(v))
