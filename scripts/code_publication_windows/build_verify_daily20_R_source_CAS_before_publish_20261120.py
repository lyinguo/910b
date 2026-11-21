from pathlib import Path
import ast, hashlib, json

P = Path(__file__).resolve().parent
packet = json.loads((P / 'R_daily20_current_declared_primary_sources_v2_20261120.json').read_bytes())
current = json.loads((P / 'R_AutoGLMPhone9BSingle_current_v1_v2_actual3_20261120.json').read_bytes())
review = json.loads((P / 'daily_20261120_plan_review.json').read_bytes())
assert review['passed'] and review['reviewed_changes'] == 319 and review['reviewed_explicit_paths'] == 2339
data = dict(records=[{k: v[k] for k in ('path', 'SHA256', 'bytes')} for v in packet['records']],
            policy_before_SHA256=packet['policy_before_SHA256'], pending=packet['pending_manifests'][0],
            canonical=current['canonical_SHA256'], reviewed_plan_SHA256=review['reviewed_plan_SHA256'])
code = 'DATA=' + repr(data) + '\n' + '''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(DATA['pending']['path'])==DATA['pending']['SHA256']
for row in DATA['records']:
 p=(R/row['path']).resolve();assert p.is_relative_to(R) and p.stat().st_size==row['bytes'] and sha(p)==row['SHA256'],row['path']
for name,h in DATA['canonical'].items():assert sha(name)==h,(name,h)
assert sha(R/'configs/publication/code_only.json')==DATA['policy_before_SHA256']
jobs=[]
for name in ['AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v2_exact_frontend_path_real_FLOCK_20261120','AutoGLMPhone9BSingle_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120']:
 J=B/name/'job.json'
 if not J.exists():continue
 job=json.loads(J.read_bytes());assert 'configs/publication/code_only.json' not in job['spec']['source_sha256']
 for rel,h in job['spec']['source_sha256'].items():assert sha(R/rel)==h,rel
 assert not any(str(R/'configs/publication/code_only.json')==str(k) for k in job.get('protected_sha256',{}))
 jobs.append(dict(path=str(J),SHA256=sha(J),source_count=len(job['spec']['source_sha256'])))
assert jobs,'Current own Single job must remain available'
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,declared_source_files_fresh_SHA256_checked=len(DATA['records']),canonical_SHA256=DATA['canonical'],policy_before_SHA256=DATA['policy_before_SHA256'],current_own_Single_job_sources_unchanged=jobs,code_publication_policy_not_an_active_model_source=True,reviewed_plan_SHA256=DATA['reviewed_plan_SHA256'],no_model_or_canonical_or_source_write=True),ensure_ascii=False))
'''
ast.parse(code)
out = P / 'verify_daily20_R_source_CAS_before_publish_20261120_remote.py'
out.write_text(code, encoding='utf-8', newline='\n')
print(json.dumps(dict(helper=out.name, SHA256=hashlib.sha256(code.encode()).hexdigest(), count=len(data['records']))))
