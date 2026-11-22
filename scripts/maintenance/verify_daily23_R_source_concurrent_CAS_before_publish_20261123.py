"""Fresh explicit source hashes and actual current waiters before the reviewed code-only transaction."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=Path(sys.argv[1]);assert sha(p)==sys.argv[2];q=json.loads(p.read_bytes())
ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());assert sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json')==q['pointer_SHA256']
assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
for n,h in q['source_SHA256'].items():
    p=R/n;assert p.resolve().is_relative_to(R) and p.is_file() and sha(p)==h,n
policy=R/'configs/publication/code_only.json';assert sha(policy)==q['policy_before_SHA256']
cfg=R/'configs/queues/BeeSFT_completed_storage_NF_capacity_v5_verified185_exact137_wait_actual_capacity_serial_20261122.json'
bq=json.loads(cfg.read_bytes());assert str(policy) not in bq['canonical_after_verified185_and_exact137_cleanup_SHA256']
assert 'configs/publication/code_only.json' not in bq['source_sha256']
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
procs=project_processes();owners=runner_lock_owners()
result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,reviewed_plan_SHA256=q['reviewed_plan_SHA256'],
            declared_source_files_fresh_SHA256_checked=len(q['source_SHA256']),all_original_model_sources_jobs_and_canonical_unchanged=True,
            code_publication_policy_not_a_model_or_existing_Bee_waiter_protected_source=True,canonical_SHA256=ptr['canonical_SHA256'],
            original_pointer_SHA256=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'),original_policy_SHA256=sha(policy),
            actual_project_processes=procs,actual_original_FLOCK=owners,no_model_NN_NF_job_table_mutation=True)
H=C/'daily_20261123_actual_source_concurrency_CAS_before_publish';H.mkdir(exist_ok=False)
(H/'actual_source_CAS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
