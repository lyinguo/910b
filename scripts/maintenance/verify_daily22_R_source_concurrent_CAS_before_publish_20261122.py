"""Fresh explicitly declared original source and active model protection audit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
def main():
    plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=json.loads(plan.read_bytes());assert q['reviewed_plan_SHA256']=='93b931d17235cbeeac79d32e620eb6fb31434f4d63ce6e58d3410ccbf8a447e0'
    pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(pointer)=='eb655ed07ad02616695caed245f33c6faa064741c9db4051f3a86161091a1dcd';ptr=json.loads(pointer.read_bytes());canonical=ptr['canonical_SHA256'];assert all(sha(path)==digest for path,digest in canonical.items())
    for name,digest in q['source_SHA256'].items():
        path=R/name;assert path.resolve().is_relative_to(R) and path.is_file() and sha(path)==digest,name
    B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'LlavaMini_own_original_FP16_2048_true3_v2_actual_reserved_devices_real_FLOCK_20261122';assert sha(J/'job.json')=='99994d88480b6b9fef627de652cbf1837b05080650389d5243676e3a2787f9b7';job=json.loads((J/'job.json').read_bytes())
    assert 'configs/publication/code_only.json' not in job['spec']['source_sha256'] and str(R/'configs/publication/code_only.json') not in job['protected_canonical_hashes']
    assert all(sha(R/name)==digest for name,digest in job['spec']['source_sha256'].items())
    assert all(sha(path)==digest for path,digest in canonical.items())
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,reviewed_plan_SHA256=q['reviewed_plan_SHA256'],declared_source_files_fresh_SHA256_checked=len(q['source_SHA256']),all_active_LlavaMini_sources_job_and_canonical_unchanged=True,code_publication_policy_not_an_active_model_source=True,canonical_SHA256=canonical,original_pointer_SHA256=sha(pointer),original_policy_SHA256=sha(R/'configs/publication/code_only.json'),no_model_NN_NF_or_job_or_table_mutation=True)
    home=C/'daily_20261122_actual_source_concurrency_CAS_before_publish';home.mkdir(exist_ok=False);(home/'actual_source_CAS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
