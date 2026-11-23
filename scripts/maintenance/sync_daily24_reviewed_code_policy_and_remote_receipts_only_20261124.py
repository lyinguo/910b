"""After verified normal push, CAS only the actual publisher policy and archive exact receipts."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=Path(sys.argv[1]);assert sha(p)==sys.argv[2];q=json.loads(p.read_bytes());receipt=q['remote_verified_receipt']
assert receipt['passed'] and receipt['day']=='2026-11-24' and receipt['commit']==receipt['remote_commit']
assert receipt['repository']=='https://github.com/lyinguo/910b.git' and receipt['branch']=='codex/910b-source-layout-20261114'
ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());canonical=ptr['canonical_SHA256']
assert all(sha(n)==h for n,h in canonical.items())
policy=R/'configs/publication/code_only.json';before=policy.read_bytes();assert hashlib.sha256(before).hexdigest()==q['policy_before_SHA256']
after=base64.b64decode(q['policy_after_base64'],validate=True);assert hashlib.sha256(after).hexdigest()==q['policy_after_SHA256']
sys.path[:0]=[str(R/'scripts')]
import publish_code_daily as pub
assert sha(R/'scripts/publish_code_daily.py')=='e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30'
new=json.loads(after);assert new['repository']==receipt['repository'] and new['branch']==receipt['branch']
for n,e in new['files'].items():pub.validate_path(n,e['role'])
for n,h in q['source_SHA256'].items():assert sha(R/n)==h,n
cfg=R/'configs/queues/BeeSFT_completed_storage_NF_capacity_v5_verified185_exact137_wait_actual_capacity_serial_20261122.json';bq=json.loads(cfg.read_bytes())
assert str(policy) not in bq['canonical_after_verified185_and_exact137_cleanup_SHA256'] and 'configs/publication/code_only.json' not in bq['source_sha256']
H=C/'daily_20261124_actual_reviewed_commit_remote_receipts';H.mkdir(exist_ok=False)
(H/'code_only_before.original.json').write_bytes(before)
for z in q['local_original_records']:
    b=base64.b64decode(z['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==z['SHA256']
    p=H/z['name'];assert p.resolve().is_relative_to(H);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
assert policy.read_bytes()==before and all(sha(n)==h for n,h in canonical.items())
t=policy.with_name(policy.name+'.tmp.'+str(os.getpid()));t.write_bytes(after);os.replace(t,policy)
assert sha(policy)==q['policy_after_SHA256'] and all(sha(n)==h for n,h in canonical.items())
result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,Code_Day='2026-11-24',commit=receipt['commit'],remote_commit=receipt['remote_commit'],
            reviewed_plan_SHA256=receipt['reviewed_plan_SHA256'],policy_before_SHA256=q['policy_before_SHA256'],policy_after_SHA256=q['policy_after_SHA256'],
            only_R_publication_policy_CAS_changed=True,canonical_SHA256=canonical,all_model_sources_jobs_tables_registry_state_PAUSE_and_existing_waiters_unchanged=True,
            receipt_home=str(H),receipt_SHA256=sha(H/'daily_20261124_receipt.json'),no_NN_acceptance_or_G_claim=True,no_second_commit=True)
(H/'R_policy_exact_CAS_actual.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
