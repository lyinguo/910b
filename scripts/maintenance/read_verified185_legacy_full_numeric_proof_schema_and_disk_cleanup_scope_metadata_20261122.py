"""Inspect old accepted-proof schema and physical dedup scope, never infer numeric acceptance from score."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def rec(p):
    b=Path(p).read_bytes()
    try:v=json.loads(b)
    except ValueError:v=b.decode('utf8',errors='replace')
    return dict(path=str(p),SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),complete_value=v)
ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
old=B/'verified185_current_full_accepted_correct_nonexample_metadata_reaudit_Turkish_completed_20261122/discovery.json'
assert sha(old)=='2eec9de701b82477a646d7e2762277c73ca21f5fda93185cdf4d1673f63da068'
previous=js(old);proofs=[]
for row in previous['skipped']:
    if row['reason']!='Numeric-only legacy proof is excluded; complete original full numerical qualification not declared':continue
    p=Path(row['proof']);q=js(p)
    scalars={k:v for k,v in q.items() if isinstance(v,(str,bool,int,float)) or v is None}
    containers={k:dict(type=type(v).__name__,count=len(v),keys=list(v)[:50] if isinstance(v,dict) else None) for k,v in q.items() if isinstance(v,(dict,list))}
    proofs.append(dict(model_id=row['model_id'],proof=str(p),SHA256=sha(p),bytes=p.stat().st_size,all_scalar_fields=scalars,all_container_fields=containers))
cfgfile=R/'configs/maintenance/batch_910b_immediate_disk_recovery_cleanup_v3_20261120.json';cfg=js(cfgfile)
consumer=R/cfg['urgent_policy_consumer'];assert consumer.resolve().is_relative_to(R/'scripts')
watch=D/'immediate_disk_recovery_cleanup_watch_20261120';state=js(watch/'state.json')
records={str(cfgfile):rec(cfgfile),str(consumer):rec(consumer),str(watch/'state.json'):rec(watch/'state.json')}
last=state.get('last_cleanup') or {}
for key in ['request','stdout','stderr']:
    p=last.get(key)
    if p and Path(p).is_file():records[p]=rec(Path(p))
pub=B/'Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122'
audit=js(pub/'full_target_dual_publication_audit.json')
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),legacy_proofs_not_assumed_numeric_only_or_accepted=proofs,cleanup_actual_scope_and_current_full_error_context=records,current185_target_audit_keys=list(audit),current185_target_audit_scalar_fields={k:v for k,v in audit.items() if not isinstance(v,(dict,list))},current185_target_audit_container_keys={k:list(v)[:100] if isinstance(v,dict) else dict(type=type(v).__name__,count=len(v)) for k,v in audit.items() if isinstance(v,(dict,list))},no_deletion_no_acceptance_no_array_replay_or_model_source_change=True),ensure_ascii=False))
