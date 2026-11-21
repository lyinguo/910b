"""Read all ownership/dependency references before considering exact NPZ removal."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'verified183_current_accepted_correct_nonexample_metadata_reaudit_continue_GLM41Thinking_20261120'
from maintenance import retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119 as base
sha,js,put=base.sha,base.js,base.put
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid()]
 d=js(H/'discovery.json');helpers=base.old_helpers(dict(prior_helper_SHA256=d['prior_pure_helper_source_SHA256']));reg=js(D/'registry.json')
 candidates={r['path']:r for s in d['scopes'] for r in s['remove']};accepted={r['model_id']:dict(proof=r['proof'],report_dir=r['report_dir']) for r in d['all_selected_proof_metadata']}
 foreign,metadata=base.sources_and_dependencies(reg,accepted,candidates,helpers)
 selected=[r for n,r in candidates.items() if n not in foreign]
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,discovery_SHA256=sha(H/'discovery.json'),selected=selected,selected_count=len(selected),selected_bytes=sum(r['bytes'] for r in selected),selected_allocated_bytes=sum(r['allocated_bytes'] for r in selected),foreign_references_preserved=foreign,all_dependency_metadata_files=sorted(map(str,metadata)),all_accepted_own_proofs_reports_current_registry_static_dependencies_private_jobs_and_all_process_FD_checked=True,no_NPZ_or_weight_contents_read=True,no_delete=True)
 assert not (H/'foreign_FD_metadata_review.json').exists();put(H/'foreign_FD_metadata_review.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['selected','all_dependency_metadata_files']},ensure_ascii=False))
if __name__=='__main__':main()
