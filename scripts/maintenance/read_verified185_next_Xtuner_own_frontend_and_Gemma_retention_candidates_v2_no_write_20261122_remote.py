import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def read(p):
 if not p.is_file():return None
 b=p.read_bytes()
 try:return json.loads(b)
 except Exception:return b.decode('utf-8',errors='replace')
H=B/'XtunerLlava3_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_original_tokenizer_fallback_20261122';I=B/'XtunerLlava3_own_frontend200_meta_v2_original_tokenizer_fallback_actual_launcher_no_NN_20261122';out=dict(UTC=datetime.now(timezone.utc).isoformat(),next_own_frontend={str(p):read(p) for p in [H/'progress.json',H/'source_progress.json',H/'actual_completion.json',H/'readiness.json',H/'own_original_legacy_mapping_meta_review.json',I/'stderr.log',I/'launch.json']},actual_procs=project_processes(),original_FLOCK=runner_lock_owners(),free_bytes=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize)
pub=B/'Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122';audit=read(pub/'full_target_dual_publication_audit.json');assert audit['passed']
proof=read(Path(audit['target_audits'][0].get('formal_proof',str(B/'Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122/formal200/verification_stdout.json'))))
report=read(Path(proof['report_dir'])/'eval_report.json');line=report['operator_lineage']['actual_execution'];trace_path=Path(line['sample_trace']);trace=[json.loads(s) for s in trace_path.read_text().splitlines()];assert [r['index'] for r in trace]==list(range(200))
wrong={i for i,v in enumerate(report['detailed_results']) if str(v['prediction']).strip()!=str(v['reference']).strip()}
keep=wrong|set(proof['invalid_indices'])|set(proof['empty_indices'])|set(proof['truncated_indices'])|set(range(10))|set(range(90,110))|set(range(190,200))
for e in trace:
 if e.get('error'):keep.add(e['index'])
out['Gemma4MLX8_trace_all_keys']=list(trace[0]);out['Gemma4MLX8_first_last_trace']=[trace[0],trace[-1]];out['Gemma4MLX8_retention_scope']=dict(fullNF200_passed=True,full_dual_target_audit_passed=True,wrong_indices=sorted(wrong),all_wrong_and_original_outer10_complete_first_middle_last_group_indices=sorted(keep),correct_nonexample_candidate_indices=sorted(set(range(200))-keep),no_deletion_yet=True)
stderr=B/'Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122/formal200/verification_stderr.log';out['formal200_verification_stderr_full']=read(stderr)
wait=B/'BeeSFT_completed_storage_NF_capacity_v4_actual_original_disk_waiter_serial_20261121';out['BeeSFT_existing_waiter_actual']={str(p):read(p) for p in [wait/'state.json',wait/'actual_completion.json',wait/'stderr.log']}
print(json.dumps(out,ensure_ascii=False))
