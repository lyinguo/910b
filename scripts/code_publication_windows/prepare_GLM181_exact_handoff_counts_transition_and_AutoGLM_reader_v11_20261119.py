"""Extend exact post-publication handoff metadata; never change model artifacts."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent
old='close_alignment_handoff_exact_ZaiGLM46_qualified181_dual_publication_only_v10_actual_verifier_20261119.py'
new='close_alignment_handoff_exact_ZaiGLM46_qualified181_dual_publication_only_v11_actual_counts_20261119.py'
s=(P/old).read_text(encoding='utf8')
anchor="ptr=js(pointer);assert sha(pointer)==result['pointer_SHA256'];ptr.update(canonical_SHA256=current,"
replacement="counts=dict(delivered=181,numeric_within=166,numeric_outside=15,complete_weight_pending_delivery=209,confirmed_missing=310,full_protocol_qualified=67,numeric_only=99,complete_weight_still_needing_adaptation_qualification_or_repair=323)\n ptr=js(pointer);assert sha(pointer)==result['pointer_SHA256'];ptr.update(canonical_counts=counts,current_canonical_counts=counts,verified_delivered_count=181,last_complete_result_BJT='2026-11-19 22:31:45.646595+08:00',current_formal_acceptance_completed=True,current_formal_acceptance_home=str(J),canonical_SHA256=current,"
assert s.count(anchor)==1;s=s.replace(anchor,replacement).replace('closure_actual_receipt_v10_exact_qualified_publication','closure_actual_receipt_v11_exact_qualified_publication')
ast.parse(s);p=P/new;assert not p.exists();p.write_text(s,encoding='utf8',newline='\n')
oldreader='read_AutoGLM_serial_v3_actual_preparation_job_trace_source_and_locks_20261119_remote.py'
newreader='read_AutoGLM_serial_v3_actual_all_child_FDs_job_trace_sources_and_locks_v2_20261119_remote.py'
s=(P/oldreader).read_text(encoding='utf8');s=s.replace('procs={}\n','pids.update(p[\'PID\'] for p in project_processes() if \'AutoGLM\' in str(p) or p.get(\'role\') in [\'alignment_runner\',\'model\',\'CPU_verifier_or_audit\'])\nprocs={}\n');ast.parse(s);p=P/newreader;assert not p.exists();p.write_text(s,encoding='utf8',newline='\n')
print(json.dumps(dict(new_exact_count_closure=new,closure_SHA256=hashlib.sha256((P/new).read_bytes()).hexdigest(),new_actual_child_reader=newreader,reader_SHA256=hashlib.sha256((P/newreader).read_bytes()).hexdigest(),no_model_or_canonical_write=True)))
