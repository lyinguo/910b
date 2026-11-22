"""Collect only the SHA-bound current pending source inventory and the actual Day23 consumers."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());pending=Path(ptr['new_primary_pending'])
h=hashlib.sha256(pending.read_bytes()).hexdigest()
assert h==ptr['new_primary_pending_SHA256']=='bc5f4275df7abc884b3f021f096cb0a5934ce061fb6d7843f310dab587059ef8'
assert ptr['current_verified_count']==185
assert hashlib.sha256((R/'scripts/collect_code_publication_primary_sources.py').read_bytes()).hexdigest()=='260122f226c69a9b3a349115ea83fe6654fe236e1f5c98aa3f6862fd933ed604'
extra=['scripts/maintenance/collect_daily23_verified185_explicit_primary_sources_20261123.py','scripts/maintenance/run_daily23_actual_R_source_tests_no_model_20261123_remote.py','scripts/maintenance/run_daily23_actual_R_source_tests_no_model_v2_original_source_arg_20261123_remote.py','scripts/maintenance/verify_daily23_R_source_concurrent_CAS_before_publish_20261123.py','scripts/maintenance/sync_daily23_reviewed_code_policy_and_remote_receipts_only_20261123.py','scripts/code_publication_windows/daily23_reviewed_code_publication_controls_20261123.py','scripts/code_publication_windows/build_daily23_controls_and_explicit_collection_20261123.py']
args=['python3','-B',str(R/'scripts/collect_code_publication_primary_sources.py'),'--pending',str(pending),
      '--pending-sha',h,'--review-home',str(C/'daily_20261123_explicit_primary_source_review_verified185')]
for n in extra:args+=['--include',n]
r=subprocess.run(args,cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
