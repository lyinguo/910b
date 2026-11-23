"""Collect only the SHA-bound current pending source inventory and the actual Day24 consumers."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());pending=Path(ptr['new_primary_pending'])
h=hashlib.sha256(pending.read_bytes()).hexdigest()
assert h==ptr['new_primary_pending_SHA256']=='73032ebd63354be27e338950415f9c089e7f044cf0ddffe762b8eca21ee146fd'
assert ptr['current_verified_count']==185
assert hashlib.sha256((R/'scripts/collect_code_publication_primary_sources.py').read_bytes()).hexdigest()=='260122f226c69a9b3a349115ea83fe6654fe236e1f5c98aa3f6862fd933ed604'
extra=['scripts/maintenance/collect_daily24_verified185_explicit_primary_sources_20261124.py', 'scripts/maintenance/run_daily24_actual_R_source_tests_no_model_20261124_remote.py', 'scripts/maintenance/verify_daily24_R_source_concurrent_CAS_before_publish_20261124.py', 'scripts/maintenance/sync_daily24_reviewed_code_policy_and_remote_receipts_only_20261124.py', 'scripts/code_publication_windows/daily24_reviewed_code_publication_controls_20261124.py', 'scripts/code_publication_windows/build_daily24_controls_and_explicit_collection_20261124.py']
args=['python3','-B',str(R/'scripts/collect_code_publication_primary_sources.py'),'--pending',str(pending),
      '--pending-sha',h,'--review-home',str(C/'daily_20261124_explicit_primary_source_review_verified185')]
for n in extra:args+=['--include',n]
r=subprocess.run(args,cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
