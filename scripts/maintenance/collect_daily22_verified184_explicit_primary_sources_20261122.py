import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
pointer=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());pending=Path(pointer['new_primary_pending']);digest=hashlib.sha256(pending.read_bytes()).hexdigest()
assert digest==pointer['new_primary_pending_SHA256']=='f541f6ed94f928b8b695686587c2829d669c60739f312172b848c97f9782b0d8'
assert pointer['current_verified_count']==184
assert hashlib.sha256((R/'scripts/collect_code_publication_primary_sources.py').read_bytes()).hexdigest()=='260122f226c69a9b3a349115ea83fe6654fe236e1f5c98aa3f6862fd933ed604'
extra=['scripts/maintenance/read_heartbeat2352_verified184_mandatory_full_and_latest_20261121_remote.py', 'scripts/maintenance/read_verified184_pending_owned_model_inventory_20261121_2352_remote.py', 'scripts/maintenance/read_next_owned_GLMBase_GemmaAssistant_LlavaMini_saved_deployment_metadata_20261122_remote.py', 'scripts/maintenance/read_GLMBase_Assistant_LlavaMini_original_adapter_and_D_contract_20261122_remote.py', 'scripts/maintenance/read_LlavaMini_own_processor_meta_interface_for_original_FP16_20261122_remote.py', 'scripts/maintenance/collect_daily22_verified184_explicit_primary_sources_20261122.py']
args=['python3','-B',str(R/'scripts/collect_code_publication_primary_sources.py'),'--pending',str(pending),'--pending-sha',digest,'--review-home',str(C/'daily_20261122_explicit_primary_source_review_verified184')]
for p in extra:args+=['--include',p]
r=subprocess.run(args,cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)
