"""Read actual sole NN, next own exact input proof and source-only CPU constructor."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def rec(p):
 b=p.read_bytes()
 try:v=json.loads(b)
 except ValueError:v=b.decode('utf8',errors='replace')
 return dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),complete_value=v)
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_GLM41Base_own_probability_true3_serial_full_actual_v1_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr;x=json.loads(r.stdout);records={}
for stem in ['TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122','TurkishLlava_own_original_frontend200_no_NN_actual_launcher_v1_20261122','TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_exact_own_template_20261122','TurkishLlava_own_original_frontend200_v2_exact_own_template_no_NN_actual_launcher_20261122','TurkishLlava_own_original_auto_FP16_actual_CPU_loader_resize_remap_no_forward_v1_20261122','TurkishLlava_own_frontend_v2_then_original_CPU_resize128258_no_NN_20261122','TurkishLlava_own_CPU_constructor_source_no_forward_actual_launcher_v1_20261122']:
 for name in ['state.json','progress.json','readiness.json','actual_completion.json','actual_failure.json','actual_original_loader_resize_remap_review.json','original_from_pretrained_loading_info_collection_types.json','original_from_pretrained_loading_info.json','own_original_legacy_mapping_meta_review.json','launch.json','stderr.log','stdout.json','constructor_stderr.log','constructor_stdout.json']:
  p=B/stem/name
  if p.is_file():records[str(p)]=rec(p)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),same_GLM41Base_only_actual_serial_model=x,Turkish_own_exact_frontend_and_CPU_source_only_complete_records=records,Turkish_no_learned_CPU_or_NPU_forward=True,no_source_job_config_deadline_signal_or_canonical_mutation=True),ensure_ascii=False))
