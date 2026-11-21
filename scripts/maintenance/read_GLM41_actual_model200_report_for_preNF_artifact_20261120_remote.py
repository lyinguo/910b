import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
s=json.loads((J/'state.json').read_bytes())
if s.get('phase')=='running_formal200':
 print(json.dumps(dict(model200_complete=False,phase=s['phase'],not_fullNF_or_qualification=True)));raise SystemExit(0)
folder=Path(s.get('report_dir','/nonexistent'))
if not (folder/'eval_report.json').is_file():
 print(json.dumps(dict(model200_complete=False,phase=s.get('phase'),report_not_yet_closed=True)));raise SystemExit(0)
report=json.loads((folder/'eval_report.json').read_bytes());config=report['pipeline_config']
if config.get('sample_count')!=200:
 print(json.dumps(dict(model200_complete=False,phase=s.get('phase'),state_report_is_previous_fresh3=True)));raise SystemExit(0)
assert set(x.name for x in folder.iterdir())=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
assert config['model']['name']=='zai-org/GLM-4.1V-9B-Thinking' and config['sample_range']=='0:199'
metric=report['metrics']
print(json.dumps(dict(model200_complete=True,report_dir=str(folder),metrics=metric,pipeline_config=config,model_returncode=s['model_returncode'],phase=s['phase'],report_SHA256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()},not_fullNF_or_qualification_until_real_independent_verification=True),ensure_ascii=False))
