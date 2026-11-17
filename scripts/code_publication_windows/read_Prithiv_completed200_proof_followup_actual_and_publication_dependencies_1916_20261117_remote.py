import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,os
os.chdir('/root');R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'Prithiv_completed_card2_true3_verified179_capacity_approved_NEWfresh3_fixed200_original_real_FLOCK_20261117';H=B/'after_Prithiv_completed_card2_real_release_immediate_original_resource_queue_20261117';o={}
for p in H.iterdir():
 if p.is_file() and p.suffix in ['.json','.log'] and p.stat().st_size<200000:
  raw=p.read_bytes()
  try:v=json.loads(raw)
  except Exception:v=raw.decode(errors='replace')
  o[str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=v)
done=json.loads((J/'completion.json').read_bytes());a=done['accepted'][-1];proof=json.loads(Path(a['proof']).read_bytes());assert hashlib.sha256(Path(a['proof']).read_bytes()).hexdigest()==a['proof_sha256'];f=Path(a['report_dir']);rr=json.loads((f/'eval_report.json').read_bytes());paths=['scripts/verified_A100_G_cell_writer_20261103.py','scripts/publish_MervePaliCompact_verified179_dual_transaction_v3_exact_negative_fixture_20261117.py','scripts/publish_Granite41_verified178_dual_transaction_20261117.py']
sources={}
for name in paths:
 p=R/name
 if p.exists():sources[name]=dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text())
print(json.dumps(dict(followup_actual_files=o,full_proof=proof,report_summary=rr['summary'],statistics=rr['statistics'],actual_execution=json.loads((f/'operator_lineage.json').read_bytes())['actual_execution'],source_files=sources),ensure_ascii=False))
