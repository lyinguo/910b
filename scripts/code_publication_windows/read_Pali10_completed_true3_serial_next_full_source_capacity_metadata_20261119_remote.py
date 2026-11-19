import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,zipfile
from datetime import datetime,timezone
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J=B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117'
js=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
job=js(J/'job.json');done=js(J/'completion.json');out=dict(UTC=datetime.now(timezone.utc).isoformat(),job=job,job_SHA256=sha(J/'job.json'),completion=done,proof=None,sources={},archives=[],current_state=js(D/'state.json')['models']['google/paligemma2-10b-pt-448'])
assert done['actual_returncode']==0 and done['verifier_returncode']==0
accepted,=done['accepted'];p=Path(accepted['proof']);assert sha(p)==accepted['proof_sha256'];proof=js(p);out['proof']={k:v for k,v in proof.items() if k not in ['numeric_sample_checks','external_evidence_sha256']};out['proof_external_keys']=list(proof.get('external_evidence_sha256',{}));out['numeric_keys']=list(proof.get('numeric_sample_checks',[{}])[0])
for rel in job['spec']['source_sha256']:
 if any(s in rel for s in ['pali10pt448_full_native','named_worker_RPC','verify_pali10pt','full_native_value']):
  p=R/rel
  if p.suffix in ['.py','.yaml','.json']:out['sources'][rel]=dict(SHA256=sha(p),text=p.read_text())
line=js(Path(accepted['report_dir'])/'operator_lineage.json')['actual_execution'];out['lineage_execution']=line
trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()]
out['trace']=trace
for row in trace:
 p=Path(row['actual_execution_file']);e=js(p);out['archives'].append(dict(execution=e,execution_path=str(p),execution_SHA256=sha(p)))
print(json.dumps(out,ensure_ascii=False))
