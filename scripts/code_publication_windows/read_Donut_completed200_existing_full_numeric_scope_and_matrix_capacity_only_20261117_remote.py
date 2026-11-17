import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';folder=R/'workspace/reports/katanaml-org_invoices-donut-model-v1/image-to-text/20261117_045854';js=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
line=js(folder/'operator_lineage.json')['actual_execution'];rows=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()];assert len(rows)==200;keep=set(range(10))|set(range(90,100))|set(range(190,200));files=[]
for e in rows:
 p=Path(e['archive']);s=p.stat();assert p.is_relative_to(R/'workspace/diagnostics/model_runs');files.append(dict(index=e['index'],path=str(p),bytes=s.st_size,keep_as_original_group_example=e['index'] in keep,original_archive_SHA256=e['archive_SHA256']))
diag=B/'DonutInvoice_existing_complete200_v6_present_raw_full_numeric_independent_diagnostic_real_FLOCK_20261117';meta={}
for p in diag.iterdir():
 if not p.is_file() or p.suffix not in ['.json','.log']:continue
 raw=p.read_bytes()
 try:v=json.loads(raw)
 except Exception:v=raw.decode()
 if isinstance(v,dict):v={k:x for k,x in v.items() if k not in ['numeric_sample_checks','external_evidence_sha256']}
 meta[str(p)]=dict(SHA256=sha(p),value=v)
p=Path(line['protocol_file']);protocol=js(p);job=Path(protocol['evidence_job']);home=job.parent
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),own_actual200_count=200,actual_forward_count=sum(e['actual_native_forwards'] for e in rows),full_five_SHA256={n:sha(folder/n) for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']},matrix_stat_only=files,total_matrix_bytes=sum(x['bytes'] for x in files),retained_original30_examples_bytes=sum(x['bytes'] for x in files if x['keep_as_original_group_example']),potential_nonexample170_bytes=sum(x['bytes'] for x in files if not x['keep_as_original_group_example']),existing_NF6_diagnostic_metadata=meta,original_job=js(job),original_completion=js(home/'completion.json'),original_formal200_exit_metadata={p.name:js(p) for p in (home/'formal200').iterdir() if p.is_file() and p.name in ['exit.json','returncodes.json','accepted.json']},own_lineage=line,own_protocol=protocol,free_bytes=shutil.disk_usage(R).free,no_model_or_NF_run_or_file_or_array_deletion=True),ensure_ascii=False))
