import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'PrithivGemma4E4B_existing_model_first_independent_card2_true3_parallel_v1_20261117'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();js=lambda p:json.loads(Path(p).read_bytes())
j=js(J/'job.json');c=js(J/'completion.json');assert c['actual_returncode']==c['verifier_returncode']==0;a=c['accepted'][0];assert a['proof_sha256']=='94f5649a4117467baae93dee177a5b794fc4d171b4ae8d962c9067bfc251f645' and sha(a['proof'])==a['proof_sha256'];proof=js(a['proof']);folder=Path(a['report_dir']);line=js(folder/'operator_lineage.json')['actual_execution'];trace=Path(line['sample_trace']);rows=[json.loads(x) for x in trace.read_text().splitlines()];archives=[]
for x in rows:
 e=js(x['actual_execution_file']);p=Path(e['archive']);assert sha(p)==e['archive_SHA256']
 with zipfile.ZipFile(p) as z:parts=[dict(name=i.filename,uncompressed=i.file_size,compressed=i.compress_size) for i in z.infolist()]
 archives.append(dict(index=e['index'],archive=str(p),archive_SHA256=sha(p),archive_bytes=p.stat().st_size,array_occurrence_count=len(parts),complete_uncompressed_bytes=sum(i['uncompressed'] for i in parts),largest_complete_values=sorted(parts,key=lambda v:v['uncompressed'],reverse=True)[:12],execution_metadata=x['actual_execution_file'],execution_metadata_SHA256=sha(x['actual_execution_file'])))
sources={}
for n,h in j['spec']['source_sha256'].items():
 if 'prithiv' in n.lower() or 'Prithiv' in n or 'lossless_npz_aliases' in n or 'readonly_npz' in n:
  p=R/n;assert sha(p)==h;sources[n]=dict(SHA256=h,text=p.read_text())
np=R/'universal_eval/utils/readonly_npz_indexed.py';sources[str(np.relative_to(R))]=dict(SHA256=sha(np),text=np.read_text())
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),job=j,completion=c,proof_keys=list(proof),source_files=sources,lineage_actual_execution=line,archives=archives,free_bytes=shutil.disk_usage(R).free,no_NN_no_old_source_or_value_or_report_or_proof_mutation=True),ensure_ascii=False))
