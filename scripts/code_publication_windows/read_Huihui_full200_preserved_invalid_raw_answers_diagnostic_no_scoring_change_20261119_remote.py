import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119'
js=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=J/'formal200/verification_stdout.json';assert sha(proof)=='7dd295080b26203c38fb75519a418162ef35bf865c3902ab66701902a5fb5e54';p=js(proof);folder=Path(p['report_dir']);r=js(folder/'eval_report.json');line=js(folder/'operator_lineage.json')['actual_execution'];trace={x['index']:x for x in [json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()]}
out=[]
for i in p['invalid_indices']:
 x=r['detailed_results'][i];e=js(trace[i]['actual_execution_file']);assert x['index']==i and Path(e['archive']).exists()
 out.append(dict(index=i,prediction=x['prediction'],reference=x['reference'],raw_output=x['raw_output'],actual_model_prompt=e['actual_model_prompt'],own_generated_tokens=e['generated_tokens'],native_forward_count=e['actual_native_forwards'],complete_original_numeric_archive_preserved=e['archive'],archive_original_SHA256=e['archive_SHA256']))
assert len(out)==43 and all(sha(folder/n)==h for n,h in p['report_sha256'].items())
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),kind='Read-only original invalid raw answer inspection; no alternative extraction or scoring',full200_original_score=r['metrics'],records=out,all43_original_invalid_values_retained=True,all_five_original_SHA_unchanged=True,no_new_model_no_score_or_extractor_change=True),ensure_ascii=False))
