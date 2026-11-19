import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,re,shutil
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119'
js=lambda p:json.loads(Path(p).read_bytes())
fresh=js(J/'fresh3/verification_stdout.json');folder=R/'workspace/reports/huihui-ai_Huihui-GLM-4.6V-Flash-abliterated/image-text-to-text/20261119_152641';report=js(folder/'eval_report.json');line=js(folder/'operator_lineage.json')['actual_execution'];trace=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()]
correct=[x['index'] for x in report['detailed_results'] if isinstance(x['prediction'],str) and re.fullmatch('[A-E]',x['prediction'].strip().upper()) and x['prediction'].strip().upper()==str(x['reference']).strip().upper()]
outer=report['pipeline_config']['algorithm']['batch_size'];base=set(range(10))|set(range(90,110))|set(range(190,200));examples={i for i in range(200) if i//outer in {n//outer for n in base}}
remove=[]
for x in trace:
 e=js(x['actual_execution_file']);p=Path(e['archive'])
 if x['index'] in correct and x['index'] not in examples and not e.get('error'):remove.append(dict(index=x['index'],path=str(p),bytes=p.stat().st_size,allocated=p.stat().st_blocks*512))
out=dict(UTC=datetime.now(timezone.utc).isoformat(),fresh3_proof_top_level_keys=list(fresh),fresh3_small_fields={k:v for k,v in fresh.items() if not isinstance(v,(list,dict))},fresh3_numeric_check0=fresh['numeric_sample_checks'][0],fresh3_external_map_count=len(fresh['external_evidence_sha256']),fresh3_external_key_first20=list(fresh['external_evidence_sha256'])[:20],correct_count=len(correct),incorrect_indices=sorted(set(range(200))-set(correct)),original_outer_group_size=outer,conservative_full_first_middle_last_original_group_indices=sorted(examples),possible_remove_count=len(remove),possible_remove_bytes=sum(x['bytes'] for x in remove),possible_remove_allocated_bytes=sum(x['allocated'] for x in remove),candidates_metadata=remove,actual_free_bytes=shutil.disk_usage(R).free,no_deletion=True,all_wrong_or_unresolved_values_to_be_preserved=True)
print(json.dumps(out,ensure_ascii=False))
