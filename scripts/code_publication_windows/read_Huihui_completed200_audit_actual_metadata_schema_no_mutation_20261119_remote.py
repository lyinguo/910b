import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119';F=R/'workspace/reports/huihui-ai_Huihui-GLM-4.6V-Flash-abliterated/image-text-to-text/20261119_152641'
js=lambda p:json.loads(Path(p).read_bytes())
j=js(J/'job.json');a=js(j['spec']['alignment_approval_review']);cap=js(a['capacity_review']);l=js(F/'operator_lineage.json')['actual_execution'];t=json.loads(Path(l['sample_trace']).read_text().splitlines()[0]);e=js(t['actual_execution_file']);old=js(B/'PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117/active180_full_report_stat_baseline.json');mid=next(iter(old['reports']))
print(json.dumps(dict(approval=a,capacity_small={k:v for k,v in cap.items() if not isinstance(v,(list,dict))},trace0=t,sample0_fields=list(e),sample0_small={k:v for k,v in e.items() if k in ['index','archive','archive_SHA256','error']},baseline_first={mid:old['reports'][mid]},statistics=js(F/'eval_report.json')['statistics'],free=shutil.disk_usage(R).free),ensure_ascii=False))
