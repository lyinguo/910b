import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
js=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
state=js(D/'state.json');reg=js(D/'registry.json');out={}
for mid,pub in [('zai-org/GLM-4.6V-Flash','ZaiGLM46Flash_original_BF16_thinking4096_LZMA_verified181_blank_G_publication_20261119'),('zai-org/AutoGLM-Phone-9B-Multilingual','AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120')]:
 row=state['models'][mid];entry=reg[mid];folder=Path(row['report_dir']);rp=js(folder/'eval_report.json');ex=js(folder/'operator_lineage.json')['actual_execution'];trace=[json.loads(l) for l in Path(ex['sample_trace']).read_text().splitlines()]
 audit=js(B/pub/'full_target_audit.json') if (B/pub/'full_target_audit.json').exists() else None
 out[mid]=dict(state=row,registry=entry,statistics=rp['statistics'],pipeline_config=rp['pipeline_config'],details=rp['detailed_results'],execution=ex,trace=trace,publication_files=[p.name for p in (B/pub).iterdir()],full_target_audit=audit)
print(json.dumps(dict(models=out,counts={s:sum(v['status']==s for v in state['models'].values()) for s in ['done','needs_adaptation','skipped_no_complete_weights']},ledger_SHA256=sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')),ensure_ascii=False))
