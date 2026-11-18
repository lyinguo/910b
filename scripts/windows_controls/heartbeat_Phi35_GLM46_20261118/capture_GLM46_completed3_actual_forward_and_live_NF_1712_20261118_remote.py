import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118';F=R/'workspace/reports/zai-org_GLM-4.6V-Flash/image-text-to-text/20261118_165457'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
line=js(F/'operator_lineage.json')['actual_execution'];trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()];summary=[]
for row in trace:
 r=js(row['actual_execution_file']);summary.append({k:r[k] for k in ['index','actual_native_forwards','raw_output','error','original_budget_hit','archive_bytes','lossless_uncompressed_bytes']})
s=js(J/'state.json');child=s.get('child_PID');proc={}
if child and Path(f'/proc/{child}').exists():
 for name in ['stat','io','status','cmdline','wchan']:proc[name]=Path(f'/proc/{child}/{name}').read_bytes().decode(errors='replace').replace('\x00',' ')
weight=js(line['weight_evidence']);complete=js(line['final_trace_manifest'])
v=dict(UTC=datetime.now(timezone.utc).isoformat(),model_id='zai-org/GLM-4.6V-Flash',report_dir=str(F),full_five_SHA256={p.name:sha(p) for p in F.iterdir()},actual_completed_samples=len(trace),actual_completed_native_forwards=sum(r['actual_native_forwards'] for r in summary),per_sample=summary,actual_NPU_source_count=len(weight['weights']),actual_all_NPU_source_SHA_equal=all(r['actual_NPU_SHA256']==r['cast_SHA256'] for r in weight['weights'].values()),own_parameter_source_proof=str(line['weight_evidence']),own_parameter_source_proof_SHA256=sha(line['weight_evidence']),original_outer_groups=complete['actual_outer_batches'],state=s,completion=js(J/'completion.json') if (J/'completion.json').exists() else None,live_child_proc=proc,verifier_stderr=(J/'fresh3/verification_stderr.log').read_text() if (J/'fresh3/verification_stderr.log').exists() else None)
print(json.dumps(v,ensure_ascii=False))
