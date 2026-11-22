"""Read completed original NN200 measurement, current full NF and publication inputs. No write."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import json,hashlib,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='unsloth/gemma-4-E4B-it-MLX-8bit'
J=B/'Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122';H=B/'Gemma4E4BMLX8_completed_genuine3_formal_NF2_actual_profiler_schema_then_capacity_serial_v2_20261122'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
job=js(J/'job.json');state=js(J/'state.json');folder=R/'workspace/reports/unsloth_gemma-4-E4B-it-MLX-8bit/image-text-to-text/20261122_024715';report=js(folder/'eval_report.json');cfg=report['pipeline_config'];p=js(cfg['model']['kwargs']['protocol_file'])
assert cfg['model']['name']==MID and cfg['sample_count']==200 and cfg['sample_range']=='0:199' and report['statistics']['total_samples']==200
outs=[json.loads(s) for s in (folder/'sample_outputs.jsonl').read_text().splitlines()];assert [r['index'] for r in outs]==list(range(200))
score=report['metrics']['Accuracy'];relative={k:str(abs(Decimal(str(score))-Decimal(str(d)))/abs(Decimal(str(d)))) for k,d in p['current_A100_D_values'].items()}
line=report['operator_lineage']['actual_execution'];trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()];assert len(trace)==200
cap=js(H/'own_completed3_all_ZIP_all200_shape_capacity.json')
result=dict(UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,job=str(J),job_SHA256=sha(J/'job.json'),job_state=state,actual_completion=js(J/'completion.json') if (J/'completion.json').is_file() else None,controller_actual_completion=js(H/'actual_completion.json') if (H/'actual_completion.json').is_file() else None,report_dir=str(folder),report_summary=report['summary'],metrics=report['metrics'],statistics=report['statistics'],three_D=p['current_A100_D_values'],relative_error_to_three_D=relative,all_three_D_numeric_within5pct=all(Decimal(x)<=Decimal('.05') for x in relative.values()),actual_native_forwards=sum(r['actual_native_forwards'] for r in trace),actual_NN200_completed_not_implying_fullNF_or_qualification=True,report_SHA256={f.name:sha(f) for f in folder.iterdir() if f.is_file()},fresh3_accepted=state['accepted'],wrong_indices=[i for i,r in enumerate(outs) if str(r['prediction']).strip()!=str(r['reference']).strip()],error_indices=[i for i,r in enumerate(outs) if r.get('error')],capacity=dict(empirical_estimate_bytes=cap['empirical_estimate_bytes'],required_GiB=str(Decimal(cap['empirical_estimate_bytes'])/Decimal(2**30)),original_free_bytes=cap['actual_free_bytes'],capacity_passed=cap['capacity_passed'],ZIP_parts=len(cap['all_complete_ZIP_parts']),SHA256=sha(H/'own_completed3_all_ZIP_all200_shape_capacity.json')),current_free_bytes=shutil.disk_usage(R).free,protocol_path=cfg['model']['kwargs']['protocol_file'],protocol_SHA256=sha(cfg['model']['kwargs']['protocol_file']),all_model_config_source_SHA256_unchanged=all(sha(R/n)==h for n,h in job['spec']['source_sha256'].items()),no_canonical_or_workbook_write=True)
print(json.dumps(result,ensure_ascii=False))
