import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
J3=B/'AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v2_exact_frontend_path_real_FLOCK_20261120';J200=B/'AutoGLMPhone9BSingle_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120'
sources={}
for j in [J3,J200]:
 p=j/'job.json'
 if p.exists():
  v=json.loads(p.read_bytes());sources.update(v['spec']['source_sha256'])
  c=v['spec']['config'];smoke=str(Path(c).with_name(Path(c).stem+'-smoke.yaml'))
  if (R/smoke).exists():sources[smoke]=sha(R/smoke)
rows=[]
for n,h in sorted(sources.items()):
 p=R/n;assert not Path(n).is_absolute() and p.resolve().is_relative_to(R) and sha(p)==h
 raw=p.read_bytes();rows.append(dict(relative=n,SHA256=h,bytes=len(raw),base64=base64.b64encode(raw).decode()))
H=B/'AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120'
print(json.dumps(dict(actual_bound_runtime_sources=rows,verified182_export_path=str(H/'verified182_export_complete_metadata.json'),verified182_export_SHA256=sha(H/'verified182_export_complete_metadata.json'),verified182_full_report_stat_baseline_SHA256=sha(H/'active182_full_report_stat_baseline.json'),no_old_model_or_CPU200_or_weight_array_scan=True),ensure_ascii=False))
