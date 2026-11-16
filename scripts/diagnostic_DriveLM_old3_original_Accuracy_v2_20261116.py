HOME='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/MiniInternVL2_DriveLM_v2_original_Accuracy_and_explicit_own_tokenizer_old3_readonly_diagnostic_v3_original_YAML_BOS_20261116'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,traceback,contextlib,importlib.util,hashlib
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
H=Path(HOME)
try:
 p=R/'scripts/verify_miniinternvl2_drivelm_own_native_BF16_2048_v2_accuracy_20261116.py';spec=importlib.util.spec_from_file_location('drive_Accuracy_readonly_diagnostic',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 folder=R/'workspace/reports/OpenGVLab_Mini-InternVL2-4B-DA-DriveLM/image-text-to-text/20261116_130939'
 with contextlib.redirect_stdout(sys.stderr):q=m.verify(folder,3)
 q.update(actual_NPU_inference=False,NEW_real_NPU_inference=False,historical_record_reuse=True,diagnostic_only=True,not_new_protocol_qualification=True,original_failed_job_returncode_unchanged=True)
 (H/'old_record_readonly_diagnostic.json').write_text(json.dumps(q,ensure_ascii=False),encoding='utf-8');rc=0
except BaseException:rc=1;(H/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:(H/'actual_completion.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=rc,no_new_NPU_forward=True)),encoding='utf-8')
