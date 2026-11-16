"""Read-only diagnostic re-verification of old complete3; no new NPU claim."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,contextlib,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
H=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Borealis4B_v1_complete3_readonly_full_numeric_diagnostic_v2_prediction_schema_20261116'
from scripts.verify_borealis_own_original_BF16_native_v2_original_prediction_schema_20261116 import verify,sha
F=R/'workspace/reports/NbAiLab_borealis-4b-instruct-preview/image-text-to-text/20261116_162651'
def main():
    before={str(p):sha(p) for p in F.iterdir() if p.is_file()}
    with contextlib.redirect_stdout(sys.stderr):proof=verify(F,3)
    assert before=={str(p):sha(p) for p in F.iterdir() if p.is_file()}
    proof.update(NEW_real_NPU_inference=False,actual_NPU_inference=False,readonly_diagnostic_of_existing_failed_job_data=True,old_failed_job_not_reclassified=True,formal_alignment_within_5pct=False,original_protocol_final_qualification_complete=False,diagnostic_source_SHA256=sha(__file__))
    (H/'review.json').write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False));(H/'actual_completion.json').write_text(json.dumps(dict(actual_returncode=0,UTC=datetime.now(timezone.utc).isoformat(),NEW_NPU=False)))
if __name__=='__main__':
    H.mkdir(exist_ok=True)
    try:main()
    except BaseException:
        (H/'actual_completion.json').write_text(json.dumps(dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),NEW_NPU=False,error=traceback.format_exc())));raise
