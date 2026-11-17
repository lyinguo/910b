import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
H=B/'MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117';I=B/'MervePaliCompact_verified179_publication_review_inputs_v3_exact_negative_fixture_20261117'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={}
for p in [I/'publication_actual_completion.json',I/'publication_stdout.log',I/'publication_stderr.log',H/'full_target_dual_publication_audit.json',H/'transaction.json',H/'merve--paligemma_vqav2/isolated_11_writer_tests.json',H/'active179_full_report_stat_baseline.json']:
 if p.exists():
  raw=p.read_bytes()
  try:v=json.loads(raw)
  except Exception:v=raw.decode()
  files[str(p)]={'SHA256':sha(p),'value':v}
export=H/'verified179_export_complete_metadata.json'
book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
cards=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
o=dict(UTC=datetime.now(timezone.utc).isoformat(),files=files,book_SHA256=sha(book),canonical_SHA256={n:sha(D/n) for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json']},processes=project_processes(),FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(R).free,npu_smi=dict(returncode=cards.returncode,stdout=cards.stdout,stderr=cards.stderr))
if export.exists():o.update(export_path=str(export),export_SHA256=sha(export),export_original_bytes_base64=base64.b64encode(export.read_bytes()).decode())
print(json.dumps(o,ensure_ascii=False))
