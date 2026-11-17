import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
names=['scripts/tests/Merve_actual_formal_acceptance_manifest_writer_11_cases_template_20261117.py','scripts/tests/test_merve_paligemma_vqav2_verified179_v2_formal_acceptance_writer_isolated_20261117.py']
files={n:dict(SHA256=hashlib.sha256((R/n).read_bytes()).hexdigest(),text=(R/n).read_text()) for n in names}
books=[Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'),D/'state.json',D/'A100_ALIGNMENT_DUAL_WORKBOOK_BINDING.json']
print(json.dumps(dict(files=files,books={str(p):dict(exists=p.exists(),SHA256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None) for p in books},processes=project_processes(),locks=runner_lock_owners()),ensure_ascii=False))
