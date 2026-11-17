import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,subprocess,datetime,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
J=B/'MervePali_own603_original_native_2048_missing_full_numeric_true3_v1_real_FLOCK_20261117';job=js(J/'job.json')
import yaml
cfg=yaml.safe_load((R/job['spec']['config']).read_bytes());pp=Path(cfg['model']['kwargs']['protocol_file'])
names=['universal_eval/model/merve_pali_full_native_value_20261117/worker_named_RPC.py','universal_eval/model/merve_pali_full_native_value_20261117/worker_lossless_v2.py','universal_eval/model/merve_pali_full_native_value_20261117/worker.py','scripts/batch_910b.py','universal_eval/model/adapters/prithiv_gemma4_e4b_own_original_BF16_native_v1_20261116.py']
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
accepted=js(J/'completion.json')['accepted'][0]
result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,config=cfg,protocol=dict(path=str(pp),SHA256=sha(pp),value=js(pp)),completion=js(J/'completion.json'),proof=js(accepted['proof']),proof_SHA256=sha(accepted['proof']),sources={n:dict(SHA256=sha(R/n),text=(R/n).read_text()) for n in names},project_processes=project_processes(),runner_lock_owners=runner_lock_owners(),disk_free=shutil.disk_usage(R).free,current_canonical={str(p):sha(p) for p in [D/'state.json',D/'registry.json',D/'PAUSE',D/'POLICY.json',D/'A100_ALIGNMENT_PUBLICATION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json']})
for p in R.parent.glob('C500_MLU_*20261029.xlsx'):
 result['current_canonical'][str(p)]=sha(p)
print(json.dumps(result,ensure_ascii=False))
