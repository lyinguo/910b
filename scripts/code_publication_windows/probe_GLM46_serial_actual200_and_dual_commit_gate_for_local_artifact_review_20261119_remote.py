import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
J=B/'GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119';H=B/'GLM46_own_passed3_after_Huihui_full_NF_baselinegap_verified_correct_nonexample_cleanup_serial_v2_20261119'
HOME=R/'workspace/diagnostics/model_runs/GLM46Flash_own_BF16_thinking_20261119_180502_691852_1574589'
def maybe(p):
 p=Path(p)
 if not p.is_file():return None
 try:return json.loads(p.read_bytes())
 except ValueError:return dict(partial_live_JSON_not_a_completed_result=True)
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
assert hashlib.sha256((J/'job.json').read_bytes()).hexdigest()=='18c9d24f9992311ea3261e82cc56c3121db03ccd0655e21aaafc4d2b5ee4f323'
raw=(HOME/'actual_sample_trace.jsonl').read_bytes() if (HOME/'actual_sample_trace.jsonl').is_file() else b'';lines=raw.splitlines(keepends=True);records=[json.loads(v) for v in lines if v.endswith(b'\n')];assert [x['index'] for x in records]==list(range(len(records)))
done=maybe(J/'completion.json');control=maybe(H/'actual_completion.json');pf=None
if done and done.get('accepted') and done['accepted'][-1]['sample_count']==200:
 accepted=done['accepted'][-1];assert hashlib.sha256(Path(accepted['proof']).read_bytes()).hexdigest()==accepted['proof_sha256'];pf=maybe(accepted['proof'])
procs={}
for pid in [1573127,1573205,1574589]:
 p=Path('/proc')/str(pid)
 if p.exists():
  try:procs[pid]=dict(argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x],cwd=os.readlink(p/'cwd'))
  except FileNotFoundError:pass
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),sample_count=len(records),committed_actual_forwards=sum(x['actual_native_forwards'] for x in records),incomplete_tail_bytes=len(lines[-1]) if lines and not lines[-1].endswith(b'\n') else 0,failure_records=[dict(path=str(p),value=maybe(p)) for p in HOME.glob('*_failed.json')],actual_procs=procs,original_FLOCK=runner_lock_owners(),job_completion=done,controller_completion=control,full200_proof_summary=None if pf is None else {k:pf[k] for k in ['passed','sample_count','actual_forward_count','metrics','formal_alignment_within_5pct','original_protocol_final_qualification_complete','relative_error_to_current_A100_D']},free_bytes=shutil.disk_usage(R).free,no_NEW_NN_or_G_write=True),ensure_ascii=False))
