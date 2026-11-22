"""Read actual current private LLaVA-Mini stages, processes, locks and output."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
out=dict(UTC=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(R).free,original_kernel_FLOCK=runner_lock_owners(),stages={},processes=[],captures=[])
names=['LlavaMini_own_original_FP16_2048_source_readiness_v1_20261122','LlavaMini_own_original_FP16_2048_true3_v1_real_FLOCK_20261122','LlavaMini_own_original_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122','LlavaMini_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261122']
names += ['LlavaMini_own_original_FP16_2048_source_readiness_v2_actual_reserved_devices_20261122','LlavaMini_own_original_FP16_2048_true3_v2_actual_reserved_devices_real_FLOCK_20261122','LlavaMini_own_original_FP16_true3_capacity_fresh3_fixed200_serial_v2_actual_reserved_devices_20261122','LlavaMini_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v2_actual_reserved_devices_real_FLOCK_20261122']
for name in names:
    home=B/name;record=dict(exists=home.exists(),files={})
    for item in ['state.json','actual_failure.json','actual_completion.json','readiness.json','completion.json','own_passed3_complete_all200_shape_capacity.json']:
        path=home/item
        if path.is_file():
            raw=path.read_bytes();record['files'][item]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
    record['logs']={}
    if home.exists():
        for path in sorted(home.rglob('*.log')):
            if path.stat().st_size<=8*2**20:record['logs'][str(path.relative_to(home))]=path.read_text(errors='replace')
            else:record['logs'][str(path.relative_to(home))]=dict(bytes=path.stat().st_size,full_log_retained=True,tail=path.read_bytes()[-65536:].decode(errors='replace'))
    out['stages'][name]=record
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
        args=[v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v]
        if not args or not Path(args[0]).name.startswith('python'):continue
        if not any(arg.startswith(str(R)+'/') and ('llava_mini' in arg.lower() or Path(arg).name in ['run.py','A100_alignment_evidence_runner_20261103.py']) for arg in args[1:]) and not any(Path(arg).name.startswith('verify_llava_mini_') for arg in args[1:]) and not (any(Path(arg).name=='run.py' for arg in args[1:]) and os.readlink(path/'cwd')==str(R)):continue
        fds=[]
        for fd in (path/'fd').iterdir():
            try:
                target=os.readlink(fd)
                if target.startswith((str(R),'/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaMiniLlamaForCausalLM/ICTNLP/llava-mini-llama-3.1-8b')):fds.append(target)
            except OSError:pass
        out['processes'].append(dict(PID=int(path.name),argv=args,cwd=os.readlink(path/'cwd'),stat=(path/'stat').read_text(),io=(path/'io').read_text(),open_project_FDs=fds,status=(path/'status').read_text()))
    except OSError:pass
for home in sorted((D.parent/'model_runs').glob('LlavaMini_own_original_FP16_2048_*')):
    record=dict(path=str(home),files=[dict(name=p.name,bytes=p.stat().st_size) for p in home.iterdir() if p.is_file()])
    for name in ['sample_execution.jsonl','generation_progress.jsonl']:
        path=home/name
        if path.is_file():
            rows=[json.loads(line) for line in path.read_text().splitlines() if line];record[name]=dict(rows=len(rows),last=rows[-1] if rows else None,full_original_retained=True)
    out['captures'].append(record)
process=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=20);out['npu_smi']=dict(actual_returncode=process.returncode,stdout=process.stdout,stderr=process.stderr)
print(json.dumps(out,ensure_ascii=False))
