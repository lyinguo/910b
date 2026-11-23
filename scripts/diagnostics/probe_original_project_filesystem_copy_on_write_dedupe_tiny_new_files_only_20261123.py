"""Test kernel extent dedupe on two new tiny synthetic files, never model evidence."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import errno,fcntl,hashlib,json,os,random,struct,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
H=D/'filesystem_extent_dedupe_capability_tiny_synthetic_only_20261123'
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
def main():
    frozen=R/'scripts/A100_alignment_incremental_monitor_20261103.py'
    assert sha(frozen)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
    sys.path[:0]=[str(R/'scripts'),str(R)]
    from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
    assert not runner_lock_owners()
    assert not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
    ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes())
    assert all(sha(p)==h for p,h in ptr['canonical_SHA256'].items())
    H.mkdir(exist_ok=False)
    fs=subprocess.run(['stat','-f','-c','%T',str(R)],capture_output=True,text=True,check=True).stdout.strip()
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),filesystem=fs,synthetic_only=True,
        no_existing_matrix_or_report_source_config_log_or_weight_opened_for_write=True,
        no_model_or_NF=True,no_new_storage_method_applied_to_model_evidence=True)
    lock=D/'runner.lock';assert lock.is_file()
    with lock.open('r+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inode=lock.stat().st_ino
        lines=[s for s in Path('/proc/locks').read_text().splitlines() if 'FLOCK' in s and len(s.split())>5 and s.split()[4]==str(os.getpid()) and s.split()[5].split(':')[-1]==str(inode)]
        assert lines,('Original kernel EX not found',os.getpid(),inode)
        result['original_real_kernel_EX']=lines
        common=random.Random(11312026).randbytes(16384)
        source=b'S'*4096+common+b'T'*4096;target=b'D'*4096+common+b'X'*4096
        a=H/'synthetic_source.bin';b=H/'synthetic_target.bin'
        assert a.resolve().is_relative_to(H.resolve()) and b.resolve().is_relative_to(H.resolve())
        with a.open('xb') as f:f.write(source)
        with b.open('xb') as f:f.write(target)
        initial={str(a):sha(a),str(b):sha(b)}
        sa=a.stat();sb=b.stat();assert sa.st_ino!=sb.st_ino and sa.st_dev==sb.st_dev and sa.st_nlink==sb.st_nlink==1
        with a.open('rb') as sf,b.open('r+b') as df:
            request=bytearray(struct.pack('=QQHHI',4096,16384,1,0,0)+struct.pack('=qQQiI',df.fileno(),4096,0,0,0))
            try:
                fcntl.ioctl(sf.fileno(),0xc0189436,request,True)
                fd,offset,deduped,status,reserved=struct.unpack_from('=qQQiI',request,24)
                result['ioctl']=dict(returned=True,destination_status=status,bytes_deduped=deduped)
                assert sha(a)==initial[str(a)] and sha(b)==initial[str(b)]
                if status==0 and deduped==16384:
                    df.seek(4096);original=df.read(1);df.seek(4096);df.write(bytes([original[0]^1]));df.flush();os.fsync(df.fileno())
                    assert sha(a)==initial[str(a)] and sha(b)!=initial[str(b)]
                    df.seek(4096);df.write(original);df.flush();os.fsync(df.fileno())
                    assert sha(a)==initial[str(a)] and sha(b)==initial[str(b)]
                    result['copy_on_write_isolation_verified_on_new_tiny_files']=True
                else:result['copy_on_write_isolation_verified_on_new_tiny_files']=False
            except OSError as e:
                result['ioctl']=dict(returned=False,errno=e.errno,error=str(e))
                result['copy_on_write_isolation_verified_on_new_tiny_files']=False
        assert a.stat().st_ino==sa.st_ino and b.stat().st_ino==sb.st_ino
        assert a.stat().st_nlink==b.stat().st_nlink==1
        assert sha(a)==initial[str(a)] and sha(b)==initial[str(b)]
        result['complete_synthetic_file_SHA256_unchanged']=initial
    assert not runner_lock_owners()
    assert all(sha(p)==h for p,h in ptr['canonical_SHA256'].items())
    result.update(UTC=datetime.now(timezone.utc).isoformat(),original_EX_released=True,canonical_SHA256_unchanged=True,
        actual_project_bytes_freed_not_claimed=True,no_existing_file_changed_or_deleted=True)
    p=H/'review.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(**result,review=str(p),review_SHA256=sha(p)),ensure_ascii=False))
if __name__=='__main__':main()
