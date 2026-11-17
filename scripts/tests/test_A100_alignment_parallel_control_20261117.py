"""Real temporary-kernel FLOCK/concurrency and reservation tests; never NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import tempfile,fcntl,subprocess,time,json,os,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from A100_alignment_parallel_control_20261117 import can_admit,validate_cards,real_exclusive_owner
def main():
 assert can_admit(100,{},60,2) and can_admit(100,{'a':60},40,2) and not can_admit(100,{'a':60},41,2) and not can_admit(100,{'a':10,'b':10},10,2)
 for rows in [[dict(card=0)],[dict(card=2),dict(card=2)],[dict(card=16)]]:
  try:validate_cards(rows)
  except AssertionError:pass
  else:raise AssertionError('Unsafe cards admitted')
 assert validate_cards([dict(card=2),dict(card=4)])==[2,4]
 with tempfile.TemporaryDirectory(prefix='parallel_real_lock_test_') as root:
  root=Path(root);globalfile=root/'global.lock';held=globalfile.open('a+');fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);real_exclusive_owner(os.getpid(),globalfile)
  child=root/'child.py';child.write_text("import sys,fcntl,time,json,os\nfrom pathlib import Path\nr=Path(sys.argv[1]);card=sys.argv[2];f=(r/(card+'.lock')).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nw=(r/'global.lock').open('a+')\ntry:fcntl.flock(w,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError:pass\nelse:raise AssertionError('Publisher lock incorrectly released')\n(r/(card+'.ready')).write_text(str(os.getpid()))\nwhile not (r/'go').exists():time.sleep(.01)\na=time.monotonic();time.sleep(.5);(r/(card+'.json')).write_text(json.dumps([a,time.monotonic()]))\n".replace('\n',chr(10)))
  p1=subprocess.Popen([sys.executable,'-B',str(child),str(root),'2']);p2=subprocess.Popen([sys.executable,'-B',str(child),str(root),'4']);deadline=time.monotonic()+10
  while not ((root/'2.ready').exists() and (root/'4.ready').exists()):assert time.monotonic()<deadline;time.sleep(.01)
  duplicate=(root/'2.lock').open('a+')
  try:fcntl.flock(duplicate,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise AssertionError('Same card double reservation admitted')
  (root/'go').touch();assert p1.wait()==p2.wait()==0;a=json.loads((root/'2.json').read_text());b=json.loads((root/'4.json').read_text());assert min(a[1],b[1])>max(a[0],b[0]);overlap=min(a[1],b[1])-max(a[0],b[0]);fcntl.flock(held,fcntl.LOCK_UN);held.close();fcntl.flock(duplicate,fcntl.LOCK_EX|fcntl.LOCK_NB);duplicate.close()
 print(json.dumps(dict(passed=True,test_count=12,real_disjoint_card_lock_holders_overlapped_seconds=overlap,global_publisher_exclusion_verified=True,same_card_duplicate_refused=True,conservative_combined_storage_and_max_parallel_gates_verified=True,production_locks_and_NPU_and_canonical_files_never_touched=True)))
if __name__=='__main__':main()
