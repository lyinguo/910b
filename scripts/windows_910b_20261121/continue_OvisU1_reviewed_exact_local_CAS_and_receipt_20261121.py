"""Complete the authorized local half only after the exact server original-lock transaction."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,os,time,traceback
V=Path(__file__).resolve().parent;H=V/'OvisU1_local_result_sync_verified184_20261121'
PY=Path(sys.executable);STRICT=Path('D:/code-lv/.tmp/batch_910b_20261030/remote_strict_config_copy_0721_20261114.py');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(argv,stdout,stderr):
 with Path(stdout).open('xb') as out,Path(stderr).open('xb') as err:r=subprocess.run([str(x) for x in argv],cwd=V,stdout=out,stderr=err)
 assert r.returncode==0,(r.returncode,str(stderr))
def main():
 assert sha(STRICT)=='b0167e9d5ecc18f825abee59aa6b4d7e6bff847397299987fbfadea74bcbadda'
 q=json.loads((H/'plan.json').read_bytes());assert q['expected_before_sha256']=='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81' and q['changes'][0]['new_value']==.895 and q['changes'][0]['cell']=='G144'
 reader=V/'read_OvisU1_verified184_server_book_for_local_CAS_20261121_remote.py';packet=V/'R_OvisU1_verified184_server_book_for_local_CAS_20261121.json'
 n=0
 while True:
  n+=1;run([PY,'-X','utf8','-B',STRICT,reader,packet],H/('server_poll_%04d.stdout.log'%n),H/('server_poll_%04d.stderr.log'%n));j=json.loads(packet.read_bytes())
  if j.get('ready_for_local_CAS'):break
  time.sleep(20)
 run([PY,'-X','utf8','-B',V/'apply_OvisU1_verified184_exact_server_bytes_local_CAS_20261121.py'],H/'actual_local_CAS_stdout.json',H/'actual_local_CAS_stderr.log')
 run([PY,'-X','utf8','-B',V/'build_OvisU1_verified184_exact_local_receipt_upload_20261121.py'],H/'local_receipt_upload_builder_stdout.json',H/'local_receipt_upload_builder_stderr.log')
 run([PY,'-X','utf8','-B',STRICT,V/'upload_OvisU1_verified184_exact_local_CAS_receipt_20261121_remote.py',V/'R_OvisU1_verified184_exact_local_CAS_receipt_uploaded_20261121.json'],H/'actual_receipt_upload_stdout.log',H/'actual_receipt_upload_stderr.log')
 (H/'local_driver_actual_completion.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,local_receipt_SHA256=sha(H/'local_publication_receipt.json'),only_reviewed_G144_changed=True,server_original_FLOCK_receipt_handshake_completed=True),indent=2)+'\n',encoding='utf8')
if __name__=='__main__':
 try:main()
 except BaseException:
  (H/'local_driver_actual_completion.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),original_reports_and_evidence_retained=True),indent=2)+'\n',encoding='utf8');raise
