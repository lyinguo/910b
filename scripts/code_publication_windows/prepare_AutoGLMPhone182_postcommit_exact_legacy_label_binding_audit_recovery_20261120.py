from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;src=P/'publish_AutoGLMPhone9B_verified182_original_single_lock_dual_transaction_v1_20261120.py';raw=src.read_bytes();assert hashlib.sha256(raw).hexdigest()=='fab4bab195788d36959c6d5ac4ad39f8e170d2456772cd1727690b52928ea6e8';tree=ast.parse(raw)
names=['sha','now','glm46_capacity_review_passed','finish_local_and_full_target_audit','full_numeric_head_verified'];functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
class Labels(ast.NodeTransformer):
 def visit_keyword(self,n):
  if n.arg:n.arg=n.arg.replace('old180','old181').replace('prior180','prior181').replace('prior_180','prior_181')
  return self.generic_visit(n)
functions=[Labels().visit(n) for n in functions];f=ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[]));body=ast.unparse(f)
assert 'write_verified_A100_G' not in body
header='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
from datetime import datetime,timezone
from collections import Counter
import os,json,hashlib,base64,zipfile,re,xml.etree.ElementTree as ET,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120';I=B/'AutoGLMPhone9B_verified182_reviewed_single_lock_dual_publication_20261120';sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import atomic_json,workbook_rows
from eager_corrective_control_safety_2203 import core_unchanged,kernel_lock_owned
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
cache={}
def read(p):
 p=Path(p);v=json.loads(p.read_bytes())
 if p==H/'publication_receipt.json':
  assert v['old180_state_registry_and_G_preserved'] and v['canonical_counts']=={'done':182,'skipped_no_complete_weights':310,'needs_adaptation':208}
  old=json.loads((H/'before_verified_write/state.json').read_bytes());assert sum(x['status']=='done' for x in old['models'].values())==181
  v=dict(v,old181_state_registry_and_G_preserved=True)
 return v
'''
tail='''
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller']]
original=R/'scripts/publish_AutoGLMPhone9B_verified182_original_single_lock_dual_transaction_v1_20261120.py';assert sha(original)=='fab4bab195788d36959c6d5ac4ad39f8e170d2456772cd1727690b52928ea6e8'
failed=read(I/'publication_actual_completion.json');assert failed['actual_returncode']==1 and failed['source_SHA256']==sha(original)
stderr=(I/'publication_stderr.log').read_text();assert "KeyError: 'old181_state_registry_and_G_preserved'" in stderr and 'finish_local_and_full_target_audit' in stderr
assert sha(H/'publication_receipt.json')=='bf766c39dee1928d7ed96e783d15039d8d7169b1111a0efd705351b7d76fa696'
assert sha('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')=='0b2fcf21fe5a56e5b33111c4e65c2ec0ddf53dae43d451dec756a342df810340'
assert sha(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json')=='0ae99984e7e1f37f31f6c17d586d2f41c38abe838760900dba49ac5c5abbb931'
assert not (H/'dual_publication_completed.json').exists() and not (H/'full_target_dual_publication_audit.json').exists()
q=read(H/'reviewed_local_receipt_ready.json');assert q['local_receipt_SHA256']=='d6addb964b36b3649f38307d18fe510bddde149f7489969b281d61a327620bbf'
receipt_before=sha(H/'publication_receipt.json');before={str(p):sha(p) for p in [Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'),D/'state.json',D/'registry.json']}
with (D/'runner.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged()
 result=finish_local_and_full_target_audit(q)
 for p,h in before.items():assert sha(p)==h
 assert sha(H/'publication_receipt.json')==receipt_before and read(I/'publication_actual_completion.json')['actual_returncode']==1
 value=dict(UTC=now(),actual_returncode=0,source=str(Path(__file__).resolve()),source_SHA256=sha(Path(__file__)),prior_real_publisher_returncode_preserved=1,only_legacy_receipt_schema_read_alias_and_exact_binding_final_audit_recovered=True,no_new_model_forward_or_G_state_registry_write=True,original_server_receipt_bytes_unchanged=True,local_and_server_book_SHA256=before['/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'],full_target_audit=result['audit_path'],full_target_audit_SHA256=result['audit_SHA256'],counts=result['audit']['canonical_counts'],qualified_delivered=result['audit']['qualified_delivered'])
 atomic_json(H/'postcommit_exact_legacy_label_binding_audit_recovery_actual.json',value);print(json.dumps(value,ensure_ascii=False))
'''
s=header+body+tail;ast.parse(s);name='finish_AutoGLMPhone182_postcommit_exact_legacy_label_binding_audit_recovery_20261120.py';(P/name).write_text(s,encoding='utf8',newline='\n');digest=hashlib.sha256(s.encode()).hexdigest();data=dict(source='scripts/'+name,SHA256=digest,base64=base64.b64encode(s.encode()).decode())
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,subprocess,os
R=Path('/root/projects/comperror');p=R/DATA['source'];raw=base64.b64decode(DATA['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['SHA256'];ast.parse(raw)
assert not p.exists();p.write_bytes(raw)
r=subprocess.run(['python3','-B',str(p)],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=subprocess.PIPE,stderr=subprocess.PIPE);sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)
'''
ast.parse(remote);(P/'execute_AutoGLMPhone182_postcommit_exact_binding_audit_recovery_20261120_remote.py').write_text(remote,encoding='utf8',newline='\n');print(json.dumps(dict(source=name,SHA256=digest,no_G_rewrite_or_NN_replay=True,original_failed_source_and_receipt_will_not_be_modified=True)))
