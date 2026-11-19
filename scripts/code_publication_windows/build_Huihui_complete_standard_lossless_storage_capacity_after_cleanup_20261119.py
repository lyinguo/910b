from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
old=(P/'diagnostic_glm46flash_completed_true3_complete_LZMA_storage_capacity_only_1714_20261118.py').read_text()
replacements={
"J=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118'":"J=B/'HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118'",
"OLD=B/'GLM46Flash_own_ready_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v2_actual_D965_1653_20261118'":"OLD=B/'HuihuiGLM46_completed_true3_serial_efficiency_original_capacity_resumer_v2_1155_20261119'",
"H=B/'GLM46Flash_completed_true3_complete_standard_Fortran_LZMA_storage_capacity_diagnostic_only_1714_20261118'":"H=B/'HuihuiGLM46_completed_true3_complete_standard_Fortran_LZMA_capacity_after_verified4972_cleanup_20261119'",
"done['model_returncode']==done['verifier_returncode']==0":"done['model_returncode'] in [0,1] and done['verifier_returncode']==0",
"'d089c0ded120ee0df0be6ad28d9a02fe0b61697330d6f3b4a3178bc667916950'":"'7bbc04ae6b05c7845ca01dc3e682f94da39000230beea45f6fbf5dc6f773cc59'",
"unchanged(job['protected_canonical_hashes'])":"verified_cleanup_ledger_transition_guard(job['protected_canonical_hashes'])",
}
for a,b in replacements.items():
 assert a in old,a;old=old.replace(a,b)
needle='def main():'
guard="""def verified_cleanup_ledger_transition_guard(before):
 from A100_alignment_evidence_runner_20261103 import unchanged
 ledger=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
 cleanup=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119'
 assert sha(cleanup/'actual_completion.json')=='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3'
 actual=js(cleanup/'actual_completion.json');assert actual['actual_returncode']==0 and actual['all_protected_SHA256_unchanged'] and actual['removed_files']==4972
 assert sha(ledger)=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'
 assert before[str(ledger)]=='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5'
 unchanged({p:h for p,h in before.items() if p!=str(ledger)})
 assert js(ledger)['events'][-1]['receipt_SHA256']==sha(cleanup/'actual_completion.json')
 return dict(only_authorized_retention_ledger_changed=True,ledger_SHA256=sha(ledger),cleanup_receipt_SHA256=sha(cleanup/'actual_completion.json'))
def main():"""
old=old.replace(needle,guard)
old=old.replace("H.mkdir(exist_ok=False);folder", "H.mkdir(exist_ok=False);put('authorized_cleanup_ledger_transition.json',verified_cleanup_ledger_transition_guard(job['protected_canonical_hashes']));folder")
name='diagnostic_huihui_completed_true3_complete_standard_LZMA_capacity_after_verified_cleanup_20261119.py';ast.parse(old);(P/name).write_text(old,encoding='utf8',newline='\n')
b=(P/name).read_bytes();data=dict(relative='scripts/'+name,SHA256=sha(b),base64=base64.b64encode(b).decode())
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,runpy,json
R=Path('/root/projects/comperror');p=R/DATA['relative'];b=base64.b64decode(DATA['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==DATA['SHA256'];ast.parse(b);assert p.resolve().is_relative_to(R)
if p.exists():assert p.read_bytes()==b
else:p.write_bytes(b)
runpy.run_path(str(p),run_name='__main__')
'''
ast.parse(remote);(P/'deploy_run_Huihui_complete_standard_LZMA_capacity_after_verified_cleanup_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(source=data['relative'],SHA256=data['SHA256'],no_NEW_model_NN=True,original_complete_three_and_all_guards_unchanged=True),ensure_ascii=False))
