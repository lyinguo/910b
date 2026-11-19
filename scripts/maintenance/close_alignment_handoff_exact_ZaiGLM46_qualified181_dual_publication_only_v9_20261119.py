"""Close only the actual qualified zai GLM dual publication, no retention change."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import hashlib,json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'ZaiGLM46Flash_original_BF16_thinking4096_LZMA_verified181_blank_G_publication_20261119';I=B/'ZaiGLM46Flash_verified181_publication_review_inputs_v2_20261119';J=B/'GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def execute(plan_path,expected):
 assert sha(plan_path)==expected;plan=js(plan_path);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(pointer)==plan['expected_pointer_SHA256'];before=js(pointer);old=before['canonical_SHA256'];current=plan['canonical_SHA256'];assert set(old)==set(current)
 book='/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';required={book,str(D/'state.json'),str(D/'registry.json'),str(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json')};changed={n for n,h in current.items() if old[n]!=h};assert changed==required,changed
 publisher='scripts/publish_ZaiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py';assert sha(R/publisher)=='f245f5bd16476b341d9330877aa750cf98fb6a8d0da7f808c35f9163a551b273'
 assert sha(R/'scripts/verified_A100_G_cell_writer_20261103.py')=='83134eec524c27bc4e9fd4569193e72038a7b248b379c12cf6999e1dcff798b8' and sha(R/'scripts/A100_alignment_evidence_runner_20261103.py')=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 assert sha(J/'job.json')=='18c9d24f9992311ea3261e82cc56c3121db03ccd0655e21aaafc4d2b5ee4f323'
 pub=js(H/'publication_receipt.json');audit=js(H/'full_target_dual_publication_audit.json');dual=js(H/'dual_publication_completed.json');run=js(I/'publication_actual_completion.json')
 assert run['actual_returncode']==0 and run['source_SHA256']==sha(R/publisher)
 assert pub['completed'] and pub['old180_state_registry_and_G_preserved'] and dual['completed'] and audit['passed'] and audit['numeric_within5pct']==166 and audit['numeric_outside5pct']==15 and audit['qualified_delivered']==67 and audit['complete_weights_pending_formal']==209 and audit['confirmed_missing']==310
 target,=pub['targets'];assert target['model_id']=='zai-org/GLM-4.6V-Flash';pf=js(target['formal_proof']);assert sha(target['formal_proof'])==target['formal_proof_sha256'] and pf['passed'] and pf['sample_count']==200 and pf['actual_NPU_inference'] and pf['formal_alignment_within_5pct'] and pf['original_protocol_final_qualification_complete'] and pf['source_tensor_count']==704
 assert target['new_G']==pf['metrics']['Accuracy'] and target['actual_forward_count']==pf['actual_forward_count'] and all(Decimal(str(e))<=Decimal('.05') for e in pf['relative_error_to_current_A100_D'].values())
 accepted,=[v for v in js(J/'completion.json')['accepted'] if v['sample_count']==200];assert accepted['proof_sha256']==target['formal_proof_sha256'] and accepted['report_dir']==target['new_report']
 for n in required:assert old[n]==pub['canonical_before_sha256'][n] and current[n]==audit['canonical_SHA256'][n]==sha(n)
 assert dual['local_and_server_workbook_sha256']==current[book] and old[book]=='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34'
 for n,h in current.items():assert sha(n)==h,n
 from maintenance.close_alignment_handoff_official_readback_current_owned_stage_v4_20261118 import execute as close
 result=close(Path(plan_path),expected);note=Path(result['latest_actual_note']).parent
 transition=dict(authority='Direct user authorizes original-protocol full NPU200 and all three D<=5% only dual publication',publisher=publisher,publisher_SHA256=sha(R/publisher),publication_receipt=str(H/'publication_receipt.json'),publication_receipt_SHA256=sha(H/'publication_receipt.json'),full_target_audit=str(H/'full_target_dual_publication_audit.json'),full_target_audit_SHA256=sha(H/'full_target_dual_publication_audit.json'),actual_publication_returncode=0,only_model=target['model_id'],only_G_cell='G697',canonical_before=old,canonical_after=current,no_retention_or_policy_transition=True)
 supplement=note/'exact_authorized_ZaiGLM46_qualified181_canonical_transition.json';put(supplement,transition)
 closing=Path(plan['handoff_directory'])/'closing_addendum.json';c=js(closing);c.update(exact_authorized_ZaiGLM46_qualified181_canonical_transition=str(supplement),exact_authorized_ZaiGLM46_qualified181_canonical_transition_SHA256=sha(supplement),current_canonical_SHA256=current);put(closing,c)
 ptr=js(pointer);assert sha(pointer)==result['pointer_SHA256'];ptr.update(canonical_SHA256=current,exact_authorized_ZaiGLM46_qualified181_canonical_transition=str(supplement),exact_authorized_ZaiGLM46_qualified181_canonical_transition_SHA256=sha(supplement),closing_addendum_SHA256=sha(closing));put(pointer,ptr)
 result.update(pointer_SHA256=sha(pointer),closing_addendum_SHA256=sha(closing),canonical_exact_authorized_transition_only=True,transition_SHA256=sha(supplement));put(note/'closure_actual_receipt_v9_exact_qualified_publication.json',result);return result
if __name__=='__main__':print(json.dumps(execute(Path(sys.argv[1]),sys.argv[2]),ensure_ascii=False))
