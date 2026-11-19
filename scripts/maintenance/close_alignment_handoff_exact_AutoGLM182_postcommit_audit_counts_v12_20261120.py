"""Exact already-delivered AutoGLM182 canonical transition; no G rewrite."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def execute(path,expected):
 assert sha(path)==expected;plan=js(path);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(pointer)==plan['expected_pointer_SHA256'];oldptr=js(pointer);old=oldptr['canonical_SHA256'];new=plan['canonical_SHA256'];required={str(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')),str(D/'state.json'),str(D/'registry.json'),str(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json')};assert {p for p in new if old[p]!=new[p]}==required
 pub=js(H/'publication_receipt.json');audit=js(H/'full_target_dual_publication_audit.json');dual=js(H/'dual_publication_completed.json');recovery=js(H/'postcommit_exact_legacy_label_binding_audit_recovery_actual.json')
 assert sha(H/'publication_receipt.json')=='bf766c39dee1928d7ed96e783d15039d8d7169b1111a0efd705351b7d76fa696' and sha(H/'full_target_dual_publication_audit.json')=='68b43b7c08f7095e6086a26b1813752572d4f1320e06121df5329a28d3aca9e2'
 assert recovery['actual_returncode']==0 and recovery['prior_real_publisher_returncode_preserved']==1 and recovery['no_new_model_forward_or_G_state_registry_write'] and recovery['source_SHA256']==sha(recovery['source'])=='d91103fe3e666bf906d7a4c7e3dbafb442d73c87e64dbe46d3ca258b259ab2f6'
 assert pub['completed'] and pub['old180_state_registry_and_G_preserved'] and audit['passed'] and dual['completed'] and audit['canonical_counts']=={'done':182,'skipped_no_complete_weights':310,'needs_adaptation':208} and audit['qualified_delivered']==68 and audit['numeric_within5pct']==167 and audit['numeric_outside5pct']==15
 assert sum(v['status']=='done' for v in js(H/'before_verified_write/state.json')['models'].values())==181
 target,=pub['targets'];pf=js(target['formal_proof']);assert target['model_id']=='zai-org/AutoGLM-Phone-9B-Multilingual' and target['new_G']==.82 and target['actual_forward_count']==2601 and sha(target['formal_proof'])==target['formal_proof_sha256']=='498c5f03c923bdcd12f8ea40022f03a4c8bae44ea6ca0eabde00687b8566c95d' and pf['passed'] and pf['sample_count']==200 and pf['original_protocol_final_qualification_complete'] and set(pf['relative_error_to_current_A100_D'].values())=={'0'}
 for p in required:assert old[p]==pub['canonical_before_sha256'][p] and new[p]==audit['canonical_SHA256'][p]==sha(p)
 for p,h in new.items():assert sha(p)==h
 assert old['/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx']=='4e740da3981c17fd7b599875ab95e60c5f709804f08c22c9fb64c3c8a5454ed8'
 from maintenance.close_alignment_handoff_official_readback_current_owned_stage_v4_20261118 import execute as close
 result=close(Path(path),expected);note=Path(result['latest_actual_note']).parent;counts=dict(delivered=182,numeric_within=167,numeric_outside=15,complete_weight_pending_delivery=208,confirmed_missing=310,full_protocol_qualified=68,numeric_only=99,complete_weight_still_needing_adaptation_qualification_or_repair=322)
 supplement=note/'exact_authorized_AutoGLM182_recovered_dual_canonical_transition.json';put(supplement,dict(canonical_before=old,canonical_after=new,counts=counts,only_target_G693=True,original_publisher_actual_returncode_preserved=1,postcommit_recovery_actual_returncode=0,full_target_audit=str(H/'full_target_dual_publication_audit.json'),full_target_audit_SHA256=sha(H/'full_target_dual_publication_audit.json'),no_model_G_or_retention_change_by_this_closure=True))
 closing=Path(plan['handoff_directory'])/'closing_addendum.json';c=js(closing);c.update(current_canonical_SHA256=new,current_canonical_counts=counts,exact_authorized_AutoGLM182_transition=str(supplement),exact_authorized_AutoGLM182_transition_SHA256=sha(supplement));put(closing,c)
 ptr=js(pointer);assert sha(pointer)==result['pointer_SHA256'];ptr.update(canonical_SHA256=new,canonical_counts=counts,current_canonical_counts=counts,verified_delivered_count=182,last_complete_result_BJT='2026-11-20 02:12:26.027587+08:00',closing_addendum_SHA256=sha(closing),exact_authorized_AutoGLM182_transition=str(supplement),exact_authorized_AutoGLM182_transition_SHA256=sha(supplement));put(pointer,ptr);result.update(pointer_SHA256=sha(pointer),closing_addendum_SHA256=sha(closing),canonical_exact_authorized_transition_only=True,transition_SHA256=sha(supplement));put(note/'closure_actual_receipt_v12_exact_qualified_publication.json',result);return result
if __name__=='__main__':print(json.dumps(execute(Path(sys.argv[1]),sys.argv[2]),ensure_ascii=False))
