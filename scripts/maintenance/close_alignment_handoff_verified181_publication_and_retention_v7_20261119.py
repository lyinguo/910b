"""Close only the exact audited Huihui publication and optional owned retention transition."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'HuihuiGLM46Flash_original_BF16_2048_lossless_verified181_blank_G_publication_20261119';RET=B/'authorized_verified181_Huihui_only_accepted_nonexample_serial_next_20261119';LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)
def execute(plan_path,expected):
 assert sha(plan_path)==expected;plan=js(plan_path);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(pointer)==plan['expected_pointer_SHA256'];before=js(pointer);old=before['canonical_SHA256'];current=plan['canonical_SHA256']
 changed={n for n,h in current.items() if old[n]!=h};book='/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';required={book,str(D/'state.json'),str(D/'registry.json'),str(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json')}
 assert changed in [required,required|{str(LED)}],changed
 assert sha(R/'scripts/publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_next182_metadata_20261119.py')=='be06184a15a41354a18b164d99dad2f76359578a0160ea73f4060cb51d3299c7'
 assert sha(R/'scripts/verified_A100_G_cell_writer_20261103.py')=='83134eec524c27bc4e9fd4569193e72038a7b248b379c12cf6999e1dcff798b8'
 pub=js(H/'publication_receipt.json');audit=js(H/'full_target_dual_publication_audit.json');dual=js(H/'dual_publication_completed.json');run=js(B/'HuihuiGLM46Flash_verified181_publication_review_inputs_v2_20261119/publication_actual_completion.json')
 assert run['actual_returncode']==0 and run['source_SHA256']=='be06184a15a41354a18b164d99dad2f76359578a0160ea73f4060cb51d3299c7'
 assert pub['completed'] and pub['old180_state_registry_and_G_preserved'] and dual['completed'] and audit['passed'] and audit['numeric_within5pct']==166 and audit['numeric_outside5pct']==15 and audit['qualified_delivered']==67
 t,=pub['targets'];assert t['model_id']=='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated';pf=js(t['formal_proof']);assert sha(t['formal_proof'])==t['formal_proof_sha256'] and pf['passed'] and pf['sample_count']==200 and pf['actual_NPU_inference'] and pf['formal_alignment_within_5pct'] and pf['original_protocol_final_qualification_complete']
 for n in required:assert old[n]==pub['canonical_before_sha256'][n] and current[n]==audit['canonical_SHA256'][n]==sha(n)
 assert dual['local_and_server_workbook_sha256']==current[book]
 transition=dict(authority='Direct user authorizes qualified full200-only dual publication and accepted-only nonexample retention',publication_receipt=str(H/'publication_receipt.json'),publication_receipt_SHA256=sha(H/'publication_receipt.json'),full_target_audit=str(H/'full_target_dual_publication_audit.json'),full_target_audit_SHA256=sha(H/'full_target_dual_publication_audit.json'),actual_publication_returncode=run['actual_returncode'],only_model=t['model_id'],only_G_cell='G355',canonical_before=old,canonical_after=current)
 if str(LED) in changed:
  assert sha(RET/'ledger_before_owner_reference_v2.original')==old[str(LED)]=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'
  rec=js(RET/'actual_completion.json');assert rec['actual_returncode']==0 and rec['all_protected_SHA256_unchanged'] and 0<rec['removed_files']<=160 and rec['all_five_and_all_records_sources_configs_snapshots_logs_and_required_full_examples_retained']
  assert rec['manifest_SHA256']==sha(rec['manifest']) and js(LED)['events'][-1]['receipt_SHA256']==sha(RET/'actual_completion.json')
  keep=set(range(10))|set(range(90,110))|set(range(190,200))
  rows=[json.loads(s) for s in Path(rec['manifest']).read_text().splitlines()];assert len(rows)==rec['removed_files']
  for row in rows:assert row['model_id']==t['model_id'] and not keep&set(row['logical_indices']) and row['proof_SHA256']==t['formal_proof_sha256']
  transition['retention_receipt']=str(RET/'actual_completion.json');transition['retention_receipt_SHA256']=sha(RET/'actual_completion.json');transition['manifest_SHA256']=rec['manifest_SHA256'];transition['removed_files']=rec['removed_files'];transition['pruned_full200_arrays_not_available_for_reverification']=True
 for n,h in current.items():assert sha(n)==h,n
 from maintenance.close_alignment_handoff_official_readback_current_owned_stage_v4_20261118 import execute as close
 result=close(Path(plan_path),expected);note=Path(result['latest_actual_note']).parent
 supplement=note/'exact_authorized_canonical_publication_and_retention_transition.json';put(supplement,transition)
 closing=Path(plan['handoff_directory'])/'closing_addendum.json';c=js(closing);c.update(authorized_verified181_publication_and_retention_transition=str(supplement),authorized_verified181_publication_and_retention_transition_SHA256=sha(supplement),current_canonical_SHA256=current);put(closing,c)
 ptr=js(pointer);assert sha(pointer)==result['pointer_SHA256'];ptr.update(canonical_SHA256=current,authorized_verified181_publication_and_retention_transition=str(supplement),authorized_verified181_publication_and_retention_transition_SHA256=sha(supplement),closing_addendum_SHA256=sha(closing));put(pointer,ptr)
 result.update(pointer_SHA256=sha(pointer),closing_addendum_SHA256=sha(closing),canonical_exact_authorized_transition_only=True,transition_SHA256=sha(supplement));put(note/'closure_actual_receipt_v7_authorized_transitions.json',result);return result
if __name__=='__main__':print(json.dumps(execute(Path(sys.argv[1]),sys.argv[2]),ensure_ascii=False))
