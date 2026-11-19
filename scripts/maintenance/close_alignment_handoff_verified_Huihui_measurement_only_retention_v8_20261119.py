"""Bind only the exact approved numerical-measurement retention transition."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Huihui_completed200_full_numerical_baselinegap_target_audit_and_correct_nonexample_retention_20261119';LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def execute(path,expected):
 assert sha(path)==expected;plan=js(path);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(pointer)==plan['expected_pointer_SHA256'];before=js(pointer);old=before['canonical_SHA256'];current=plan['canonical_SHA256']
 assert {n for n,h in current.items() if old[n]!=h}=={str(LED)} and old[str(LED)]=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32' and current[str(LED)]=='7978cbde39e32b7d28ecb05cf69148e451cbe262493b39d6b07e6fdb785b9b71'
 assert sha(H/'measurement_target_audit.json')=='b831aacc983f0a03c5aaa2cafa20e5853b7ff8ba0b51a6538027177054a85d36';audit=js(H/'measurement_target_audit.json')
 assert audit['passed'] and audit['sample_count']==200 and audit['actual_forward_count']==4325 and audit['score']==.695 and not audit['G_publication_permitted'] and not audit['original_protocol_final_qualification_complete']
 assert audit['canonical_counts']=={'done':180,'skipped_no_complete_weights':310,'needs_adaptation':210} and audit['numeric_within5pct']==165 and audit['numeric_outside5pct']==15 and audit['qualified_delivered']==66
 assert sha(audit['formal_proof'])==audit['formal_proof_SHA256']=='7dd295080b26203c38fb75519a418162ef35bf865c3902ab66701902a5fb5e54';pf=js(audit['formal_proof']);assert pf['passed'] and pf['sample_count']==200 and pf['actual_NPU_inference'] and not pf['formal_alignment_within_5pct'] and not pf['original_protocol_final_qualification_complete']
 assert sha(H/'retention_actual_completion.json')=='25e0f1b5767ec73d748cf3d1fac44556a9b4d3bc34a2a1a307649c69c8f8de91';rec=js(H/'retention_actual_completion.json');assert rec['actual_returncode']==0 and rec['all_protected_SHA256_unchanged'] and rec['removed_files']==105 and rec['removed_bytes']==10617515541 and rec['all61_incorrect_debug_control_cases_and_full_original_examples_retained'] and rec['canonical_scores_state_registry_binding_and_both_workbooks_unchanged']
 assert sha(H/'ledger_before_selective_retention.original')==old[str(LED)] and sha(rec['manifest'])==rec['manifest_SHA256']=='f957592949489091fb0173800a8fa4a9525938bfdfc2800119f008fdb7acdddc'
 rows=[json.loads(x) for x in Path(rec['manifest']).read_text().splitlines()];assert len(rows)==105;records={x['index']:x for x in audit['all200_archive_original_metadata']};keep=set(audit['selection']['keep']);assert len(keep)==95 and set(audit['selection']['incorrect'])<=keep
 assert {x['index'] for x in rows}==set(audit['selection']['remove']) and all(x==records[x['index']] and x['index'] not in keep and not Path(x['path']).exists() for x in rows)
 assert js(LED)['events'][-1]['receipt_SHA256']==sha(H/'retention_actual_completion.json') and all(sha(n)==h for n,h in current.items())
 for n,h in old.items():
  if n!=str(LED):assert audit['canonical_SHA256'][n]==h==current[n]
 from maintenance.close_alignment_handoff_official_readback_current_owned_stage_v4_20261118 import execute as close
 result=close(Path(path),expected);note=Path(result['latest_actual_note']).parent
 transition=dict(authority='Explicit user fully verified nonexample retention consent; numerical measurement acceptance is not5pct qualification',measurement_target_audit=str(H/'measurement_target_audit.json'),measurement_target_audit_SHA256=sha(H/'measurement_target_audit.json'),formal_NF200_proof_SHA256=audit['formal_proof_SHA256'],retention_receipt=str(H/'retention_actual_completion.json'),retention_receipt_SHA256=sha(H/'retention_actual_completion.json'),manifest_SHA256=rec['manifest_SHA256'],removed_correct_nonexample_files=105,all61_incorrect_cases_and_full_original_outer_examples_retained=True,canonical_before=old,canonical_after=current,no_G_state_registry_binding_or_workbook_change=True,intentionally_pruned105_arrays_no_longer_reverifiable=True)
 supplement=note/'exact_authorized_measurement_retention_only_transition.json';put(supplement,transition)
 closing=Path(plan['handoff_directory'])/'closing_addendum.json';c=js(closing);c.update(authorized_measurement_retention_transition=str(supplement),authorized_measurement_retention_transition_SHA256=sha(supplement),current_canonical_SHA256=current);put(closing,c)
 ptr=js(pointer);assert sha(pointer)==result['pointer_SHA256'];ptr.update(canonical_SHA256=current,authorized_measurement_retention_transition=str(supplement),authorized_measurement_retention_transition_SHA256=sha(supplement),closing_addendum_SHA256=sha(closing));put(pointer,ptr)
 result.update(pointer_SHA256=sha(pointer),closing_addendum_SHA256=sha(closing),canonical_exact_authorized_retention_transition_only=True,transition_SHA256=sha(supplement));put(note/'closure_actual_receipt_v8_measurement_retention_transition.json',result);return result
if __name__=='__main__':print(json.dumps(execute(Path(sys.argv[1]),sys.argv[2]),ensure_ascii=False))
