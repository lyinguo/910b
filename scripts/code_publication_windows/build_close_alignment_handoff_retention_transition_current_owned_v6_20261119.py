from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent;s=(P/'close_alignment_handoff_official_readback_current_owned_stage_v4_20261118.py').read_text()
needle="assert sha(H/'handoff.txt')==ptr['handoff_SHA256'] and sha(H/'facts.json')==ptr['facts_SHA256']"
add=needle+"""
 old_canonical=ptr['canonical_SHA256'];changed={n:dict(old=old_canonical[n],new=h) for n,h in plan['canonical_SHA256'].items() if old_canonical.get(n)!=h}
 ledger=str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json');transition=plan['authorized_retention_ledger_transition']
 assert set(changed)=={ledger} and changed[ledger]==dict(old='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5',new='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32')
 receipt=Path(transition['receipt']);assert sha(receipt)==transition['receipt_SHA256']=='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3';actual=js(receipt)
 assert actual['actual_returncode']==0 and actual['removed_files']==4972 and actual['all_protected_SHA256_unchanged'] and sha(actual['manifest'])==actual['manifest_SHA256']=='f955add85f6012e7b1f8f4ad5fc0b78af043b80d2d0797315c21d59b35478382'
 assert js(ledger)['events'][-1]['receipt_SHA256']==sha(receipt)
"""
assert s.count(needle)==1;s=s.replace(needle,add.rstrip())
s=s.replace("all_canonical_and_preexisting_source_SHA256_unchanged=True,no_model_signal_restart_replay_or_matrix_deletion=True","all_canonical_unchanged_after_exact_authorized_ledger_transition=True,authorized_retention_ledger_transition=transition,preexisting_source_SHA256_unchanged=True,closure_only_no_model_signal_restart_replay_or_matrix_deletion=True")
needle="ptr.update(UTC=fresh['UTC'],"
assert s.count(needle)==1;s=s.replace(needle,"ptr['canonical_SHA256']=plan['canonical_SHA256'];ptr['authorized_retention_ledger_transition']=transition\n "+needle)
s=s.replace("all_canonical_unchanged=True)","all_canonical_unchanged_except_reviewed_exact_retention_ledger_transition=True)")
ast.parse(s);p=P/'close_alignment_handoff_official_readback_authorized_retention_current_owned_v6_20261119.py';p.write_text(s,encoding='utf8',newline='\n');print(hashlib.sha256(p.read_bytes()).hexdigest())
