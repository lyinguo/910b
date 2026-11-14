"""Final own Smol22 verifier: bind independently measured A100/NPU controls and original source."""
import json,sys
from pathlib import Path
from verify_Smol22_original_HF_FP16_v1_2109_20261111 import verify as parent_verify,check,sha,js,EXTERNAL,R,MID
from strict_mc_independent_reference_910b import independent_choice
def verify(folder,count):
 proof=parent_verify(folder,count);p=js(js(folder/'operator_lineage.json')['actual_execution']['protocol_file'])
 nr=js(check(p['numerical_review'],p['numerical_review_SHA256']));br=js(check(p['original_B10_gate_review'],p['original_B10_gate_review_SHA256']))
 assert nr['passed'] and nr['full_head_rows']==7 and nr['relative_error_same3']=='0' and br['passed'] and br['same10_relative_error']=='0'
 for label,mapping in [('numeric',nr['external_evidence_sha256']),('B10',br['external_evidence_sha256'])]:
  for name,digest in mapping.items():
   path=Path(name);W=Path(p['weight_path']);root=W if path.is_relative_to(W) else R if path.is_relative_to(R) else Path('/usr/local/python3.12.16/lib/python3.12/site-packages/transformers')
   check(path,digest,root=root)
 old=js(check(p['original_AA_rows_and_runtime']));rows=old['rows'];assert len(rows)==200 and sum(independent_choice(x['raw_output'])==str(x['reference']).strip().upper() for x in rows)==134
 new=[json.loads(x) for x in check(p['actual_AA_new200_rows']).read_text().splitlines()];assert len(new)==200 and sum(independent_choice(x['raw_output'])==x['reference'] for x in new)==139
 report=js(folder/'eval_report.json');changed_old=[];changed_current=[]
 for i,x in enumerate(report['detailed_results']):
  assert x['prompt']==rows[i]['prompt'] and x['reference']==rows[i]['reference']==new[i]['reference']
  if x['raw_output']!=rows[i]['raw_output']:changed_old.append(i)
  if x['raw_output']!=new[i]['raw_output']:changed_current.append(i)
 proof.update(original_AA_stored_and_same_raw_strict_accuracy='.67',current_original_HF_AA_new_diagnostic200_accuracy='.695',original_and_current_AA_raw_difference_disclosed=True,old_original_AA_raw_different_indices=changed_old,current_AA_diagnostic_raw_different_indices=changed_current,numerical_review_verified=True,original_B10_gate_verified=True,numerical_disclosure=nr['nonzero_numerical_differences_disclosed'],historical_full_untracked_source_bytes_not_claimed_captured=True,independent_verifier_SHA256=sha(__file__),external_evidence_sha256=EXTERNAL,smol22_numerical_diagnostic_evidence_sha256=nr['external_evidence_sha256'])
 return proof
if __name__=='__main__':
 folder=Path(sys.argv[1]).resolve();proof=verify(folder,int(sys.argv[2]));home=Path(js(folder/'operator_lineage.json')['actual_execution']['evidence_root']);out=home/'independent_report_verification.json'
 if out.exists():assert js(out)==proof
 else:out.write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 print(json.dumps(proof,ensure_ascii=False,allow_nan=False))
