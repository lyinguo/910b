"""Preserve completed original GLM200 evidence and the current serial Meta run."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
from collections import Counter
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
G=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
M=B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_actual_capacity_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261120'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def main():
 from batch_910b_support import validate_report
 import batch_910b as batch
 p=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before=p.read_bytes();ptr=json.loads(before)
 canon=ptr['canonical_SHA256'];assert all(sha(p)==h for p,h in canon.items())
 job=js(G/'job.json');assert sha(G/'job.json')=='756c7f9221f17a39e85f00cf064d640d197fa9ec42e1af0482eac85f17c7d163'
 batch.check_sources(job['spec'])
 done=js(G/'completion.json');assert done['actual_returncode']==done['verifier_returncode']==0 and done['model_returncode']==1
 accepted=done['accepted'];assert [v['sample_count'] for v in accepted]==[3,200]
 a=accepted[-1];assert sha(a['proof'])==a['proof_sha256']=='af10ad5191779c0732038c95b369fa6cef27ec2478e813eafa59f7c87c97c101'
 proof=js(a['proof']);assert proof['passed'] and proof['actual_NPU_inference'] and not proof['historical_record_reuse'] and proof['sample_count']==200
 assert proof['source_tensor_count']==704 and proof['actual_forward_count']==55771
 checks=proof['numeric_sample_checks'];assert len(checks)==200 and {v['index'] for v in checks}==set(range(200)) and sum(v['full_head_rows'] for v in checks)==55771
 assert all(v['all40_native_cache_values_reconstructed'] and v['original_input_pixels_token_ids_checked'] and v['all_masks_3D_positions_and_own_visual_injection_checked'] and len(v['head_reference_checks'])==v['full_head_rows'] for v in checks)
 assert proof['correct']==111 and len(proof['invalid_indices'])==76 and not proof['error_indices'] and not proof['empty_indices'] and not proof['truncated_indices']
 report=Path(a['report_dir']);five=['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']
 assert {f.name for f in report.iterdir()}==set(five)
 assert all(sha(report/n)==proof['report_sha256'][n] for n in five)
 score=validate_report(R,report,job['model_id'],'image-text-to-text',200,proof);assert Decimal(str(score))==Decimal('.555')
 exact={k:str(abs(Decimal(d)-Decimal(str(score)))/abs(Decimal(d))) for k,d in [('server','0.7500'),('local','0.7500'),('preserved_original','0.75')]}
 assert set(exact.values())=={'0.26'} and proof['formal_alignment_within_5pct'] is False and proof['original_protocol_final_qualification_complete'] is False
 # This receipt records completed acceptance; it does not repeat numerical verification.
 reader=R/'scripts/maintenance/read_MetaMllama_current_serial_fixed200_and_GLM41_completed200_20261120.py'
 assert sha(reader)=='3349cda45e4bb0dc3632d7faaaff9dd180479f2ea9317e1d7744951158b33100'
 r=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 assert r.returncode==0,r.stderr.decode();actual=json.loads(r.stdout)
 meta_job=js(M/'job.json');assert sha(M/'job.json')=='581c7c7c9a4f6a270f5c9021ee9c6824144dc7c437473b7eb3ab81fce6141757'
 batch.check_sources(meta_job['spec'])
 active=[v for v in actual['Meta_captures'] if v['current_model_PID_matches_capture']]
 owners=actual['original_FLOCK'];procs={v['PID']:v for v in actual['actual_processes']}
 exact_owner=[]
 for own in owners:
  z=procs.get(own['PID'])
  if z and str(M) in z['argv'] and any(Path(a).name=='A100_alignment_evidence_runner_20261103.py' for a in z['argv']):
   children=[v for v in procs.values() if v['PPid']==z['PID'] and any(Path(a).name=='run.py' for a in v['argv'])]
   exact_owner.append(dict(owner=z,actual_model_children=children,root_cwd_explains_only_this_frozen_monitor_filter=z['cwd']=='/root'))
 assert len(active)==len(exact_owner)==1 and len(exact_owner[0]['actual_model_children'])==1
 console=next(v['complete_text'] for k,v in actual['files'].items() if k==str(M/'formal200/console.log'))
 lines=console.splitlines();hits=[i for i,s in enumerate(lines) if 'self.npu_exception' in s]
 assert all(i>0 and 'SyntaxWarning' in lines[i-1] for i in hits) and 'Traceback (most recent call last)' not in console
 state=js(D/'state.json');counts=Counter(v['status'] for v in state['models'].values())
 assert counts['done']==183 and counts['needs_adaptation']==207 and counts['skipped_no_complete_weights']==310
 assert all(sha(p)==h for p,h in canon.items()) and p.read_bytes()==before
 home=B/('completed_GLM41_full200_pass_gap_and_current_Meta_user_results_logs_preservation_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'));home.mkdir()
 (home/'current_actual_snapshot.json').write_text(json.dumps(actual,ensure_ascii=False),encoding='utf-8')
 receipt=dict(UTC=actual['UTC'],passed=True,completed_GLM41_full200=dict(report=str(report),proof=a['proof'],proof_SHA256=a['proof_sha256'],actual_completed_at=done['completed_at'],sample_count=200,actual_native_forwards=55771,Accuracy=score,three_D_relative_error=exact,correct=111,invalid=76,error=0,empty=0,truncated=0,full_numerical_independent_acceptance=True,original_protocol_qualification=False,G_unchanged_blank=True,all_five_report_files_SHA256={n:sha(report/n) for n in five},all_200_original_denominators_preserved=True,all_full_numeric_proof_bytes_retained=True),current_Meta=dict(job=str(M),job_SHA256=sha(M/'job.json'),actual_progress=active,actual_original_runner_binding=exact_owner,approved_fresh3_proof_SHA256='b8c1e2f51b99518bf1059c44567ff810687b917711b3cc388e275fa20fa2db72',current_full200_acceptance_not_yet_claimed=True),library_SyntaxWarning_regex_literal_candidates_reviewed=len(hits),no_new_model_traceback_in_complete_console=True,canonical_counts=dict(counts),canonical_SHA256=canon,source_code_configs_snapshots_reports_logs_and_required_numeric_examples_retained=True,no_arrays_deleted=True,no_new_NN_or_old_NF_replay=True,no_source_job_yaml_deadline_signal_or_G_change=True,current_actual_snapshot=str(home/'current_actual_snapshot.json'),current_actual_snapshot_SHA256=sha(home/'current_actual_snapshot.json'))
 (home/'preservation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(passed=True,home=str(home),receipt_SHA256=sha(home/'preservation_receipt.json'),current_Meta=active,GLM41_full200_NF_passed_score555_threeD26pct_G_blank=True,canonical_unchanged=True),ensure_ascii=False))
if __name__=='__main__':main()
