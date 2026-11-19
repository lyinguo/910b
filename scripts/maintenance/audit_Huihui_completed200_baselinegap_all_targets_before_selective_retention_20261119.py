"""Audit one completed numerical measurement without qualifying or publishing it.

The exact independent verifier, model, scoring, reports and proof stay immutable.
All incorrect cases remain numerical debugging controls for this baseline gap.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
from collections import Counter
import hashlib,json,os,re,zipfile,xml.etree.ElementTree as ET
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
MID='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated'
J=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119'
HC=B/'HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v4_approval_schema_20261119'
H=B/'Huihui_completed200_full_numerical_baselinegap_target_audit_and_correct_nonexample_retention_20261119'
OLD=B/'PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117'
LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
BOOK=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'
FIVE={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def full_head(p):
 c=p.get('numeric_sample_checks',[])
 return len(c)==p['sample_count'] and {x['index'] for x in c}==set(range(p['sample_count'])) and sum(x['full_head_rows'] for x in c)==p['actual_forward_count'] and all(x['full_head_rows']>0 and len(x['head_reference_checks'])==x['full_head_rows'] and x['all40_native_cache_values_reconstructed'] is True and x['original_input_pixels_token_ids_checked'] is True and x['all_masks_3D_positions_and_own_visual_injection_checked'] is True for x in c)
def measurement_contract(p):
 assert p['passed'] is True and p['sample_count']==200 and p['actual_NPU_inference'] is True and p['NEW_real_NPU_inference'] is True and not p['historical_record_reuse']
 assert p['source_tensor_count']==704 and p['own_original_BF16_preserved'] and p['all_original_active_and_unused_source_bytes_verified'] and p['all_own_source_actual_NPU_SHA256_verified']
 assert p['all_native_KV_values_losslessly_reconstructed'] and p['all_full_inputs_tokens_raw_references_errors_preserved'] and full_head(p)
 assert p['independent_verifier_SHA256']=='22d2919d3bca51c68a6f3de45141ed55999d781aefccb9b22c18affc79e61a1d'
 assert not p['A100_historical_engine_or_tensor_capture_claimed'] and not p['formal_alignment_within_5pct'] and not p['original_protocol_final_qualification_complete']
 assert p['metrics']=={'Accuracy':.695} and p['actual_forward_count']==4325 and not p['error_indices']
 return True
def preserve_and_remove_indices(details,outer,special=()):
 assert outer>=1 and [x['index'] for x in details]==list(range(200))
 correct={x['index'] for x in details if isinstance(x['prediction'],str) and re.fullmatch('[A-E]',x['prediction'].strip().upper()) and x['prediction'].strip().upper()==str(x['reference']).strip().upper()}
 base=set(range(10))|set(range(90,110))|set(range(190,200));groups={i//outer for i in base};examples={i for i in range(200) if i//outer in groups}
 wrong=set(range(200))-correct;keep=examples|wrong|set(special)
 keep.update(x['index'] for x in details if x.get('extra',{}).get('error') or x.get('error') or not x.get('raw_output'))
 return dict(correct=sorted(correct),incorrect=sorted(wrong),examples=sorted(examples),keep=sorted(keep),remove=sorted(correct-keep))
def quiet():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 others=[p for p in project_processes() if p['PID']!=os.getpid()]
 assert not others and not runner_lock_owners(),('Still live: no measurement audit or cleanup boundary',others,runner_lock_owners())
def main(local_snapshot_SHA256):
 import batch_910b as batch
 from batch_910b_support import workbook_rows,validate_report
 from A100_alignment_evidence_runner_20261103 import unchanged
 from universal_eval.algorithm.tinyllava_strict_mc import StrictMCAccuracy
 quiet();assert not (H/'measurement_target_audit.json').exists()
 assert sha(J/'job.json')=='a1bf90557a326523990a6004396e2dbe336e85d72139b216fe00f5337340c0fd'
 job=js(J/'job.json');done=js(J/'completion.json');sup=js(HC/'actual_completion.json')
 assert done['actual_returncode']==done['verifier_returncode']==sup['actual_returncode']==0 and done['model_returncode']==1
 assert [x['sample_count'] for x in done['accepted']]==[3,200] and job['stages']==[3,200] and job['spec']['approved'] and not job['write_workbook']
 unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec'])
 a=js(job['spec']['alignment_approval_review']);assert sha(job['spec']['alignment_approval_review'])==job['spec']['alignment_approval_review_sha256']
 assert a['approved'] and a['real_NPU3_independent_proof_verified'] and a['numerical_review_verified'] and a['first200_measurement_only_not_preclaimed_qualification']
 assert sha(a['true3_proof'])==a['true3_proof_SHA256']=='7bbc04ae6b05c7845ca01dc3e682f94da39000230beea45f6fbf5dc6f773cc59' and sha(a['capacity_review'])==a['capacity_review_SHA256']
 cap=js(a['capacity_review']);assert cap['passed'] and cap['original_factor4_plus6GiB_and_all_sample_guards_unchanged']
 run=done['accepted'][-1];assert sha(run['proof'])==run['proof_sha256'];p=js(run['proof']);measurement_contract(p)
 folder=Path(run['report_dir']);assert set(x.name for x in folder.iterdir())==FIVE and set(p['report_sha256'])==FIVE
 score=validate_report(R,str(folder),MID,'image-text-to-text',200,p);assert Decimal(str(score))==Decimal('.695')
 report=js(folder/'eval_report.json');details=report['detailed_results'];assert len(details)==200
 assert StrictMCAccuracy().compute([x['prediction'] for x in details],[x['reference'] for x in details])==score
 assert report['statistics']['total_samples']==200 and report['statistics']['success_count']==200 and report['statistics']['error_count']==0
 special=set(p['error_indices'])|set(p['empty_indices'])|set(p['invalid_indices'])|set(p['truncated_indices'])
 selection=preserve_and_remove_indices(details,report['pipeline_config']['algorithm']['batch_size'],special)
 assert len(selection['correct'])==139 and len(selection['incorrect'])==61 and not set(selection['remove'])&set(selection['keep'])
 line=js(folder/'operator_lineage.json')['actual_execution'];trace=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()]
 assert [x['index'] for x in trace]==list(range(200)) and sum(x['actual_native_forwards'] for x in trace)==4325
 home=Path(line['evidence_root']).resolve();assert home.is_relative_to(R/'workspace/diagnostics/model_runs') and home.name=='HuihuiGLM46Flash_own_original_BF16_2048_20261119_153254_363360_1567344'
 records=[]
 for row in trace:
  e=js(row['actual_execution_file']);i=row['index'];archive=Path(e['archive']).resolve();assert archive.parent==home and archive.name=='sample_%05d.npz'%i and not e['error']
  assert p['external_evidence_sha256'][str(archive)]==e['archive_SHA256'] and sha(row['actual_execution_file'])==row['actual_execution_SHA256']
  s=archive.stat();records.append(dict(index=i,path=str(archive),home=str(home),model_id=MID,logical_indices=[i],bytes=s.st_size,allocated_bytes=s.st_blocks*512,SHA256=e['archive_SHA256'],proof=run['proof'],proof_SHA256=run['proof_sha256']))
 # All new sources, all external files and all three genuine phases get full SHA checks.
 cache={};maps=[]
 def checkmap(label,m):
  for n,h in m.items():
   path=Path(n);path=path if path.is_absolute() else R/path
   key=str(path)
   if key not in cache:cache[key]=sha(path)
   if cache[key]!=h:
    assert path==LED and h=='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5' and cache[key]=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32',(label,n,'SHA changed')
    receipt=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119/actual_completion.json';assert sha(receipt)=='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3' and js(receipt)['all_protected_SHA256_unchanged'] and js(LED)['events'][-1]['receipt_SHA256']==sha(receipt)
  maps.append(dict(label=label,count=len(m)))
 for label,q in [('original_true3',js(a['true3_proof']))]+[(str(x['sample_count']),js(x['proof'])) for x in done['accepted']]:
  assert q['passed'] and q['actual_NPU_inference'] and not q['historical_record_reuse'];validate_report(R,q['report_dir'],MID,'image-text-to-text',q['sample_count'],q)
  checkmap(label+'_external',q['external_evidence_sha256'])
  for k,v in q.items():
   if k!='external_evidence_sha256' and k.endswith(('evidence_sha256','evidence_SHA256')) and isinstance(v,dict):checkmap(label+'_'+k,v)
 checkmap('job_bound_source',{str(R/n):h for n,h in job['spec']['source_sha256'].items()})
 assert sha(H/'local_current_snapshot.json')==local_snapshot_SHA256;local=js(H/'local_current_snapshot.json')
 assert local['book_SHA256']==sha(BOOK)=='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34' and local['model_id']==MID and local['task']=='image-text-to-text' and local['G'] in ['',None]
 rows,sheet,_=workbook_rows(BOOK);row,=[x for x in rows if x['model_id']==MID];assert row['row']==355 and row['task']=='image-text-to-text' and row['score'] is None
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(BOOK) as z:current_D=ET.fromstring(z.read(sheet)).find('.//s:c[@r="D355"]',ns).findtext('s:v',namespaces=ns)
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');ref=Path(binding['local_A100_reference_snapshot']);assert sha(ref)==binding['local_A100_reference_snapshot_sha256']=='700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba'
 original=js(ref)['models'][MID];assert original['task']=='image-text-to-text';bases=dict(server_current_D=current_D,local_current_D=local['D'],preserved_original_full_precision_D=original['A100_D']);errors={k:str(abs(Decimal(v)-Decimal(str(score)))/abs(Decimal(v))) for k,v in bases.items()}
 assert all(Decimal(v)==Decimal('.945') for v in bases.values()) and all(Decimal(v)>Decimal('.05') for v in errors.values())
 state=js(D/'state.json');reg=js(D/'registry.json');counts=dict(Counter(x['status'] for x in state['models'].values()));assert counts=={'done':180,'skipped_no_complete_weights':310,'needs_adaptation':210}
 assert state['models'][MID]['status']=='needs_adaptation' and state['models'][MID]['score'] is None and MID not in reg and state['current'] is None
 export=js(OLD/'verified180_export_complete_metadata.json');assert sha(OLD/'verified180_export_complete_metadata.json')=='33de521bbe14ccd680edbc0da1a5b08a5d0b075248ce5841bc9d7d8bd0d9757b'
 byid={x['model_id']:x for x in rows};assert len(rows)==700 and sum(x['score'] is not None for x in rows)==180
 for x in export['records']:
  r=byid[x['model_id']];assert r['row']==x['current_row'] and r['task']==x['task'] and Decimal(str(r['score']))==Decimal(x['current_G'])
 assert all(byid[m]['score'] is None for m,x in state['models'].items() if x['status']=='skipped_no_complete_weights')
 baseline=js(OLD/'active180_full_report_stat_baseline.json');assert sha(OLD/'active180_full_report_stat_baseline.json')=='2ad3869d9152b05f7235a649525fdd54328c6ac157ae938cae23f8f62254dc37'
 for mid,m in baseline['reports'].items():
  for name,old in m.items():
   f=Path(state['models'][mid]['report_dir'])/name;s=f.stat();actual=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
   assert actual==old,(mid,name,'old stat changed')
 unchanged(job['protected_canonical_hashes']);quiet()
 audit=dict(passed=True,UTC=now(),kind='Full numerical acceptance and target audit of an outside5pct measurement; no qualification or G publication',job=str(J),job_SHA256=sha(J/'job.json'),completion_SHA256=sha(J/'completion.json'),controller_completion_SHA256=sha(HC/'actual_completion.json'),formal_proof=run['proof'],formal_proof_SHA256=run['proof_sha256'],report_dir=str(folder),all_five_original_SHA256=p['report_sha256'],sample_count=200,actual_forward_count=4325,NPU_card=4,score=score,three_D=bases,relative_errors=errors,original_protocol_final_qualification_complete=False,G_publication_permitted=False,no_G_state_registry_binding_or_workbook_write=True,canonical_counts=counts,numeric_within5pct=165,numeric_outside5pct=15,qualified_delivered=66,all180_C_B_D_G_and_full_five_stats_preserved=True,all_source_and_external_maps_checked=maps,distinct_full_SHA_files_checked=len(cache),canonical_SHA256=job['protected_canonical_hashes'],home=str(home),trace=str(line['sample_trace']),trace_SHA256=sha(line['sample_trace']),selection=selection,all200_archive_original_metadata=records,full_correct_and_incorrect_original_records_retained=True,all_incorrect_and_original_first_middle_last_full_outer_groups_must_be_retained=True,all_small_fresh3_failed_unverified_values_retained=True,original_true3_proof=a['true3_proof'],fresh3_proof=done['accepted'][0]['proof'],local_snapshot_SHA256=local_snapshot_SHA256)
 put(H/'measurement_target_audit.json',audit)
 return dict(passed=True,audit=str(H/'measurement_target_audit.json'),audit_SHA256=sha(H/'measurement_target_audit.json'),score=score,three_D=bases,relative_errors=errors,incorrect_count=len(selection['incorrect']),keep_numeric_count=len(selection['keep']),eligible_correct_nonexample_count=len(selection['remove']),all_external_files=len(cache),publication_permitted=False,no_deletion_yet=True)
if __name__=='__main__':print(json.dumps(main(sys.argv[1]),ensure_ascii=False))
