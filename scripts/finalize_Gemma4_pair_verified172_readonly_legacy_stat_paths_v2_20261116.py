"""Read-only post-commit audit; never modify canonical files or an active model.
The two authorized G writes/local CAS/binding completed under the original lock.
The original finalizer failed on legacy optional export metadata and is preserved.
This independent finalizer writes diagnostics/export only, reuses exact unchanged
old170 five-file stats and never reruns old CPU200 or claims pruned arrays exist.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
from collections import Counter
import json,hashlib,base64,zipfile,xml.etree.ElementTree as ET,re,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'Gemma4_pair_original_BF16_verified172_blank_G_publication_20261116';A=H/'readonly_legacy_stat_path_finalization_v2';sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import atomic_json,workbook_rows
from eager_corrective_control_safety_2203 import core_unchanged
def read(p):return json.loads(Path(p).read_bytes())
cache={}
def sha(p):
 p=Path(p);s=p.stat();k=(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if k not in cache:
  with p.open('rb') as f:cache[k]=hashlib.file_digest(f,'sha256').hexdigest()
 return cache[k]
core_unchanged();Q=read(H/'reviewed_local_receipt_ready.json');local=base64.b64decode(Q['local_receipt_base64'],validate=True);assert hashlib.sha256(local).hexdigest()==Q['local_receipt_SHA256'];lp=json.loads(local);pub=read(H/'publication_receipt.json');dual=read(H/'dual_publication_completed.json');book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');p=D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json'
assert pub['completed'] and set(lp['changed_G_cells'])=={'G300','G305'} and sha(book)==lp['after_sha256']==Q['after_book_SHA256']==dual['local_and_server_workbook_sha256']=='0c377455da279683db143f5d167ff431db8378b66c2fafb77950dda47a820686'
assert sha(p)==dual['local_binding_sha256'] and read(p)['real_record_count']==read(p)['real_measured_record_count']==172
original=read(B/'Gemma4_pair_verified172_publication_review_inputs_v1_20261116/publication_actual_completion_v2.json');assert original['actual_returncode']==1
protected=[book,D/'state.json',D/'registry.json',p,D/'PAUSE',D/'POLICY.json',D/'A100_ALIGNMENT_PUBLICATION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'];before={str(x):sha(x) for x in protected}
A.mkdir(exist_ok=False)
def readonly_finalize():
 # Fresh target/source/external checks acknowledge only this authorized batch's canonical pre/post transition.
 transitions=[];maps=[];after=dict(pub['canonical_after_sha256']);after[str(p)]=sha(p)
 def checkmap(label,m):
  for path,h in m.items():
   assert re.fullmatch('[0-9a-f]{64}',h)
   if path in pub['canonical_before_sha256']:
    assert h==pub['canonical_before_sha256'][path] and sha(path)==after[path];transitions.append(dict(path=path,before=h,after=sha(path),reason='Authorized own Gemma4E2B original BF16 publication; original inference and proof bytes remain unchanged'))
   else:assert sha(path)==h,(label,path)
  maps.append(dict(label=label,entries=len(m)))
 state=read(D/'state.json');registry=read(D/'registry.json');counts=Counter(v['status'] for v in state['models'].values());assert dict(counts)=={'done':172,'skipped_no_complete_weights':310,'needs_adaptation':218}
 rows,sheet,bh=workbook_rows(book);assert len(rows)==len({r['model_id'] for r in rows})==700 and state['current'] is None
 oldstate=read(H/'before_verified_write/state.json');oldreg=read(H/'before_verified_write/registry.json');targets={x['model_id'] for x in pub['targets']}
 assert all(v==state['models'][m] for m,v in oldstate['models'].items() if m not in targets) and all(v==registry[m] for m,v in oldreg.items()) and set(registry)==set(oldreg)|targets
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:values={c.get('r'):c.findtext('s:v',namespaces=ns) for c in ET.fromstring(z.read(sheet)).findall('.//s:c',ns)}
 within=sum(bool(Decimal(values[f"D{r['row']}"]) and abs(Decimal(values[f"D{r['row']}"])-Decimal(r['score']))/abs(Decimal(values[f"D{r['row']}"]))<=Decimal('.05')) for r in rows if r['score'] is not None)
 assert within==157 and sum(r['score'] is not None for r in rows)==172
 for mid,e in state['models'].items():
  if e['status']=='skipped_no_complete_weights':assert next(r for r in rows if r['model_id']==mid)['score'] is None
 targetaudits=[]
 for t in pub['targets']:
  mid=t['model_id'];spec=registry[mid];entry=state['models'][mid];proof=read(t['formal_proof']);assert sha(t['formal_proof'])==t['formal_proof_sha256']==spec['accepted_formal200_proof_sha256'] and entry==read(D/'receipts'/(mid.replace('/','_')+'.json'))
  assert proof['passed'] and proof['sample_count']==200 and proof['actual_NPU_inference'] and proof['formal_alignment_within_5pct'] and proof['original_protocol_final_qualification_complete'] and not proof['historical_record_reuse']
  row,=[r for r in rows if r['model_id']==mid];assert Decimal(row['score'])==Decimal(str(t['new_G']))==Decimal(str(entry['score']))
  folder=Path(t['new_report']);assert {f.name for f in folder.iterdir()}==set(proof['report_sha256'])=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
  checkmap(mid+' complete five',{str(folder/n):h for n,h in proof['report_sha256'].items()});checkmap(mid+' external',proof['external_evidence_sha256'])
  for name,h in spec['source_sha256'].items():assert not Path(name).is_absolute() and '..' not in Path(name).parts and sha(R/name)==h
  approval=read(spec['alignment_approval_review']);assert sha(spec['alignment_approval_review'])==spec['alignment_approval_review_sha256'] and approval['approved']
  for name,key in [('true3_proof','true3_proof_SHA256'),('capacity_review','capacity_review_SHA256')]:assert sha(approval[name])==approval[key] and read(approval[name])['passed']
  tests=read(H/mid.replace('/','--')/'isolated_11_writer_tests.json');assert tests['passed'] and tests['test_count']==11 and tests['production_workbook_unchanged']
  for k,m in proof.items():
   if k!='external_evidence_sha256' and k.endswith('evidence_sha256') and isinstance(m,dict):checkmap(mid+' '+k,m)
  targetaudits.append(dict(model_id=mid,cell=f"G{row['row']}",G=row['score'],D=values[f"D{row['row']}"],proof_SHA256=t['formal_proof_sha256'],three_D_errors=t['exact_three_D_errors'],forward_count=t['actual_forward_count']))
 with zipfile.ZipFile(next(x for x in (H/'before_verified_write').iterdir() if x.suffix=='.xlsx')) as a,zipfile.ZipFile(book) as b:
  assert a.namelist()==b.namelist() and a.comment==b.comment
  for info in a.infolist():
   j=b.getinfo(info.filename)
   for attr in ('date_time','compress_type','external_attr','create_system','extra','comment'):assert getattr(info,attr)==getattr(j,attr)
   if info.filename!=sheet:assert a.read(info.filename)==b.read(info.filename)
  ac={x.get('r'):ET.tostring(x) for x in ET.fromstring(a.read(sheet)).findall('.//s:c',ns)};bc={x.get('r'):ET.tostring(x) for x in ET.fromstring(b.read(sheet)).findall('.//s:c',ns)}
  assert {k for k in set(ac)|set(bc) if ac.get(k)!=bc.get(k)}=={'G300','G305'}
 baseline=read(B/'OSAtlas4B_original_probability_verified170_blank_G_publication_20261116/verified170_export_complete_metadata.json');assert baseline['measured_count']==len(baseline['records'])==170
 for rec in baseline['records']:
  row,=[r for r in rows if r['model_id']==rec['model_id']];assert row['task']==rec['task'] and Decimal(row['score'])==Decimal(rec['current_G']) and Decimal(values[f"D{row['row']}"])==Decimal(rec['current_D'])
 for t in pub['targets']:
  row,=[r for r in rows if r['model_id']==t['model_id']];e=state['models'][t['model_id']];s=registry[t['model_id']]
  baseline['records'].append(dict(model_id=t['model_id'],task=row['task'],current_row=row['row'],current_D=values[f"D{row['row']}"],current_G=row['score'],actual_canonical_metric=e['score'],accepted_report=e['report_dir'],active_report_files=e['active_report_files'],accepted_formal_proof=s['accepted_formal200_proof'],accepted_formal_proof_SHA256=s['accepted_formal200_proof_sha256']))
 baseline.update(measured_count=172,numeric_within5pct=157,numeric_outside5pct=15,complete_weights_pending_formal=218,confirmed_missing=310,original_protocol_qualified=58,workbook_SHA256=bh,UTC=datetime.now(timezone.utc).isoformat(),source='Incremental verified172 after own Gemma4 E2B and E4B same-baseline publication; old170 proof reuse')
 for k in list(baseline):
  if k.startswith('next_new') or k.startswith('prior_') or k in ['target_full_audit','report_stat_baseline']:baseline.pop(k)
 baseline.update(current_original_A100_protocol_qualified=58,blank_count=528,server_book_SHA256=bh,source_canonical_SHA256={str(x):sha(x) for x in [book,D/'state.json',D/'registry.json',p]},prior_170_record_evidence_not_rerun=True,next_new173_count_branch_review_mandatory=True)
 # Reuse unchanged old170 stat/proof evidence; do not rerun old CPU200 or arrays.
 oldstats_path=B/'OSAtlas4B_original_probability_verified170_blank_G_publication_20261116/active170_full_report_stat_baseline.json'
 assert sha(oldstats_path)=='84353a17f7c5be919f786180339525439682b491511a2985a022b16099e3c9d5'
 oldstats=read(oldstats_path);stats={};legacy_optional_file_maps_reused=[]
 for rec in baseline['records']:
  mid=rec['model_id'];stats[mid]={}
  files=rec.get('active_report_files')
  if files is None:
   assert mid not in targets and mid in oldstats['reports'] and isinstance(rec['accepted_report'],str)
   files={name:{'path':str(Path(rec['accepted_report'])/name)} for name in oldstats['reports'][mid]}
   legacy_optional_file_maps_reused.append(mid)
  assert set(files)=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
  for name,info in files.items():
   q=Path(info['path']);st=q.stat();value=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns];stats[mid][name]=value
   if mid not in targets:assert value==oldstats['reports'][mid][name],(mid,name,'Old accepted report stat changed: targeted deep review required before acceptance')
 assert len(stats)==172 and len(oldstats['reports'])==170
 stat_path=H/'active172_full_report_stat_baseline.json';assert not stat_path.exists()
 atomic_json(stat_path,dict(reports=stats,prior170_report_stats_exactly_unchanged=True,prior170_stat_file_SHA256=sha(oldstats_path),stat_only_cache_invalidation_not_new_acceptance=True))
 baseline['report_stat_baseline']=str(stat_path)
 export=H/'verified172_export_complete_metadata.json';assert not export.exists();atomic_json(export,baseline)
 statmeta=read(stat_path);statmeta['export_SHA256']=sha(export);atomic_json(stat_path,statmeta)
 audit=dict(passed=True,readonly_post_commit_finalization=True,no_workbook_state_registry_binding_PAUSE_policy_or_model_job_write=True,original_publication_process_returncode_1_preserved=True,original_error='Legacy accepted records may have active_report_files=None; accepted_report and frozen unchanged five-file stat baseline supply exact original paths',legacy_optional_file_map_models_reused=legacy_optional_file_maps_reused,UTC=datetime.now(timezone.utc).isoformat(),local_and_server_book_SHA256=bh,canonical_counts=dict(counts),numeric_within5pct=157,numeric_outside5pct=15,complete_weights_pending_formal=218,confirmed_missing=310,qualified_delivered=58,old170_state_registry_G_C_B_D_preserved=True,all_non_G_formulas_styles_rows_ZIP_members_and_metadata_preserved=True,confirmed310_missing_G_all_blank=True,target_audits=targetaudits,all_target_external_maps_checked=maps,distinct_SHA256_files_checked=len(cache),authorized_canonical_pre_to_post_transitions=transitions,only_new_targets_deep_audited_no_old_model_CPU200_rerun=True,verified172_export=str(export),verified172_export_SHA256=sha(export),canonical_SHA256={str(x):sha(x) for x in [book,D/'state.json',D/'registry.json',p]})
 atomic_json(H/'full_target_dual_publication_audit_v2_legacy_stat_paths_readonly.json',audit)
 return dict(dual=dual,audit=audit,audit_path=str(H/'full_target_dual_publication_audit_v2_legacy_stat_paths_readonly.json'),audit_SHA256=sha(H/'full_target_dual_publication_audit_v2_legacy_stat_paths_readonly.json'),export=baseline)

try:
 result=readonly_finalize()
 assert before=={str(x):sha(x) for x in protected}
 completion=dict(actual_returncode=0,UTC=datetime.now(timezone.utc).isoformat(),read_only_post_commit_audit=True,original_publication_actual_returncode_1_preserved=True,canonical_before_and_after_identical=before,old170_full_reports_stat_unchanged=True,new_targets_full_proofs_sources_external_SHA_checked=True,full_target_audit=result['audit_path'],full_target_audit_SHA256=result['audit_SHA256'],both_workbooks_identical_SHA256=sha(book),no_new_NPU_or_old_CPU200_rerun=True)
 atomic_json(A/'actual_completion.json',completion);print(json.dumps(completion,ensure_ascii=False))
except BaseException:
 import traceback
 atomic_json(A/'actual_completion.json',dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc(),no_canonical_write_by_this_source=True));raise
