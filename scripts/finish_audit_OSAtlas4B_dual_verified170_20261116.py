import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
from collections import Counter
import json,hashlib,base64,fcntl,copy,zipfile,xml.etree.ElementTree as ET,re,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'OSAtlas4B_original_probability_verified170_blank_G_publication_20261116'
Q=json.loads(Path(sys.argv[1]).read_bytes())
sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import atomic_json,workbook_rows
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from eager_corrective_control_safety_2203 import kernel_lock_owned,core_unchanged
def read(p):return json.loads(Path(p).read_bytes())
hc={}
def sha(p):
 p=Path(p);s=p.stat();k=(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if k not in hc:
  with p.open('rb') as f:hc[k]=hashlib.file_digest(f,'sha256').hexdigest()
 return hc[k]
assert not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
with (D/'runner.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged()
 local=base64.b64decode(Q['local_receipt_base64'],validate=True);assert hashlib.sha256(local).hexdigest()==Q['local_receipt_SHA256'];lp=json.loads(local)
 pub=read(H/'publication_receipt.json');assert pub['completed'] and pub['old169_state_registry_and_G_preserved'] and lp['changed_G_cells']==['G563']
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)==lp['after_sha256']==Q['after_book_SHA256']=='195e9af1a5284c764c795516853256bca54e9880ddd7b388cfefaa9a3c31b496'
 p=D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json';before=p.read_bytes();assert hashlib.sha256(before).hexdigest()=='782e32c5c9aa7a5fea92c9a683f5dba045bf760063c855ec9f856868ebdcbae3';binding=json.loads(before)
 assert binding['local_workbook_sha256']==lp['before_sha256'] and binding['real_record_count']==binding['real_measured_record_count']==169
 assert binding['local_A100_reference_snapshot_sha256']=='700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba' and binding['current_user_workbook_D_reference_snapshot_sha256']=='14ab07ef221d0d9deb6da157bcb3e7cadd9577b6a7d0110eb333235c4728c49b'
 (H/'before_local_binding_verbatim.json').write_bytes(before);(H/'user_local_publication_receipt.json').write_bytes(local)
 binding.update(local_workbook_sha256=lp['after_sha256'],real_record_count=170,real_measured_record_count=170,numeric_within5pct=155,numeric_outside5pct=15,blank_count=539,original_reference_numeric_within5pct=155,original_reference_numeric_outside5pct=15,last_model_update='OS-Copilot/OS-Atlas-Base-4B',last_local_publication_receipt_SHA256=Q['local_receipt_SHA256'],this_publication_preserved_all_non_G_content=True)
 assert p.read_bytes()==before;atomic_json(p,binding)
 dual=dict(completed=True,UTC=datetime.now(timezone.utc).isoformat(),models=[x['model_id'] for x in pub['targets']],server_publication_receipt_sha256=sha(H/'publication_receipt.json'),local_publication_receipt_sha256=Q['local_receipt_SHA256'],local_and_server_workbook_sha256=sha(book),local_binding_sha256=sha(p),old169_exactly_preserved=True,only_one_G_value_changed=True,current_D_counts=dict(within=155,outside=15),preserved_original_D_counts=dict(within=155,outside=15))
 atomic_json(H/'dual_publication_completed.json',dual)
 # Fresh target/source/external checks acknowledge only this authorized batch's canonical pre/post transition.
 transitions=[];maps=[];after=dict(pub['canonical_after_sha256']);after[str(p)]=sha(p)
 def checkmap(label,m):
  for path,h in m.items():
   assert re.fullmatch('[0-9a-f]{64}',h)
   if path in pub['canonical_before_sha256']:
    assert h==pub['canonical_before_sha256'][path] and sha(path)==after[path];transitions.append(dict(path=path,before=h,after=sha(path),reason='Authorized OSAtlas original MCScore publication; original inference and proof bytes remain unchanged'))
   else:assert sha(path)==h,(label,path)
  maps.append(dict(label=label,entries=len(m)))
 state=read(D/'state.json');registry=read(D/'registry.json');counts=Counter(v['status'] for v in state['models'].values());assert dict(counts)=={'done':170,'skipped_no_complete_weights':310,'needs_adaptation':220}
 rows,sheet,bh=workbook_rows(book);assert len(rows)==len({r['model_id'] for r in rows})==700 and state['current'] is None
 oldstate=read(H/'before_verified_write/state.json');oldreg=read(H/'before_verified_write/registry.json');targets={x['model_id'] for x in pub['targets']}
 assert all(v==state['models'][m] for m,v in oldstate['models'].items() if m not in targets) and all(v==registry[m] for m,v in oldreg.items()) and set(registry)==set(oldreg)|targets
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:values={c.get('r'):c.findtext('s:v',namespaces=ns) for c in ET.fromstring(z.read(sheet)).findall('.//s:c',ns)}
 within=sum(bool(Decimal(values[f"D{r['row']}"]) and abs(Decimal(values[f"D{r['row']}"])-Decimal(r['score']))/abs(Decimal(values[f"D{r['row']}"]))<=Decimal('.05')) for r in rows if r['score'] is not None)
 assert within==155 and sum(r['score'] is not None for r in rows)==170
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
  assert {k for k in set(ac)|set(bc) if ac.get(k)!=bc.get(k)}=={'G563'}
 baseline=read(B/'GLMOCR_pair_common_baseline_verified169_G_publication_20261116/verified169_export_complete_metadata.json');assert baseline['measured_count']==len(baseline['records'])==169
 for rec in baseline['records']:
  row,=[r for r in rows if r['model_id']==rec['model_id']];assert row['task']==rec['task'] and Decimal(row['score'])==Decimal(rec['current_G']) and Decimal(values[f"D{row['row']}"])==Decimal(rec['current_D'])
 for t in pub['targets']:
  row,=[r for r in rows if r['model_id']==t['model_id']];e=state['models'][t['model_id']];s=registry[t['model_id']]
  baseline['records'].append(dict(model_id=t['model_id'],task=row['task'],current_row=row['row'],current_D=values[f"D{row['row']}"],current_G=row['score'],actual_canonical_metric=e['score'],accepted_report=e['report_dir'],active_report_files=e['active_report_files'],accepted_formal_proof=s['accepted_formal200_proof'],accepted_formal_proof_SHA256=s['accepted_formal200_proof_sha256']))
 baseline.update(measured_count=170,numeric_within5pct=155,numeric_outside5pct=15,complete_weights_pending_formal=220,confirmed_missing=310,original_protocol_qualified=56,workbook_SHA256=bh,UTC=datetime.now(timezone.utc).isoformat(),source='Incremental verified170 after authorized OSAtlas original probability publication; old169 proof reuse')
 for k in list(baseline):
  if k.startswith('next_new') or k.startswith('prior_') or k in ['target_full_audit','report_stat_baseline']:baseline.pop(k)
 baseline.update(current_original_A100_protocol_qualified=56,blank_count=530,server_book_SHA256=bh,source_canonical_SHA256={str(x):sha(x) for x in [book,D/'state.json',D/'registry.json',p]},prior_169_record_evidence_not_rerun=True,next_new171_count_branch_review_mandatory=True)
 export=H/'verified170_export.json';atomic_json(export,baseline)
 audit=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),local_and_server_book_SHA256=bh,canonical_counts=dict(counts),numeric_within5pct=155,numeric_outside5pct=15,complete_weights_pending_formal=220,confirmed_missing=310,qualified_delivered=56,old169_state_registry_G_C_B_D_preserved=True,all_non_G_formulas_styles_rows_ZIP_members_and_metadata_preserved=True,confirmed310_missing_G_all_blank=True,target_audits=targetaudits,all_target_external_maps_checked=maps,distinct_SHA256_files_checked=len(hc),authorized_canonical_pre_to_post_transitions=transitions,only_new_targets_deep_audited_no_old_model_CPU200_rerun=True,verified170_export=str(export),verified170_export_SHA256=sha(export),canonical_SHA256={str(x):sha(x) for x in [book,D/'state.json',D/'registry.json',p]})
 atomic_json(H/'full_target_dual_publication_audit.json',audit)
 result={'dual':dual,'audit':audit,'audit_path':str(H/'full_target_dual_publication_audit.json'),'audit_SHA256':sha(H/'full_target_dual_publication_audit.json'),'export':baseline}
print(json.dumps(result,ensure_ascii=False))
