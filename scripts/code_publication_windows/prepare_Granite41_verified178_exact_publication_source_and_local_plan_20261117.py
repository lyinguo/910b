import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import json,ast,hashlib,base64,re,zipfile,shutil,xml.etree.ElementTree as ET
P=Path(__file__).parent;x=json.loads((P/'R_Granite41_completed_full_evidence_verified178_publication_20261117.json').read_bytes());MID='ibm-granite/granite-vision-4.1-4b';prior='deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit';proof=x['formal_proof'];score=proof['metrics']['Accuracy'];row=x['current_row']['row'];assert row==364
assert x['completion']['actual_returncode']==x['completion']['model_returncode']==x['completion']['verifier_returncode']==x['controller_completion']['actual_returncode']==0
assert proof['passed'] and proof['original_protocol_final_qualification_complete'] and proof['sample_count']==200 and proof['all1083_own_source_actual_NPU_SHA_verified'] and proof['all100352_head_rows_independently_recomputed'] and proof['all40_native_KV_values_losslessly_reconstructed'] and not proof['error_indices'] and all(Decimal(v)<=Decimal('.05') for v in proof['relative_error_to_current_A100_D'].values())
s=(P/'publish_DeadbydawnGemma4E2B_verified177_single_original_lock_dual_transaction_v2_finish_reference_scope_20261117.py').read_text()
s=s.replace("t['all2011_own_source_SHA_verified_with60_original_unused_sharedKV']","t['all1083_own_source_actual_NPU_SHA_verified']").replace("proof['all2011_own_source_SHA_verified_with60_original_unused_sharedKV']","proof['all1083_own_source_actual_NPU_SHA_verified']").replace("proof['qualification_basis']",repr('User-confirmed A100 workbook D; own original Granite4.1 HF BF16/4096 and completed full independent NPU200 audit'))
s=s.replace("and read(a['capacity_review'])['passed']","and granite_capacity_review_passed(read(a['capacity_review']))")
s=s.replace("assert sha(approval[name]) == approval[key] and read(approval[name])['passed']","assert sha(approval[name]) == approval[key] and (granite_capacity_review_passed(read(approval[name])) if name=='capacity_review' else read(approval[name])['passed'])")
mapping={prior:MID,'6d73108e181887c9dfd95b7c851981c1ac316238705fe37e8c5bf59c15dc9a82':'f2f05d07e779323349b936d505c72984027ef147ba44ddf2f5fed751e34e164d','3825072ab23d52237acf0f4db96fd4f7101128cebe21274205d1bb31019f5c26':'4bd737d83ebae4c86727d802c00cf87a81d2ff1c24354b6dea90a5c0be18ba12','13b0835212cc55ac129236c325e3db472c16366ccc997c5a9cc4c9edb9f94b2a':'15a2409e539948e56ed93ab1af369c8c8240ef0aec8ec7164de5b3a107c94324','e29a7c2725574f522ea0b1cdcc69404906a5932ff25b7c8bd7e39533ce930598':x['prior177_stat_SHA256'],'CoderGemma4E4B_original_HF_BF16_verified176_blank_G_publication_20261117/verified176_export_complete_metadata.json':'DeadbydawnGemma4E2B_original_HF_BF16_verified177_blank_G_publication_20261117/verified177_export_complete_metadata.json','CoderGemma4E4B_original_HF_BF16_verified176_blank_G_publication_20261117/active176_full_report_stat_baseline.json':'DeadbydawnGemma4E2B_original_HF_BF16_verified177_blank_G_publication_20261117/active177_full_report_stat_baseline.json','all1951_own_source_actual_NPU_SHA256_verified':'all1083_own_source_actual_NPU_SHA_verified','all_own_source_complete_no_foreign_or_placeholder_tensors':'all1083_own_source_actual_NPU_SHA_verified','all_full_head_rows_independently_recomputed':'all100352_head_rows_independently_recomputed','all_native_KV_values_losslessly_reconstructed':'all40_native_KV_values_losslessly_reconstructed','all_full_inputs_tokens_raw_references_errors_preserved':'all_original_inputs_tokens_raw_references_errors_preserved'}
class Bind(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str):
   if n.value in mapping:n.value=mapping[n.value];return n
   if re.fullmatch('[0-9a-f]{64}',n.value):return n
   n.value=n.value.replace('DeadbydawnGemma4E2B','Granite41').replace('Deadbydawn Gemma4E2B','Granite4.1').replace('Deadbydawn Gemma4 E2B','Granite4.1').replace('G239','G364');n.value=re.sub(r'(?<!\d)(176|177)(?!\d)',lambda m:{'176':'177','177':'178'}[m.group()],n.value)
  elif type(n.value) is int:n.value={176:177,177:178,162:163,213:212,63:64,532:531,523:522,2011:1083,1951:1083}.get(n.value,n.value)
  return n
 def visit_keyword(self,n):
  self.generic_visit(n)
  if n.arg:
   n.arg=n.arg.replace('next_new178','next_new179');n.arg=re.sub(r'(?<!\d)(176|177)(?!\d)',lambda m:{'176':'177','177':'178'}[m.group()],n.arg)
   if n.arg=='all1951_active_NPU_and60_original_unused_stored_source_SHA256_verified':n.arg='all1083_own_stored_source_actual_NPU_SHA256_verified'
   if n.arg=='original_unused_sharedKV_source_count':n.arg='original_unloaded_source_count';n.value=ast.Constant(0)
  return n
tree=Bind().visit(ast.parse(s));s=ast.unparse(ast.fix_missing_locations(tree))+'\n'
helper="""def granite_capacity_review_passed(cap):
    return cap.get('passed_independent_full_numeric_stage_and_five_reports_review') is True and cap.get('capacity_passed') is True and cap.get('no_matrix_deleted') is True and cap.get('no_model_math_precision_budget_or_retention_change') is True and cap['factor']==4 and cap['extra_budget_bytes']==6*1024**3 and len(cap['full_all200_own_input_shapes'])==200 and len(cap['all_complete_small3_ZIP_parts'])==3
"""
pos=s.index('def finish_local_and_full_target_audit');s=s[:pos]+helper+'\n'+s[pos:];ast.parse(s)
assert '6d73108e' not in s and 'G239' not in s and 'all1951_' not in s and "proof['qualification_basis']" not in s
pub='scripts/publish_Granite41_verified178_single_original_lock_dual_transaction_v1_20261117.py';(P/Path(pub).name).write_bytes(s.encode());ph=hashlib.sha256(s.encode()).hexdigest();remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,base64,hashlib
R=Path('/root/projects/comperror');n=__N__;raw=base64.b64decode(__RAW__);assert hashlib.sha256(raw).hexdigest()==__H__;ast.parse(raw);p=R/n
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(source=str(p),SHA256=__H__,staged_only_not_executed=True)))
'''.replace('__N__',repr(pub)).replace('__RAW__',repr(base64.b64encode(s.encode()).decode())).replace('__H__',repr(ph));ast.parse(remote);(P/'stage_Granite41_verified178_dual_publication_primary_not_execute_20261117_remote.py').write_bytes(remote.encode())
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(book)=='f2f05d07e779323349b936d505c72984027ef147ba44ddf2f5fed751e34e164d';ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
 wb=ET.fromstring(z.read('xl/workbook.xml'));first=wb.find('s:sheets/s:sheet',ns);sheetname=first.get('name');rid=first.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id');rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));target=next(n.get('Target') for n in rels if n.get('Id')==rid);sheetxml=target.lstrip('/') if target.startswith('/') else 'xl/'+target;strings=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else [];vals={}
 for c in ET.fromstring(z.read(sheetxml)).findall('.//s:c',ns):
  v=c.findtext('s:v',namespaces=ns);vals[c.get('r')]=strings[int(v)] if c.get('t')=='s' else ''.join(c.find('s:is',ns).itertext()) if c.get('t')=='inlineStr' else v
ids={v:int(k[1:]) for k,v in vals.items() if k.startswith('C') and k!='C1' and v};assert len(ids)==700 and ids[MID]==row and not vals.get('G364') and vals['B364']=='image-text-to-text' and Decimal(vals['D364'])==Decimal('.9150')
old=json.loads((P/'DeadbydawnGemma4E2B_verified177_incremental_real_results_export_20261117.json').read_bytes());assert old['measured_count']==len(old['records'])==177
for r in old['records']:
 i=ids[r['model_id']];assert vals[f'B{i}']==r['task'] and Decimal(vals[f'D{i}'])==Decimal(r['current_D']) and Decimal(vals[f'G{i}'])==Decimal(r['current_G'])
H=P/'Granite41_local_result_sync_verified178_20261117';H.mkdir(exist_ok=False);backup=H/'original_user_workbook.xlsx';shutil.copy2(book,backup);st=book.stat();q=dict(target=str(book),source_backup=str(backup),expected_before_sha256=sha(book),sheet_name=sheetname,sheet_xml=sheetxml,last_model_row=max(ids.values()),changes=[dict(model_id=MID,task='image-text-to-text',row=row,cell='G364',old_value=None,new_value=score,A100_full_precision_cached_D=vals['D364'],relative_error=str(abs(Decimal(vals['D364'])-Decimal(str(score)))/abs(Decimal(vals['D364']))),new_real_NPU200=True,A100_reference_from_user_confirmed_D=True,report_dir=x['completion']['accepted'][-1]['report_dir'],formal_proof_sha256=x['formal_proof_SHA256'])],real_measured_count=178,numeric_within5pct=163,numeric_outside5pct=15,complete_weights_pending_formal=212,confirmed_missing=310,original_protocol_qualified=64,model_blank_count=522,all_prior177_C_B_D_G_checked=True,complete_numeric_verifier_passed=True,local_stat_before=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns));(H/'plan.json').write_bytes(json.dumps(q,ensure_ascii=False,indent=2).encode())
mjs=(P/'deadbydawn_gemma4_e2b_workbook_cell_verified177_20261117.mjs').read_text().replace('B237:G241','B362:G366');(P/'granite41_workbook_cell_verified178_20261117.mjs').write_bytes(mjs.encode());print(json.dumps(dict(publisher=pub,publisher_SHA256=ph,plan=str(H/'plan.json'),target_cell='G364',score=score,old177_preserved=True,no_production_write_yet=True)))
