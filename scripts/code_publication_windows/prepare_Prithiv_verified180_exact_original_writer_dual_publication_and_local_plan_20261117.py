"""Bind the audited one-target publisher to genuine accepted Prithiv200 only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import ast,json,hashlib,re,shutil,zipfile,xml.etree.ElementTree as ET,base64
P=Path(__file__).resolve().parent;x=json.loads((P/'R_heartbeat1916_Prithiv_verified180_exact_publication_inputs_20261117.json').read_bytes());MID='prithivMLmods/gemma-4-E4B-it-Uncensored-MAX';row=x['current_row']['row'];assert row==572;proof=x['formal_proof'];assert x['completion']['actual_returncode']==x['completion']['model_returncode']==x['completion']['verifier_returncode']==x['controller_completion']['actual_returncode']==0 and not x['actual_procs'] and not x['FLOCK']
assert proof['sample_count']==200 and proof['metrics']=={'Accuracy':.88} and proof['actual_forward_count']==18669 and proof['formal_alignment_within_5pct'] and proof['original_protocol_final_qualification_complete'] and proof['all2130_own_source_actual_NPU_and_unused_SHA256_verified'] and proof['all54_unused_shared_KV_source_preserved'] and proof['all_full_head_rows_independently_recomputed'] and proof['all_native_KV_values_losslessly_reconstructed'] and all(Decimal(v)<=Decimal('.05') for v in proof['relative_error_to_current_A100_D'].values())
s=(P/'publish_Granite41_verified178_single_original_lock_dual_transaction_v1_20261117.py').read_text()
oldpub='DeadbydawnGemma4E2B_original_HF_BF16_verified177_blank_G_publication_20261117';newpub='MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117'
mapping={'ibm-granite/granite-vision-4.1-4b':MID,'f2f05d07e779323349b936d505c72984027ef147ba44ddf2f5fed751e34e164d':'9bd689a4a763bb716c6f9d4c3869aed95bb8d4b1ea9ede4b418591e3226a859c','4bd737d83ebae4c86727d802c00cf87a81d2ff1c24354b6dea90a5c0be18ba12':'3534a1dc1762fcb276d9c074d52bc47b04cf82459bb58424c2893804dfd301d1','15a2409e539948e56ed93ab1af369c8c8240ef0aec8ec7164de5b3a107c94324':x['true3_proof_SHA256'],'10ab41844e705abe094c5c4c4151d53bfac6a310674097719feef3157254c88a':x['prior179_stat_SHA256'],oldpub+'/verified177_export_complete_metadata.json':newpub+'/verified179_export_complete_metadata.json',oldpub+'/active177_full_report_stat_baseline.json':newpub+'/active179_full_report_stat_baseline.json','all1083_own_source_actual_NPU_SHA_verified':'all2130_own_source_actual_NPU_and_unused_SHA256_verified','all100352_head_rows_independently_recomputed':'all_full_head_rows_independently_recomputed','all40_native_KV_values_losslessly_reconstructed':'all_native_KV_values_losslessly_reconstructed','all_original_inputs_tokens_raw_references_errors_preserved':'all_full_inputs_tokens_raw_references_errors_preserved','User-confirmed A100 workbook D; own original Granite4.1 HF BF16/4096 and completed full independent NPU200 audit':'User-confirmed A100 workbook D; own original Prithiv Gemma4E4B BF16/card2/2048 and completed full independent NPU200 audit'}
nums={177:179,178:180,163:165,212:210,64:66,531:529,522:520,1083:2130}
class Bind(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str):
   if n.value in mapping:n.value=mapping[n.value];return n
   if re.fullmatch('[0-9a-f]{64}',n.value):return n
   n.value=n.value.replace('Granite41','PrithivGemma4E4B').replace('Granite4.1','Prithiv Gemma4E4B').replace('G364','G572')
   n.value=re.sub(r'(?<!\d)(177|178)(?!\d)',lambda m:{'177':'179','178':'180'}[m.group()],n.value)
   return n
  if type(n.value) is int:n.value=nums.get(n.value,n.value)
  return n
 def visit_Name(self,n):
  if n.id=='granite_capacity_review_passed':n.id='prithiv_capacity_review_passed'
  return n
 def visit_FunctionDef(self,n):
  if n.name=='granite_capacity_review_passed':
   return ast.parse("def prithiv_capacity_review_passed(cap):\n return cap.get('passed') is True and cap.get('capacity_passed') is True and cap.get('no_storage_format_change') is True and cap.get('no_model_math_dtype_input_budget_or_retention_change') is True and cap.get('original_factor4_reserve6GiB_and_all_runtime_guards_preserved') is True and cap.get('all200_own_original_input_shapes_checked') is True and len(cap['own_original_prompt_lengths'])==200 and len(cap['entries'])==3 and len(cap['forecast197'])==197 and cap['empirical_estimate_bytes']==123344571732\n").body[0]
  return self.generic_visit(n)
 def visit_keyword(self,n):
  self.generic_visit(n)
  if n.arg:
   n.arg=n.arg.replace('next_new179','next_new181');n.arg=re.sub(r'(?<!\d)(177|178)(?!\d)',lambda m:{'177':'179','178':'180'}[m.group()],n.arg)
   if n.arg=='active_source_tensor_count':n.value=ast.Constant(2076)
   if n.arg=='original_unloaded_source_count':n.arg='original_unused_sharedKV_source_count';n.value=ast.Constant(54)
   if n.arg=='actual_NPU_card':n.value=ast.Constant(2)
   if n.arg=='all1083_own_stored_source_actual_NPU_SHA256_verified':n.arg='all2130_own_stored_source_and_actual_NPU_or_original_unused_SHA256_verified'
  return n
tree=Bind().visit(ast.parse(s))
# The unused54 are original stored source, not falsely claimed active NPU weights.
for n in ast.walk(tree):
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROFILES' for t in n.targets):n.value=ast.parse(repr({MID:dict(total=2130,source_guard='all2130_own_source_actual_NPU_and_unused_SHA256_verified',unused_guard='all54_unused_shared_KV_source_preserved')}),mode='eval').body
source=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(source);assert 'G364' not in source and 'all1083_' not in source and 'all40_native' not in source
pub='scripts/publish_PrithivGemma4E4B_verified180_single_original_lock_dual_transaction_v1_20261117.py';(P/Path(pub).name).write_bytes(source.encode());ph=hashlib.sha256(source.encode()).hexdigest()
remote='import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport os,json,base64,hashlib,ast\nos.chdir("/root");R=Path("/root/projects/comperror");p=R/'+repr(pub)+';raw=base64.b64decode('+repr(base64.b64encode(source.encode()).decode())+');assert hashlib.sha256(raw).hexdigest()=='+repr(ph)+';ast.parse(raw)\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps({"source":str(p),"SHA256":'+repr(ph)+',"staged_only":True}))\n'
(P/'stage_Prithiv_verified180_original_writer_dual_publication_primary_20261117_remote.py').write_text(remote,encoding='utf8',newline='\n')
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(book)=='9bd689a4a763bb716c6f9d4c3869aed95bb8d4b1ea9ede4b418591e3226a859c';ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
 wb=ET.fromstring(z.read('xl/workbook.xml'));first=wb.find('s:sheets/s:sheet',ns);sheetname=first.get('name');rid=first.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id');rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));target=next(n.get('Target') for n in rels if n.get('Id')==rid);sheetxml=target.lstrip('/') if target.startswith('/') else 'xl/'+target;strings=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else [];vals={}
 for c in ET.fromstring(z.read(sheetxml)).findall('.//s:c',ns):
  v=c.findtext('s:v',namespaces=ns);vals[c.get('r')]=strings[int(v)] if c.get('t')=='s' else ''.join(c.find('s:is',ns).itertext()) if c.get('t')=='inlineStr' else v
ids={v:int(k[1:]) for k,v in vals.items() if k.startswith('C') and k!='C1' and v};assert len(ids)==700 and ids[MID]==572 and not vals.get('G572') and vals['B572']=='any-to-any' and Decimal(vals['D572'])==Decimal('.87')
old=json.loads((P/'MervePaliCompact_verified179_incremental_real_results_export_20261117.json').read_bytes());assert old['measured_count']==len(old['records'])==179
for r in old['records']:
 i=ids[r['model_id']];assert vals[f'B{i}']==r['task'] and Decimal(vals[f'D{i}'])==Decimal(r['current_D']) and Decimal(vals[f'G{i}'])==Decimal(r['current_G'])
H=P/'PrithivGemma4E4B_local_result_sync_verified180_20261117';H.mkdir(exist_ok=False);backup=H/'original_user_workbook.xlsx';shutil.copy2(book,backup);st=book.stat();q=dict(target=str(book),source_backup=str(backup),expected_before_sha256=sha(book),sheet_name=sheetname,sheet_xml=sheetxml,last_model_row=max(ids.values()),changes=[dict(model_id=MID,task='any-to-any',row=row,cell='G572',old_value=None,new_value=.88,A100_full_precision_cached_D=vals['D572'],relative_error=str(abs(Decimal(vals['D572'])-Decimal('.88'))/abs(Decimal(vals['D572']))),new_real_NPU200=True,A100_reference_from_user_confirmed_D=True,report_dir=x['completion']['accepted'][-1]['report_dir'],formal_proof_sha256=x['formal_proof_SHA256'])],real_measured_count=180,numeric_within5pct=165,numeric_outside5pct=15,complete_weights_pending_formal=210,confirmed_missing=310,original_protocol_qualified=66,model_blank_count=520,all_prior179_C_B_D_G_checked=True,complete_numeric_verifier_passed=True,local_stat_before=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns));(H/'plan.json').write_bytes(json.dumps(q,ensure_ascii=False,indent=2).encode())
mjs=(P/'granite41_workbook_cell_verified178_20261117.mjs').read_text().replace('B362:G366','B570:G574');(P/'prithiv_workbook_cell_verified180_20261117.mjs').write_bytes(mjs.encode());print(json.dumps(dict(publisher=pub,publisher_SHA256=ph,plan=str(H/'plan.json'),target_cell='G572',score=.88,old179_preserved=True,production_not_written=True)))
