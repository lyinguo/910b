"""Prepare the narrowly bound original-writer transaction and a disposable authoring plan.
Measured numeric success is not publication permission until the real full NF ends.
"""
from pathlib import Path
from decimal import Decimal
import ast,copy,hashlib,json,re,shutil,zipfile,xml.etree.ElementTree as E
V=Path(__file__).resolve().parent;BASE=Path('D:/code-lv/.tmp/cd22s1')
MID='unsloth/gemma-4-E4B-it-MLX-8bit';OLDMID='prithivMLmods/gemma-4-E4B-it-Uncensored-MAX'
PUB='Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122'
PRIOR='OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121'
OLDPRIOR='MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117'
BEFORE='0104ccaaab17fef845cfd172a7274c0a7c20faf51f858f21c0aef4e4a71992a1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
measurement=json.loads((V/'R_Gemma4MLX8_completed200_measurement_current_fullNF_20261122.json').read_bytes());assert measurement['metrics']=={'Accuracy':.84} and measurement['actual_native_forwards']==13383 and measurement['all_three_D_numeric_within5pct'] and measurement['error_indices']==[]
original=BASE/'scripts/publish_PrithivGemma4E4B_verified180_single_original_lock_dual_transaction_v1_20261117.py';assert sha(original)=='b6630776c326931926e9c6bbe58da77619875954086929050add7090b7863123';old=original.read_text(encoding='utf-8')
mapping={'9bd689a4a763bb716c6f9d4c3869aed95bb8d4b1ea9ede4b418591e3226a859c':BEFORE,'3534a1dc1762fcb276d9c074d52bc47b04cf82459bb58424c2893804dfd301d1':'f66b7500bbb8c5f5a99a4c35b248f8203b5fd9bde4bbaa15c7f32f21af58526c','94f5649a4117467baae93dee177a5b794fc4d171b4ae8d962c9067bfc251f645':'b84890225152fc6d92601a4c47a4e625e5ed227f4ffe271a8ee32873986beb16','e577683e744c71712ad748f3003b5d5958a191aad0cdc1fcfa59872e7972ce98':'1549f33251985bdd1c1c35bb3752e4dc992ae9f87be700ae3076c8ee899147ee'}
def string(s):
 if s in mapping:return mapping[s]
 if re.fullmatch('[0-9a-f]{64}',s):return s
 if s.startswith(OLDPRIOR+'/'):
  return PRIOR+'/'+s.split('/',1)[1].replace('verified179','verified184').replace('active179','active184')
 if s=='PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117':return PUB
 return s.replace(OLDMID,MID).replace('PrithivGemma4E4B','Gemma4E4BMLX8').replace('Prithiv Gemma4E4B','Gemma4 E4B MLX8-named actual BF16 source').replace('G572','G664').replace('old179','old184').replace('prior179','prior184').replace('prior_179','prior_184').replace('verified180','verified185').replace('active180','active185').replace('next_new181','next_new186').replace('card2','card4').replace('20261117','20261122')
class Adapt(ast.NodeTransformer):
 def visit_Constant(self,n):
  value=n.value
  if isinstance(value,str):value=string(value)
  elif type(value)==int:value={179:184,180:185,165:170,210:205,66:71,529:524,520:515}.get(value,value)
  return ast.copy_location(ast.Constant(value),n)
 def visit_Name(self,n):
  if n.id=='prithiv_capacity_review_passed':n.id='gemma4_MLX8_capacity_review_passed'
  return n
 def visit_keyword(self,n):
  n=self.generic_visit(n)
  if n.arg:
   n.arg=string(n.arg)
   if n.arg=='actual_NPU_card':n.value=ast.Constant(4)
  return n
tree=Adapt().visit(ast.parse(old));ast.fix_missing_locations(tree)
capacity='''def gemma4_MLX8_capacity_review_passed(cap):
 return cap.get('passed') is True and cap.get('capacity_passed') is True and cap.get('no_matrix_deleted') is True and cap.get('no_model_math_precision_budget_or_retention_change') is True and cap.get('original_factor')==4 and cap.get('original_reserve_bytes')==6*2**30 and cap.get('hard_reserve_bytes')==3*2**30 and cap.get('log_reserve_bytes')==2**30 and cap.get('precheck_reserve_bytes')==512*2**20 and cap.get('per_sample_guard_bytes')==2*2**30 and cap.get('configured_outer10_actual_inner1_unchanged') is True and len(cap.get('all200_original_input_shapes',[]))==200 and len(cap.get('all200_original_prompt_lengths',[]))==200 and len(cap.get('all197_forecast_mandatory_common_bytes',[]))==197 and len(cap.get('all_complete_ZIP_parts',[]))==1552 and [e['index'] for e in cap.get('entries',[])]==[0,1,2] and cap.get('empirical_estimate_bytes')==42861614112==4*(2*sum(e['archive_bytes'] for e in cap['entries'])+sum(cap['all197_forecast_mandatory_common_bytes']))+6*2**30 and cap.get('actual_free_bytes',0)>=42861614112
'''
for i,n in enumerate(tree.body):
 if isinstance(n,ast.FunctionDef) and n.name=='prithiv_capacity_review_passed':tree.body[i]=ast.parse(capacity).body[0]
source=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(source)
assert 'prithiv_capacity_review_passed' not in source and OLDMID not in source and 'G572' not in source and OLDPRIOR not in source
pubname='publish_Gemma4E4BMLX8_verified185_single_original_lock_dual_transaction_v1_20261122.py';(V/pubname).write_text(source,encoding='utf-8',newline='\n')

# Authoring in this temporary copy never rewrites the required user workbook.
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)==BEFORE
home=V/'Gemma4E4BMLX8_local_result_sync_verified185_20261122';home.mkdir(exist_ok=False);backup=home/'original_user_workbook.xlsx';shutil.copy2(book,backup)
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
 wb=E.fromstring(z.read('xl/workbook.xml'));first=wb.find('s:sheets/s:sheet',ns);sheetname=first.get('name');rid=first.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id');rels=E.fromstring(z.read('xl/_rels/workbook.xml.rels'));target=next(n.get('Target') for n in rels if n.get('Id')==rid);sheet=target.lstrip('/') if target.startswith('/') else 'xl/'+target
 strings=[''.join(t.itertext()) for t in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else [];values={}
 for c in E.fromstring(z.read(sheet)).findall('.//s:c',ns):
  val=c.findtext('s:v',namespaces=ns);values[c.get('r')]=strings[int(val)] if c.get('t')=='s' else ''.join(c.find('s:is',ns).itertext()) if c.get('t')=='inlineStr' else val
ids={v:int(k[1:]) for k,v in values.items() if k.startswith('C') and k!='C1' and v};assert len(ids)==700 and ids[MID]==664 and not values.get('G664') and values['B664']=='image-text-to-text' and Decimal(values['D664'])==Decimal('.835')
baseline=json.loads(Path('D:/code-lv/.tmp/batch_910b_20261030/OvisU1_verified184_incremental_real_results_export_20261121.json').read_bytes());assert baseline['measured_count']==len(baseline['records'])==184
for r in baseline['records']:
 i=ids[r['model_id']];assert values[f'B{i}']==r['task'] and Decimal(values[f'D{i}'])==Decimal(r['current_D']) and Decimal(values[f'G{i}'])==Decimal(r['current_G'])
st=book.stat();plan=dict(target=str(book),source_backup=str(backup),expected_before_sha256=BEFORE,sheet_name=sheetname,sheet_xml=sheet,last_model_row=max(ids.values()),changes=[dict(model_id=MID,task='image-text-to-text',row=664,cell='G664',old_value=None,new_value=.84,A100_full_precision_cached_D=values['D664'],relative_error=str(abs(Decimal('.84')-Decimal(values['D664']))/abs(Decimal(values['D664']))),new_real_NPU200=True,A100_reference_from_user_confirmed_D=True,report_dir=measurement['report_dir'],formal_proof_sha256=None)],real_measured_count=185,numeric_within5pct=170,numeric_outside5pct=15,complete_weights_pending_formal=205,confirmed_missing=310,original_protocol_qualified=71,model_blank_count=515,all_prior184_C_B_D_G_checked=True,complete_numeric_verifier_passed=False,production_publication_not_authorized_until_actual_fullNF_exit_and_all_quiet=True,local_stat_before=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns))
(home/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
mjs=(BASE/'scripts/code_publication_windows/prithiv_workbook_cell_verified180_20261117.mjs').read_text(encoding='utf-8').replace('B570:G574','B662:G666')
(V/'gemma4_MLX8_workbook_cell_verified185_20261122.mjs').write_text(mjs,encoding='utf-8',newline='\n')
review=dict(publisher='scripts/'+pubname,publisher_SHA256=sha(V/pubname),source_parent_SHA256=sha(original),source_bound_to_only_model=MID,prior184_baseline_preserved=True,original_frozen_writer_import_and_raw_one_G_write_unchanged=True,all_original_evidence_external_11_tests_EX_CAS_protection_retained=True,metadata_transition=dict(before=184,after=185,within_before=169,within_after=170,pending_after=205,qualified_before=70,qualified_after=71),nativeBF16_source_not_native8bit=True,plan=str(home/'plan.json'),MJS='scripts/windows_controls/gemma4_MLX8_workbook_cell_verified185_20261122.mjs',MJS_SHA256=sha(V/'gemma4_MLX8_workbook_cell_verified185_20261122.mjs'),production_workbook_untouched=True,fullNF_still_required=True)
(V/'Gemma4MLX8_verified185_publisher_artifact_plan_source_review_20261122.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(review,ensure_ascii=False))
