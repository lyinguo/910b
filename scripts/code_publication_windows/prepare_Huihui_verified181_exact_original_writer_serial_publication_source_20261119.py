"""Bind the previous full audited transaction to this own Huihui job; stage only."""
from pathlib import Path
import ast,base64,hashlib,json,re
P=Path(__file__).resolve().parent
source=(P/'publish_PrithivGemma4E4B_verified180_single_original_lock_dual_transaction_v1_20261117.py').read_text(encoding='utf8')
MID='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated'
oldpub='MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117'
newpub='PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117'
mapping={
 'prithivMLmods/gemma-4-E4B-it-Uncensored-MAX':MID,
 'PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117':'HuihuiGLM46Flash_original_BF16_2048_lossless_verified181_blank_G_publication_20261119',
 '9bd689a4a763bb716c6f9d4c3869aed95bb8d4b1ea9ede4b418591e3226a859c':'689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34',
 '3534a1dc1762fcb276d9c074d52bc47b04cf82459bb58424c2893804dfd301d1':'e3a02a66da363e3ca15649f1cdab2d553f693297dc1ccbad25ea8495d12ca88b',
 '94f5649a4117467baae93dee177a5b794fc4d171b4ae8d962c9067bfc251f645':'7bbc04ae6b05c7845ca01dc3e682f94da39000230beea45f6fbf5dc6f773cc59',
 'e577683e744c71712ad748f3003b5d5958a191aad0cdc1fcfa59872e7972ce98':'2ad3869d9152b05f7235a649525fdd54328c6ac157ae938cae23f8f62254dc37',
 oldpub+'/verified179_export_complete_metadata.json':newpub+'/verified180_export_complete_metadata.json',
 oldpub+'/active179_full_report_stat_baseline.json':newpub+'/active180_full_report_stat_baseline.json',
 'all2130_own_source_actual_NPU_and_unused_SHA256_verified':'all_own_source_actual_NPU_SHA256_verified',
 'all54_unused_shared_KV_source_preserved':'all_original_active_and_unused_source_bytes_verified',
 'User-confirmed A100 workbook D; own original Prithiv Gemma4E4B BF16/card2/2048 and completed full independent NPU200 audit':'User-confirmed A100 workbook D; own original Huihui GLM4.6V BF16/card4/2048/existing thinkingFalse, complete original math and full independent NPU200 audit; only lossless numeric storage changed',
 'first200_is_measurement_not_predetermined_5pct_qualification':'first200_measurement_only_not_preclaimed_qualification',
}
nums={179:180,180:181,165:166,210:209,66:67,529:528,520:519}
class Bind(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str):
   if n.value in mapping:n.value=mapping[n.value];return n
   if re.fullmatch('[0-9a-f]{64}',n.value):return n
   n.value=n.value.replace('PrithivGemma4E4B','HuihuiGLM46Flash').replace('Prithiv Gemma4E4B','Huihui GLM4.6V').replace('G572','G355')
   n.value=re.sub(r'(?<!\d)(179|180)(?!\d)',lambda m:{'179':'180','180':'181'}[m.group()],n.value)
   return n
  if type(n.value) is int:n.value=nums.get(n.value,n.value)
  return n
 def visit_Name(self,n):
  if n.id=='prithiv_capacity_review_passed':n.id='huihui_capacity_review_passed'
  return n
 def visit_FunctionDef(self,n):
  if n.name=='prithiv_capacity_review_passed':
   return ast.parse('''def huihui_capacity_review_passed(cap):
 return cap.get('passed') is True and cap.get('empirical_estimate_bytes')==146869927228 and cap.get('all333_names_shapes_dtypes_every_raw_bit_equal') is True and cap.get('only_standard_lossless_NPY_order_ZIP_codec_change') is True and cap.get('original_factor4_plus6GiB_and_all_sample_guards_unchanged') is True and cap.get('original_factor4_plus6GiB_and_hard3_log1_precheckHalf_sample2GiB_preserved') is True and len(cap['all200_original_prompt_lengths'])==200 and len(cap['all200_original_pixel_elements'])==200
''').body[0]
  return self.generic_visit(n)
 def visit_keyword(self,n):
  self.generic_visit(n)
  if n.arg:
   n.arg=re.sub(r'(?<!\d)(179|180)(?!\d)',lambda m:{'179':'180','180':'181'}[m.group()],n.arg)
   if n.arg=='active_source_tensor_count':n.value=ast.Constant(704)
   if n.arg=='original_unused_sharedKV_source_count':n.arg='original_unused_MTP_source_tensor_count';n.value=ast.Constant(0)
   if n.arg=='actual_NPU_card':n.value=ast.Constant(4)
   if n.arg=='all2130_own_stored_source_and_actual_NPU_or_original_unused_SHA256_verified':n.arg='all704_own_stored_source_and_actual_NPU_SHA256_verified'
  return n
 def visit_Subscript(self,n):
  if isinstance(n.slice,ast.Constant) and n.slice.value=='all_full_head_rows_independently_recomputed':
   return ast.Call(func=ast.Name(id='full_numeric_head_verified',ctx=ast.Load()),args=[n.value],keywords=[])
  return self.generic_visit(n)
tree=Bind().visit(ast.parse(source))
for n in ast.walk(tree):
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROFILES' for t in n.targets):
  n.value=ast.parse(repr({MID:dict(total=704,source_guard='all_own_source_actual_NPU_SHA256_verified',unused_guard='all_original_active_and_unused_source_bytes_verified')}),mode='eval').body
source=ast.unparse(ast.fix_missing_locations(tree))+'\n'
helpers='''
def full_numeric_head_verified(proof):
 checks=proof.get('numeric_sample_checks',[])
 return len(checks)==proof['sample_count'] and {x['index'] for x in checks}==set(range(proof['sample_count'])) and sum(x['full_head_rows'] for x in checks)==proof['actual_forward_count'] and all(x['full_head_rows']>0 and len(x['head_reference_checks'])==x['full_head_rows'] and x['all40_native_cache_values_reconstructed'] is True and x['original_input_pixels_token_ids_checked'] is True and x['all_masks_3D_positions_and_own_visual_injection_checked'] is True for x in checks)

def check_original_external_hash(path, expected):
 actual=sha(path)
 if actual==expected:return
 ledger=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
 assert str(path)==str(ledger) and expected=='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5' and actual=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32', (path,'External evidence changed outside exact user-authorized cleanup transition')
 receipt=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119/actual_completion.json'
 assert sha(receipt)=='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3'
 value=read(receipt);assert value['actual_returncode']==0 and value['all_protected_SHA256_unchanged'] is True and value['removed_files']==4972 and value['manifest_SHA256']=='f955add85f6012e7b1f8f4ad5fc0b78af043b80d2d0797315c21d59b35478382'
 assert read(ledger)['events'][-1]['receipt_SHA256']==sha(receipt)
'''
source=source.replace("assert not H.exists() and (not runner_lock_owners())",helpers+"\nassert not H.exists() and (not runner_lock_owners())",1)
old="""assert path not in all_external or all_external[path] == h
                all_external[path] = h"""
new="""check_original_external_hash(path,h)
                if path in all_external and all_external[path]!=h:
                    check_original_external_hash(path,all_external[path])
                all_external[path] = sha(path)"""
assert old in source;source=source.replace(old,new,1)
# Require the frozen original numerical verifier itself, all 704 actual sources,
# full per-head and cache checks. The model's partial/failed run is never accepted.
marker="proof = read(run['proof'])\n        assert proof['passed']"
assert marker in source
source=source.replace(marker,"proof = read(run['proof'])\n        assert proof['independent_verifier_SHA256']=='22d2919d3bca51c68a6f3de45141ed55999d781aefccb9b22c18affc79e61a1d' and proof['source_tensor_count']==704 and proof['original_unused_MTP_source_tensor_count']==0 and proof['own_original_BF16_preserved'] and full_numeric_head_verified(proof)\n        assert proof['passed']",1)
ast.parse(source)
assert all(s not in source for s in ['G572','2130','2076','all54_unused','prithiv_capacity_review_passed','Prithiv Gemma4E4B'])
# Every original production writer call remains byte-for-byte identical AST.
calls=lambda s:[ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write_verified_A100_G']
assert calls(source)==calls((P/'publish_PrithivGemma4E4B_verified180_single_original_lock_dual_transaction_v1_20261117.py').read_text())
name='publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py'
(P/name).write_text(source,encoding='utf8',newline='\n')
digest=hashlib.sha256(source.encode()).hexdigest()
review=dict(source=name,SHA256=digest,staged_only=True,frozen_G_writer_calls_AST_identical=True,own704_and_original_NF_full_head_KV_source_guards=True,only_legal_retention_ledger_transition_explicitly_bound=True,counts_before=dict(done=180,within=165,outside=15,pending=210,missing=310,qualified=66),counts_after_if_and_only_if_full_NF_and_original_qualification_and_threeD5pct=dict(done=181,within=166,outside=15,pending=209,missing=310,qualified=67))
(P/'Huihui_verified181_publisher_scope_review_20261119.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(review,ensure_ascii=False))
