import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,copy,json,hashlib,re,base64
P=Path(__file__).parent
MID='deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit'
oldmid='coder3101/gemma-4-E4B-it-qat-q4_0-unquantized-heretic'
s=(P/'publish_CoderGemma4E4B_verified176_single_lock_dual_transaction_v2_exact_loaded_source_reference_20261117.py').read_text(encoding='utf8')
# No alias is needed: this model's accepted real3 and formal job both bind the
# unchanged verified176 book. All original external files must match directly.
tree=ast.parse(s)
class RemoveOnlyEarlierModelAlias(ast.NodeTransformer):
 def visit_If(self,n):
  if ast.unparse(n.test).startswith('p is t and path == str(book)'):return []
  return self.generic_visit(n)
tree=RemoveOnlyEarlierModelAlias().visit(tree)
s=ast.unparse(ast.fix_missing_locations(tree))
start="alias = read(H / 'original_real3_immutable_equal_reference_review.json')"
i=s.index(start);e=s.index('\n    local = ',i)
s=s[:i]+"alias = read(H / 'own_original_real3_full_source_and_current_reference_review.json')\n    assert alias['passed'] and alias['model_id'] == "+repr(MID)+" and alias['original_true3_proof_SHA256'] == '13b0835212cc55ac129236c325e3db472c16366ccc997c5a9cc4c9edb9f94b2a' and alias['all_original_true3_external_SHA_match_fresh_files'] and alias['original_current_reference_SHA256'] == '6d73108e181887c9dfd95b7c851981c1ac316238705fe37e8c5bf59c15dc9a82'"+s[e:]
old="assert reference_alias_review and reference_alias_review['passed']"
assert old in s
s=s.replace(old,"reference_alias_review = dict(passed=True, model_id=mid, original_true3_proof=a['true3_proof'], original_true3_proof_SHA256=a['true3_proof_SHA256'], original_current_reference_SHA256=Q['expected_before_sha256'], all_original_true3_external_SHA_match_fresh_files=True, no_reference_alias_used_or_needed=True, all_other_external_SHA_mandatory=True)\n    assert reference_alias_review['passed']")
s=s.replace("'original_real3_immutable_equal_reference_review.json'","'own_original_real3_full_source_and_current_reference_review.json'")
mapping={
oldmid:MID,
'2f05d90e41b69081eb347bf8d40d0761b5988504a660e4846cefe905eb3001cd':'6d73108e181887c9dfd95b7c851981c1ac316238705fe37e8c5bf59c15dc9a82',
'ad249a0860a84299dc773dfc4ff2d5e5a202cd64f1b2370cb9653bbca8ec8766':'3825072ab23d52237acf0f4db96fd4f7101128cebe21274205d1bb31019f5c26',
'6a7b87eb520264e6733757a103c38c8a6e0042bc2de9a0b205860fa5f8a6f9cc':'e29a7c2725574f522ea0b1cdcc69404906a5932ff25b7c8bd7e39533ce930598',
'RedHatGemma4E2B_original_HF_BF16_verified175_blank_G_publication_20261116/verified175_export_complete_metadata.json':'CoderGemma4E4B_original_HF_BF16_verified176_blank_G_publication_20261117/verified176_export_complete_metadata.json',
'RedHatGemma4E2B_original_HF_BF16_verified175_blank_G_publication_20261116/active175_full_report_stat_baseline.json':'CoderGemma4E4B_original_HF_BF16_verified176_blank_G_publication_20261117/active176_full_report_stat_baseline.json',
'all2076_own_source_actual_NPU_SHA256_verified':'all1951_own_source_actual_NPU_SHA256_verified',
}
class BindNewOnly(ast.NodeTransformer):
 def visit_keyword(self,n):
  self.generic_visit(n)
  if n.arg:
   if n.arg=='next_new177_count_branch_review_mandatory':n.arg='next_new178_count_branch_review_mandatory'
   else:n.arg=re.sub(r'(?<!\d)(175|176)(?!\d)',lambda m:{'175':'176','176':'177'}[m.group()],n.arg)
  return n
 def visit_Constant(self,n):
  if isinstance(n.value,str):
   if n.value in mapping:n.value=mapping[n.value];return n
   if re.fullmatch('[0-9a-f]{64}',n.value):return n
   value=n.value
   value=value.replace('CoderGemma4E4B','DeadbydawnGemma4E2B').replace('Coder Gemma4E4B','Deadbydawn Gemma4E2B').replace('Coder Gemma4 E4B','Deadbydawn Gemma4 E2B')
   value=re.sub(r'(?<!\d)(175|176)(?!\d)',lambda m:{'175':'176','176':'177'}[m.group()],value)
   value=value.replace('G212','G239')
   n.value=value
  elif type(n.value) is int:n.value={175:176,176:177,161:162,214:213,62:63,533:532,524:523,2076:2011}.get(n.value,n.value)
  return n
tree=BindNewOnly().visit(ast.parse(s));s=ast.unparse(ast.fix_missing_locations(tree))+'\n'
s=s.replace("assert t[profile['source_guard']]", "assert t['all2011_own_source_SHA_verified_with60_original_unused_sharedKV'] and t[profile['source_guard']]")
s=s.replace("assert proof[profile['source_guard']]", "assert proof['all2011_own_source_SHA_verified_with60_original_unused_sharedKV'] and proof[profile['source_guard']]")
s=s.replace('all_own_source_values_loaded_to_NPU_SHA256_verified=True','all1951_active_NPU_and60_original_unused_stored_source_SHA256_verified=True, active_source_tensor_count=1951, original_unused_sharedKV_source_count=60')
assert oldmid not in s and 'G212' not in s and 'RedHatGemma4E2B' not in s and 'original_reference_SHA256' not in s
assert "no_reference_alias_used_or_needed=True" in s and "all_original_true3_external_SHA_match_fresh_files=True" in s
ast.parse(s)
finish=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='finish_local_and_full_target_audit')
local_receipt_keys={ast.literal_eval(n.slice) for n in ast.walk(finish) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='Q'}
assert local_receipt_keys=={'local_receipt_base64','local_receipt_SHA256','after_book_SHA256'},local_receipt_keys
pub='scripts/publish_DeadbydawnGemma4E2B_verified177_single_original_lock_dual_transaction_v2_finish_reference_scope_20261117.py';(P/Path(pub).name).write_bytes(s.encode())
reader=(P/'read_CoderE4_v2_completed_full_evidence_for_exact_publication_20261117_remote.py').read_text(encoding='utf8').replace('CoderGemma4E4B_completed_true3_capacity_recovered_approved_fresh3_fixed200_v2_static_current_binding_real_FLOCK_20261117','DeadbydawnGemma4E2B_completed_true3_419_capacity_recovered_approved_fresh3_fixed200_v3_real_FLOCK_20261117').replace('CoderGemma4E4B_completed_true3_six_retention_capacity_recovery_v2_static_current_binding_20261117','DeadbydawnGemma4E2B_completed_true3_419_exact_retention_capacity_recovered_approved_fresh3_fixed200_v3_continuation_20261117')
ast.parse(reader);(P/'read_Deadbydawn_v3_completed_full_evidence_for_exact_publication_20261117_remote.py').write_bytes(reader.encode())
payload=base64.b64encode(s.encode()).decode();digest=hashlib.sha256(s.encode()).hexdigest()
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
R=Path('/root/projects/comperror');n=__NAME__;raw=base64.b64decode(__DATA__);assert hashlib.sha256(raw).hexdigest()==__SHA__;ast.parse(raw);p=R/n
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(source=str(p),SHA256=__SHA__,staged_only_not_executed_no_canonical_write=True)))
'''.replace('__NAME__',repr(pub)).replace('__DATA__',repr(payload)).replace('__SHA__',repr(digest))
ast.parse(remote);(P/'stage_Deadbydawn_verified177_dual_publication_primary_v2_not_execute_20261117_remote.py').write_bytes(remote.encode())
print(json.dumps(dict(publication_primary=pub,SHA256=digest,only_source_generated_not_published=True)))
