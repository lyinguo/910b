"""Stage the audited one-G transaction for this real GLM job; never execute it here."""
from pathlib import Path
import ast,copy,hashlib,json,re
P=Path(__file__).resolve().parent
base=P/'publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_next182_metadata_20261119.py';raw=base.read_bytes();assert hashlib.sha256(raw).hexdigest()=='be06184a15a41354a18b164d99dad2f76359578a0160ea73f4060cb51d3299c7'
MID='zai-org/GLM-4.6V-Flash'
cap_source='''def glm46_capacity_review_passed(cap):
 bound=cap.get('all_complete_original_values_still_available_and_SHA_bound',{})
 return cap.get('passed') is True and cap.get('capacity_passed') is True and cap.get('empirical_estimate_bytes')==130401389967 and cap.get('actual_free_bytes',0)>=130401389967 and cap.get('original_factor')==4 and cap.get('original_extra_reserve_bytes')==6*1024**3 and cap.get('all_original_hard3GiB_log1GiB_precheckHalfGiB_per_sample2GiB_times_outer10_guards_preserved') is True and cap.get('all200_original_shapes_used_not_only_200_div3') is True and cap.get('no_tensor_value_or_occurrence_or_model_parameter_changed') is True and isinstance(bound,dict) and len(bound)==8 and len([n for n in bound if n.endswith('.npz')])==3 and bound.get(cap.get('original_true3_proof'))=='d089c0ded120ee0df0be6ad28d9a02fe0b61697330d6f3b4a3178bc667916950' and cap.get('existing_standard_LZMA_storage_review_SHA256')=='a0ec96ab83110aef79d6c5327fb86e8ed05ebf80656dc61271afd8e8bf9e88dc' and len(cap.get('all_complete_ZIP_parts_individually_classified',[]))==2569 and len(cap.get('all_original200_prompt_lengths',[]))==len(cap.get('all_original200_pixel_elements',[]))==200 and cap.get('original_true3_proof_SHA256')=='d089c0ded120ee0df0be6ad28d9a02fe0b61697330d6f3b4a3178bc667916950'
'''
ledger_source='''def check_original_external_hash(path, expected):
 actual=sha(path)
 if actual==expected:return
 ledger=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
 assert str(path)==str(ledger) and expected in ['4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5','5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'] and actual=='7978cbde39e32b7d28ecb05cf69148e451cbe262493b39d6b07e6fdb785b9b71', (path,'External change outside exact two user-authorized retention events')
 first=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119/actual_completion.json'
 second=B/'Huihui_completed200_full_numerical_baselinegap_target_audit_and_correct_nonexample_retention_20261119/retention_actual_completion.json'
 assert sha(first)=='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3' and sha(second)=='25e0f1b5767ec73d748cf3d1fac44556a9b4d3bc34a2a1a307649c69c8f8de91'
 a,b=read(first),read(second);assert a['actual_returncode']==b['actual_returncode']==0 and a['all_protected_SHA256_unchanged'] and b['all_protected_SHA256_unchanged'] and a['removed_files']==4972 and b['removed_files']==105 and b['all61_incorrect_debug_control_cases_and_full_original_examples_retained'] and b['canonical_scores_state_registry_binding_and_both_workbooks_unchanged']
 assert sha(second.parent/'ledger_before_selective_retention.original')=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32' and read(ledger)['events'][-2]['receipt_SHA256']==sha(first) and read(ledger)['events'][-1]['receipt_SHA256']==sha(second)
 assert a['manifest_SHA256']=='f955add85f6012e7b1f8f4ad5fc0b78af043b80d2d0797315c21d59b35478382' and b['manifest_SHA256']=='f957592949489091fb0173800a8fa4a9525938bfdfc2800119f008fdb7acdddc'
'''
def strings(s):
 return s.replace('huihui-ai/Huihui-GLM-4.6V-Flash-abliterated',MID).replace('HuihuiGLM46Flash','ZaiGLM46Flash').replace('Huihui GLM4.6V','zai GLM4.6V').replace('BF16_2048_lossless_verified181','BF16_thinking4096_LZMA_verified181').replace('BF16/card4/2048/existing thinkingFalse','BF16/card4/4096/original thinkingTrue').replace('7bbc04ae6b05c7845ca01dc3e682f94da39000230beea45f6fbf5dc6f773cc59','d089c0ded120ee0df0be6ad28d9a02fe0b61697330d6f3b4a3178bc667916950').replace('G355','G697').replace('B355','B697').replace('D355','D697').replace('all_original_true3_external_SHA_match_fresh_files','all_original_true3_external_SHA_exact_or_exact_authorized_retention_transition')
class Bind(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str):n.value=strings(n.value)
  elif type(n.value)is int and n.value==355:n.value=697
  return n
 def visit_Name(self,n):
  if n.id=='huihui_capacity_review_passed':n.id='glm46_capacity_review_passed'
  return n
 def visit_keyword(self,n):
  self.generic_visit(n)
  if n.arg:n.arg=n.arg.replace('all_original_true3_external_SHA_match_fresh_files','all_original_true3_external_SHA_exact_or_exact_authorized_retention_transition')
  return n
 def visit_FunctionDef(self,n):
  if n.name=='huihui_capacity_review_passed':return ast.parse(cap_source).body[0]
  if n.name=='check_original_external_hash':return ast.parse(ledger_source).body[0]
  return self.generic_visit(n)
tree=Bind().visit(ast.parse(raw));source=ast.unparse(ast.fix_missing_locations(tree))+'\n'
marker="and glm46_capacity_review_passed(read(a['capacity_review']))"
assert marker in source
end=source.index('\n',source.index(marker))
source=source[:end]+"\n        for capacity_path,capacity_hash in read(a['capacity_review'])['all_complete_original_values_still_available_and_SHA_bound'].items():\n            check_original_external_hash(capacity_path,capacity_hash)"+source[end:]
ast.parse(source)
assert 'G355' not in source and 'huihui-ai/' not in source and 'thinkingFalse' not in source and 'be06184a' not in source
calls=lambda s:[ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write_verified_A100_G']
assert calls(source)==calls(raw)
helpers=ast.Module(body=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ['full_numeric_head_verified','glm46_capacity_review_passed']],type_ignores=[]);ns={};exec(compile(helpers,'owned_GLM_publication_scope','exec'),ns)
current=json.loads((P/'R_user_continue_GLM_serial_actual1_20261119.json').read_bytes());cap=current['states']['capacity'];pf=json.loads((P/'R_GLM46_serial_v2_full_NF_complete_logs_trace3_20261119.json').read_bytes())['phases']['fresh3']['verification_stdout.json'];assert ns['full_numeric_head_verified'](pf) and ns['glm46_capacity_review_passed'](cap)
cases=2
for name,value in [('original_factor',3),('original_extra_reserve_bytes',5*1024**3),('empirical_estimate_bytes',130401389966),('capacity_passed',False),('all_original_hard3GiB_log1GiB_precheckHalfGiB_per_sample2GiB_times_outer10_guards_preserved',False)]:
 bad=copy.deepcopy(cap);bad[name]=value;assert not ns['glm46_capacity_review_passed'](bad);cases+=1
for name in ['all_complete_ZIP_parts_individually_classified','all_original200_prompt_lengths','all_original200_pixel_elements']:
 bad=copy.deepcopy(cap);bad[name]=bad[name][:-1];assert not ns['glm46_capacity_review_passed'](bad);cases+=1
bad=copy.deepcopy(pf);bad['numeric_sample_checks'][0]['full_head_rows']+=1;assert not ns['full_numeric_head_verified'](bad);cases+=1
bad=copy.deepcopy(pf);bad['numeric_sample_checks'][0]['all40_native_cache_values_reconstructed']=False;assert not ns['full_numeric_head_verified'](bad);cases+=1
bad=copy.deepcopy(cap);bad['all_complete_original_values_still_available_and_SHA_bound']=True;assert not ns['glm46_capacity_review_passed'](bad);cases+=1
name='publish_ZaiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py';out=source.encode();(P/name).write_bytes(out);digest=hashlib.sha256(out).hexdigest()
# Windows and pure readback consumers keep their actual source paths and hashes.
names=['prepare_Huihui_verified181_local_baseline_preview_no_predicted_score_20261119.py','prepare_Huihui_verified181_completed200_actual_score_local_plan_20261119.py','apply_Huihui_verified181_exact_server_bytes_local_CAS_20261119.py','build_Huihui_verified181_exact_local_receipt_upload_20261119.py','build_launch_Huihui_verified181_actual_reviewed_publication_20261119.py','read_Huihui_verified181_server_book_for_local_CAS_20261119_remote.py','run_Huihui_verified181_reviewed_dual_publication_actual_20261119.py','render_Huihui_verified181_actual_published_20261119.mjs','read_Huihui_completed200_exact_publication_inputs_no_live_assumptions_20261119_remote.py','huihui_glm46_workbook_cell_verified181_20261119.mjs']
made={name:digest}
for old in names:
 s=(P/old).read_text(encoding='utf8');new=old.replace('Huihui','ZaiGLM46').replace('huihui_glm46','zai_glm46');s=strings(s).replace('Huihui_verified','ZaiGLM46_verified').replace('read_Huihui','read_ZaiGLM46').replace('run_Huihui','run_ZaiGLM46').replace('launch_Huihui','launch_ZaiGLM46').replace('R_Huihui_','R_ZaiGLM46_').replace('huihui_glm46_workbook','zai_glm46_workbook').replace('before_355','before_697').replace('after_355','after_697').replace('[353]','[695]').replace('B353:G357','B695:G699').replace("Decimal('.945')","Decimal('.965')").replace('be06184a15a41354a18b164d99dad2f76359578a0160ea73f4060cb51d3299c7',digest).replace('publish_ZaiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_next182_metadata_20261119.py',name)
 s=s.replace('HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119','GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119').replace('HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v4_approval_schema_20261119','GLM46_own_passed3_after_Huihui_full_NF_baselinegap_verified_correct_nonexample_cleanup_serial_v2_20261119').replace('a1bf90557a326523990a6004396e2dbe336e85d72139b216fe00f5337340c0fd','18c9d24f9992311ea3261e82cc56c3121db03ccd0655e21aaafc4d2b5ee4f323').replace('/proc/1565741','/proc/1573127').replace('continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v4_approval_schema_20261119.py','continue_GLM46_own_passed3_after_Huihui_numerical_gap_verified_cleanup_serial_v2_20261119.py')
 if new.endswith('.py'):
  class CellOnly(ast.NodeTransformer):
   def visit_Constant(self,n):
    if type(n.value)is int and n.value==355:n.value=697
    return n
  s=ast.unparse(ast.fix_missing_locations(CellOnly().visit(ast.parse(s))))+'\n'
  assert all(n not in s for n in ['G355','B355','D355','B353:G357','before_355','after_355'])
 (P/new).write_text(s,encoding='utf8',newline='\n');made[new]=hashlib.sha256(s.encode()).hexdigest()
review=dict(passed=True,scope_checks=cases,full_head_and_704_source_positive_incomplete_or_corrupt_refusals=True,complete2569_parts_and_all200_shapes_positive_reduced_guards_refused=True,original_frozen_Gwriter_calls_AST_identical=True,own_real_true3_d089_and_exact_4972_plus105_retention_chain=True,target_model=MID,target_cell='G697',current_three_D='.9650/.9650/.965',source_SHA256=made,only_staged_no_production_publication=True,no_forecasted_full200_score=True)
(P/'ZaiGLM46_verified181_publication_sources_scope_review_20261119.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(review,ensure_ascii=False))
