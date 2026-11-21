"""Prepare a narrow publication guard and reviewed transaction for genuine Ovis200/NF2."""
from pathlib import Path
import ast, copy, hashlib, json, difflib
V=Path(__file__).resolve().parent
sha=lambda x:hashlib.sha256(x).hexdigest()
MID='AIDC-AI/Ovis-U1-3B'
OLD='AutoGLMPhone9BSingle_original_BF16_2048_LZMA_verified183_blank_G_publication_20261120'
PRIOR='AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120'
PUB='OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121'
LOCAL='OvisU1_local_result_sync_verified184_20261121'
WRITER='verified_OvisU1_genuine200_formal_NF2_original_raw_fallback_G_writer_20261121'
PUBLISHER='publish_OvisU1_genuine200_fullNF2_verified184_single_original_lock_dual_transaction_20261121.py'
J='OvisU1_completed_storage_fullNF_passed_exact23_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261121'
NF='OvisU1_completed_genuine200_formal_NF2_original_raw_fallback_metadata_after_Meta_serial_no_NN_replay_20261121'
TRUE3='221e4c3060bfce23733937f1679a7f89b0e58f0be148650738b99b0da016f39d'
FRESH3='a3422cc1cf1b45002e910374e655d832941702741a9cfe7ec9064c6d987f7443'
BEFORE='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81'
NFHASH='60c0d21fd51f9ad8e712890330e9b7c887d72e8154ace6a14b11ab5f6fc20052'
header='''"""Authorize only the original failed Ovis NN200 plus its actual complete NF2 acceptance.

The original job remains failed. The original raw one-G ZIP write function is
byte-for-byte identical to the frozen writer; only this explicit authorization
knows the separately completed formal acceptance, with zero new NN forwards.
"""
import pathlib,json,copy,hashlib,io,math,os,re,tempfile,zipfile,decimal
from batch_910b_support import workbook_rows,digest
BOOK=pathlib.Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
R=pathlib.Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
MID='AIDC-AI/Ovis-U1-3B'
J=B/''' + repr(J) + '''
NF=B/''' + repr(NF) + '''
def read(p):return json.loads(pathlib.Path(p).read_bytes())

def full_numeric_head_verified(proof):
 checks=proof.get('numeric_sample_checks',[]);count=proof.get('sample_count')
 limits={'LM_all512_head_CPU_BF16_reference_relative_RMSE':.02,'visual_full_head_relative_RMSE':.02,'visual_softmax_relative_RMSE':.002,'visual_embedding_relative_RMSE':.02}
 return count in [3,200] and len(checks)==count and {x['index'] for x in checks}==set(range(count)) and proof.get('actual_forward_count')==count*512 and all(x.get('all28_KV_and_all_masks_positions_preserved') is True and x.get('CPU_references_diagnostic_only_not_grade') is True and all(math.isfinite(x[k]) and 0<=x[k]<lim for k,lim in limits.items()) for x in checks)

def check_formal_metadata(old,acceptance,completion,proof,fresh):
 assert old['phase']=='failed_evidence_preserved' and old['actual_returncode']==old['verifier_returncode']==1 and old['model_returncode']==0 and old['sample_count']==200
 assert [x['sample_count'] for x in old['accepted']]==[3] and old['accepted'][0]['proof_sha256']==''' + repr(FRESH3) + '''
 assert acceptance['passed'] is True and acceptance['actual_NEW_model_forwards']==0 and acceptance['original_job_remains_failed'] is True
 assert acceptance['original_model_returncode']==0 and acceptance['original_verifier_returncode']==1 and acceptance['current_verifier_returncode']==0
 assert acceptance['original_sample_count']==200 and acceptance['original_native_forwards']==102400 and acceptance['score']==.895
 assert acceptance['own_approved_NEWfresh3_proof_SHA256']==''' + repr(FRESH3) + '''
 assert acceptance['original_full_five_reports_unchanged'] and acceptance['all_own1060_source_and_all200_full_numeric_head_KV_inputs_raw_ref_error_denominators_verified'] and acceptance['only_original_raw_prediction_fallback_expected_metadata_corrected']
 assert acceptance['original_protocol_qualified'] is True and acceptance['no_G_or_canonical_write'] is True
 assert completion['actual_returncode']==0 and completion['actual_NEW_model_forwards']==0 and completion['no_G_or_canonical_write']
 for p,count in [(fresh,3),(proof,200)]:
  assert p['passed'] and p['sample_count']==count and p['actual_NPU_inference'] and p['NEW_real_NPU_inference'] and not p['historical_record_reuse']
  assert p['source_tensor_count']==1060 and p['active_physical_source_tensor_count']==487 and p['original_unexecuted_generator_source_tensor_count']==573
  assert p['all_own_source_actual_NPU_SHA256_verified'] and p['all_native_KV_values_losslessly_reconstructed'] and p['all_full_inputs_tokens_raw_references_errors_preserved'] and full_numeric_head_verified(p)
  assert p['error_indices']==p['empty_indices']==[] and p['truncated_indices']==list(range(count)) and not p['A100_historical_engine_or_tensor_capture_claimed']
 assert proof['metrics']['Accuracy']==.895 and proof['invalid_indices']==[198,199] and proof['formal_alignment_within_5pct'] and proof['original_protocol_final_qualification_complete']
 assert set(proof['relative_error_to_current_A100_D'])=={'server','local','preserved_original'} and all(decimal.Decimal(x)<=decimal.Decimal('.05') for x in proof['relative_error_to_current_A100_D'].values())
 assert proof['independent_verifier_SHA256']==''' + repr(NFHASH) + '''
 return True

def verified_evidence(manifest):
 assert pathlib.Path(manifest['job_completion']).resolve()==J/'completion.json' and pathlib.Path(manifest['formal_acceptance']).resolve()==NF/'formal_acceptance.json'
 assert digest(J/'job.json')=='11f0c684e4dfe6f39bff484c2386dca3e7371f778230faa6ca2a65ca64affc7d'
 job=read(J/'job.json');old=read(J/'completion.json');a=read(NF/'formal_acceptance.json');c=read(NF/'actual_completion.json')
 assert digest(NF/'formal_acceptance.json')==manifest['formal_acceptance_SHA256']==c['formal_acceptance_SHA256']
 assert c['formal_acceptance']==str(NF/'formal_acceptance.json') and a['original_genuine_NEW_NPU200_job']==str(J) and a['original_job_SHA256']==digest(J/'job.json')
 assert manifest['proof']==a['proof']==str(NF/'verification_stdout.json') and manifest['proof_sha256']==a['proof_SHA256']==digest(manifest['proof'])
 p=read(manifest['proof']);f=old['accepted'][0];assert digest(f['proof'])==f['proof_sha256']==''' + repr(FRESH3) + ''';fresh=read(f['proof'])
 assert manifest['report_dir']==old['report_dir']==p['report_dir'] and pathlib.Path(p['report_dir']).name=='20261121_093507'
 check_formal_metadata(old,a,c,p,fresh)
 assert job['stages']==[3,200] and job['model_id']==MID and job['spec']['approved'] and not job['write_workbook']
 for n,h in job['spec']['source_sha256'].items():
  assert not pathlib.Path(n).is_absolute() and '..' not in pathlib.Path(n).parts and digest(R/n)==h
 assert digest(R/'scripts/verify_ovisu1_completed200_original_raw_prediction_fallback_metadata_v2_20261121.py')==p['independent_verifier_SHA256']
 for proof in [fresh,p]:
  folder=pathlib.Path(proof['report_dir']);files={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
  assert folder.resolve().is_relative_to(R/'workspace/reports') and {x.name for x in folder.iterdir()}==set(proof['report_sha256'])==files
  for n,h in proof['report_sha256'].items():assert digest(folder/n)==h
 r=read(pathlib.Path(p['report_dir'])/'eval_report.json');cfg=r['pipeline_config'];assert cfg['model']['name']==MID and cfg['task_type']==manifest['task']=='image-text-to-text' and cfg['sample_count']==200 and cfg['sample_range']=='0:199' and cfg['algorithm']['batch_size']==4
 assert r['metrics']['Accuracy']==p['metrics']['Accuracy']==manifest['new_score']==.895 and r['statistics']['total_samples']==200 and r['statistics']['error_count']==0
 return job,old,a,c,p,fresh

def authorize(book,model_id,score,decision_path,expected_book_hash,manifest_path):
 d=read(decision_path);m=read(manifest_path);book=pathlib.Path(book).resolve()
 if d.get('test_fixture_only'):
  assert d['origin']=='isolated_test_fixture' and book.is_relative_to(pathlib.Path(d['test_root']).resolve()) and book!=BOOK
 else:
  assert book==BOOK and d['origin']=='actual_user_chat_instruction' and d['overwrite_existing_G_when_verified'] and d['reask_per_model_not_required'] and d['required_sample_count']==200 and d['required_real_smoke_count']==3
 assert expected_book_hash==digest(book),'Concurrent workbook change'
 assert model_id==m['model_id']==MID and float(m['new_score'])==float(score)==.895 and m['new_real_NPU_inference'] and not m['historical_record_reuse']
 verified_evidence(m)
 if not d.get('test_fixture_only'):
  from local_A100_reference_gate_20261103 import check_local_A100_reference
  m['local_A100_full_precision_gate']=check_local_A100_reference(model_id,m['task'],score)
 return m

'''
frozen=(V/'frozen_verified_A100_G_cell_writer_20261103.py').read_text(encoding='utf8')
assert sha((V/'frozen_verified_A100_G_cell_writer_20261103.py').read_bytes())=='83134eec524c27bc4e9fd4569193e72038a7b248b379c12cf6999e1dcff798b8'
start=frozen.index('def write_verified_A100_G(')
rawfn=frozen[start:]
writer=header+rawfn
(V/(WRITER+'.py')).write_text(writer,encoding='utf8',newline='\n')
assert writer[writer.index('def write_verified_A100_G('):]==rawfn

parent=V/'publish_AutoGLMPhone9BSingle_verified183_single_original_lock_dual_transaction_v1_20261120.py'
assert sha(parent.read_bytes())=='504a20985debd6d956a37451e373683819677e86b5dc2aa56963b0b439842755'
changes={'0b2fcf21fe5a56e5b33111c4e65c2ec0ddf53dae43d451dec756a342df810340':BEFORE,'4718832822525fdd4516fd50b8f45d5863c21974b9b11bd1ab30608f44c53605':'c1c867fbfdf2637c03757612c2c49a43fefefa5be6886b30d979198db6bcf0cb','3448fd5d8eb8f7d93a1b48ab7c7e3720608dd3f937388fc408c6d9c555e21c4d':'218c75abd8d84caced7308f3a521504eb6b76682eb2a655d8a068ee63ec59fb7','da99ef8ddab138186eb4368204a6917041f126d3fb03730fe8163352a211c259':TRUE3}
def string(s):
 if s in changes:return changes[s]
 if s.startswith(PRIOR+'/'):return OLD+'/'+s.split('/',1)[1].replace('182','183')
 return s.replace(OLD,PUB).replace('zai-org/AutoGLM-Phone-9B',MID).replace('old182','old183').replace('prior182','prior183').replace('prior_182','prior_183').replace('G692','G144').replace('verified183','verified184').replace('active183_full_report_stat_baseline','active184_full_report_stat_baseline').replace('next_new182','next_new185').replace('own AutoGLM-Phone-9B HF publication','own Ovis-U1 BF16 EOSNone512 genuine200/formal NF2 publication')
class Adapt(ast.NodeTransformer):
 def visit_Constant(self,n):
  v=n.value
  if type(v)==int:v={182:183,183:184,168:169,207:206,69:70,526:525,517:516,692:144}.get(v,v)
  elif isinstance(v,str):v=string(v)
  return ast.copy_location(ast.Constant(v),n)
 def visit_keyword(self,n):
  n=self.generic_visit(n)
  if n.arg:n.arg=n.arg.replace('old182','old183').replace('prior182','prior183').replace('prior_182','prior_183').replace('verified183_export','verified184_export').replace('next_new182','next_new185')
  return n
oldtree=ast.parse(parent.read_bytes());tree=Adapt().visit(copy.deepcopy(oldtree));ast.fix_missing_locations(tree)
capacity='''def glm46_capacity_review_passed(cap):
 return cap.get('passed') is True and cap.get('empirical_estimate_bytes')==160797505832 and cap.get('actual_free_bytes',0)>=160797505832 and cap.get('original_factor4_extra6GiB_and_all_runtime_guards_preserved') is True and cap.get('own_original512_and_EOSNone_preserved') is True and cap.get('all200_own_original_shapes_checked') is True and len(cap.get('all200_projected_complete_archives',[]))==200 and cap.get('all_complete_existing_standard_LZMA_values_full_NF_accepted') is True and cap.get('formal_storage_acceptance_SHA256')=='c93806c03194e5137a6c5f76f52109ad27aace0f05a0fa9527e4f6f3fde922bc'
'''
external="def check_original_external_hash(path, expected):\n    assert sha(path) == expected, (path, 'Own Ovis fresh3/formal200 external evidence changed')\n"
tree.body=[ast.parse(capacity).body[0] if isinstance(n,ast.FunctionDef) and n.name=='glm46_capacity_review_passed' else ast.parse(external).body[0] if isinstance(n,ast.FunctionDef) and n.name=='check_original_external_hash' else n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='full_numeric_head_verified')]
pub=ast.unparse(tree)+'\n'
pub=pub.replace('from verified_A100_G_cell_writer_20261103 import write_verified_A100_G','from '+WRITER+' import write_verified_A100_G, verified_evidence, full_numeric_head_verified')
pub=pub.replace('Authorized own AutoGLM-Phone-9B original BF16 publication','Authorized own Ovis-U1 original BF16 EOSNone512 publication')
pub=pub.replace("PROFILES = {'AIDC-AI/Ovis-U1-3B': {'total': 704, 'source_guard': 'all_own_source_actual_NPU_SHA256_verified', 'unused_guard': 'all_original_active_and_unused_source_bytes_verified'}}","PROFILES = {'AIDC-AI/Ovis-U1-3B': {'total':1060}}")
pub=pub.replace("for capacity_path, capacity_hash in read(a['capacity_review'])['all_complete_original_values_still_available_and_SHA_bound'].items():", "for capacity_path, capacity_hash in a['storage_full_NF_acceptance']['all_complete_existing_new_storage_archives_SHA256'].items():")
pub=pub.replace("'_verified184_original_writer_isolated_20261119.py'", "'_verified184_original_writer_isolated_20261121.py'")
a=pub.index('        t = read(a[\'true3_proof\'])');b=pub.index("        score = validate_report",a)
pub=pub[:a]+'''        t=read(a['true3_proof'])
        storage=a['storage_full_NF_acceptance']
        assert storage['passed'] and storage['same_original_genuine3_no_model_replay'] and storage['all1060_own_original_source_and_full1536_steps_numerically_verified'] and storage['original_genuine3_proof_SHA256']==a['true3_proof_SHA256']
        assert sha(storage['complete_storage_full_NF_proof'])==storage['complete_storage_full_NF_proof_SHA256']=='9c98b387b233037220b1e43fa666f57c6de57be85da5da984a79bdde7604e97f'
        assert sha(read(a['capacity_review'])['formal_storage_acceptance'])=='c93806c03194e5137a6c5f76f52109ad27aace0f05a0fa9527e4f6f3fde922bc'
        old_done=read(J/'completion.json')
        assert sha(J/'completion.json')=='e1395963e8ef3861025886313cf18e0dc44d3f3a772806e1a215ff37609f3c4d'
        original_job, original_done, acceptance, actual_acceptance, proof, fresh=verified_evidence(dict(job_completion=str(J/'completion.json'),formal_acceptance=expected['formal_acceptance'],formal_acceptance_SHA256=expected['formal_acceptance_SHA256'],proof=c['formal_proof'],proof_sha256=c['formal_proof_sha256'],report_dir=c['report_dir'],task=c['task'],new_score=c['new_value']))
        assert original_job==job and original_done==old_done and len(old_done['accepted'])==1 and old_done['accepted'][0]['score']==expected['fresh3_score']
        run=dict(sample_count=200,report_dir=proof['report_dir'],proof=c['formal_proof'],proof_sha256=c['formal_proof_sha256'],score=proof['metrics']['Accuracy'],original_NPU_job_completion=str(J/'completion.json'),formal_acceptance=expected['formal_acceptance'])
        done=dict(accepted=old_done['accepted']+[run],completed_at=actual_acceptance['UTC'],publication_evidence_aggregation_not_a_job_completion=True,original_job_remains_failed=True,actual_NEW_model_forwards_in_formal_acceptance=0)
        assert proof['actual_forward_count']==expected['expected_forward_count']==102400 and proof['NPU_profiler_kernel_events']==expected['expected_kernels']
        assert proof['truncated_indices']==expected['expected_truncated']==list(range(200)) and proof['empty_indices']==expected['expected_empty']==[] and proof['invalid_indices']==expected['expected_invalid']==[198,199]
        spec['accepted_formal200_verifier']='scripts/verify_ovisu1_completed200_original_raw_prediction_fallback_metadata_v2_20261121.py'
        spec['source_sha256'][spec['accepted_formal200_verifier']]=proof['independent_verifier_SHA256']
        spec['formal_independent_acceptance']=expected['formal_acceptance'];spec['formal_independent_acceptance_SHA256']=expected['formal_acceptance_SHA256'];spec['original_genuine_NPU_job_remains_failed']=True
''' + pub[b:]
pub=pub.replace("for p in [t] + [read(x['proof']) for x in done['accepted']]:","for p in [read(x['proof']) for x in done['accepted']]:")
old="reference_alias_review = dict(passed=True, model_id=mid, original_true3_proof=a['true3_proof'], original_true3_proof_SHA256=a['true3_proof_SHA256'], original_current_reference_SHA256=Q['expected_before_sha256'], all_original_true3_external_SHA_exact_or_exact_authorized_retention_transition=True, no_reference_alias_used_or_needed=True, all_other_external_SHA_mandatory=True)"
new="""alias=storage['original_D_reference_alias_review']
    assert alias['passed'] and alias['all700_C_B_row_bindings_and_entire_D_column_unchanged'] and alias['all170_prior_G_values_preserved'] and alias['current183_book_SHA256']==Q['expected_before_sha256'] and sha(alias['old170_exact_book_backup'])==alias['old170_book_SHA256']=='195e9af1a5284c764c795516853256bca54e9880ddd7b388cfefaa9a3c31b496'
    oldrows,oldsheet,_=workbook_rows(alias['old170_exact_book_backup'])
    assert [(r['row'],r['model_id'],r['task']) for r in oldrows]==[(r['row'],r['model_id'],r['task']) for r in rows] and all(r['score']==next(x for x in rows if x['model_id']==r['model_id'])['score'] for r in oldrows if r['score'] is not None)
    with zipfile.ZipFile(alias['old170_exact_book_backup']) as oz,zipfile.ZipFile(book) as nz:
        ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        dc=lambda z,s:{c.get('r'):ET.tostring(c) for c in ET.fromstring(z.read(s)).findall('.//s:c',ns) if re.fullmatch('D[0-9]+',c.get('r',''))}
        assert dc(oz,oldsheet)==dc(nz,workbook_rows(book)[1])
    reference_alias_review = dict(passed=True,model_id=mid,original_true3_proof=a['true3_proof'],original_true3_proof_SHA256=a['true3_proof_SHA256'],original_current_reference_SHA256=Q['expected_before_sha256'],all_original_true3_external_SHA_exact_or_exact_authorized_retention_transition=True,original_reference_alias=alias,complete_storage_full_NF_proof=storage['complete_storage_full_NF_proof'],complete_storage_full_NF_proof_SHA256=storage['complete_storage_full_NF_proof_SHA256'],fresh3_and_formal200_all_current_external_SHA_exact=True,all_other_external_SHA_mandatory=True)"""
assert pub.count(old)==1;pub=pub.replace(old,new)
pub=pub.replace("proof_sha256=v['run']['proof_sha256'], new_real_NPU_inference=True", "proof_sha256=v['run']['proof_sha256'], formal_acceptance=Q['models'][mid]['formal_acceptance'],formal_acceptance_SHA256=Q['models'][mid]['formal_acceptance_SHA256'],new_real_NPU_inference=True")
pub=pub.replace("proof['qualification_basis']", "'Own BF16 EOSNone512 genuine200 fullNF2 formal acceptance; original job remains failed and no new model inference in acceptance'")
pub=pub.replace('all704_own_stored_source_and_actual_NPU_SHA256_verified=True, active_source_tensor_count=704, original_unused_MTP_source_tensor_count=0','all1060_own_stored_source_and_actual_NPU_SHA256_verified=True, active_source_tensor_count=487, original_unexecuted_generator_source_tensor_count=573, formal_acceptance=Q[\'models\'][v[\'change\'][\'model_id\']][\'formal_acceptance\'], formal_acceptance_SHA256=Q[\'models\'][v[\'change\'][\'model_id\']][\'formal_acceptance_SHA256\'], original_NPU_job_remains_failed=True, actual_NEW_model_forwards_in_formal_acceptance=0')
ast.parse(pub)
(V/PUBLISHER).write_text(pub,encoding='utf8',newline='\n')
(V/'OvisU1_NF2_verified184_publisher_complete_parent_diff_20261121.diff').write_text(''.join(difflib.unified_diff(parent.read_text(encoding='utf8').splitlines(keepends=True),pub.splitlines(keepends=True),fromfile=parent.name,tofile=PUBLISHER)),encoding='utf8')

template=(V/'verified_same_baseline_writer_11_cases_template_20261116.py').read_text(encoding='utf8').replace('from verified_A100_G_cell_writer_20261103 import write_verified_A100_G','from '+WRITER+' import write_verified_A100_G')
assert template.count('from '+WRITER)==1
(V/'OvisU1_NF2_verified184_writer_11_cases_template_20261121.py').write_text(template,encoding='utf8',newline='\n')
for old,new in [('apply_AutoGLMPhone9BSingle_verified183_exact_server_bytes_local_CAS_20261120.py','apply_OvisU1_verified184_exact_server_bytes_local_CAS_20261121.py'),('build_AutoGLMPhone9BSingle_verified183_exact_local_receipt_upload_20261120.py','build_OvisU1_verified184_exact_local_receipt_upload_20261121.py'),('render_AutoGLMPhone9BSingle_verified183_actual_published_20261120.mjs','render_OvisU1_verified184_actual_published_20261121.mjs')]:
 s=(V/old).read_text(encoding='utf8').replace(OLD,PUB).replace('AutoGLMPhone9BSingle_local_result_sync_verified183_20261120',LOCAL).replace('R_AutoGLMPhone9BSingle_verified183_server_book_for_local_CAS_20261120.json','R_OvisU1_verified184_server_book_for_local_CAS_20261121.json').replace('G692','G144').replace('B692:D692','B144:D144').replace('B690:G694','B142:G146').replace('[690]','[142]').replace('before_692','before_144').replace('after_692','after_144').replace('all_prior182','all_prior183').replace('verified183_rendered','verified184_rendered').replace('upload_AutoGLMPhone9BSingle_verified183_exact_local_CAS_receipt_20261120_remote.py','upload_OvisU1_verified184_exact_local_CAS_receipt_20261121_remote.py').replace('owned_ZaiGLM46_verified181','owned_OvisU1_verified184')
 if new.endswith('.py'):ast.parse(s)
 (V/new).write_text(s,encoding='utf8',newline='\n')
s=(V/'autoglm_phone_single_workbook_cell_verified183_20261120.mjs').read_text(encoding='utf8').replace('B690:G694','B142:G146')
(V/'ovisu1_workbook_cell_verified184_20261121.mjs').write_text(s,encoding='utf8',newline='\n')
s=(V/'read_AutoGLMPhone9BSingle_verified183_server_book_for_local_CAS_20261120_remote.py').read_text(encoding='utf8').replace(OLD,PUB).replace('zai-org/AutoGLM-Phone-9B',MID).replace('old182','old183').replace('publish_AutoGLMPhone9BSingle_verified183_single_original_lock_dual_transaction_v1_20261120.py',PUBLISHER).replace('504a20985debd6d956a37451e373683819677e86b5dc2aa56963b0b439842755',sha(pub.encode()))
(V/'read_OvisU1_verified184_server_book_for_local_CAS_20261121_remote.py').write_text(s,encoding='utf8',newline='\n')
scope=dict(passed=True,model_id=MID,publisher=PUBLISHER,publisher_SHA256=sha(pub.encode()),writer=WRITER+'.py',writer_SHA256=sha(writer.encode()),parent_publisher_SHA256=sha(parent.read_bytes()),frozen_raw_G_write_function_byte_identical=True,original_job_remains_failed=True,new_acceptance_actual_NEW_model_forwards=0,no_model_source_or_old_job_or_original_verifier_changed=True,publication_home=PUB,local_home=LOCAL,formalNF2_not_yet_claimed_complete=True)
(V/'OvisU1_NF2_verified184_publication_source_scope_20261121.json').write_text(json.dumps(scope,indent=2)+'\n',encoding='utf8')
print(json.dumps(scope))
