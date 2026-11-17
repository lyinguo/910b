import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,ast,base64
P=Path(__file__).parent;x=json.loads((P/'R_frozen_G_writer_exact_source_schema_20261117.json').read_bytes());old=x['text'];assert hashlib.sha256(old.encode()).hexdigest()=='83134eec524c27bc4e9fd4569193e72038a7b248b379c12cf6999e1dcff798b8'
helper='''
def check_formal_acceptance_metadata(completed,p,job,source):
 assert completed['phase']=='completed_formal_independent_acceptance_of_original_NPU200' and completed['actual_returncode']==0 and completed['verifier_returncode']==0 and completed['actual_new_model_forwards_this_phase']==0
 assert p['passed'] and p['sample_count']==200 and p['actual_NPU_inference'] and not p['historical_record_reuse'] and p['formal_alignment_within_5pct']
 assert p['formal_independent_acceptance_phase_not_diagnostic_only'] and p['no_new_NPU_or_learned_CPU_inference_in_this_acceptance_phase'] and p['all200_input_pixels_exact_original_native_media_decode_verified'] and p['original_full_numeric_model_checks_unchanged']
 assert job['model_id']=='merve/paligemma_vqav2' and job['execution_kind']=='formal_independent_acceptance_of_completed_original_NPU200_only' and job['stages']==[3,200] and job['spec']['approved'] and not job['write_workbook']
 assert source['model_returncode']==0 and source['verifier_returncode']==1 and source['actual_returncode']==1
 assert [x['sample_count'] for x in completed['accepted']]==[3,200]
 return True

def authorize_actual_merve_formal_acceptance(completed,m,model_id):
 assert model_id=='merve/paligemma_vqav2'
 p=json.loads(pathlib.Path(m['proof']).read_text());ctx=pathlib.Path(p['current_execution_job']).resolve();assert ctx.is_relative_to('/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103');assert digest(ctx)==p['current_execution_job_SHA256'];job=json.loads(ctx.read_text());assert pathlib.Path(m['job_completion']).resolve()==ctx.parent/'completion.json'
 source_home=pathlib.Path(job['source_original_NPU200_job']);assert str(source_home)==p['actual_original_NPU200_model_job'];source=json.loads((source_home/'completion.json').read_text());assert digest(source_home/'job.json')==job['source_original_NPU200_job_SHA256'] and digest(source_home/'completion.json')==job['source_original_NPU200_completion_SHA256'];check_formal_acceptance_metadata(completed,p,job,source)
 assert completed['model_returncode_source_original_job']==str(source_home/'completion.json') and completed['model_returncode']==source['model_returncode']==p['actual_original_NPU200_model_returncode']==0
 assert "assert matches,'No own original processor branch reproduced actual native input pixels'" in (source_home/'formal200/verification_stderr.log').read_text()
 original_job=json.loads((source_home/'job.json').read_text());assert original_job['spec']['config']==job['spec']['config'] and original_job['spec']['alignment_approval_review_sha256']==job['spec']['alignment_approval_review_sha256'];assert not job['write_workbook']
 for name,h in original_job['spec']['source_sha256'].items():assert digest(pathlib.Path('/root/projects/comperror')/name)==h and job['spec']['source_sha256'][name]==h
 from A100_alignment_resume_completed_approved_fresh3_guard_20261117 import validate_resume
 prior=validate_resume(job,pathlib.Path('/root/projects/comperror'));assert completed['accepted'][0]['proof_sha256']==prior['proof_sha256']
 for name,h in job['spec']['source_sha256'].items():assert digest(pathlib.Path('/root/projects/comperror')/name)==h
 assert digest(pathlib.Path('/root/projects/comperror')/job['spec']['verifier'])==p['independent_verifier_SHA256']
 return True
'''
needle="completed=json.loads(pathlib.Path(m['job_completion']).read_text());assert completed['actual_returncode']==0 and completed['phase']=='completed_evidence_only' and [x['sample_count'] for x in completed['accepted']]==[3,200]";assert old.count(needle)==1
replacement="completed=json.loads(pathlib.Path(m['job_completion']).read_text());assert completed['actual_returncode']==0 and [x['sample_count'] for x in completed['accepted']]==[3,200]\n if completed['phase']!='completed_evidence_only':assert authorize_actual_merve_formal_acceptance(completed,m,model_id)"
new=old.replace(needle,replacement);pos=new.index('def authorize(');new=new[:pos]+helper+'\n'+new[pos:];ast.parse(new);assert new[new.index('def write_verified_A100_G'):]==old[old.index('def write_verified_A100_G'):]
W='scripts/verified_Merve_actual_formal_acceptance_G_cell_writer_schema_v2_20261117.py';(P/Path(W).name).write_bytes(new.encode());wh=hashlib.sha256(new.encode()).hexdigest()
pub=(P/'publish_MervePaliCompact_verified179_single_original_lock_dual_transaction_v1_20261117.py').read_text(encoding='utf-8');pub=pub.replace('from verified_A100_G_cell_writer_20261103 import write_verified_A100_G','from verified_Merve_actual_formal_acceptance_G_cell_writer_schema_v2_20261117 import write_verified_A100_G');pub=pub.replace("H = B / 'MervePaliCompact_original_HF_BF16_verified179_blank_G_publication_20261117'","H = B / 'MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v2_eligible_phase_schema_20261117'");pub=pub.replace("policy = read(D / 'A100_ALIGNMENT_PUBLICATION_POLICY.json')","assert sha(R/'"+W+"')=='"+wh+"'\npolicy = read(D / 'A100_ALIGNMENT_PUBLICATION_POLICY.json')");pub=pub.replace('_verified179_original_writer_isolated_','_verified179_v2_formal_acceptance_writer_isolated_');ast.parse(pub);PN='scripts/publish_MervePaliCompact_verified179_dual_transaction_v2_real_formal_acceptance_schema_20261117.py';(P/Path(PN).name).write_bytes(pub.encode());ph=hashlib.sha256(pub.encode()).hexdigest()
tests='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from verified_Merve_actual_formal_acceptance_G_cell_writer_schema_v2_20261117 import check_formal_acceptance_metadata
B=Path('/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct');H=B/'Merve_completed_original200_formal_independent_acceptance_v4_native_media_reference_no_NN_replay_20261117';completed=json.loads((H/'completion.json').read_bytes());job=json.loads((H/'job.json').read_bytes());p=json.loads(Path(completed['accepted'][-1]['proof']).read_bytes());source=json.loads((Path(job['source_original_NPU200_job'])/'completion.json').read_bytes());assert check_formal_acceptance_metadata(completed,p,job,source);cases=[]
for target,key,value in [('completed','phase','failed'),('completed','verifier_returncode',1),('completed','actual_new_model_forwards_this_phase',1),('proof','actual_NPU_inference',False),('proof','sample_count',3),('proof','formal_alignment_within_5pct',False),('proof','all200_input_pixels_exact_original_native_media_decode_verified',False),('proof','original_full_numeric_model_checks_unchanged',False),('job','model_id','different/model'),('source','model_returncode',1)]:
 values=[copy.deepcopy(x) for x in [completed,p,job,source]];i={'completed':0,'proof':1,'job':2,'source':3}[target];values[i][key]=value;cases.append(values)
for values in cases:
 try:check_formal_acceptance_metadata(*values)
 except AssertionError:pass
 else:raise AssertionError('Invalid independent acceptance admitted')
print(json.dumps(dict(passed=True,test_count=1+len(cases),no_production_or_original_evidence_mutation=True,frozen_shared_writer_unchanged=True)))
'''
TN='scripts/tests/test_Merve_actual_formal_acceptance_writer_metadata_rejections_20261117.py';ast.parse(tests);(P/Path(TN).name).write_bytes(tests.encode());files={}
for n,s in [(W,new),(PN,pub),(TN,tests)]:files[n]=dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,ast,json,base64,hashlib,subprocess
R=Path('/root/projects/comperror');F=__F__
for n,v in F.items():
 p=R/n;assert not p.exists();raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);p.write_bytes(raw)
old=R/'scripts/tests/verified_Gemma4E2B_manifest_task_writer_11_cases_template_20261116.py';raw=old.read_bytes();assert hashlib.sha256(raw).hexdigest()=='8e595053cedf0ff3fc46b1f1a76fde787ed268ac190bc037e7521882e048e318';text=raw.decode().replace('from verified_A100_G_cell_writer_20261103 import write_verified_A100_G','from verified_Merve_actual_formal_acceptance_G_cell_writer_schema_v2_20261117 import write_verified_A100_G');assert text!=raw.decode();tp=R/'scripts/tests/Merve_actual_formal_acceptance_manifest_writer_11_cases_template_20261117.py';assert not tp.exists();tp.write_bytes(text.encode())
r=subprocess.run(['python3','-B','__TN__'],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert hashlib.sha256((R/'scripts/verified_A100_G_cell_writer_20261103.py').read_bytes()).hexdigest()=='83134eec524c27bc4e9fd4569193e72038a7b248b379c12cf6999e1dcff798b8';print(json.dumps(dict(actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_SHA256={n:v['SHA256'] for n,v in F.items()},new_test_template=str(tp.relative_to(R)),new_test_template_SHA256=hashlib.sha256(tp.read_bytes()).hexdigest(),raw_G_write_function_byte_identical_to_frozen_writer=True,no_completion_phase_or_canonical_or_report_or_proof_changed=True),ensure_ascii=False));raise SystemExit(r.returncode)
'''.replace('__F__',repr(files)).replace('__TN__',TN);ast.parse(remote);(P/'deploy_test_Merve_real_formal_acceptance_writer_v2_without_any_workbook_edit_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(new_writer=W,new_writer_SHA256=wh,publisher=PN,publisher_SHA256=ph,frozen_writer_not_modified=True,raw_G_mutation_function_byte_identical=True,no_fake_completed_phase_certificate_created=True),indent=2))
