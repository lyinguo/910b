"""Derive one-target publication from actual completed AutoGLM evidence only."""
from pathlib import Path
import ast,base64,copy,hashlib,json,re
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
packet=json.loads((P/'R_AutoGLM_serial_v3_after_qualified_GLM181_actual8_20261120.json').read_bytes());files=packet['files']
def record(suffix):
 rows=[v for n,v in files.items() if n.endswith(suffix) and isinstance(v,dict) and 'complete_text' in v];assert len(rows)==1,(suffix,len(rows));v=rows[0];return json.loads(v['complete_text']),v['SHA256']
pf,pfsha=record('formal200/verification_stdout.json');cap,capsha=record('own_passed3_complete_shape_capacity_review.json');assert pfsha=='498c5f03c923bdcd12f8ea40022f03a4c8bae44ea6ca0eabde00687b8566c95d' and pf['passed'] and pf['metrics']=={'Accuracy':.82} and pf['actual_forward_count']==2601 and pf['independent_verifier_SHA256']=='05543ba8f1ea79ee3a22b602e8e2b0afc764528eda2783c0568227399b74a93b'
PUB='AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120';I='AutoGLMPhone9B_verified182_reviewed_single_lock_dual_publication_20261120'
MID='zai-org/AutoGLM-Phone-9B-Multilingual';OLDMID='zai-org/GLM-4.6V-Flash';BEFORE='4e740da3981c17fd7b599875ab95e60c5f709804f08c22c9fb64c3c8a5454ed8';TRUE3='9676feeab2999d51ddac07d4900ec1947342795fba8db322b694de589efd89c3'
OLDPUB='ZaiGLM46Flash_original_BF16_thinking4096_LZMA_verified181_blank_G_publication_20261119';LOCAL='AutoGLMPhone9B_local_result_sync_verified182_20261120'
bookchanges={'689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34':BEFORE,'e3a02a66da363e3ca15649f1cdab2d553f693297dc1ccbad25ea8495d12ca88b':'0ae99984e7e1f37f31f6c17d586d2f41c38abe838760900dba49ac5c5abbb931','2ad3869d9152b05f7235a649525fdd54328c6ac157ae938cae23f8f62254dc37':'46915b35ef9b942783ed09f8664eaf7bd6351d9d90c1eef62dd793bdb0642243','d089c0ded120ee0df0be6ad28d9a02fe0b61697330d6f3b4a3178bc667916950':TRUE3,'148afbeee4860b7a980608ae53b933b81afea6173cfa40aad635ce80f21cb91b':pf['independent_verifier_SHA256']}
class Adapt(ast.NodeTransformer):
 def visit_Constant(self,n):
  v=n.value
  if type(v)==int:v={180:181,181:182,166:167,209:208,67:68,528:527,519:518,697:693}.get(v,v)
  elif isinstance(v,str):
   if v in bookchanges:v=bookchanges[v]
   elif v.startswith('PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117/'):
    v=OLDPUB+'/'+v.split('/',1)[1].replace('180','181')
   elif v==OLDPUB:v=PUB
   elif not v.startswith('authorized_verified180_current_accepted_nonexample_reaudit_'):
    v=v.replace(OLDMID,MID).replace('G697','G693').replace('next_new182','next_new183')
    v=re.sub(r'(?<![0-9])(180|181)(?![0-9])',lambda m:str(int(m.group())+1),v)
  return ast.copy_location(ast.Constant(v),n)
source=P/'publish_ZaiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_actual_verifier_20261119.py';assert sha(source.read_bytes())=='dc4160569fe1d8b694da5d0f0d01c112a81579ab6e01ccd0586f0db8e229954a'
tree=Adapt().visit(ast.parse(source.read_bytes()));ast.fix_missing_locations(tree)
parts=len(cap['all_complete_ZIP_parts_individually_classified']);boundlen=len(cap['all_complete_original_values_still_available_and_SHA_bound'])
capcode=f'''def glm46_capacity_review_passed(cap):
    bound=cap.get('all_complete_original_values_still_available_and_SHA_bound',{{}})
    return cap.get('passed') is True and cap.get('capacity_passed') is True and cap.get('model_id')=={MID!r} and cap.get('empirical_estimate_bytes')==83391788264 and cap.get('actual_free_bytes',0)>=83391788264 and cap.get('original_factor')==4 and cap.get('original_extra_reserve_bytes')==6*1024**3 and cap.get('all_original_hard3GiB_log1GiB_precheckHalfGiB_per_sample2GiB_times_outer10_guards_preserved') is True and cap.get('all200_original_shapes_used_not_only_200_div3') is True and cap.get('no_tensor_value_or_occurrence_or_model_parameter_changed') is True and cap.get('all200_inputs_are_this_models_own_original_frontend') is True and cap.get('no_NN_or_compression_replay_in_capacity_review') is True and isinstance(bound,dict) and len(bound)=={boundlen} and len([n for n in bound if n.endswith('.npz')])==3 and bound.get({next(n for n,h in cap['all_complete_original_values_still_available_and_SHA_bound'].items() if h==TRUE3)!r})=={TRUE3!r} and len(cap.get('all_complete_ZIP_parts_individually_classified',[]))=={parts} and len(cap.get('all_original200_prompt_lengths',[]))==len(cap.get('all_original200_pixel_elements',[]))==200 and cap.get('original_true3_proof_SHA256')=={TRUE3!r}
'''
fn=ast.parse(capcode).body[0]
tree.body=[fn if isinstance(n,ast.FunctionDef) and n.name=='glm46_capacity_review_passed' else n for n in tree.body]
newname='publish_AutoGLMPhone9B_verified182_original_single_lock_dual_transaction_v1_20261120.py';pub=ast.unparse(tree)+'\n';ast.parse(pub)
# Unchanged frozen writer entry point and all full-head checks remain mandatory.
getfn=lambda t,n:ast.dump(next(v for v in t.body if isinstance(v,ast.FunctionDef) and v.name==n))
assert getfn(tree,'full_numeric_head_verified')==getfn(ast.parse(source.read_bytes()),'full_numeric_head_verified')
g={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<own-complete-capacity>','exec'),g);assert g[fn.name](cap)
for key,val in [('capacity_passed',False),('model_id',OLDMID),('original_factor',3),('all200_inputs_are_this_models_own_original_frontend',False),('empirical_estimate_bytes',83391788263)]:
 bad=copy.deepcopy(cap);bad[key]=val;assert not g[fn.name](bad)
(P/newname).write_text(pub,encoding='utf8',newline='\n')
# Existing exact local preservation/CAS logic, with reviewed target/count/schema binding.
def textadapt(s):
 for a,b in bookchanges.items():s=s.replace(a,b)
 return s.replace('ZaiGLM46Flash_local_result_sync_verified181_20261119',LOCAL).replace('R_ZaiGLM46_verified181_server_book_for_local_CAS_20261119.json','R_AutoGLMPhone9B_verified182_server_book_for_local_CAS_20261120.json').replace('old180','old181').replace('prior180','prior181').replace('G697','G693').replace('[695]','[691]').replace(OLDMID,MID)
names={}
for old,new in [('apply_ZaiGLM46_verified181_exact_server_bytes_local_CAS_20261119.py','apply_AutoGLMPhone9B_verified182_exact_server_bytes_local_CAS_20261120.py'),('render_ZaiGLM46_verified181_actual_published_20261119.mjs','render_AutoGLMPhone9B_verified182_actual_published_20261120.mjs'),('build_ZaiGLM46_verified181_exact_local_receipt_upload_20261119.py','build_AutoGLMPhone9B_verified182_exact_local_receipt_upload_20261120.py')]:
 s=textadapt((P/old).read_text(encoding='utf8')).replace(OLDPUB,PUB).replace('B697:D697','B693:D693').replace('B695:G699','B691:G695').replace('published_after_697','published_after_693').replace('before_697','before_693').replace('after_697','after_693').replace('upload_ZaiGLM46_verified181_exact_local_CAS_receipt_20261119_remote.py','upload_AutoGLMPhone9B_verified182_exact_local_CAS_receipt_20261120_remote.py')
 if new.endswith('.py'):ast.parse(s)
 (P/new).write_text(s,encoding='utf8',newline='\n');names[new]=sha(s.encode())
node=(P/'zai_glm46_workbook_cell_verified181_20261119.mjs').read_text(encoding='utf8').replace('B695:G699','B691:G695');(P/'autoglm_phone_workbook_cell_verified182_20261120.mjs').write_text(node,encoding='utf8',newline='\n')
wrapper=f'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/{I!r};source=R/'scripts/{newname}';review=I/'reviewed_inputs.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=={sha(pub.encode())!r} and hashlib.sha256(review.read_bytes()).hexdigest()==sys.argv[1]
start=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout.json').open('xb') as out,(I/'publication_stderr.log').open('xb') as err:r=subprocess.run(['python3','-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=r.returncode,source=str(source),source_SHA256={sha(pub.encode())!r},reviewed_inputs_SHA256=sys.argv[1]);(I/'publication_actual_completion.json').write_text(json.dumps(v,indent=2)+'\\n',encoding='utf8');raise SystemExit(r.returncode)
'''
ast.parse(wrapper);wrapname='run_AutoGLMPhone9B_verified182_original_single_lock_dual_publication_20261120.py';(P/wrapname).write_text(wrapper,encoding='utf8',newline='\n')
reader=(P/'read_ZaiGLM46_verified181_server_book_for_local_CAS_v2_actual_verifier_20261119_remote.py').read_text(encoding='utf8').replace(OLDPUB,PUB).replace('publish_ZaiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_actual_verifier_20261119.py',newname).replace('dc4160569fe1d8b694da5d0f0d01c112a81579ab6e01ccd0586f0db8e229954a',sha(pub.encode())).replace('old180','old181').replace(OLDMID,MID);ast.parse(reader);(P/'read_AutoGLMPhone9B_verified182_server_book_for_local_CAS_20261120_remote.py').write_text(reader,encoding='utf8',newline='\n')
result=dict(passed=True,model_id=MID,actual_full200_proof_SHA256=pfsha,publisher=newname,publisher_SHA256=sha(pub.encode()),wrapper=wrapname,wrapper_SHA256=sha(wrapper.encode()),publication_home=PUB,supervisor_home=I,own_capacity_SHA256=capsha,all_complete_ZIP_parts=parts,complete_bound_files=boundlen,full_head_verification_AST_exactly_unchanged=True,capacity_positive_and_five_negative_scope_checks_passed=True,source_only_no_NN_replay_or_canonical_write=True,local_controls=names)
(P/'AutoGLMPhone9B_verified182_actual_publication_source_scope_20261120.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
