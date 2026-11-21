"""Prepare the own-model narrow publication implementation before NN completion."""
from pathlib import Path
import ast, copy, difflib, hashlib, json
V=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
packet=json.loads((V/'R_GLM41_verified183_publication_contract_20261120_1810.json').read_bytes())
files=packet['files']
def item(suffix):
 a=[v for n,v in files.items() if n.endswith(suffix) and 'complete_text' in v]
 assert len(a)==1,(suffix,len(a))
 return json.loads(a[0]['complete_text']),a[0]['SHA256']
cap,capsha=item('own_passed3_complete_shape_capacity_review.json')
assert capsha=='00f142bfd671abee441fa612b1013afb4ebaeede7acc0ab2334e04bf6da550b3'
MID='zai-org/GLM-4.1V-9B-Thinking'
OLDMID='zai-org/AutoGLM-Phone-9B'
OLDPUB='AutoGLMPhone9BSingle_original_BF16_2048_LZMA_verified183_blank_G_publication_20261120'
PRIORPUB='AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120'
PUB='GLM41Thinking_original_VLLM_BF16_8192_LZMA_verified184_blank_G_publication_20261120'
LOCAL='GLM41Thinking_local_result_sync_verified184_20261120'
TRUE3='48d8a3ad392c380c134d1185a800e25039689d4cf92dd820bb460db8a1ca9478'
BEFORE='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81'
changes={'0b2fcf21fe5a56e5b33111c4e65c2ec0ddf53dae43d451dec756a342df810340':BEFORE,
 '4718832822525fdd4516fd50b8f45d5863c21974b9b11bd1ab30608f44c53605':'c1c867fbfdf2637c03757612c2c49a43fefefa5be6886b30d979198db6bcf0cb',
 '3448fd5d8eb8f7d93a1b48ab7c7e3720608dd3f937388fc408c6d9c555e21c4d':'218c75abd8d84caced7308f3a521504eb6b76682eb2a655d8a068ee63ec59fb7',
 'da99ef8ddab138186eb4368204a6917041f126d3fb03730fe8163352a211c259':TRUE3,
 '3a08758fe52fbb86415daebbee4afe667ba3cbaf3224d1828e6fb95e7fb580ce':'48db53f3f64f7149eb11cdcb1278b8136c17c9a7d432fc82b7c914fb027b1203'}
def string(v):
 if v in changes:return changes[v]
 if v.startswith(PRIORPUB+'/'):return OLDPUB+'/'+v.split('/',1)[1].replace('182','183')
 v=v.replace(OLDPUB,PUB).replace(OLDMID,MID)
 v=v.replace('old182','old183').replace('prior182','prior183').replace('prior_182','prior_183')
 return v.replace('G692','G696').replace('verified183','verified184').replace('active183_full_report_stat_baseline','active184_full_report_stat_baseline').replace('next_new182','next_new185').replace('own AutoGLM-Phone-9B HF publication','own GLM-4.1V-9B-Thinking original VLLM8192 native-HF publication')
class Adapt(ast.NodeTransformer):
 def visit_Constant(self,n):
  v=n.value
  if type(v)==int:v={182:183,183:184,168:169,207:206,69:70,526:525,517:516,692:696}.get(v,v)
  elif isinstance(v,str):v=string(v)
  return ast.copy_location(ast.Constant(v),n)
 def visit_keyword(self,n):
  n=self.generic_visit(n)
  if n.arg:n.arg=n.arg.replace('old182','old183').replace('prior182','prior183').replace('prior_182','prior_183').replace('verified183_export','verified184_export').replace('next_new182','next_new185')
  return n
parent=V/'publish_AutoGLMPhone9BSingle_verified183_single_original_lock_dual_transaction_v1_20261120.py'
assert sha(parent.read_bytes())=='504a20985debd6d956a37451e373683819677e86b5dc2aa56963b0b439842755'
oldtree=ast.parse(parent.read_bytes());tree=Adapt().visit(copy.deepcopy(oldtree));ast.fix_missing_locations(tree)
capacity=f'''def glm46_capacity_review_passed(cap):
 bound=cap.get('all_complete_original_values_still_available_and_SHA_bound',{{}})
 return cap.get('passed') is True and cap.get('capacity_passed') is True and cap.get('model_id')=={MID!r} and cap.get('empirical_estimate_bytes')==136041918364 and cap.get('actual_free_bytes',0)>=136041918364 and cap.get('original_factor')==4 and cap.get('original_extra_reserve_bytes')==6*1024**3 and cap.get('all_original_hard3GiB_log1GiB_precheckHalfGiB_per_sample2GiB_times_outer10_guards_preserved') is True and cap.get('all200_original_shapes_used_not_only_200_div3') is True and cap.get('no_tensor_value_or_occurrence_or_model_parameter_changed') is True and cap.get('all200_inputs_are_this_models_own_original_frontend') is True and cap.get('no_NN_or_compression_replay_in_capacity_review') is True and len(bound)==7 and len([n for n in bound if n.endswith('.npz')])==3 and len(cap.get('all_complete_ZIP_parts_individually_classified',[]))==3009 and cap.get('original_true3_proof_SHA256')=={TRUE3!r} and len(cap.get('all_original200_prompt_lengths',[]))==len(cap.get('all_original200_pixel_elements',[]))==200
'''
capfn=ast.parse(capacity).body[0]
external=ast.parse("def check_original_external_hash(path, expected):\n    assert sha(path) == expected, (path, 'Own GLM41 original/current external evidence changed')\n").body[0]
tree.body=[capfn if isinstance(n,ast.FunctionDef) and n.name==capfn.name else external if isinstance(n,ast.FunctionDef) and n.name==external.name else n for n in tree.body]
pub=ast.unparse(tree)+'\n'
old_admission="assert not H.exists() and (not runner_lock_owners()) and (not [p for p in project_processes() if p['role'] in roles])"
assert pub.count(old_admission)==1
pub=pub.replace(old_admission,"assert not H.exists() and '_reviewed_original_lock' in globals() and (not [p for p in project_processes() if p['role'] in roles])\nassert {p['PID'] for p in runner_lock_owners()} == {os.getpid()}\nkernel_lock_owned(os.getpid(), D / 'runner.lock')")
assert pub.count("with (D / 'runner.lock').open('a+') as lock:")==1
pub=pub.replace("with (D / 'runner.lock').open('a+') as lock:","with __import__('contextlib').nullcontext(_reviewed_original_lock) as lock:")
ast.parse(pub)
fn=lambda t,n:ast.dump(next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n),include_attributes=False)
assert fn(ast.parse(pub),'full_numeric_head_verified')==fn(oldtree,'full_numeric_head_verified')
g={};exec(compile(ast.Module(body=[capfn],type_ignores=[]),'<own-GLM41-capacity>','exec'),g);assert g[capfn.name](cap)
for k,v in [('capacity_passed',False),('model_id',OLDMID),('original_factor',3),('all200_inputs_are_this_models_own_original_frontend',False),('empirical_estimate_bytes',1),('original_true3_proof_SHA256','0'*64)]:
 bad=copy.deepcopy(cap);bad[k]=v;assert not g[capfn.name](bad)
pubname='publish_GLM41Thinking_verified184_same_reserved_original_lock_dual_transaction_v1_20261120.py'
(V/pubname).write_text(pub,encoding='utf-8',newline='\n')
(V/'GLM41Thinking_verified184_publisher_full_parent_diff_20261120.diff').write_text(''.join(difflib.unified_diff(parent.read_text(encoding='utf-8').splitlines(keepends=True),pub.splitlines(keepends=True),fromfile=parent.name,tofile=pubname)),encoding='utf-8')
# Preserve the previously reviewed local CAS code; only the target and prior count change.
for old,new in [('apply_AutoGLMPhone9BSingle_verified183_exact_server_bytes_local_CAS_20261120.py','apply_GLM41Thinking_verified184_exact_server_bytes_local_CAS_20261120.py'),('build_AutoGLMPhone9BSingle_verified183_exact_local_receipt_upload_20261120.py','build_GLM41Thinking_verified184_exact_local_receipt_upload_20261120.py'),('render_AutoGLMPhone9BSingle_verified183_actual_published_20261120.mjs','render_GLM41Thinking_verified184_actual_published_20261120.mjs')]:
 s=(V/old).read_text(encoding='utf-8').replace(OLDPUB,PUB).replace('AutoGLMPhone9BSingle_local_result_sync_verified183_20261120',LOCAL).replace('R_AutoGLMPhone9BSingle_verified183_server_book_for_local_CAS_20261120.json','R_GLM41Thinking_verified184_server_book_for_local_CAS_20261120.json').replace('G692','G696').replace('B692:D692','B696:D696').replace('B690:G694','B694:G698').replace('[690]','[694]').replace('before_692','before_696').replace('after_692','after_696').replace('all_prior182','all_prior183').replace('verified183_rendered','verified184_rendered').replace('upload_AutoGLMPhone9BSingle_verified183_exact_local_CAS_receipt_20261120_remote.py','upload_GLM41Thinking_verified184_exact_local_CAS_receipt_20261120_remote.py')
 if new.endswith('.py'):ast.parse(s)
 (V/new).write_text(s,encoding='utf-8',newline='\n')
s=(V/'autoglm_phone_single_workbook_cell_verified183_20261120.mjs').read_text(encoding='utf-8').replace('B690:G694','B694:G698')
(V/'glm41thinking_workbook_cell_verified184_20261120.mjs').write_text(s,encoding='utf-8',newline='\n')
reader=(V/'read_AutoGLMPhone9BSingle_verified183_server_book_for_local_CAS_20261120_remote.py').read_text(encoding='utf-8').replace(OLDPUB,PUB).replace(OLDMID,MID).replace('old182','old183')
reader=reader.replace("source = R / 'scripts/publish_AutoGLMPhone9BSingle_verified183_single_original_lock_dual_transaction_v1_20261120.py'", "source = R / 'scripts/"+pubname+"'\nreservation = R / 'scripts/reserve_GLM41_completed200_reviewed_dual_publication_original_lock_v2_actual_runner_directory_20261120.py'")
reader=reader.replace("assert str(source) in argv and sha(source) == '504a20985debd6d956a37451e373683819677e86b5dc2aa56963b0b439842755'", "assert str(reservation) in argv and sha(reservation) == 'f28e5bff07285b8d6f7fe33ca5e5602a00504f35e9e4a57df9f419ae6036fbb7' and sha(source) == "+repr(sha(pub.encode())))
ast.parse(reader)
(V/'read_GLM41Thinking_verified184_server_book_for_local_CAS_20261120_remote.py').write_text(reader,encoding='utf-8',newline='\n')
scope=dict(passed=True,model_id=MID,publisher=pubname,publisher_SHA256=sha(pub.encode()),parent_publisher_SHA256=sha(parent.read_bytes()),capacity_SHA256=capsha,capacity_positive_and_six_negative_checks_passed=True,full_head_function_AST_unchanged=True,own_original_true3_proof_SHA256=TRUE3,no_model_or_NF_source_change=True,only_own_model_binding_prior183_new184_and_existing_original_lock_admission_changed=True,publication_home=PUB,local_home=LOCAL,reviewed_input_not_yet_created_formal200_still_running=True)
(V/'GLM41Thinking_verified184_publication_source_scope_20261120.json').write_text(json.dumps(scope,indent=2)+'\n',encoding='utf-8')
print(json.dumps(scope))
