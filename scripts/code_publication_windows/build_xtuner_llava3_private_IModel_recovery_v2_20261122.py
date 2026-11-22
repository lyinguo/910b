"""New private IModel binding after completed zero-forward v1 loader rejection."""
from pathlib import Path
import ast,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def out(name,text):
 p=V/name; b=text.encode('utf-8');ast.parse(text) if name.endswith('.py') else json.loads(text)
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
 return b
model1='xtuner_llava3_own_original_Transformers_FP16_2048_native_v1_20261122.py';model2=model1.replace('_v1_','_v2_IModel_')
prep1='prepare_xtuner_llava3_own_original_Transformers_FP16_2048_true3_v1_20261122.py';prep2=prep1.replace('_v1_','_v2_IModel_')
ctrl1='continue_xtuner_llava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122.py';ctrl2=ctrl1.replace('_v1_','_v2_IModel_')
q1='xtuner_llava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122.json';q2=q1.replace('_v1_','_v2_IModel_')
old=(V/model1).read_text();new=old.replace('from torch import nn','from torch import nn\nfrom universal_eval.model.interface import IModel').replace('class OwnXtunerLlava3OriginalFP16:','class OwnXtunerLlava3OriginalFP16(IModel):')
a=ast.parse(old);b=ast.parse(new)
ca=next(x for x in a.body if isinstance(x,ast.ClassDef) and x.name=='OwnXtunerLlava3OriginalFP16');cb=next(x for x in b.body if isinstance(x,ast.ClassDef) and x.name==ca.name)
assert ast.dump(ast.Module(body=ca.body,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=cb.body,type_ignores=[]),include_attributes=False)
assert len(b.body)==len(a.body)+1
out(model2,new)
prep=(V/prep1).read_text().replace(model1,model2).replace(prep1,prep2).replace(ctrl1,ctrl2).replace(q1,q2)
prep=prep.replace('readiness_v1_20261122','readiness_v2_IModel_20261122').replace('true3_v1_real_FLOCK_20261122','true3_v2_IModel_real_FLOCK_20261122').replace('native-v1-20261122','native-v2-IModel-20261122').replace('native_v1_20261122.json','native_v2_IModel_20261122.json')
needle=" H.mkdir(exist_ok=False);ready=js(FRONT/'readiness.json');ctor=js(CPU/'actual_original_loader_resize_remap_review.json')"
replacement=""" old=B/'XtunerLlava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122';old_done=js(old/'actual_completion.json')
 assert old_done['actual_returncode']==1 and old_done['phase']=='own_genuine_NEWtrue3_or_fullNF_failed_evidence_preserved'
 old_job=B/'XtunerLlava3_own_original_Transformers_FP16_2048_true3_v1_real_FLOCK_20261122';console=(old_job/'fresh3/console.log').read_text()
 assert 'TypeError: OwnXtunerLlava3OriginalFP16 必须继承自 IModel' in console and 'Own Xtuner complete sample' not in console
 import inspect
 from universal_eval.model.interface import IModel
 from universal_eval.model.loader import ModelLoader
 actual_class=ModelLoader.load_class(MODEL+':OwnXtunerLlava3OriginalFP16')
 assert issubclass(actual_class,IModel) and not inspect.isabstract(actual_class) and not actual_class.__abstractmethods__
 old_tree=ast.parse((R/'universal_eval/model/adapters/xtuner_llava3_own_original_Transformers_FP16_2048_native_v1_20261122.py').read_bytes());new_tree=ast.parse((R/MODEL).read_bytes())
 old_class=next(x for x in old_tree.body if isinstance(x,ast.ClassDef) and x.name=='OwnXtunerLlava3OriginalFP16');new_class=next(x for x in new_tree.body if isinstance(x,ast.ClassDef) and x.name==old_class.name)
 assert ast.dump(ast.Module(body=old_class.body,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=new_class.body,type_ignores=[]),include_attributes=False)
 H.mkdir(exist_ok=False);create(H/'zero_forward_v1_IModel_recovery_review.json',dict(passed=True,old_actual_completion=str(old/'actual_completion.json'),old_actual_completion_SHA256=sha(old/'actual_completion.json'),old_console=str(old_job/'fresh3/console.log'),old_console_SHA256=sha(old_job/'fresh3/console.log'),all_model_method_AST_unchanged=True,only_IModel_import_and_base_added=True,actual_framework_ModelLoader_class_acceptance_passed=True,no_model_forward_in_interface_check=True))
 ready=js(FRONT/'readiness.json');ctor=js(CPU/'actual_original_loader_resize_remap_review.json')"""
assert needle in prep;prep=prep.replace(needle,replacement)
prep=prep.replace("'universal_eval/model/interface.py','universal_eval/algorithm/inference.py'","'universal_eval/model/interface.py','universal_eval/model/loader.py','universal_eval/model/manager.py','universal_eval/model/adapters/xtuner_llava3_own_original_Transformers_FP16_2048_native_v1_20261122.py','universal_eval/algorithm/inference.py'")
prep=prep.replace("CPU/'original_actual_CPU_buffer_layout.json']","CPU/'original_actual_CPU_buffer_layout.json',H/'zero_forward_v1_IModel_recovery_review.json',old/'actual_completion.json',old_job/'fresh3/console.log']")
out(prep2,prep)
ctrl=(V/ctrl1).read_text().replace(ctrl1,ctrl2).replace(prep1,prep2).replace(q1,q2).replace('serial_v1_20261122','serial_v2_IModel_20261122').replace('readiness_v1_20261122','readiness_v2_IModel_20261122').replace('true3_v1_real_FLOCK_20261122','true3_v2_IModel_real_FLOCK_20261122').replace('first200_v1_20261122','first200_v2_IModel_20261122').replace('first200-v1-20261122','first200-v2-IModel-20261122').replace('serial_v1_real_FLOCK_20261122','serial_v2_IModel_real_FLOCK_20261122')
ctrl=ctrl.replace('Serial own LLaVA-Mini genuine3/full NF','Serial own Xtuner IModel repair genuine3/full NF')
out(ctrl2,ctrl)
primary={
 'universal_eval/model/adapters/'+model2:sha((V/model2).read_bytes()),
 'scripts/'+prep2:sha((V/prep2).read_bytes()),
 'scripts/'+ctrl2:sha((V/ctrl2).read_bytes())}
base=json.loads((V/q1).read_bytes());base['exact_source_SHA256']={k:h for k,h in base['exact_source_SHA256'].items() if k not in ['universal_eval/model/adapters/'+model1,'scripts/'+prep1,'scripts/'+ctrl1]};base['exact_source_SHA256'].update(primary);base['completed_v1_zero_forward_IModel_rejection_preserved']=True
out(q2,json.dumps(base,ensure_ascii=False,indent=2)+'\n');primary['configs/queues/'+q2]=sha((V/q2).read_bytes())
primary['scripts/code_publication_windows/'+Path(__file__).name]=sha(Path(__file__).read_bytes())
out('XtunerLlava3_private_IModel_v2_primary_paths_20261122.json',json.dumps(primary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(primary=primary,checks=dict(all_methods_AST_identical=True,only_interface_import_and_base_added=True,old_versions_unchanged=True,new_unique_private_jobs=True,real_ModelLoader_and_abstract_methods_checked_before_NN=True)),ensure_ascii=False))
