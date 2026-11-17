import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,base64,json
P=Path(__file__).parent;old='prepare_phi35_own_original_BF16_4096_HD_frontend200_v2_exact_own_EOS_metadata_no_NN_20261117.py';s=(P/old).read_text()
needle="prompt=row['prompt'];media=Path(row['media_path']);assert media.is_file();clean=re.sub(r'<image\\d+>\\s*','',prompt).strip()";assert s.count(needle)==1
new=s.replace('frontend200_v2_exact_own_EOS_metadata_no_NN','frontend200_v3_original_dataset_prompt_no_NN').replace('import os,json,hashlib,struct,re,math,traceback','import os,json,hashlib,struct,re,math,traceback,ast,copy')
new=new.replace(needle,"prompt=row['prompt'];media=Path(row['media_path']);media=(media if media.is_absolute() else dataset.parent/media).resolve();assert media.is_relative_to(dataset.parent) and media.is_file();clean=re.sub(r'<image\\d+>\\s*','',add_instruction(prompt)).strip()")
key="index=[]\n with (H/"
assert new.count(key)==1
func="""algorithm=R/'universal_eval/algorithm/inference.py';assert sha(algorithm)=='5405d7c05e39bc2bc1cba84c7fb0064b85c93c3ef69919842341e60b28994097';tree=ast.parse(algorithm.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GreedyInference');f=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_add_answer_instruction'));f.decorator_list=[];scope={'re':re};exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(algorithm),'exec'),scope);add_instruction=scope['_add_answer_instruction'];resources[str(algorithm)]=dict(SHA256=sha(algorithm))
 index=[]
 with (H/"""
new=new.replace(key,func).replace('original_prompt=prompt,original_native_message=','original_prompt=prompt,original_model_prompt=clean,original_native_message=')
ast.parse(new)
name='scripts/prepare_phi35_own_original_BF16_4096_HD_frontend200_v3_original_dataset_prompt_no_NN_20261117.py';h=hashlib.sha256(new.encode()).hexdigest();(P/Path(name).name).write_bytes(new.encode())
deploy=(P/'deploy_run_Phi35_own_original_frontend200_v2_exact_own_EOS_metadata_no_NN_20261117_remote.py').read_text();tree=ast.parse(deploy)
for node in ast.walk(tree):
 if isinstance(node,ast.Constant) and isinstance(node.value,str):
  if node.value=='scripts/'+old:node.value=name
  elif node.value==hashlib.sha256(s.encode()).hexdigest():node.value=h
  elif node.value==base64.b64encode(s.encode()).decode():node.value=base64.b64encode(new.encode()).decode()
  elif node.value=='Phi35_own_HD_frontend200_v2_exact_own_EOS_metadata_no_NN_actual_launch_20261117':node.value='Phi35_own_HD_frontend200_v3_original_dataset_prompt_no_NN_actual_launch_20261117'
deploy=ast.unparse(tree);assert name in deploy and h in deploy
(P/'deploy_run_Phi35_own_original_frontend200_v3_original_dataset_prompt_no_NN_20261117_remote.py').write_bytes(deploy.encode());print(json.dumps(dict(primary=name,SHA256=h,no_original_asset_or_math_changed=True)))
