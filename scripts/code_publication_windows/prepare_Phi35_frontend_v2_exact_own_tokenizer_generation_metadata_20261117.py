import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,base64,json
P=Path(__file__).parent;old='prepare_phi35_own_original_BF16_4096_HD_frontend200_v1_no_NN_20261117.py';raw=(P/old).read_text();line="tokenizer=processor.tokenizer;assert tokenizer.eos_token_id==2";assert raw.count(line)==1
new=raw.replace('frontend200_v1_no_NN','frontend200_v2_exact_own_EOS_metadata_no_NN').replace(line,"tokenizer=processor.tokenizer;assert js(W/'generation_config.json')['eos_token_id']==2 and tokenizer.eos_token_id==tokenizer.convert_tokens_to_ids(tokenizer.eos_token)")
new=new.replace("own_original_4096_outer10_Greedy_and_config_values_unchanged=True,","own_original_4096_outer10_Greedy_and_config_values_unchanged=True,own_tokenizer_EOS_id=tokenizer.eos_token_id,own_generation_config_EOS_id=js(W/'generation_config.json')['eos_token_id'],tokenizer_and_generation_EOS_difference_preserved_not_rewritten=True,actual_native_engine_stop_ids_still_require_observation=True,")
ast.parse(new)
a=ast.parse(raw);b=ast.parse(new)
for name in ['independent_pixels','tsha','tm']:
 lhs=next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name==name);rhs=next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name==name);assert ast.dump(lhs,include_attributes=False)==ast.dump(rhs,include_attributes=False)
name='scripts/prepare_phi35_own_original_BF16_4096_HD_frontend200_v2_exact_own_EOS_metadata_no_NN_20261117.py';h=hashlib.sha256(new.encode()).hexdigest();(P/Path(name).name).write_bytes(new.encode())
deploy=(P/'deploy_run_Phi35_own_original_frontend200_v1_no_NN_20261117_remote.py').read_text();tree=ast.parse(deploy)
for node in ast.walk(tree):
 if isinstance(node,ast.Constant) and isinstance(node.value,str):
  if node.value=='scripts/'+old:node.value=name
  elif node.value==hashlib.sha256(raw.encode()).hexdigest():node.value=h
  elif node.value==base64.b64encode(raw.encode()).decode():node.value=base64.b64encode(new.encode()).decode()
  elif node.value=='Phi35_own_HD_frontend200_v1_no_NN_actual_launch_20261117':node.value='Phi35_own_HD_frontend200_v2_exact_own_EOS_metadata_no_NN_actual_launch_20261117'
deploy=ast.unparse(tree);assert name in deploy and h in deploy
(P/'deploy_run_Phi35_own_original_frontend200_v2_exact_own_EOS_metadata_no_NN_20261117_remote.py').write_bytes(deploy.encode());print(json.dumps(dict(primary=name,SHA256=h,pixel_and_numeric_implementations_AST_identical=True,old_failure_retained=True)))
