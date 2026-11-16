"""Bind the weight-free MobileNet graph to this model's original FP16 input.
Only its initial dtype cast and static tensor dtype annotations differ. Every
operator, edge, other constant, intrinsic FP32 operation and own tensor shape is
checked. No learned values, calibration, quantization or math changes.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,re,copy
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='nicoboss/MedraN-E4B-Uncensored-EP7';H=B/'MedraNE4B_original_FP16_weight_free_vision_graph_binding_20261116'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def main():
 import torch
 H.mkdir(exist_ok=False);parent=js(R/'configs/protocols/unsloth_gemma3n_e4bit_native_v1.json');src=Path(parent['vision_graph']);old=src.read_text();assert sha(src)=='65e0b3b8f48e88cb393ed06723827781fd986251a94bd2525433e61efe855608'
 g=torch._C.parse_ir(old);nodes=list(g.nodes());cast=[]
 for node in nodes:
  if node.kind()=='prim::Constant' and node.hasAttribute('value'):assert node.kindOf('value')!='t','A weight-free graph may not contain Tensor constants'
  if node.kind()=='prim::Constant' and node.output().toIValue()==15:
   uses=list(node.output().uses());assert len(uses)==1 and uses[0].user.kind()=='aten::to' and uses[0].offset==1 and list(uses[0].user.inputs())[0]==list(g.inputs())[0];cast.append(node)
 assert len(cast)==1
 pattern=r'(?m)^(\s*%[\w.]+ : int = prim::Constant\[value=)15(\]\(\).*)$';new,count=re.subn(pattern,lambda m:m.group(1)+'5'+m.group(2),old);assert count==1
 new=new.replace('BFloat16','Half');ng=torch._C.parse_ir(new);nn=list(ng.nodes());assert len(nn)==len(nodes)==14874
 def structure(graph,normalize_old):
  values={v:i for i,v in enumerate(graph.inputs())};rows=[]
  for node in graph.nodes():
   attrs={}
   for key in node.attributeNames():
    kind=node.kindOf(key);assert kind!='t';value=getattr(node,kind)(key)
    if normalize_old and node.kind()=='prim::Constant' and key=='value' and value==15 and str(node.output().type())=='int':value=5
    attrs[key]=repr(value)
   row=[node.kind(),[values[v] for v in node.inputs()],attrs,[str(v.type()).replace('BFloat16','Half') for v in node.outputs()]]
   for value in node.outputs():values[value]=len(values)
   rows.append(row)
  return dict(inputs=[str(x.type()).replace('BFloat16','Half') for x in graph.inputs()],nodes=rows,outputs=[values[x] for x in graph.outputs()])
 assert structure(g,True)==structure(ng,False),'An operator, edge or non-dtype constant changed'
 W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma3nForConditionalGeneration')/MID;cfg=js(W/'config.json');assert cfg['torch_dtype']=='float16' and cfg['vision_config']['torch_dtype']=='float16'
 front=B/'MedraNE4B_own_original_FP16_chat_frontend200_assets_20261116';hdr=js(front/'source_tensor_headers.json');layout=copy.deepcopy(js(parent['vision_graph_layout']));assert len(layout['own_state'])==548
 for name,rec in layout['own_state'].items():
  original=hdr['model.vision_tower.timm_model.'+name];assert rec['dtype']=='torch.bfloat16' and original['dtype']=='F16' and original['shape']==rec['shape'];rec['dtype']='torch.float16'
 path=R/'universal_eval/model/medran_own_FP16_20261116/original_MobileNetV5_parameterized_FP16_v1.ir';assert not path.exists();path.write_text(new,encoding='utf-8');layout.update(model_id=MID,own_original_graph_SHA256=sha(path),scope='Original unweighted MobileNet operator graph; exact original ownFP16 entry instead of unrelatedBF16, all548 own F16 parameter/buffer names+shapes verified, no learned values reused',source_architecture_graph_SHA256=sha(src))
 pp=R/'configs/protocols/medran_e4b_own_original_FP16_vision_layout_v1_20261116.json';assert not pp.exists();pp.write_text(json.dumps(layout,ensure_ascii=False,indent=2),encoding='utf-8')
 proof=dict(passed=True,model_id=MID,own_original_dtype='FP16',old_weight_free_graph=str(src),old_graph_SHA256=sha(src),own_FP16_graph=str(path),own_FP16_graph_SHA256=sha(path),layout=str(pp),layout_SHA256=sha(pp),operator_nodes=14874,all_operator_edges_and_non_dtype_constants_exact=True,only_initial_input_cast_original_FP16_and_dtype_annotations_differ=True,all_intrinsic_FP32_operations_preserved=True,tensor_constants_in_graph=0,all548_own_vision_source_shapes_and_original_F16_verified=True,no_model_NPU_or_learned_CPU_inference=True)
 (H/'review.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(proof,ensure_ascii=False))
if __name__=='__main__':main()
