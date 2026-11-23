"""Bind original source and unweighted operators to this model's own metadata."""
from pathlib import Path
import ast,base64,hashlib,json,re
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
own=json.loads((V/'R_Kanana_owned_source_config_headers_prior_stages_no_NN_20261123.json').read_bytes())
aa=json.loads((V/'AA_Kanana_own_RegStage_weightfree_BF16_operator_graph_v5_full_control_values_20261123.json').read_bytes())
old=json.loads((V/'AA_Kanana_own_RegStage_weightfree_BF16_operator_graph_control_v4_complete_inline_20261123.json').read_bytes())
assert aa['passed'] and aa['no_checkpoint_tensor_values_read'] and aa['no_learned_CPU_GPU_NPU_forward']
assert [(s['name'],s['graph_SHA256']) for s in aa['stages']]==[(s['name'],s['graph_SHA256']) for s in old['stages']]
pkg='universal_eval/model/kanana_own_original_graphtext_v2_20261123/'
stage={pkg+'__init__.py':b''};original_owned={}
for name,r in own['own_source_records'].items():
    p=Path(name)
    if p.suffix!='.py':continue
    b=r['complete_value'].encode('utf8');assert sha(b)==r['SHA256']
    original_owned[p.name]=dict(original_path=name,SHA256=r['SHA256'])
    if p.name=='modeling.py':
        text=b.decode();repl={
          'from timm.layers import LayerNorm, LayerNorm2d':'from .original_parameterized_RegStage_compat_20261123 import LayerNorm, LayerNorm2d',
          'from timm.layers.pos_embed import resample_abs_pos_embed':'from .original_parameterized_RegStage_compat_20261123 import resample_abs_pos_embed',
          'from timm.models.regnet import RegStage':'from .original_parameterized_RegStage_compat_20261123 import RegStage'}
        for a,z in repl.items():assert text.count(a)==1;text=text.replace(a,z)
        a=ast.parse(b);z=ast.parse(text)
        def bodies(t):return [ast.dump(n) for n in t.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        assert bodies(a)==bodies(z)
        b=text.encode()
    stage[pkg+p.name]=b
prior=json.loads((V/'AA_MedraN_original_exact_timm_source_readonly_attempt_20261123.json').read_bytes())['exact_original_source_records']
norm=next(r for p,r in prior.items() if p.endswith('/timm/layers/norm.py'))
fast=next(r for p,r in prior.items() if p.endswith('/timm/layers/fast_norm.py'))
fb=fast['complete_source_text'].encode();assert sha(fb)==fast['SHA256'];stage[pkg+'original_fast_norm_20261123.py']=fb
ns=norm['complete_source_text'];nt=ast.parse(ns)
classes=[ast.get_source_segment(ns,n) for n in nt.body if isinstance(n,ast.ClassDef) and n.name in ['LayerNorm','LayerNorm2d']]
assert len(classes)==2
ps=aa['original_pos_embed_source']['complete_source_text'];assert sha(ps.encode())==aa['original_pos_embed_source']['SHA256']
pos=next(n for n in ast.parse(ps).body if isinstance(n,ast.FunctionDef) and n.name=='resample_abs_pos_embed')
fx=json.loads((V/'AA_original_timm_FX_decorator_source_readonly_20261123.json').read_bytes());fxb=fx['complete_source_text'].encode();assert sha(fxb)==fx['SHA256'];stage[pkg+'original_timm_FX_20261123.py']=fxb
first_line=min([pos.lineno]+[d.lineno for d in pos.decorator_list]);pos_text='\n'.join(ps.splitlines()[first_line-1:pos.end_lineno])
layers='''"""Exact original active timm normalization classes and position resampling."""
from typing import List,Optional
import logging,math
import torch
from torch import nn
from torch.nn import functional as F
from .original_fast_norm_20261123 import is_fast_norm,fast_layer_norm
from .original_timm_FX_20261123 import register_notrace_function
_logger=logging.getLogger(__name__)
'''+ '\n\n'.join(classes)+'\n\n'+pos_text+'\n'
stage[pkg+'original_timm_layers_20261123.py']=layers.encode()
for n in ast.parse(layers).body:
    if isinstance(n,ast.ClassDef):assert any(ast.dump(n)==ast.dump(x) for x in nt.body if isinstance(x,ast.ClassDef))
    if isinstance(n,ast.FunctionDef):assert ast.dump(n)==ast.dump(pos)
stage[pkg+'original_parameterized_RegStage_compat_20261123.py']=(V/'kanana_original_parameterized_RegStage_compat_20261123.py').read_bytes().replace(b'kanana_own_original_RegStage_BF16_parameterized_graphs_20261123.json',b'kanana_own_original_RegStage_BF16_parameterized_graphs_v2_graphtext_20261123.json')
manifest=dict(model_id=own['model_id'],own_original_config_SHA256='500c82ccfd5f8368c7a4dbbf3af13139a379db853a21a85c42df42f127d81e52',
    own_original_index_SHA256='930f14ada407b77f2404b79ad5dcbdc30e5dd3544ed4ddf78e821e67108d190c',
    checkpoint_values_embedded=False,tensor_constants_in_graph=0,actual_original_fast_norm=False,
    original_owned_source=original_owned,original_RegStage_external_source_SHA256=aa['original_RegStage_source_SHA256'],
    original_LayerNorm2d_external_source_SHA256=aa['original_LayerNorm2d_source_SHA256'],
    all_imported_AA_timm_external_source_SHA256=aa['all_imported_timm_external_source_SHA256'],
    original_pos_embed_external_source_SHA256=aa['original_pos_embed_source']['SHA256'],
    source_only_not_model_acceptance=True,stages=[])
for s in aa['stages']:
    n=pkg+'original_RegStage'+str(s['input_channels'])+'_parameterized_BF16.ir';raw=s['complete_weight_free_IR'];assert sha(raw.encode())==s['graph_SHA256'];clean,count=re.subn(r', scope: __module[^#\n]*(?=#|$)',' ',raw,flags=re.MULTILINE);assert count==raw.count(', scope:');b=clean.encode();stage[n]=b
    parameters=[{k:p[k] for k in ['own_state_key','relative_key','shape','dtype']} for p in s['parameter_inputs']]
    manifest['stages'].append(dict(name=s['name'],input_channels=s['input_channels'],output_channels=1024,depth=2,
        graph_relative_path=n,graph_SHA256=sha(b),raw_export_graph_SHA256=s['graph_SHA256'],debug_scope_annotations_removed_only=count,parameter_inputs=parameters,original_state_keys=s['original_state_keys']))
mp='configs/protocols/kanana_own_original_RegStage_BF16_parameterized_graphs_v2_graphtext_20261123.json'
stage[mp]=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
controls=['read_Kanana_owned_original_source_config_headers_and_previous_stages_no_NN_20261123.py',
 'read_Kanana_original_canonical_adapter_and_completed_failed_import_context_20261123.py',
 'read_Kanana_original_reference_and_native_Qwen2_interfaces_no_model_20261123.py',
 'export_Kanana_original_RegStage_weight_free_BF16_graphs_readonly_CPU_synthetic_control_20261123.py',
 'export_Kanana_original_RegStage_weight_free_BF16_graphs_v2_original_inline_before_lower_readonly_CPU_synthetic_control_20261123.py',
 'export_Kanana_original_RegStage_weight_free_BF16_graphs_v3_explicit_all_parameters_readonly_CPU_synthetic_control_20261123.py',
 'export_Kanana_original_RegStage_weight_free_BF16_graphs_v4_complete_inline_explicit_parameters_readonly_CPU_synthetic_control_20261123.py',
 'export_Kanana_original_RegStage_weight_free_BF16_graphs_v5_full_synthetic_control_values_and_pos_source_readonly_20261123.py',
 'diagnose_private_TorchScript_inlining_with_tiny_synthetic_modules_readonly_20261123.py']
controls += ['read_original_timm_FX_decorator_source_readonly_20261123.py','build_Kanana_owned_private_real_RegStage_graphs_and_meta_validation_20261123.py','build_Kanana_owned_private_real_RegStage_graphs_and_meta_validation_v2_original_decorators_20261123.py','build_Kanana_owned_private_real_RegStage_graphs_and_meta_validation_v3_graphtext_20261123.py']
for n in controls:stage['scripts/code_publication_windows/'+n]=(V/n).read_bytes()
validator='verify_Kanana_private_RegStage_full_synthetic_controls_and_own739_meta_only_v2_graphtext_20261123.py'
stage['scripts/diagnostics/'+validator]=(V/validator).read_bytes()
stage['scripts/code_publication_windows/'+Path(__file__).name]=Path(__file__).read_bytes()
for n,b in stage.items():
    if n.endswith('.py'):ast.parse(b)
headers={}
for z in own['own_shard_header_records'].values():headers.update({n:m for n,m in z['complete_header'].items() if n!='__metadata__'})
assert len(headers)==739
q=dict(new_primary_source_SHA256={n:sha(b) for n,b in stage.items()},parent_control_plan='/root/projects/comperror/workspace/diagnostics/batch_910b/heartbeat_handoff_plans/Kanana_private_original_RegStage_complete_synthetic_control_and739_meta_v2_original_decorators_plan_20261123.json',parent_control_plan_SHA256='02d4ae1b5dae2dd07db8286ff7b3d53d8945b92a35d5952b2391ff759d62e26e',original_owned_source=original_owned,own_modeling_nonimport_AST_exact=True,only_graph_text_debug_scope_annotations_removed=True)
qb=json.dumps(q,ensure_ascii=False,indent=2).encode();plan='Kanana_private_original_RegStage_complete_synthetic_control_and739_meta_v3_graphtext_overlay_plan_20261123.json';(V/plan).write_bytes(qb)
(V/'Kanana_private_original_RegStage_and_control_primary_source_SHA_v3_graphtext_20261123.json').write_text(json.dumps(q['new_primary_source_SHA256'],indent=2)+'\n')
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
s="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,re,os,subprocess\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256']\n if n.endswith('.py'):ast.parse(b)\n p=R/n;p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\np=R/'workspace/diagnostics/batch_910b/heartbeat_handoff_plans'/"+repr(plan)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(qb))+"\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/diagnostics/"+validator+"'),str(p),"+repr(sha(qb))+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(s);(V/'stage_Kanana_private_real_RegStage_full_controls_and739_meta_v3_graphtext_20261123_remote.py').write_text(s,encoding='utf8')
print(json.dumps(dict(primary_source_count=len(stage),plan_SHA256=sha(qb),plan_bytes=len(qb),own_math_AST_exact=True,no_checkpoint_value_payload=True)))
