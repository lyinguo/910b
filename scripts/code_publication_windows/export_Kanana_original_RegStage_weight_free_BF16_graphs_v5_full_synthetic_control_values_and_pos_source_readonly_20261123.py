"""Export original RegStage operators, never checkpoint values or a model score.

Runs only in the existing original A100 Python environment. All arithmetic is
CPU evaluation of freshly initialized synthetic parameters. The graph has one
explicit input for every tensor and cannot contain a learned tensor constant.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,struct,time
EXPECTED_HEADER_SHA256 = {'model-00001-of-00002.safetensors': 'fd492152a75ca5a60eb3e5b8a4111cfcb321a051ecc941a6e18007450de225d8', 'model-00002-of-00002.safetensors': '8c4c63ad5d2efd4e53db924123e36cc08fde14fec7b3f17335315b865c195724'}
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/KananaVForConditionalGeneration/kakaocorp/kanana-1.5-v-3b-instruct')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert sha(W/'config.json')=='500c82ccfd5f8368c7a4dbbf3af13139a379db853a21a85c42df42f127d81e52'
    assert sha(W/'model.safetensors.index.json')=='930f14ada407b77f2404b79ad5dcbdc30e5dd3544ed4ddf78e821e67108d190c'
    headers={}
    assert len(EXPECTED_HEADER_SHA256)==2
    for name,h in EXPECTED_HEADER_SHA256.items():
        with (W/name).open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0];assert 0<n<64*1024*1024;b=f.read(n)
        assert hashlib.sha256(b).hexdigest()==h
        headers.update({k:v for k,v in json.loads(b).items() if k!='__metadata__'})
    assert len(headers)==739 and all(x['dtype']=='BF16' for x in headers.values())
    import torch,timm
    from timm.models.regnet import RegStage
    from timm.layers import LayerNorm2d
    from timm.layers.fast_norm import is_fast_norm
    assert not is_fast_norm()
    torch.set_num_threads(4);torch.manual_seed(11312026)
    torch._C._set_graph_executor_optimize(False)
    def bits(t):return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    def digest(t):return hashlib.sha256(bits(t)).hexdigest()
    stages=[];start=time.monotonic()
    for name,in_channels in [('abstractor.net.0',1280),('abstractor.net.2',4096)]:
        stage=RegStage(2,in_channels,1024,stride=1,dilation=1,act_layer=torch.nn.SiLU,norm_layer=LayerNorm2d).eval().to(dtype=torch.bfloat16)
        state=stage.state_dict()
        expected={k.removeprefix(name+'.'):v for k,v in headers.items() if k.startswith(name+'.')}
        assert set(state)==set(expected),(name,set(state)-set(expected),set(expected)-set(state))
        for k,t in state.items():assert list(t.shape)==expected[k]['shape'] and t.dtype==torch.bfloat16
        shape=(1,in_channels,4,6);example=torch.randn(shape,dtype=torch.bfloat16)
        with torch.inference_mode():traced=torch.jit.trace(stage,example,check_trace=False,strict=True)
        graph=traced.inlined_graph.copy()
        inline_counts=[]
        for inline_round in range(32):
            remaining=sum(n.kind()=='prim::CallMethod' for n in graph.nodes());inline_counts.append(remaining)
            if remaining==0:break
            torch._C._jit_pass_inline(graph)
        assert all(n.kind()!='prim::CallMethod' for n in graph.nodes()),(inline_counts,str(graph))
        original_inputs=list(graph.inputs());assert len(original_inputs)==2
        resolved={original_inputs[0]:('',traced)}
        lifted={};values=[];parameters=[];module_nodes=[]
        for node in list(graph.nodes()):
            if node.kind()!='prim::GetAttr':continue
            parent,=list(node.inputs());assert parent in resolved
            parent_path,parent_object=resolved[parent];attribute=node.s('name')
            key=parent_path+'.'+attribute if parent_path else attribute
            value=getattr(parent_object,attribute)
            if isinstance(value,torch.Tensor):
                assert key in state and key in expected
                assert list(value.shape)==expected[key]['shape'] and value.dtype==torch.bfloat16 and torch.equal(value,state[key])
                if key not in lifted:
                    arg=graph.addInput('own_parameter_'+str(len(values)));arg.setType(node.output().type());lifted[key]=arg
                    values.append(value)
                    parameters.append(dict(own_state_key=name+'.'+key,relative_key=key,shape=list(value.shape),dtype=str(value.dtype),synthetic_control_SHA256=digest(value),complete_synthetic_parameter_raw_base64=base64.b64encode(bits(value)).decode()))
                node.output().replaceAllUsesWith(lifted[key]);node.destroy()
            else:
                assert hasattr(value,'_c') or isinstance(value,torch.nn.Module),(key,type(value))
                resolved[node.output()]=(key,value);module_nodes.append(node)
        for node in reversed(module_nodes):
            assert not list(node.output().uses());node.destroy()
        assert not list(original_inputs[0].uses());graph.eraseInput(0)
        assert len(list(graph.inputs()))==1+len(values)
        assert set(x['relative_key'] for x in parameters)==set(state)
        def inspect_nodes(block):
            result=[]
            for node in block.nodes():
                assert node.kind()!='prim::GetAttr'
                if node.kind()=='prim::Constant' and node.hasAttribute('value'):assert node.kindOf('value')!='t','A tensor constant would violate parameterization'
                result.append(node.kind())
                for b in node.blocks():result.extend(inspect_nodes(b))
            return result
        operators=inspect_nodes(graph)
        fn=torch._C._create_function_from_graph('own_Kanana_RegStage_complete_parameterized_'+name.replace('.','_'),graph)
        controls=[]
        for h,w in [(4,6),(6,4),(8,8)]:
            x=torch.randn((1,in_channels,h,w),dtype=torch.bfloat16)
            with torch.inference_mode():native=stage(x);actual=fn(x,*values)
            assert torch.equal(native,actual),(name,h,w,(native.float()-actual.float()).abs().max())
            assert torch.isfinite(native).all() and native.dtype==torch.bfloat16
            controls.append(dict(input_shape=list(x.shape),output_shape=list(actual.shape),dtype='torch.bfloat16',
                all_raw_bits_identical=True,input_SHA256=digest(x),native_SHA256=digest(native),graph_SHA256=digest(actual),
                complete_synthetic_input_raw_base64=base64.b64encode(bits(x)).decode(),complete_synthetic_output_raw_base64=base64.b64encode(bits(actual)).decode()))
        ir=str(graph)
        stages.append(dict(name=name,input_channels=in_channels,output_channels=1024,depth=2,original_eval_only=True,
            graph_SHA256=hashlib.sha256(ir.encode()).hexdigest(),complete_weight_free_IR=ir,parameter_inputs=parameters,
            original_state_keys=sorted(state),tensor_constants_in_graph=0,all_expected_own_parameter_names_shapes_dtypes_match=True,
            CPU_synthetic_controls=controls,operator_kinds=operators,actual_inline_CallMethod_counts=inline_counts))
    external={}
    for name,module in list(sys.modules.items()):
        if not name.startswith('timm.'):continue
        f=getattr(module,'__file__',None)
        if f and Path(f).is_file() and Path(f).suffix=='.py':external[f]=sha(f)
    pos_path=Path(sys.modules[RegStage.__module__].__file__).parents[1]/'layers/pos_embed.py';pos_bytes=pos_path.read_bytes()
    result=dict(original_pos_embed_source=dict(path=str(pos_path),SHA256=hashlib.sha256(pos_bytes).hexdigest(),complete_source_text=pos_bytes.decode('utf8')),passed=True,UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
        model_id='kakaocorp/kanana-1.5-v-3b-instruct',torch_version=torch.__version__,timm_version=timm.__version__,
        original_RegStage_source_SHA256=sha(Path(sys.modules[RegStage.__module__].__file__)),
        original_LayerNorm2d_source_SHA256=sha(Path(sys.modules[LayerNorm2d.__module__].__file__)),
        all_imported_timm_external_source_SHA256=external,stages=stages,
        no_checkpoint_tensor_values_read=True,no_learned_CPU_GPU_NPU_forward=True,no_formal_model_acceptance=True,
        CPU_synthetic_operator_controls_only=True,no_package_install_download_or_A100_files_written=True,
        whole_checkpoint_value_identity_or_historical_A100_execution_not_claimed=True)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
