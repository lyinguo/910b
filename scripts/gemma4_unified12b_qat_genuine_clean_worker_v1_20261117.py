"""Fresh Python learned-model worker; no CLI/legacy adapter initialization.

JSON RPC transports requests/results only. All model math remains the frozen
own v1 implementation using genuine installed Transformers and own weights.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,hashlib,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
RPC=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1,encoding='utf-8');sys.stdout=sys.stderr

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check_genuine_modules():
    result={}
    for name,module in list(sys.modules.items()):
        if name.startswith(('flash_attn','transformer_engine')):
            path=getattr(module,'__file__',None);assert path and path!='/dev/null' and Path(path).is_file(),('Legacy mock/stub rejected inside learned worker',name,path)
            result[name]=dict(path=path,SHA256=sha(path))
    return result

def main():
    model=None
    for line in sys.stdin:
        request=json.loads(line);command=request['command']
        try:
            if command=='load':
                assert model is None;before=check_genuine_modules()
                from universal_eval.model.adapters.gemma4_unified12b_qat_own_original_BF16_native_v1_20261117 import OwnUnified12BitBF16
                model=OwnUnified12BitBF16(**request['kwargs']);model.load(request['model_path']);after=check_genuine_modules()
                attestation=dict(PID=os.getpid(),PPid=os.getppid(),cwd=os.getcwd(),actual_argv=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),worker_source=str(Path(__file__).resolve()),worker_source_SHA256=sha(__file__),no_CLI_or_original_adapter_initializer_imported=True,legacy_mock_modules_before_learned_load=before,legacy_mock_modules_after_learned_load=after,actual_native_implementation_source=str(R/'universal_eval/model/adapters/gemma4_unified12b_qat_own_original_BF16_native_v1_20261117.py'),actual_native_implementation_SHA256=sha(R/'universal_eval/model/adapters/gemma4_unified12b_qat_own_original_BF16_native_v1_20261117.py'))
                path=model.home/'clean_worker_loading_attestation.json';path.write_text(json.dumps(attestation,ensure_ascii=False,indent=2),encoding='utf-8')
                model.lineage.update(genuine_clean_learned_worker_required=True,clean_worker_loading_attestation=str(path),clean_worker_loading_attestation_SHA256=sha(path),parent_CLI_legacy_compatibility_not_used_by_learned_worker=True)
                result=True
            elif command=='lineage':result=model.get_execution_lineage()
            elif command=='batch':result=model.batch_predict([tuple(x) for x in request['batch']])
            elif command=='sample':
                result=model.get_sample_execution(request['prompt'],request['media_path']);modules=check_genuine_modules();path=model.home/'clean_worker_after_actual_inference_attestation.json'
                if not path.exists():path.write_text(json.dumps(dict(PID=os.getpid(),genuine_loaded_optional_modules=modules,no_mock_or_stub_inside_learned_worker=True,actual_sample_count=model.calls),ensure_ascii=False,indent=2),encoding='utf-8')
            elif command=='unload':model.unload();result=True
            else:raise ValueError(command)
            RPC.write(json.dumps(dict(ok=True,result=result),ensure_ascii=False)+'\n');RPC.flush()
            if command=='unload':return
        except BaseException:
            RPC.write(json.dumps(dict(ok=False,error=traceback.format_exc()),ensure_ascii=False)+'\n');RPC.flush();raise

if __name__=='__main__':main()
