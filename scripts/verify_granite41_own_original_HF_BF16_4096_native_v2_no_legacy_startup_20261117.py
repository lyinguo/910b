"""Original full numeric verifier plus explicit inactive legacy startup guard."""
import sys,json,contextlib
sys.dont_write_bytecode=True
from pathlib import Path
from verify_granite41_own_original_HF_BF16_4096_native_v1_20261117 import verify as original_verify,js,sha
def verify(folder,count):
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line['legacy_unrelated_startup_hook_disabled'] and line['original_HF_no_vllm_spawn_environment']
 value=original_verify(folder,count);value.update(independent_verifier_SHA256=sha(__file__),legacy_startup_mock_branch_not_used=True);return value
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):proof=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 print(json.dumps(proof,ensure_ascii=False,allow_nan=False))
