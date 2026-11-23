"""Run meaningful source and full-value artifact tests only; no checkpoint or model evaluation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,tempfile,unittest
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
H=D/'code_publication_source_layout_20261114/daily_20261124_actual_unit_checks_no_model'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());canonical=ptr['canonical_SHA256']
assert all(sha(p)==h for p,h in canonical.items())
assert sha(R/'scripts/publish_code_daily.py')=='e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30'
H.mkdir(exist_ok=False);os.chdir(R);sys.path[:0]=[str(R),str(R/'scripts')]
with tempfile.TemporaryDirectory(prefix='code_publication_day23_unit_') as temp:
    os.environ['TMPDIR']=temp;tempfile.tempdir=temp
    suite=unittest.TestSuite();suite.addTests(unittest.defaultTestLoader.discover(str(R/'tests')))
    extra=['scripts/tests/test_complete_module_values_tree_20261122.py','scripts/tests/test_complete_standard_LZMA_NPZ_parallel_values_20261122.py']
    for i,n in enumerate(extra):
        spec=importlib.util.spec_from_file_location('daily24_explicit_artifact_unit_'+str(i),R/n)
        mod=importlib.util.module_from_spec(spec);previous_argv=sys.argv[:]
        try:
            sys.argv=[str(R/n)]
            if 'parallel_values' in n:sys.argv += [str(R/'universal_eval/utils/complete_standard_LZMA_NPZ_parallel_values_20261122.py'),str(H/'parallel_npz_complete_synthetic_records')]
            spec.loader.exec_module(mod)
        finally:sys.argv=previous_argv
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(mod))
    with (H/'unit_checks_R_actual.log').open('x',encoding='utf8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert all(sha(p)==h for p,h in canonical.items())
# Reuse the already completed new producer/independent-reader contract checks.
# Recheck every consumed source digest; do not replay meta/model/frontend data.
completed=[]
base=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
for folder,h,count in [
 ('Kanana_complete_native_observer_synthetic_values_and_real_meta_registration_tests_v1_20261123','3ea092c400e335679619c6d951f327a30d62a8bb4933a214b7129da76d76a181',14),
 ('Kanana_independent_complete_capture_and_full32KV_reconstruction_synthetic_only_v1_20261123','91dba8a2df8d919d6f0f559e2bc4341fc0607b43dac130d3735ab35d9d42ef47',12)]:
    rp=base/folder/'review.json';assert sha(rp)==h
    z=json.loads(rp.read_bytes());assert z['passed'] and z['check_count']==count
    assert z['learned_CPU_GPU_NPU_forwards']==0 and not z['formal_NEW_NPU3'] and not z['formal_fullNF_or_200_or_qualification']
    for n,digest in z['source_sha256'].items():assert sha(R/n)==digest,n
    completed.append(dict(review=str(rp),review_SHA256=h,completed_contract_checks_reused=count,all_consumed_source_SHA_fresh=True,no_replay=True))
value=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),explicit_extra_full_value_artifact_modules=extra,completed_Kanana_contract_checks_reused=completed,completed_Kanana_check_count_reused=26,exact_R_source=True,no_checkpoint_NN_CPU200_or_canonical_or_job_change=True,canonical_SHA256=canonical,log=str(H/'unit_checks_R_actual.log'),log_SHA256=sha(H/'unit_checks_R_actual.log'))
(H/'unit_checks_R_actual.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(value,ensure_ascii=False));raise SystemExit(not result.wasSuccessful())
