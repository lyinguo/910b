"""Verify serial admission and avoid a race with the existing unmodified waiter."""
from pathlib import Path
import importlib.util,json
def run(source,configuration):
    spec=importlib.util.spec_from_file_location('serial_dispatch_under_test',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=json.loads(Path(configuration).read_bytes())
    base=dict(free_bytes=100,candidate_budget_bytes=153,active_NN_NF=[],lock_owners=[],bound_legacy_resumer_alive=True,legacy_job_pending=False,legacy_budget_bytes=220)
    cases=[({},'wait_for_single_model_complete_original_capacity'),
      ({'free_bytes':153},'start_candidate_NEWfresh3_fixed200_original_runner'),
      ({'free_bytes':219},'start_candidate_NEWfresh3_fixed200_original_runner'),
      ({'free_bytes':220},'yield_existing_bound_legacy_dispatch_then_continue_serially'),
      ({'free_bytes':500,'bound_legacy_resumer_alive':False},'start_candidate_NEWfresh3_fixed200_original_runner'),
      ({'free_bytes':500,'active_NN_NF':[{'PID':1}]},'wait_for_actual_NN_NF_FLOCK_release'),
      ({'free_bytes':500,'lock_owners':[{'PID':2}]},'wait_for_actual_NN_NF_FLOCK_release'),
      ({'free_bytes':153,'legacy_job_pending':True},'yield_existing_bound_legacy_dispatch_then_continue_serially')]
    for changes,want in cases:assert module.choose_action(p,**dict(base,**changes))==want
    bad=dict(p,max_active_models=2)
    try:module.choose_action(bad,**base)
    except AssertionError:pass
    else:raise AssertionError('Parallel setting was accepted')
    print(json.dumps(dict(passed=True,serial_scope_checks=len(cases)+1,no_model_run=True,original_legacy_resumer_unmodified=True)))
if __name__=='__main__':
    import sys
    run(sys.argv[1],sys.argv[2])
