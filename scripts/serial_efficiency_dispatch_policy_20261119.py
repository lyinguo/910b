"""Pure scheduling decisions; no model execution, signals, or file mutations."""
def validate_policy(policy):
    assert policy['mode']=='serial' and policy['max_active_models']==1
    for name in ['prefer_smallest_complete_proven_storage_budget',
                 'immediate_after_real_NN_NF_FLOCK_release',
                 'preserve_existing_bound_legacy_resumer_no_signal_or_hot_edit',
                 'yield_if_legacy_is_already_dispatching_or_its_full_budget_passes',
                 'preserve_all_existing_math_inputs_precision_budgets_full_evidence_and_publication_gates']:
        assert policy[name] is True

def choose_action(policy,*,free_bytes,candidate_budget_bytes,active_NN_NF,lock_owners,
                  bound_legacy_resumer_alive,legacy_job_pending,legacy_budget_bytes):
    validate_policy(policy)
    assert all(isinstance(v,int) and v>0 for v in [candidate_budget_bytes,legacy_budget_bytes])
    assert isinstance(free_bytes,int) and free_bytes>=0
    if active_NN_NF or lock_owners:return 'wait_for_actual_NN_NF_FLOCK_release'
    if legacy_job_pending or (bound_legacy_resumer_alive and free_bytes>=legacy_budget_bytes):
        return 'yield_existing_bound_legacy_dispatch_then_continue_serially'
    if free_bytes>=candidate_budget_bytes:return 'start_candidate_NEWfresh3_fixed200_original_runner'
    return 'wait_for_single_model_complete_original_capacity'
