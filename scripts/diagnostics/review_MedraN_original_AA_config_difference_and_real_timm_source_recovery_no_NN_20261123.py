"""Compare current source metadata; do not replace original FP16 or claim a score."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'MedraN_original_AA_current_config_difference_real_timm_source_recovery_no_NN_20261123'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for k in sorted(set(a)|set(b)):result+=diff(a.get(k),b.get(k),path+'.'+k if path else k)
        return result
    return [] if a==b else [dict(key=path,own_frozen_value=a,current_AA_value=b)]
def main():
    p=Path(sys.argv[1]);assert sha(p)==sys.argv[2];q=js(p)
    ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
    H.mkdir(exist_ok=False)
    protocol=js(R/'configs/protocols/medran_e4b_own_original_FP16_native_v1_20261116.json')
    W=Path(protocol['weight_path']);assets=protocol['own_runtime_assets_SHA256']
    assert sha(W/'config.json')==assets['config.json']=='5b880da08b0c34cbd69b5379882033dd0a092376fc86dd199b538ee07f74a9b7'
    own=js(W/'config.json');assert own['torch_dtype']==own['vision_config']['torch_dtype']=='float16'
    source_records={}
    for group in q['AA_exact_source_outputs'].values():
        for key in ['exact_original_source_records','complete_exact_sources']:
            for name,z in group.get(key,{}).items():
                if not z.get('exists'):continue
                b=z['complete_source_text'].encode('utf8');assert hashlib.sha256(b).hexdigest()==z['SHA256']
                if name in source_records:assert source_records[name]['SHA256']==z['SHA256'];continue
                dest=H/'external_original_source_snapshots'/name.removeprefix('/')
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
                source_records[name]=dict(SHA256=z['SHA256'],bytes=len(b),snapshot=str(dest),current_AA_readonly_source_not_installed_on_NPU=True)
    aa_group=q['AA_exact_source_outputs']['vllm']
    aa_config=next(z for n,z in aa_group['complete_exact_sources'].items() if n.endswith('/nicoboss/MedraN-E4B-Uncensored-EP7/config.json'))
    assert aa_config['SHA256']=='62c216531f3cdb02beb68e57036e38dafea011da254259736503aa684bc22733'
    aa=json.loads(aa_config['complete_source_text']);assert aa['torch_dtype']==aa['vision_config']['torch_dtype']=='bfloat16'
    aa_index=next(z for n,z in aa_group['complete_exact_sources'].items() if n.endswith('/model.safetensors.index.json'))
    assert aa_index['SHA256']==assets['model.safetensors.index.json']==sha(W/'model.safetensors.index.json')
    headers=q['AA_header_and_RegStage_source']['MedraN_exact_current_shard_headers']
    own_headers=js(B/'MedraNE4B_own_original_FP16_chat_frontend200_assets_20261116/source_tensor_headers.json')
    count=0
    for shard,z in headers.items():
        assert (W/shard).stat().st_size==z['file_bytes']==protocol['own_weight_files'][shard]['bytes']
        for name,m in z['tensor_metadata'].items():
            assert m['dtype']==own_headers[name]['dtype']=='F16' and m['shape']==own_headers[name]['shape']
            count+=1
    assert count==1676
    reg=q['AA_header_and_RegStage_source']['RegStage_original_dependency_source']
    rb=reg['complete_source_text'].encode();assert hashlib.sha256(rb).hexdigest()==reg['SHA256']
    assert 'class RegStage(' in reg['complete_source_text']
    dest=H/'external_original_source_snapshots'/reg['path'].removeprefix('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(rb)
    source_records[reg['path']]=dict(SHA256=reg['SHA256'],bytes=len(rb),snapshot=str(dest),original_RegStage_source_available_on_AA_not_yet_adapted=True)
    run=subprocess.run(['python3','-B',str(R/'scripts/maintenance/discover_verified185_current_full_accepted_correct_nonexample_metadata_only_v2_heartbeat0628_20261123.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    (H/'fresh_cleanup_discovery_stdout.json').write_text(run.stdout,encoding='utf8');(H/'fresh_cleanup_discovery_stderr.log').write_text(run.stderr,encoding='utf8')
    assert run.returncode==0,run.stderr
    cleanup=json.loads(run.stdout)
    watch=D/'immediate_disk_recovery_cleanup_watch_20261120/state.json';watch_value=js(watch)
    last=watch_value.get('last_cleanup') or {};watch_records={str(watch):dict(SHA256=sha(watch),complete_value=watch_value)}
    for k in ['request','stdout','stderr']:
        n=last.get(k)
        if n and Path(n).is_file():
            b=Path(n).read_bytes()
            try:z=json.loads(b)
            except ValueError:z=b.decode('utf8',errors='replace')
            watch_records[n]=dict(SHA256=hashlib.sha256(b).hexdigest(),complete_value=z)
    review=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),source_metadata_only=True,
        own_frozen_config_SHA256=assets['config.json'],own_frozen_dtype='float16',current_AA_config_SHA256=aa_config['SHA256'],
        current_AA_config_dtype='bfloat16',exact_config_differences=diff(own,aa),
        index_SHA256_identical=True,all1676_current_header_shapes_and_F16_match=True,
        whole_checkpoint_value_identity_not_tested_or_claimed=True,historical_A100_dtype_or_engine_not_proven_by_current_metadata=True,
        no_use_of_AA_checkpoint_or_current_config_to_replace_own_frozen_source=True,
        original_timm_and_VLLM_architecture_source_snapshots=source_records,
        original_RegStage_source_recovered_on_AA_only=True,no_timm_install_on_NPU_or_generic_stub=True,
        model_NPU_forwards=0,learned_CPU_GPU_forwards=0,new_model_acceptance=False,fullNF_acceptance=False,
        fresh_cleanup_discovery=dict(path=cleanup['discovery_path'],SHA256=cleanup['discovery_SHA256'],candidate_count=cleanup['candidate_count'],candidate_bytes=cleanup['candidate_bytes'],temporary_suffix_candidates=cleanup['temporary_suffix_files_preserved_pending_actual_scope_verification']),
        current_physical_dedup_watch_records=watch_records,no_deletion=True,no_model_source_dtype_math_epsilon_budget_threshold_change=True,
        canonical_SHA256=ptr['canonical_SHA256'])
    assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
    (H/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (H/'actual_completion.json').write_text(json.dumps(dict(UTC=review['UTC'],actual_returncode=0,source_metadata_only=True,no_model_acceptance=True))+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,UTC=review['UTC'],home=str(H),review_SHA256=sha(H/'review.json'),
        exact_config_differences=review['exact_config_differences'],all1676_current_header_shapes_and_F16_match=True,
        no_checkpoint_value_or_historical_A100_dtype_identity_claim=True,
        source_snapshots=len(source_records),RegStage_original_source_SHA256=reg['SHA256'],
        cleanup=review['fresh_cleanup_discovery'],no_deletion=True,no_NN_or_learned_CPU_GPU_forward=True),ensure_ascii=False))
if __name__=='__main__':main()
