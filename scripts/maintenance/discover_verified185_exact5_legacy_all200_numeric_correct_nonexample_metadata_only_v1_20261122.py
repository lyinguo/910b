import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,stat,re,importlib.util
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';S=R/'workspace/diagnostics/model_runs'
sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
source=R/'scripts/retain_verified179_accepted_current_nonexample_matrices_user_request_20261117.py'
module_spec=importlib.util.spec_from_file_location('unchanged_prior_accepted_retention_helpers',source);old=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(old)
ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');assert all(sha(p)==h for p,h in ptr['canonical_SHA256'].items())
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners();assert not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
state=js(D/'state.json');reg=js(D/'registry.json');assert sum(e['status']=='done' for e in state['models'].values())==185
KEEP=old.KEEP;scopes=[];skips=[];allaccepted=[]
entries=list(state['models'].items())
META='meta-llama/Llama-3.2-11B-Vision-Instruct';MH=B/'MetaMllama_completed_genuine200_formal_NF4_original_empty_invalid_schema_v2_actual_file_SHA_after_Ovis_serial_no_NN_replay_20261121';mp=MH/'verification_stdout.json'
assert sha(mp)=='c54902717b16dd42b1b02f851c02652ea77d9ad31884eae238d38dcc3c08cb13' and js(MH/'actual_completion.json')['actual_returncode']==0
meta=js(mp);assert meta['passed'] and meta['sample_count']==200 and meta['actual_forward_count']==2823 and meta['all906_own_source_cast_actual_NPU_SHA_verified'] and meta['all40_physical_KV_full_values_reconstructed_and_complete_SHA_verified_each_forward']
BINDINGS={'docling-project/SmolDocling-256M-preview': {'proof': '/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/SmolDocling_original_quoted_FP16_approved_fresh3_fixed200_v2_static_paths_1512_20261111/formal200/verification_stdout.json', 'SHA256': 'dfd3048905f1c6b631c05385f7ef1e9efc2b635e56a23a0b46dfcad7072e0c08'}, 'google/t5gemma-2-1b-1b': {'proof': '/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/T5Gemma2_original_HighTen_BF16_approved_fresh3_fixed200_v2_fixed_stage_budget_0123_20261113/formal200/verification_stdout.json', 'SHA256': '8e5e3c0b31cc42a1cce730b5a44c0ad3d7cf12f170cb43e657ca2202f934e705'}, 'ibm-granite/granite-docling-258M-mlx': {'proof': '/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/GraniteMLX_original_HF_BF16_approved_fresh3_fixed200_v3_0239_20261112/formal200/verification_stdout.json', 'SHA256': '551a66eec6e1e0d47d4f79c6334794a4be19548a657fda5eb62c498f81c3cbe2'}, 'OpenGVLab/InternVL3_5-2B-Flash': {'proof': '/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/InternVL35Flash2B_original_HF_BF16_approved_fresh3_fixed200_v4_capacity_resumed_0123_20261113/formal200/verification_stdout.json', 'SHA256': '1f24e53fd906fe31175b59a2b9d3f73066aba0c888db109f0ac31f84131aa2cf'}, 'OpenGVLab/InternVL3_5-4B-Flash': {'proof': '/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/InternVL35Flash4B_original_HF_BF16_approved_fresh3_fixed200_v3_exact_backup_capacity_resumed_0123_20261113/formal200/verification_stdout.json', 'SHA256': '300555b45801442df9f623faf727d23f090adb02d78f3a52693e4ea4d60a0e56'}}
for mid,e in entries:
 if mid not in BINDINGS:continue
 if e['status']!='done' and mid!=META:continue
 if mid==META:e=dict(e,report_dir=meta['report_dir']);pf=mp
 else:pf=old.proof_for(mid,e,reg)

 if pf is None:skips.append(dict(model_id=mid,reason='No current independent complete200 proof'));continue
 if not pf.is_file():skips.append(dict(model_id=mid,reason='Selected proof missing',proof=str(pf)));continue
 assert str(pf)==BINDINGS[mid]['proof'] and sha(pf)==BINDINGS[mid]['SHA256']
 proof=js(pf)
 assert proof.get('passed') and proof.get('sample_count')==200 and proof.get('actual_NPU_inference') and proof.get('NEW_real_NPU_inference') and not proof.get('historical_record_reuse') and not proof.get('error_indices')
 if mid=='docling-project/SmolDocling-256M-preview':
  assert proof['all471_own_source_FP16_cast_A100_and_NPU_weight_SHA256'] and proof['actual_KV_prefill_and_every_decode_delta_all30_reconstructed_to_final_SHA256']
 elif mid=='google/t5gemma-2-1b-1b':
  assert proof['all1119_own_source_cast_actual_A100_and_NPU_SHA256'] and proof['all_native_self_cross_cache_values_reconstructed_with_exact_NPU_change_checks'] and len(proof['numeric_sample_checks'])==200
 elif mid=='ibm-granite/granite-docling-258M-mlx':
  assert proof['all471_own_source_and_actual_AA_NPU_SHA256'] and proof['cache_values']=='original cacheFalse; no KV returned and full prefix every step independently verified' and len(proof['numeric_sample_checks'])==200
 else:
  assert mid in ['OpenGVLab/InternVL3_5-2B-Flash','OpenGVLab/InternVL3_5-4B-Flash'] and proof['all_native_KV_values_reconstructed_with_unchanged_prefix_checks'] and len(proof['numeric_sample_checks'])==200
 # Metadata discovery only: full original-protocol qualification is NOT changed or waived.
 allaccepted.append(dict(model_id=mid,proof=str(pf),proof_SHA256=sha(pf),passed=proof.get('passed'),sample_count=proof.get('sample_count'),report_dir=e['report_dir']))
 if not proof.get('passed') or proof.get('sample_count')!=200 or not proof.get('actual_NPU_inference') or proof.get('historical_record_reuse') or proof.get('error_indices'):
  skips.append(dict(model_id=mid,reason='Not a current passed nonhistorical error-free full200',proof=str(pf)));continue
 folder=Path(e['report_dir'])
 if not folder.is_dir() or set(proof.get('report_sha256',{}))!=old.FIVE:
  skips.append(dict(model_id=mid,reason='Full standard five proof schema not closed',proof=str(pf)));continue
 line=js(folder/'operator_lineage.json').get('actual_execution',{});home=old.own_home(line)
 if home is None or home.is_symlink():skips.append(dict(model_id=mid,reason='Unique own numerical home not closed',lineage_keys=list(line),proof=str(pf)));continue
 report=js(folder/'eval_report.json');details=report.get('detailed_results',[])
 if [x.get('index') for x in details]!=list(range(200)):
  skips.append(dict(model_id=mid,reason='Full current detailed0:199 selection not closed'));continue
 special=set().union(*(set(proof.get(k,[])) for k in ['error_indices','empty_indices','invalid_indices','truncated_indices']))
 if report['summary']['primary_metric']=='Accuracy':
  from maintenance.audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119 import preserve_and_remove_indices
  selection=preserve_and_remove_indices(details,report['pipeline_config']['algorithm']['batch_size'],special)
  own_keep=KEEP|set(selection['keep'])
 else:
  outer=report['pipeline_config']['algorithm']['batch_size'];example_groups={i//outer for i in KEEP}
  examples={i for i in range(200) if i//outer in example_groups}
  wrong={x['index'] for x in details if not x.get('raw_output') or x.get('prediction')!=x.get('reference') or x.get('error') or x.get('extra',{}).get('error')}
  own_keep=KEEP|examples|wrong|special;selection=dict(keep=sorted(own_keep),incorrect=sorted(wrong),examples=sorted(examples),remove=sorted(set(range(200))-own_keep))
 groups,trace=old.sample_group_index(line);remove=[];keep=[];unknown=[];missing=[];allidx=set();existing_bytes=0
 for name,h in proof.get('external_evidence_sha256',{}).items():
  p=Path(name)
  if p.suffix!='.npz' or not p.is_relative_to(home):continue
  indices=old.indices_for(p.relative_to(home),groups)
  if indices is None:unknown.append(name);continue
  assert indices and indices.issubset(set(range(200)));allidx.update(indices)
  if not p.exists():missing.append(name);continue
  assert isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h)
  s=p.stat();existing_bytes+=s.st_size
  row=dict(path=name,SHA256=h,bytes=s.st_size,allocated_bytes=s.st_blocks*512,model_id=mid,home=str(home),logical_indices=sorted(indices),relative=str(p.relative_to(home)),proof=str(pf),proof_SHA256=sha(pf))
  (keep if indices&own_keep or not old.removable_file(p,home) else remove).append(row)
 if missing:
  skips.append(dict(model_id=mid,reason='Prior precisely pruned series is excluded; do not widen or reverify absent arrays',proof=str(pf),missing_count=len(missing),existing_series_bytes=existing_bytes,unpruned_nonexample_files=len(remove)));continue
 if allidx!=set(range(200)):
  skips.append(dict(model_id=mid,reason='Full original logical0:199 mapping not closed',proof=str(pf),observed=sorted(allidx),existing_series_bytes=existing_bytes,unknown_NPZ=len(unknown)));continue
 if not remove:skips.append(dict(model_id=mid,reason='All remaining mapped NPZ are protected examples/nonremovable',proof=str(pf),existing_series_bytes=existing_bytes));continue
 # Recheck full current five bytes only for a real new deletion candidate.
 bad=[n for n,h in proof['report_sha256'].items() if sha(folder/n)!=h]
 if bad:skips.append(dict(model_id=mid,reason='Current report bytes differ from proof',files=bad));continue
 scopes.append(dict(model_id=mid,home=str(home),report_dir=str(folder),proof=str(pf),proof_SHA256=sha(pf),trace=str(trace) if trace else None,group_index_map={str(k):sorted(v) for k,v in groups.items()},all_wrong_invalid_empty_truncated_and_whole_groups_preserved=selection,remove=remove,keep=keep,unknown_or_nonseries_NPZ_preserved=unknown,bytes=sum(v['bytes'] for v in remove),allocated_bytes=sum(v['allocated_bytes'] for v in remove)))
# Metadata-only inventory of temporary suffixes. Never infer removability from a name.
temporary=[]
for root in [D,S]:
 for parent,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[n for n in dirs if not (Path(parent)/n).is_symlink()]
  for name in files:
   if name.endswith(('.part','.tmp','.temp','.download')):
    p=Path(parent)/name;s=p.lstat();temporary.append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns))
out=dict(UTC=datetime.now(timezone.utc).isoformat(),scope='Exact five SHA-bound legacy all200 numerical proofs only; no qualification reclassification, deletion or old-array replay',latest_user_verbatim='那就再检查服务器上已验收的中间矩阵和临时文件，完整报告、源码、配置、日志和必要数值例证保留，不再需要的清理掉，腾够空间马上串行继续跑，别一直等着。',prior_pure_helper_source=str(source),prior_pure_helper_source_SHA256=sha(source),keep_original_logical_indices=sorted(KEEP),all_wrong_empty_invalid_truncated_current_complete_groups_preserved=True,scopes=sorted(scopes,key=lambda x:x['allocated_bytes'],reverse=True),skipped=skips,all_selected_proof_metadata=allaccepted,temporary_suffix_files_preserved_pending_actual_scope_verification=temporary,canonical_SHA256={p:sha(p) for p in ptr['canonical_SHA256']},pointer_SHA256=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'),actual_free_bytes=shutil.disk_usage(R).free,candidate_bytes=sum(v['bytes'] for s in scopes for v in s['remove']),candidate_count=sum(len(s['remove']) for s in scopes),no_deletion=True)
H=B/'verified185_exact5_legacy_all200_numeric_metadata_reaudit_20261122';H.mkdir(exist_ok=False);p=H/'discovery.json';raw=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode();p.write_bytes(raw)
print(json.dumps(dict(**out,discovery_path=str(p),discovery_SHA256=hashlib.sha256(raw).hexdigest()),ensure_ascii=False))
