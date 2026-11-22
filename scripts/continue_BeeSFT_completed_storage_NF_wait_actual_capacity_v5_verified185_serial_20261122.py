"""After Ovis dual delivery, accept full lossless Bee3 storage and continue genuine fresh3/200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import ast,copy,contextlib,fcntl,hashlib,importlib,json,math,os,shutil,subprocess,time,traceback,zipfile,re
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
MID='Open-Bee/Bee-8B-SFT';J3=B/'Bee8BSFT_own_original_BF16_4096_true3_v1_real_FLOCK_20261116'
OLD='universal_eval/model/adapters/bee8bsft_own_native_bf16_4096_20261116.py'
MODEL='universal_eval/model/adapters/bee8bsft_own_native_bf16_4096_complete_LZMA_20261121.py'
NF='scripts/verify_bee8bsft_own_native_bf16_4096_original_raw_fallback_v2_20261121.py'
UTILITY='universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def reference_alias(proto):
 from batch_910b_support import workbook_rows
 book=Path(proto['reference_workbook']);expected=proto['reference_workbook_SHA256']
 backup=B/'VCFlash7B_original_BF16_verified167_blank_G_publication_20261116/before_verified_write/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'
 assert sha(backup)==expected=='d39447b7054fd381bb8c1801ab33ef141f17753488080249e0f50ed47bfc4a63'
 old,_,_=workbook_rows(backup);new,_,_=workbook_rows(book);assert len(old)==len(new)==700 and [(x['row'],x['model_id'],x['task']) for x in old]==[(x['row'],x['model_id'],x['task']) for x in new]
 assert all(x['score'] is None or Decimal(x['score'])==Decimal(y['score']) for x,y in zip(old,new))
 import xml.etree.ElementTree as ET
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 def dc(p):
  with zipfile.ZipFile(p) as z:return {c.get('r'):ET.tostring(c) for c in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:c',ns) if re.fullmatch('D[0-9]+',c.get('r',''))}
 assert dc(backup)==dc(book)
 row,=[x for x in new if x['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 with zipfile.ZipFile(book) as z:baseline=ET.fromstring(z.read('xl/worksheets/sheet1.xml')).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 assert all(Decimal(v)==Decimal(baseline) for v in proto['current_A100_D_values'].values())
 return dict(passed=True,original_exact_book_backup=str(backup),original_book_SHA256=expected,current_book=str(book),current_book_SHA256=sha(book),all700_C_B_row_and_entire_D_unchanged=True,all_old_nonempty_G_preserved=True,current_row=row['row'],D=baseline)
def main():
 import numpy as np
 import batch_910b as batch
 from batch_910b_support import validate_config
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 q=js(R/'configs/queues/BeeSFT_completed_storage_NF_capacity_v5_verified185_exact137_wait_actual_capacity_serial_20261122.json');H=Path(q['home']);J=Path(q['new_job']);assert not H.exists() and not J.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert quiet();unchanged({str(R/p):h for p,h in q['source_sha256'].items()})
 prev=Path(q['reuse_completed_storage_home']);doneprev=js(prev/'actual_completion.json');assert doneprev['actual_returncode']==1 and "KeyError: 'pixel_values'" in doneprev['error'] and 'line 102, in estimate' in doneprev['error']
 assert sha(prev/'complete_storage_fullNF_proof.json')=='30927e8537358e03b28aa9bee0fa8679de2c073d84f43847685ecee80261161a';pf=js(prev/'complete_storage_fullNF_proof.json');assert pf['passed'] and pf['sample_count']==3 and pf['actual_forward_count']==6 and pf['source_tensor_count']==827 and pf['all_own_source_actual_NPU_SHA256_verified'] and pf['all_native_KV_values_losslessly_reconstructed']
 parent=js(J3/'job.json');batch.check_sources(parent['spec']);cfg=js(R/parent['spec']['config']);proto=js(cfg['model']['kwargs']['protocol_file']);alias=reference_alias(proto);put(H/'exact_old_reference_to_current185.json',alias)
 protected={str(p):sha(p) for p in [Path(s) for s in parent['protected_canonical_hashes']]+[Path(proto['reference_workbook'])]}
 original,=js(J3/'completion.json')['accepted'];assert sha(original['proof'])==original['proof_sha256']=='122118431066100d6dd17a26b7d7d3aa37958de5c117cb3814f2a316a892e803' and pf['report_sha256']==js(original['proof'])['report_sha256']
 front=js(proto['own_frontend200']);assert len(front)==200 and sha(proto['own_frontend200'])=='9c41c4f712f592719b19a1664dc1a7663fafca7b0730f62cbde19900098fe62f' and all(isinstance(r['patches'],int) and r['patches']>0 for r in front)
 assert all(r['tensor_metadata']=={} for r in front),'Original saved BatchFeature tensor_metadata contract unexpectedly changed'
 (H/'complete_storage_fullNF_proof.json').write_bytes((prev/'complete_storage_fullNF_proof.json').read_bytes())
 put(H/'completed_storage_proof_reuse.json',dict(UTC=now(),original_full_storage_NF=str(prev/'complete_storage_fullNF_proof.json'),original_full_storage_NF_SHA256=sha(prev/'complete_storage_fullNF_proof.json'),passed=True,actual_NEW_model_forwards=0,new_compression_or_fullNF_replay=False,only_unfinished_capacity_pixel_axis_binding_corrected=True,all200_pixel_shapes_from_original_saved_actual_pixel_values_shape1_patch_count=True,original_BatchFeature_flatten_metadata_empty_disclosed=True))
 put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='capacity_only_resume_from_completed_storage_NF_original_saved_patch_axis_no_NN_or_NF_replay',actual_NEW_model_forwards=0))
 review=js(prev/'complete_storage_all_values_review.json');assert review['passed'] and review['all_names_shapes_dtypes_every_raw_bit_preserved'];templates=[];first_stage=0
 line=js(Path(original['report_dir'])/'operator_lineage.json')['actual_execution'];trace=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()];assert len(trace)==3
 for row,audit in zip(trace,review['audits']):
  assert sha(audit['original'])==audit['original_SHA256'] and sha(audit['new'])==audit['new_SHA256']==pf['new_complete_LZMA_archives_SHA256'][audit['new']]
  e=js(row['actual_execution_file']);assert e['archive']==audit['original'];members=[]
  with zipfile.ZipFile(audit['new']) as z:
   for m in z.infolist():
    with z.open(m.filename) as f:
     version=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
    key=m.filename[:-4];raw=int(np.prod(shape))*dtype.itemsize
    if key.startswith('stage_'):first_stage+=m.compress_size
    else:members.append(dict(key=key,shape=list(shape),dtype_bytes=dtype.itemsize,raw_bytes=raw,full_compressed_bytes=m.compress_size))
  templates.append(dict(N=len(e['input_ids']),P=front[row['index']]['patches'],steps=e['actual_native_forwards'],members=members))
 # Every original ZIP component is included; empirical compressed ratios are
 # per semantic component and the original factor4+6GiB and runtime guards stay.
 desired=max(t['steps'] for t in templates);assert desired==2
 def estimate(r,t):
  N=len(r['input_ids']);P=r['patches'];total=0
  for m in t['members']:
   key=m['key'];shape=list(m['shape']);scale=1
   if key in ['own_pixels_F32','own_pixels_BF16']:
    assert m['shape']==[1,int(t['P']),3,384,384] and m['dtype_bytes']==(4 if key=='own_pixels_F32' else 2)
    total+=math.ceil(int(np.prod([1,int(r['patches']),3,384,384]))*m['dtype_bytes']*m['full_compressed_bytes']/max(1,m['raw_bytes']));continue
   if key=='own_visual_features':shape[0]=r['image_token_count']
   elif key in ['prefill_inputs_embeds','input_ids','own_original_HF_mask']:shape[1]=N
   elif key=='native_cache_prefill':shape[3]=N
   elif key=='native_cache_decode_delta':shape[3]=desired-1
   elif key in ['all_original_full_head_logits','all_head_inputs']:shape[0]=desired
   elif key.startswith('f'):
    match=re.match(r'f(\d+)_',key);assert match,('Unclassified member',key);step=int(match.group(1))-1;shape=[N+step if v==t['N']+step else N if v==t['N'] else v for v in shape];scale=desired/max(1,t['steps'])
   else:raise AssertionError(('Unclassified member',key))
   total+=math.ceil(int(np.prod(shape))*m['dtype_bytes']*scale*m['full_compressed_bytes']/max(1,m['raw_bytes']))+256
  return total
 estimates=[max(estimate(r,t) for t in templates) for r in front];need=(sum(estimates)+first_stage)*4+6*1024**3
 cap=dict(UTC=now(),passed=shutil.disk_usage(R).free>=need,empirical_estimate_bytes=need,actual_free_bytes=shutil.disk_usage(R).free,all200_own_original_shapes_checked=True,all_original_ZIP_components_and_per_component_compressed_shapes=templates,all200_projected_complete_bytes=estimates,original_factor4_plus6GiB_and_hard3GiB_log1GiB_precheckHalfGiB_sample2GiB_outer10_guards_preserved=True,original4096_dtype_math_EOS_and_batch_order_unchanged=True,complete_storage_full_NF_proof=str(H/'complete_storage_fullNF_proof.json'),complete_storage_full_NF_proof_SHA256=sha(H/'complete_storage_fullNF_proof.json'),empirical_not_worst_case_guarantee=True,no_original_true3_NN_replay=True)
 put(H/'capacity_review.json',cap);assert cap['passed'],'Actual original project disk below complete Bee200 LZMA budget; keep accepted new storage proof and all original evidence'
 approval=dict(UTC=now(),approved=True,model_id=MID,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=original['proof'],true3_proof_SHA256=original['proof_sha256'],capacity_review=str(H/'capacity_review.json'),capacity_review_SHA256=sha(H/'capacity_review.json'),complete_storage_full_NF_proof=str(H/'complete_storage_fullNF_proof.json'),complete_storage_full_NF_proof_SHA256=sha(H/'complete_storage_fullNF_proof.json'),first200_is_measurement_not_predetermined_5pct_qualification=True,small3_not_an_extra_full200_5pct_gate=True)
 put(H/'first200_measurement_approval.json',approval);unchanged(protected);assert quiet()
 source={**parent['spec']['source_sha256'],**q['source_sha256']};source.pop(parent['spec']['config'],None);source[str(Path(q['config']).relative_to(R))]=sha(q['config'])
 p2=copy.deepcopy(proto);p2['reference_workbook_SHA256']=alias['current_book_SHA256'];p2['implementation_source_SHA256'].update(source);p2['static_evidence_SHA256'].update({str(H/'exact_old_reference_to_current185.json'):sha(H/'exact_old_reference_to_current185.json'),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json'),str(H/'capacity_review.json'):sha(H/'capacity_review.json')})
 pp=R/q['new_protocol'];cc=R/q['new_config'];create(pp,p2);cfg['name']=cc.stem;cfg['model']['adapter']=MODEL+':BeeSFTOwnNativeBF16';cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cc,cfg)
 spec=copy.deepcopy(parent['spec']);spec.update(approved=True,config=str(cc.relative_to(R)),verifier=NF,alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));spec['source_sha256']=source;spec['source_sha256'].update({str(cc.relative_to(R)):sha(cc),str(pp.relative_to(R)):sha(pp),NF:sha(R/NF)})
 validate_config(cfg,MID);entry=js(D/'state.json')['models'][MID]
 for count in [3,200]:
  path,cfgx=batch.config_for(spec,entry,count);validate_config(cfgx,MID)
  if path.resolve().is_relative_to(R/'configs'):spec['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 batch.check_sources(spec);scope(spec,[3,200],False);unchanged(protected);assert quiet() and batch.device_idle(spec) and shutil.disk_usage(R).free>=need
 newjob=copy.deepcopy(parent);newjob.update(spec=spec,stages=[3,200],created_at=now(),protected_canonical_hashes=protected,private_stage_type='own accepted genuine3 and new full LZMA storage NF, approved NEWfresh3/fixed200 serial max1');create(J/'job.json',newjob)
 runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert sha(runner)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/'runner_stdout.log').open('x') as out,(H/'runner_stderr.log').open('x') as err:
  child=subprocess.Popen(['python3','-B',str(runner),str(J)],cwd='/root',stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='approved_own_Bee_NEWfresh3_then_fixed200_actual_original_runner',job=str(J),runner_PID=child.pid,job_SHA256=sha(J/'job.json'),max_concurrent_models=1,no_G_or_canonical_write=True));rc=child.wait()
 done=js(J/'completion.json');assert rc==done['actual_returncode'];put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,job=str(J),completion=done,no_G_or_canonical_write=True));return rc

def wait_for_actual_original_capacity():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from A100_alignment_evidence_runner_20261103 import unchanged
 q=js(R/'configs/queues/BeeSFT_completed_storage_NF_capacity_v5_verified185_exact137_wait_actual_capacity_serial_20261122.json')
 W=Path(q['wait_home']);assert not W.exists();W.mkdir()
 lock=(W/'private_waiter.guard').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source={str(R/p):h for p,h in q['source_sha256'].items()}
 canonical=q['canonical_after_verified185_and_exact137_cleanup_SHA256'];unchanged(canonical)
 pub=Path(q['previous_dual_publication_home']);assert sha(pub/'full_target_dual_publication_audit.json')=='cf97094bacfa17f27c531431c7486594e0a07b57f18c02fce19e93a6751f08f7'
 actualpub=js(pub/'full_target_dual_publication_audit.json');assert actualpub['passed'] and actualpub['canonical_counts']=={'done':185,'skipped_no_complete_weights':310,'needs_adaptation':205}
 oldwait=Path(q['previous_naturally_exited_waiter']);oldexit=js(oldwait/'actual_completion.json');assert oldexit['actual_returncode']==1 and 'Protected canonical or source changed: /root/projects/comperror/workspace/diagnostics/batch_910b/state.json' in oldexit['error']
 old=Path(q['completed_capacity_home']);cap=js(old/'capacity_review.json')
 assert cap['empirical_estimate_bytes']==q['required_free_bytes']==192555984252
 assert cap['all200_own_original_shapes_checked'] and cap['original_factor4_plus6GiB_and_hard3GiB_log1GiB_precheckHalfGiB_sample2GiB_outer10_guards_preserved']
 pf=Path(q['reuse_completed_storage_home'])/'complete_storage_fullNF_proof.json'
 assert sha(pf)=='30927e8537358e03b28aa9bee0fa8679de2c073d84f43847685ecee80261161a'
 cleanup=Path(q['cleanup_actual']);assert sha(cleanup)=='0d5c89e2143683f5891033f38d1a1d56fa0d5a8d525b7fb03c9dfd2e7cb9987e'
 a=js(cleanup);assert a['actual_returncode']==0 and a['removed_files']==137 and a['all_protected_SHA256_unchanged'] and a['canonical_scores_state_registry_and_both_workbooks_unchanged']
 assert sha(cleanup.parent/'ledger_before.original')=='b3969dd7799de7a9d9a5cfa2621681b931c734c4c6f1233734dc3f803a80fbdd'
 assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='5f20ac2b35d58b09e5c4a3a8ca6a5dacf1446b9cb39876678b24458b0f1ef4b0'
 last=None
 while True:
  unchanged(source);unchanged(canonical)
  assert not Path(q['new_job']).exists() and not Path(q['home']).exists()
  owners=runner_lock_owners();procs=project_processes()
  active=[p for p in procs if any(x.endswith('/run.py') or 'verify_' in x or 'A100_alignment_evidence_runner_' in x or '/inspect_xtuner_llava3_' in x or '/prepare_xtuner_llava3_' in x for x in p.get('argv',[]))]
  free=shutil.disk_usage(R).free
  ready=free>=q['required_free_bytes'] and not owners and not active
  put(W/'state.json',dict(UTC=now(),PID=os.getpid(),phase='capacity_wait_only_no_NN_no_fullNF_no_original_runner_lock',required_free_bytes=q['required_free_bytes'],actual_free_bytes=free,missing_bytes=max(0,q['required_free_bytes']-free),original_FLOCK=owners,actual_other_NN_NF=active,ready=ready,max_concurrent_models=1,no_original_three_or_storage_NF_replay=True,no_expansion_other_disk_cleanup_or_guard_reduction=True))
  if ready:
   put(W/'actual_capacity_recovered.json',dict(UTC=now(),actual_free_bytes=free,required_free_bytes=q['required_free_bytes'],passed=True,all_NN_NF_original_FLOCK_released=True))
   rc=main()
   put(W/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,continuation_home=q['home'],job=q['new_job'],no_G_or_canonical_write=True))
   return rc
  time.sleep(20)

if __name__=='__main__':
 try:sys.exit(wait_for_actual_original_capacity())
 except SystemExit:raise
 except BaseException:
  try:
   q=js(R/'configs/queues/BeeSFT_completed_storage_NF_capacity_v5_verified185_exact137_wait_actual_capacity_serial_20261122.json');H=Path(q['home'])
   if H.exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_reports_values_sources_configs_logs_preserved=True,no_G_or_canonical_write=True))
   W=Path(q['wait_home'])
   if W.exists() and not (W/'actual_completion.json').exists():put(W/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_evidence_preserved=True,no_G_or_canonical_write=True))
  finally:raise
