"""After Ovis dual delivery, accept full lossless Bee3 storage and continue genuine fresh3/200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import ast,copy,contextlib,fcntl,hashlib,importlib,json,math,os,shutil,subprocess,time,traceback,zipfile,re
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
MID='Open-Bee/Bee-8B-RL';J3=B/'Bee8BRL_own_original_BF16_4096_true3_v1_real_FLOCK_20261116'
OLD='universal_eval/model/adapters/bee8brl_own_native_bf16_4096_20261116.py'
MODEL='universal_eval/model/adapters/bee8brl_own_native_bf16_4096_complete_LZMA_20261121.py'
NF='scripts/verify_bee8brl_own_native_bf16_4096_original_raw_fallback_v2_20261121.py'
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
 backup=B/'GLMOCR_pair_common_baseline_verified169_G_publication_20261116/before_verified_write/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'
 assert sha(backup)==expected=='364cf16d7fca909bafa4c3fef0c042c595a9be7f6bdcea236d66c1278fb4e87b'
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
 from eager_corrective_control_safety_2203 import kernel_lock_owned,core_unchanged
 from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import copy_and_check_all_values
 q=js(R/'configs/queues/BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121.json');H=Path(q['home']);J=Path(q['new_job']);assert not H.exists() and not J.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 unchanged({str(R/p):h for p,h in q['source_sha256'].items()});assert sha(J3/'job.json')=='8ba4ff0ef3a093a1140872fbe6c88658c83a0cf7c6be2433f5eadd8e9670c566'
 oldtree=ast.parse((R/OLD).read_text());newtree=ast.parse((R/MODEL).read_text())
 mathAST=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='save_npz'}
 assert mathAST(oldtree)==mathAST(newtree)
 assert (R/NF).read_text().replace("(choice or raw)==outs[i]['prediction']","(choice or '')==outs[i]['prediction']")==(R/'scripts/verify_bee8brl_own_native_bf16_4096_20261116.py').read_text()
 previous=Path(q['previous_dual_publication_home'])
 while not (previous/'dual_publication_completed.json').is_file() or not quiet():
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_Ovis_actual_dual_publication_and_all_NN_NF_original_EX_release_not_NN',actual_NEW_model_forwards=0,max_concurrent_models=1));time.sleep(20)
 assert js(previous/'dual_publication_completed.json')['completed'] and js(previous/'full_target_dual_publication_audit.json')['passed']
 assert js(previous/'full_target_dual_publication_audit.json')['canonical_counts']['done']==184
 parent=js(J3/'job.json');done=js(J3/'completion.json');assert sha(J3/'completion.json')=='dd5249192cf697cf4658b075e8b75d095510ee6817cb8ab5b66feb0a80a8f37d' and done['actual_returncode']==done['verifier_returncode']==0
 a,=done['accepted'];assert a['sample_count']==3 and sha(a['proof'])==a['proof_sha256']=='bad8d9d4ba3d126459edb2ca921be1aaea829f0c17003301b0f3029268299996';oldproof=js(a['proof']);assert oldproof['passed'] and oldproof['actual_forward_count']==11 and oldproof['source_tensor_count']==827
 batch.check_sources(parent['spec']);cfg=js(R/parent['spec']['config']);proto=js(cfg['model']['kwargs']['protocol_file']);alias=reference_alias(proto);put(H/'exact_old_reference_to_current184.json',alias)
 protected={str(p):sha(p) for p in [Path(s) for s in parent['protected_canonical_hashes']]+[Path(proto['reference_workbook'])]}
 line=js(Path(a['report_dir'])/'operator_lineage.json')['actual_execution'];trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()];front=js(proto['own_frontend200']);assert len(front)==200 and len(trace)==3
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged();unchanged(protected)
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='complete_standard_LZMA_all_existing_genuine3_values_and_full_storage_NF_no_NN_replay',actual_NEW_model_forwards=0,original_genuine3_forwards=11))
  redirects={};templates=[];audits=[];first_stage=0
  for row in trace:
   e=js(row['actual_execution_file']);src=Path(e['archive']);assert sha(src)==e['archive_SHA256'];dest=H/('sample_%05d.npz'%row['index']);review=copy_and_check_all_values(src,dest);assert review['all_names_shapes_dtypes_and_every_raw_bit_preserved'];redirects[str(src.resolve())]=dest
   review.update(original=str(src),original_SHA256=sha(src),new=str(dest),new_SHA256=sha(dest));audits.append(review)
   members=[]
   with zipfile.ZipFile(dest) as z:
    for m in z.infolist():
     with z.open(m.filename) as f:
      version=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
     key=m.filename[:-4];raw=int(np.prod(shape))*dtype.itemsize
     if key.startswith('stage_'):first_stage+=m.compress_size
     else:members.append(dict(key=key,shape=list(shape),dtype_bytes=dtype.itemsize,raw_bytes=raw,full_compressed_bytes=m.compress_size))
   templates.append(dict(N=len(e['input_ids']),P=front[row['index']]['patches'],steps=e['actual_native_forwards'],members=members))
  put(H/'complete_storage_all_values_review.json',dict(passed=True,audits=audits,all_names_shapes_dtypes_every_raw_bit_preserved=True,no_model_math_dtype_budget_or_original_evidence_changed=True))
  module=importlib.import_module('verify_bee8brl_own_native_bf16_4096_20261116');loader=module.ReadOnlyNPZ;checker=module.check
  def check(p,h=None):
   if str(Path(p))==alias['current_book'] and h==alias['original_book_SHA256']:return checker(alias['original_exact_book_backup'],h)
   return checker(p,h)
  def storage_loader(p,*args,**kwargs):return loader(redirects.get(str(Path(p).resolve()),p),*args,**kwargs)
  module.ReadOnlyNPZ=storage_loader;module.check=check
  try:
   with (H/'storage_fullNF_stderr.log').open('x') as err,contextlib.redirect_stdout(err),contextlib.redirect_stderr(err):pf=module.verify(Path(a['report_dir']),3)
  finally:module.ReadOnlyNPZ=loader;module.check=checker
  assert pf['passed'] and pf['source_tensor_count']==827 and pf['actual_forward_count']==11 and pf['report_sha256']==oldproof['report_sha256'] and pf['all_own_source_actual_NPU_SHA256_verified'] and pf['all_native_KV_values_losslessly_reconstructed']
  for p,h in oldproof['external_evidence_sha256'].items():
   if p!=alias['current_book']:assert pf['external_evidence_sha256'].get(p)==h
  pf['new_complete_LZMA_archives_SHA256']={str(p):sha(p) for p in redirects.values()};put(H/'complete_storage_fullNF_proof.json',pf);unchanged(protected)
 # Every original ZIP component is included; empirical compressed ratios are
 # per semantic component and the original factor4+6GiB and runtime guards stay.
 desired=max(t['steps'] for t in templates);assert 2<=desired<=4096
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
 approval=dict(UTC=now(),approved=True,model_id=MID,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256'],capacity_review=str(H/'capacity_review.json'),capacity_review_SHA256=sha(H/'capacity_review.json'),complete_storage_full_NF_proof=str(H/'complete_storage_fullNF_proof.json'),complete_storage_full_NF_proof_SHA256=sha(H/'complete_storage_fullNF_proof.json'),first200_is_measurement_not_predetermined_5pct_qualification=True,small3_not_an_extra_full200_5pct_gate=True)
 put(H/'first200_measurement_approval.json',approval);unchanged(protected);assert quiet()
 source={**parent['spec']['source_sha256'],**q['source_sha256']};source.pop(parent['spec']['config'],None);source[str(Path(q['config']).relative_to(R))]=sha(q['config'])
 p2=copy.deepcopy(proto);p2['reference_workbook_SHA256']=alias['current_book_SHA256'];p2['implementation_source_SHA256'].update(source);p2['static_evidence_SHA256'].update({str(H/'exact_old_reference_to_current184.json'):sha(H/'exact_old_reference_to_current184.json'),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json'),str(H/'capacity_review.json'):sha(H/'capacity_review.json')})
 pp=R/q['new_protocol'];cc=R/q['new_config'];create(pp,p2);cfg['name']=cc.stem;cfg['model']['adapter']=MODEL+':BeeRLOwnNativeBF16';cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cc,cfg)
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
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  try:
   q=js(R/'configs/queues/BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121.json');H=Path(q['home'])
   if H.exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_reports_values_sources_configs_logs_preserved=True,no_G_or_canonical_write=True))
  finally:raise
