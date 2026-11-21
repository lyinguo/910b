"""Resume only the unfinished capacity/first200 stage; reuse actual completed full storage NF."""
from pathlib import Path,PurePosixPath
import ast,json,hashlib,base64
V=Path(__file__).resolve().parent;R=PurePosixPath('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sha=lambda b:hashlib.sha256(b).hexdigest()
old='continue_BeeSFT_existing_true3_full_LZMA_NF_capacity_after_Ovis_dual_serial_v2_POSIX_20261121.py';new='continue_BeeSFT_completed_full_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121.py';cfg='BeeSFT_completed_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121.json'
s=(V/old).read_text(encoding='utf8');start=s.index('def main():');tail=s[s.index(' # Every original ZIP component'):s.index("if __name__=='__main__':")]
bad="int(np.prod(r['tensor_metadata']['pixel_values']['shape']))"
assert tail.count(bad)==1
tail=tail.replace(bad,"int(np.prod([1,int(r['patches']),3,384,384]))")
tail=tail.replace("if key in ['own_pixels_F32','own_pixels_BF16']:\n", "if key in ['own_pixels_F32','own_pixels_BF16']:\n    assert m['shape']==[1,int(t['P']),3,384,384] and m['dtype_bytes']==(4 if key=='own_pixels_F32' else 2)\n")
prefix='''def main():
 import numpy as np
 import batch_910b as batch
 from batch_910b_support import validate_config
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 q=js(R/'''+repr('configs/queues/'+cfg)+''');H=Path(q['home']);J=Path(q['new_job']);assert not H.exists() and not J.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert quiet();unchanged({str(R/p):h for p,h in q['source_sha256'].items()})
 prev=Path(q['reuse_completed_storage_home']);doneprev=js(prev/'actual_completion.json');assert doneprev['actual_returncode']==1 and "KeyError: 'pixel_values'" in doneprev['error'] and 'line 102, in estimate' in doneprev['error']
 assert sha(prev/'complete_storage_fullNF_proof.json')=='30927e8537358e03b28aa9bee0fa8679de2c073d84f43847685ecee80261161a';pf=js(prev/'complete_storage_fullNF_proof.json');assert pf['passed'] and pf['sample_count']==3 and pf['actual_forward_count']==6 and pf['source_tensor_count']==827 and pf['all_own_source_actual_NPU_SHA256_verified'] and pf['all_native_KV_values_losslessly_reconstructed']
 parent=js(J3/'job.json');batch.check_sources(parent['spec']);cfg=js(R/parent['spec']['config']);proto=js(cfg['model']['kwargs']['protocol_file']);alias=reference_alias(proto);put(H/'exact_old_reference_to_current184.json',alias)
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
'''
ending=s[s.index("if __name__=='__main__':"):].replace('BeeSFT_completed_true3_complete_LZMA_after_Ovis_dual_serial_v2_POSIX_20261121.json',cfg)
code=s[:start]+prefix+tail+ending;ast.parse(code);(V/new).write_text(code,encoding='utf8',newline='\n')
q=json.loads((V/'BeeSFT_completed_true3_complete_LZMA_after_Ovis_dual_serial_v2_POSIX_20261121.json').read_bytes());q.update(config=str(R/'configs/queues'/cfg),home=str(B/'BeeSFT_completed_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121'),new_job=str(B/'BeeSFT_completed_storage_NF_capacity_v3_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261121'),new_protocol='configs/protocols/bee8bsft_completed_storage_NF_capacity_v3_actual_saved_patch_axis_approved_20261121.json',new_config='configs/BeeForConditionalGeneration/Open-Bee--Bee-8B-SFT-own-original-BF16-4096-complete-LZMA-capacity-v3-actual-saved-patch-axis-serial-20261121.yaml',reuse_completed_storage_home=str(B/'BeeSFT_completed_true3_full_LZMA_NF_capacity_after_Ovis_verified184_dual_serial_v2_POSIX_20261121'))
q['source_sha256']['scripts/'+new]=sha(code.encode());(V/cfg).write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
records=[dict(path='scripts/'+new,SHA256=sha((V/new).read_bytes()),base64=base64.b64encode((V/new).read_bytes()).decode()),dict(path='configs/queues/'+cfg,SHA256=sha((V/cfg).read_bytes()),base64=base64.b64encode((V/cfg).read_bytes()).decode())]
transport='DATA='+repr(records)+'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast,os,subprocess,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for x in DATA:
 p=R/x['path'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'] and not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.py':ast.parse(b)
 with p.open('xb') as f:f.write(b)
q=json.loads((R/'configs/queues/BeeSFT_completed_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121.json').read_bytes());assert all(sha(R/p)==h for p,h in q['source_sha256'].items())
launch=D/'BeeSFT_capacity_v3_actual_saved_patch_axis_serial_launch_20261121';assert not launch.exists();launch.mkdir()
with (launch/'stdout.log').open('xb') as out,(launch/'stderr.log').open('xb') as err:
 c=subprocess.Popen(['python3','-B',str(R/'scripts/continue_BeeSFT_completed_full_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121.py')],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
time.sleep(1);print(json.dumps(dict(PID=c.pid,initial_returncode=c.poll(),home=q['home'],source_and_config_SHA256={x['path']:x['SHA256'] for x in DATA},no_completed_stage_replayed=True)))
'''
ast.parse(transport);(V/'stage_BeeSFT_capacity_v3_actual_saved_patch_axis_serial_20261121_remote.py').write_text(transport,encoding='utf8',newline='\n');print(json.dumps(dict(prepared=True,controller_SHA256=sha(code.encode()),config_SHA256=sha((V/cfg).read_bytes()),full_storage_proof_reused_without_replay=True)))
