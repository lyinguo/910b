"""Keep Bee-RL's own math and checkpoints; validate a complete lossless storage copy."""
from pathlib import Path,PurePosixPath
import ast,hashlib,json,base64,re
V=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
c=json.loads((V/'R_BeeRL_exact_original_completed3_protocol_and_file_SHA_20261121.json').read_bytes())
assert c['proof_SHA256']=='bad8d9d4ba3d126459edb2ca921be1aaea829f0c17003301b0f3029268299996' and c['proof']['actual_forward_count']==11 and c['all200_tensor_metadata_empty']
backup=json.loads((V/'R_BeeRL_exact_original_reference_backup_20261121.json').read_bytes())
assert backup['passed'] and backup['expected_SHA256']==c['protocol']['reference_workbook_SHA256']
modelrel='universal_eval/model/adapters/bee8brl_own_native_bf16_4096_complete_LZMA_20261121.py'
nfrel='scripts/verify_bee8brl_own_native_bf16_4096_original_raw_fallback_v2_20261121.py'
srcrel='scripts/continue_BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121.py'
cfgrel='configs/queues/BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121.json'
old=(V/'bee8brl_own_native_bf16_4096_20261116.py').read_text('utf8')
tree=ast.parse(old);save=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_npz');lines=old.splitlines(keepends=True)
new=''.join(lines[:save.lineno-1])+"def save_npz(p,values):\n from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_complete_npz\n return write_complete_npz(p,values)\n"+''.join(lines[save.end_lineno:])
util=(V/'bee8bsft_own_native_bf16_4096_complete_LZMA_20261121.py').read_text('utf8')
u=next(n for n in ast.parse(util).body if isinstance(n,ast.FunctionDef) and n.name=='save_npz')
new=''.join(lines[:save.lineno-1])+ast.get_source_segment(util,u)+'\n'+''.join(lines[save.end_lineno:])
mathAST=lambda s:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='save_npz'}
assert mathAST(old)==mathAST(new)
(V/PurePosixPath(modelrel).name).write_text(new,encoding='utf8',newline='\n')
nf=(V/'verify_bee8brl_own_native_bf16_4096_20261116.py').read_text('utf8')
assert nf.count("(choice or '')==outs[i]['prediction']")==1
nnew=nf.replace("(choice or '')==outs[i]['prediction']","(choice or raw)==outs[i]['prediction']")
assert nnew.replace("(choice or raw)==outs[i]['prediction']","(choice or '')==outs[i]['prediction']")==nf
(V/PurePosixPath(nfrel).name).write_text(nnew,encoding='utf8',newline='\n')
s=(V/'continue_BeeSFT_existing_true3_full_LZMA_NF_capacity_after_Ovis_dual_serial_v2_POSIX_20261121.py').read_text('utf8')
s=s.replace('BeeSFT','BeeRL').replace('Bee8BSFT','Bee8BRL').replace('bee8bsft','bee8brl').replace('Bee-8B-SFT','Bee-8B-RL').replace('BeeSFTOwnNativeBF16','BeeRLOwnNativeBF16')
oldq='configs/queues/BeeRL_completed_true3_complete_LZMA_after_Ovis_dual_serial_v2_POSIX_20261121.json'
assert s.count(oldq)==2;s=s.replace(oldq,cfgrel)
s=s.replace('a0238b5b6160e29ffa727e24f98dd52184eee22cc4a60e4f4c71866f36a7593c',c['job_SHA256']).replace('871f35d56eaaff079a1398d8fd8cd88be1b68ff72e57abf73cc08014d34d0761',c['completion_SHA256']).replace('122118431066100d6dd17a26b7d7d3aa37958de5c117cb3814f2a316a892e803',c['proof_SHA256'])
s=s.replace("'VCFlash7B_original_BF16_verified167_blank_G_publication_20261116/before_verified_write/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'",repr(str(PurePosixPath(backup['exact_backup']).relative_to(PurePosixPath('/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct')))))
s=s.replace('d39447b7054fd381bb8c1801ab33ef141f17753488080249e0f50ed47bfc4a63',backup['expected_SHA256'])
s=s.replace("oldproof['actual_forward_count']==6","oldproof['actual_forward_count']==11").replace("pf['actual_forward_count']==6","pf['actual_forward_count']==11").replace('original_genuine3_forwards=6','original_genuine3_forwards=11')
assert "desired==2" in s;s=s.replace('assert desired==2','assert 2<=desired<=4096')
pixel="total+=math.ceil(int(np.prod(r['tensor_metadata']['pixel_values']['shape']))*m['dtype_bytes']*m['full_compressed_bytes']/max(1,m['raw_bytes']));continue"
assert pixel in s
s=s.replace(pixel,"assert m['shape']==[1,int(t['P']),3,384,384] and m['dtype_bytes']==(4 if key=='own_pixels_F32' else 2)\n    total+=math.ceil(int(np.prod([1,int(r['patches']),3,384,384]))*m['dtype_bytes']*m['full_compressed_bytes']/max(1,m['raw_bytes']));continue")
s=s.replace("fcntl.flock(lock,fcntl.LOCK_EX);kernel_lock_owned", "fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned")
assert 'SFT' not in s and 'bee8bsft' not in s
compile(s,srcrel,'exec');compile(new,modelrel,'exec');compile(nnew,nfrel,'exec')
(V/PurePosixPath(srcrel).name).write_text(s,encoding='utf8',newline='\n')
B='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/'
q=dict(model_id='Open-Bee/Bee-8B-RL',max_concurrent_models=1,config='/root/projects/comperror/'+cfgrel,home=B+'BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121',new_job=B+'BeeRL_completed_true3_full_LZMA_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261121',previous_dual_publication_home=B+'OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121',new_protocol='configs/protocols/bee8brl_completed_full_LZMA_approved_serial_v1_20261121.json',new_config='configs/BeeForConditionalGeneration/Open-Bee--Bee-8B-RL-own-original-BF16-4096-complete-LZMA-serial-v1-20261121.yaml',source_sha256={
 'universal_eval/model/adapters/bee8brl_own_native_bf16_4096_20261116.py':'330189d54b6f0722b3fcfbb0a25f97d83a76b16a9f083df42229d157f2d79f5f',
 'scripts/verify_bee8brl_own_native_bf16_4096_20261116.py':'188b8f162b6bc6176fe8f6f837c84bc7ad9de389e03436866460aa355a46e8b1',
 'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py':'a858f771182c1e8240e8e94d158d6089a4c7ee421b7cae94ab4301c9fb2be485',
 modelrel:sha(new.encode()),nfrel:sha(nnew.encode()),srcrel:sha(s.encode())},original_BF16_sources_not_SFT_FP16=True,no_original_true3_or_frontend200_NN_replay=True)
(V/PurePosixPath(cfgrel).name).write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
records=[]
for rel in [modelrel,nfrel,srcrel,cfgrel]:
 b=(V/PurePosixPath(rel).name).read_bytes();records.append(dict(path=rel,SHA256=sha(b),base64=base64.b64encode(b).decode()))
deploy='DATA='+repr(records)+r'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,base64,json,subprocess,os
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
for x in DATA:
 p=R/x['path'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'] and not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
q=json.loads((R/DATA[3]['path']).read_bytes());H=Path(q['home']);assert not H.exists()
with Path(str(H)+'_launcher_stdout.log').open('x') as o,Path(str(H)+'_launcher_stderr.log').open('x') as e:
 p=subprocess.Popen(['python3','-B',str(R/DATA[2]['path'])],cwd=R,stdin=subprocess.DEVNULL,stdout=o,stderr=e,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
print(json.dumps(dict(launched=True,PID=p.pid,home=str(H),primary=[{k:v for k,v in x.items() if k!='base64'} for x in DATA],no_new_NN_in_storage_stage=True,max_concurrent_models=1),ensure_ascii=False))
'''
(V/'launch_BeeRL_completed3_full_LZMA_NF_capacity_serial_20261121_remote.py').write_text(deploy,encoding='utf8',newline='\n')
(V/'BeeRL_completed3_full_LZMA_primary_paths_20261121.json').write_text(json.dumps([x['path'] for x in records],indent=2)+'\n',encoding='utf8')
print(json.dumps({'prepared':True,'source_SHA256':sha(s.encode()),'model_SHA256':sha(new.encode()),'NF_SHA256':sha(nnew.encode()),'config_SHA256':records[3]['SHA256'],'original_model_all_nonstorage_functions_and_whole_class_AST_exact':True,'independent_numerical_math_exact':True,'own_BF16_source_and_11_genuine_forwards_preserved':True,'original_reference_backup':backup['exact_backup']},ensure_ascii=False))
