"""New private queue with actual POSIX paths; never hot-edit original inert v1."""
from pathlib import Path,PurePosixPath
import ast,json,hashlib,base64
V=Path(__file__).resolve().parent;R=PurePosixPath('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sha=lambda b:hashlib.sha256(b).hexdigest()
old='continue_BeeSFT_existing_true3_full_LZMA_NF_capacity_after_Ovis_dual_serial_20261121.py';new='continue_BeeSFT_existing_true3_full_LZMA_NF_capacity_after_Ovis_dual_serial_v2_POSIX_20261121.py';oldcfg='BeeSFT_completed_true3_complete_LZMA_after_Ovis_dual_serial_20261121.json';newcfg='BeeSFT_completed_true3_complete_LZMA_after_Ovis_dual_serial_v2_POSIX_20261121.json'
s=(V/old).read_text(encoding='utf8');assert s.count(oldcfg)==2;s=s.replace(oldcfg,newcfg);ast.parse(s);(V/new).write_text(s,encoding='utf8',newline='\n')
q=json.loads((V/oldcfg).read_bytes());q.update(config=str(R/'configs/queues'/newcfg),home=str(B/'BeeSFT_completed_true3_full_LZMA_NF_capacity_after_Ovis_verified184_dual_serial_v2_POSIX_20261121'),new_job=str(B/'BeeSFT_completed_true3_full_LZMA_approved_NEWfresh3_fixed200_after_Ovis_dual_serial_v2_POSIX_real_FLOCK_20261121'),previous_dual_publication_home=str(B/'OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121'),new_protocol='configs/protocols/bee8bsft_completed_true3_full_LZMA_after_Ovis_dual_approved_v2_POSIX_20261121.json',new_config='configs/BeeForConditionalGeneration/Open-Bee--Bee-8B-SFT-own-original-BF16-4096-complete-LZMA-after-Ovis-dual-serial-v2-POSIX-20261121.yaml',supersedes_inert_v1_windows_path_waiter_without_source_mutation_or_signals=True)
q['source_sha256'].pop('scripts/'+old);q['source_sha256']['scripts/'+new]=sha((V/new).read_bytes())
for k in ['config','home','new_job','previous_dual_publication_home']:assert q[k].startswith('/root/projects/comperror/') and '\\' not in q[k],k
(V/newcfg).write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
records=[dict(path='scripts/'+new,SHA256=sha((V/new).read_bytes()),base64=base64.b64encode((V/new).read_bytes()).decode()),dict(path='configs/queues/'+newcfg,SHA256=sha((V/newcfg).read_bytes()),base64=base64.b64encode((V/newcfg).read_bytes()).decode())]
transport='DATA='+repr(records)+'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,hashlib,base64,subprocess,os,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for x in DATA:
 p=R/x['path'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'];assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.py':ast.parse(b)
 with p.open('xb') as f:f.write(b)
q=json.loads((R/'configs/queues/BeeSFT_completed_true3_complete_LZMA_after_Ovis_dual_serial_v2_POSIX_20261121.json').read_bytes());assert all(sha(R/p)==h for p,h in q['source_sha256'].items())
for k in ['config','home','new_job','previous_dual_publication_home']:assert Path(q[k]).is_absolute() and Path(q[k]).is_relative_to(R) and chr(92) not in q[k]
H=Path(q['home']);assert not H.exists();launch=D/'BeeSFT_after_Ovis_dual_serial_v2_POSIX_waiter_launch_20261121';assert not launch.exists();launch.mkdir()
with (launch/'stdout.log').open('xb') as out,(launch/'stderr.log').open('xb') as err:
 c=subprocess.Popen(['python3','-B',str(R/'scripts/continue_BeeSFT_existing_true3_full_LZMA_NF_capacity_after_Ovis_dual_serial_v2_POSIX_20261121.py')],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
time.sleep(1);assert c.poll() is None
print(json.dumps(dict(passed=True,PID=c.pid,source_and_config_SHA256={x['path']:x['SHA256'] for x in DATA},home=str(H),initial_state=json.loads((H/'state.json').read_bytes()),old_v1_proven_inert_wrong_relative_windows_path_and_no_model_NN=True,no_signal_or_hot_edit=True)))
'''
ast.parse(transport);(V/'stage_BeeSFT_after_Ovis_dual_serial_v2_POSIX_20261121_remote.py').write_text(transport,encoding='utf8',newline='\n')
print(json.dumps(dict(passed=True,all_remote_paths_absolute_POSIX=True,new_source_SHA256=sha((V/new).read_bytes()),new_config_SHA256=sha((V/newcfg).read_bytes()))))
