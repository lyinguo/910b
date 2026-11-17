import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,ast,base64
P=Path(__file__).parent;oldhome='authorized_single_CoderE4_verified176_nonexample_retention_for_Deadbydawn_capacity_20261117';newhome='authorized_single_Dead177_verified178_nonexample_retention_for_Merve_complete200_capacity_20261117'
old=(P/'retain_single_CoderE4_verified176_nonexample_matrices_for_Deadbydawn_capacity_20261117.py').read_text(encoding='utf-8');new=old.replace(oldhome,newhome).replace("ALLOWED={'coder3101/gemma-4-E4B-it-qat-q4_0-unquantized-heretic'}","ALLOWED={'deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit'}").replace('only_after_verified176_full200_and_dual_publication_completed','only_after_verified178_full200_and_dual_publication_completed').replace("binding['real_record_count']==176","binding['real_record_count']==178")
aud=(P/'audit_single_CoderE4_retention_all_verified176_foreign_metadata_20261117.py').read_text(encoding='utf-8').replace(oldhome,newhome)
rows=json.loads((P/'R_recent_verified_run_matrix_capacity_Prithiv_binding_20261117.json').read_bytes())['recent_verified_runs'];record,=[r for r in rows if r['model_id']=='deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit'];assert len(record['homes'])==1
config=dict(only_after_verified178_full200_and_dual_publication_completed=True,no_live_or_failed_or_small_sample_arrays_in_scope=True,scopes=[dict(model_id=record['model_id'],home=next(iter(record['homes'])),kind='sample_serial',proof=record['proof']['independent_verification'],proof_SHA256=record['proof']['proof_sha256'])])
RS='scripts/retain_Dead177_verified178_nonexample_matrices_for_Merve_capacity_20261117.py';AS='scripts/audit_Dead177_retention_all_verified178_foreign_metadata_20261117.py';CS='configs/retention/Dead177_verified178_nonexample_for_Merve_complete200_20261117.json';files={}
for n,s in [(RS,new),(AS,aud)]:
 ast.parse(s);raw=s.encode();(P/Path(n).name).write_bytes(raw);files[n]=dict(data=base64.b64encode(raw).decode(),SHA256=hashlib.sha256(raw).hexdigest())
raw=(json.dumps(config,ensure_ascii=False,indent=2)+'\n').encode();files[CS]=dict(data=base64.b64encode(raw).decode(),SHA256=hashlib.sha256(raw).hexdigest());(P/Path(CS).name).write_bytes(raw)
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,os,subprocess
R=Path('/root/projects/comperror');os.chdir('/root');F=__F__
for n,v in F.items():
 p=R/n;assert not p.exists();raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
r=subprocess.run(['python3','-B',str(R/'__RS__'),'plan',str(R/'__CS__'),'__CH__'],cwd='/root',capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));print(json.dumps(dict(actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,read_only_plan_no_deletion=True,declared_source_SHA256={n:v['SHA256'] for n,v in F.items()}),ensure_ascii=False));raise SystemExit(r.returncode)
'''.replace('__F__',repr(files)).replace('__RS__',RS).replace('__CS__',CS).replace('__CH__',files[CS]['SHA256']);ast.parse(remote);(P/'deploy_Dead177_verified178_retention_readonly_plan_before_Merve_resume_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:v['SHA256'] for n,v in files.items()},indent=2))
