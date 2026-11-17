import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,ast,base64
P=Path(__file__).parent
old=(P/'retain_four_complete_verified_runs_nonexample_matrices_for_PrithivE4_capacity_v1_20261117.py').read_text(encoding='utf8')
new=old.replace('four owned runs','one owned run').replace("authorized_four_complete_verified_runs_nonexample_retention_v1_original_examples_for_PrithivE4_capacity_20261117","authorized_single_CoderE4_verified176_nonexample_retention_for_Deadbydawn_capacity_20261117")
start=new.index('ALLOWED=');end=new.index('\nFIVE=',start);new=new[:start]+"ALLOWED={'coder3101/gemma-4-E4B-it-qat-q4_0-unquantized-heretic'}"+new[end:]
rel='scripts/retain_single_CoderE4_verified176_nonexample_matrices_for_Deadbydawn_capacity_20261117.py'
audit=(P/'audit_four_retention_all_verified176_foreign_metadata_20261117.py').read_text(encoding='utf8').replace('authorized_four_complete_verified_runs_nonexample_retention_v1_original_examples_for_PrithivE4_capacity_20261117','authorized_single_CoderE4_verified176_nonexample_retention_for_Deadbydawn_capacity_20261117')
arel='scripts/audit_single_CoderE4_retention_all_verified176_foreign_metadata_20261117.py'
items={rel:new.encode(),arel:audit.encode()}
for n,s in items.items():ast.parse(s);(P/Path(n).name).write_bytes(s)
payload={n:dict(data=base64.b64encode(s).decode(),SHA256=hashlib.sha256(s).hexdigest()) for n,s in items.items()}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';ITEMS=__ITEMS__
for rel,item in ITEMS.items():
 raw=base64.b64decode(item['data']);assert hashlib.sha256(raw).hexdigest()==item['SHA256'];ast.parse(raw);p=R/rel
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
mid='coder3101/gemma-4-E4B-it-qat-q4_0-unquantized-heretic';reg=js(D/'registry.json');state=js(D/'state.json');proof=Path(reg[mid]['accepted_formal200_proof']);p=js(proof)
assert state['models'][mid]['status']=='done' and sha(proof)==reg[mid]['accepted_formal200_proof_sha256']=='6b2ded411703333b22a0566f09633f8b88f588ce80c68e08a39176ebea55cb24'
assert p['passed'] and p['sample_count']==200 and p['original_protocol_final_qualification_complete'] and p['formal_alignment_within_5pct'] and p['all_native_KV_values_losslessly_reconstructed'] and p['all_full_head_rows_independently_recomputed']
I=B/'single_CoderE4_verified176_retention_for_Deadbydawn_exact_inputs_actual_plan_launch_20261117';I.mkdir(exist_ok=False)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),only_after_verified176_full200_and_dual_publication_completed=True,no_live_or_failed_or_small_sample_arrays_in_scope=True,scopes=[dict(model_id=mid,proof=str(proof),proof_SHA256=sha(proof),home='/root/projects/comperror/workspace/diagnostics/model_runs/CoderGemma4E4B_own_original_BF16_20261117_024248_713108_1441945',kind='sample_serial')],reason='Restore capacity for the already passed own Deadbydawn3/419; only own complete qualified CoderE4 nonexample NPZ, keep original 0:9/90:99/190:199 groups and all full reports/records/code/YAML/logs. No other run or failed/unverified array in scope.')
f=I/'reviewed_inputs.json';f.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
with (I/'plan_stdout.json').open('xb') as out,(I/'plan_stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B','scripts/retain_single_CoderE4_verified176_nonexample_matrices_for_Deadbydawn_capacity_20261117.py','plan',str(f),sha(f)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
result=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,reviewed_input=str(f),input_SHA256=sha(f),scope_models=[mid],sources={n:x['SHA256'] for n,x in ITEMS.items()},read_only_plan_only_no_deletion=True);(I/'actual_launch.json').write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode());print(json.dumps(result,ensure_ascii=False))
'''.replace('__ITEMS__',repr(payload));ast.parse(remote);(P/'launch_single_CoderE4_verified176_retention_readonly_plan_for_Deadbydawn_capacity_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))
