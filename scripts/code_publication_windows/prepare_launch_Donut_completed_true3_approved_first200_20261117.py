import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,base64,json
P=Path(__file__).parent
s=(P/'verify_donut_invoice_own_native_fp16_v4_explicit_CPU_diagnostic_role_full_linear_KV_20261117.py').read_text(encoding='utf-8')
s=s.replace("job['stages']==[3] and not job['spec']['approved']","job['stages']==[3,200] and job['spec']['approved']")
anchor="  binding=lin['caption_metric_binding'];"
assert anchor in s
s=s.replace(anchor,"""  assert p['first200_measurement_approved'] and job['spec']['alignment_approval_review']==p['first200_measurement_approval']
  approval=js(p['first200_measurement_approval']);check(p['first200_measurement_approval'],job['spec']['alignment_approval_review_sha256']);assert approval['approved'] and approval['model_id']==MID and approval['first200_is_measurement_not_predetermined_5pct_qualification'] and approval['original_A1008_vs_fixed200_final_same_range_qualification_pending']
  ap=Path(p['approved_true3_proof']);check(ap,p['approved_true3_proof_SHA256']);prior=js(ap);assert prior['passed'] and prior['NEW_real_NPU_inference'] and prior['sample_count']==3 and prior['actual_forward_count']==453 and prior['all484_own_source_actual_NPU_SHA256_verified'] and prior['all_full57566_head_rows_independently_recomputed'] and prior['all_native_self_and_cross_KV_values_losslessly_reconstructed_and_independently_projected']
"""+anchor)
nf='scripts/verify_donut_invoice_own_native_fp16_v5_approved_measurement_full_linear_KV_20261117.py';ast.parse(s);(P/Path(nf).name).write_bytes(s.encode())
controller='scripts/continue_donut_completed_true3_v3_full_numeric_capacity_approved_fresh3_fixed200_20261117.py';c=(P/Path(controller).name).read_text(encoding='utf-8');ast.parse(c)
V={n:dict(data=base64.b64encode(t.encode()).decode(),SHA256=hashlib.sha256(t.encode()).hexdigest()) for n,t in {nf:s,controller:c}.items()}
r="""import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')];V=__V__
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
for n,x in V.items():
 f=R/n;raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw)
 if f.exists():assert f.read_bytes()==raw
 else:f.write_bytes(raw)
I=B/'DonutInvoice_completed_true3_v3_first200_capacity_continuation_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.log').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',str(R/'__CONTROLLER__')],cwd=R,stdout=out,stderr=err,start_new_session=True,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,sources={n:x['SHA256'] for n,x in V.items()},completed_453_true3_not_repeated=True,first200_only_if_original_full_numeric_and_actual_capacity_pass=True,no_G_or_canonical_write=True);(I/'launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v))
""".replace('__V__',repr(V)).replace('__CONTROLLER__',controller)
ast.parse(r);(P/'launch_Donut_completed_true3_full_numeric_capacity_first200_20261117_remote.py').write_bytes(r.encode());print(json.dumps({n:x['SHA256'] for n,x in V.items()}))
