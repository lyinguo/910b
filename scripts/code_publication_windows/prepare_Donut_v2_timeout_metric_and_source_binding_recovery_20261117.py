import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,hashlib,base64
P=Path(__file__).parent
source=(P/'verify_donut_invoice_own_native_fp16_full_head_KV_20261117.py').read_text(encoding='utf-8')
s=source.replace('def verify(folder,count):','def verify(folder,count,accept_report_metrics=True):')
old="for k,v in metrics.items():assert abs(r['metrics'][k]-v)<1.e-12,(k,r['metrics'][k],v)"
assert old in s;s=s.replace(old,"if accept_report_metrics:\n  for k,v in metrics.items():assert abs(r['metrics'][k]-v)<1.e-12,(k,r['metrics'][k],v)")
s=s.replace('return dict(passed=True,report_dir=',"return dict(passed=bool(accept_report_metrics),numeric_review_passed=True,diagnostic_only=not accept_report_metrics,original_report_metrics_preserved=r['metrics'],report_dir=")
s=s.replace('actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,','actual_NPU_inference=bool(accept_report_metrics),NEW_real_NPU_inference=bool(accept_report_metrics),historical_record_reuse=not accept_report_metrics,')
anchor=" load=js(lin['all484_own_source_actual_NPU']);"
extra=''' if accept_report_metrics:
  jobpath=Path(p['evidence_job']);job=js(jobpath);check(jobpath,sha(jobpath));assert job['model_id']==MID and job['stages']==[3] and not job['spec']['approved']
  assert all(isinstance(job['spec'][key],int) and job['spec'][key]>0 for key in ['smoke_timeout_seconds','formal_timeout_seconds','verify_timeout_seconds'])
  for name,h in job['spec']['source_sha256'].items():assert not Path(name).is_absolute() and '..' not in Path(name).parts;check(R/name,h)
  actual=Path(lin['actual_run_YAML']);assert actual.is_relative_to(R/'configs');check(actual,lin['actual_run_YAML_SHA256'])
  import yaml
  actualcfg=yaml.safe_load(actual.read_bytes())
  for key in ['task_type','sample_count','sample_range','data','model','algorithm','metrics','baseline','output']:assert actualcfg[key]==cfg[key],('Runtime YAML/report mismatch',key)
  binding=lin['caption_metric_binding'];assert binding['fixed_fixture_parity_passed'] and binding['model_forward_generate_observer_inherited_unchanged']
  for key in ['ROUGE_L_source','BLEU4_source','pure_reference_source']:check(binding[key],binding[key+'_SHA256'])
'''
assert anchor in s;s=s.replace(anchor,extra+anchor)
a=s.index("if __name__=='__main__':");s=s[:a]+'''if __name__=='__main__':
 import contextlib
 folder=Path(sys.argv[1]).resolve();count=int(sys.argv[2]);diagnostic=(len(sys.argv)>3 and sys.argv[3]=='--diagnostic-existing3-no-accept')
 assert len(sys.argv)==(5 if diagnostic else 3)
 with contextlib.redirect_stdout(sys.stderr):p=verify(folder,count,accept_report_metrics=not diagnostic)
 if diagnostic:
  dest=Path(sys.argv[4]);assert p['numeric_review_passed'] and not p['passed'] and not p['NEW_real_NPU_inference']
 else:dest=Path(js(folder/'operator_lineage.json')['actual_execution']['evidence_root'])/'independent_report_verification.json'
 assert not dest.exists();dest.write_bytes(json.dumps(p,ensure_ascii=False,allow_nan=False,indent=2).encode());print(json.dumps(p,ensure_ascii=False,allow_nan=False))
'''
nf='verify_donut_invoice_own_native_fp16_v2_source_YAML_and_exact_caption_metrics_20261117.py';ast.parse(s);(P/nf).write_bytes(s.encode())
r=(P/'prepare_Donut_invoice_own_original_FP16_native_readiness_true3_20261117.py').read_text(encoding='utf-8')
r=r.replace('DonutInvoice_own_original_FP16_poscap2048_native_v1_readiness_20261117','DonutInvoice_own_original_FP16_poscap2048_native_v2_metric_source_timeout_readiness_20261117').replace('DonutInvoice_own_original_FP16_poscap2048_true3_v1_real_FLOCK_20261117','DonutInvoice_own_original_FP16_poscap2048_true3_v2_metric_source_timeouts_real_FLOCK_20261117')
r=r.replace('from universal_eval.model.adapters.donut_invoice_own_native_fp16_20261117 import DonutInvoiceOwnFP16','from universal_eval.model.adapters.donut_invoice_own_native_fp16_v2_offline_metric_bindings_20261117 import DonutInvoiceOwnFP16MetricBindingsV2 as DonutInvoiceOwnFP16')
r=r.replace("'scripts/verify_donut_invoice_own_native_fp16_full_head_KV_20261117.py'",repr('scripts/'+nf)).replace("'scripts/prepare_Donut_invoice_own_original_FP16_native_readiness_true3_20261117.py'","'scripts/prepare_Donut_invoice_own_native_v2_metric_source_timeout_readiness_20261117.py'")
r=r.replace("source_map={n:sha(R/n) for n in sources};", "sources.extend(['universal_eval/model/adapters/donut_invoice_own_native_fp16_v2_offline_metric_bindings_20261117.py','universal_eval/metric/rouge_reference_910b.py','universal_eval/metric/text_gen.py']);source_map={n:sha(R/n) for n in sources};")
r=r.replace(" pp=R/'configs/protocols/donut_invoice_own_original_auto_FP16_poscap_native_v1_20261117.json'", " p['evidence_job']=str(J/'job.json')\n pp=R/'configs/protocols/donut_invoice_own_original_auto_FP16_poscap_native_v2_metric_source_timeouts_20261117.json'")
r=r.replace('katanaml-org--invoices-donut-model-v1-own-original-auto-FP16-poscap2048-native-v1-20261117.yaml','katanaml-org--invoices-donut-model-v1-own-original-auto-FP16-poscap2048-native-v2-metric-source-timeouts-20261117.yaml')
r=r.replace('universal_eval/model/adapters/donut_invoice_own_native_fp16_20261117.py:DonutInvoiceOwnFP16','universal_eval/model/adapters/donut_invoice_own_native_fp16_v2_offline_metric_bindings_20261117.py:DonutInvoiceOwnFP16MetricBindingsV2')
r=r.replace('model_timeout_seconds=21600,verify_timeout_seconds=21600','smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600')
r=r.replace(" batch.check_sources(job['spec']);", " assert all(isinstance(job['spec'][key],int) and job['spec'][key]>0 for key in ['smoke_timeout_seconds','formal_timeout_seconds','verify_timeout_seconds'])\n batch.check_sources(job['spec']);")
ast.parse(r);rn='prepare_Donut_invoice_own_native_v2_metric_source_timeout_readiness_20261117.py';(P/rn).write_bytes(r.encode())
names={'universal_eval/model/adapters/donut_invoice_own_native_fp16_v2_offline_metric_bindings_20261117.py':(P/'donut_invoice_own_native_fp16_v2_offline_metric_bindings_20261117.py').read_text(encoding='utf-8'),'scripts/'+nf:s,'scripts/'+rn:r}
v={}
for n,t in names.items():ast.parse(t);b=t.encode();v[n]=dict(data=base64.b64encode(b).decode(),SHA256=hashlib.sha256(b).hexdigest())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';V=__V__
for n,x in V.items():
 p=R/n;b=base64.b64decode(x['data']);assert hashlib.sha256(b).hexdigest()==x['SHA256'];ast.parse(b)
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
I=B/'DonutInvoice_v2_metric_source_timeout_recovery_readiness_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'readiness_stdout.json').open('xb') as out,(I/'readiness_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',str(R/'scripts/prepare_Donut_invoice_own_native_v2_metric_source_timeout_readiness_20261117.py')],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,staged_sources={n:x['SHA256'] for n,x in V.items()},old_failed_job_source_five_reports_and_453_forwards_untouched=True,no_new_NPU_forward=True,readiness_stdout=(I/'readiness_stdout.json').read_text(),readiness_stderr=(I/'readiness_stderr.log').read_text());(I/'actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False));raise SystemExit(p.returncode)
'''.replace('__V__',repr(v));ast.parse(remote);(P/'deploy_Donut_v2_exact_metric_timeout_source_recovery_readiness_only_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(sources={n:x['SHA256'] for n,x in v.items()},old_inference_math_and_source_v1_unchanged=True,old_453_no_accept_numeric_diagnostic_supported=True,new_required_timeout_keys_explicit=True)))
