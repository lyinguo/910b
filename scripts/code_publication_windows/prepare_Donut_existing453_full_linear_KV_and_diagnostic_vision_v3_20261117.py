import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,base64,json
P=Path(__file__).parent;s=(P/'verify_donut_invoice_own_native_fp16_v2_source_YAML_and_exact_caption_metrics_20261117.py').read_text(encoding='utf-8')
s=s.replace('def numeric(actual,reference,label):','def numeric(actual,reference,label,enforce=True):')
s=s.replace(" assert relative<=.005 and maximum<=.1,(label,rms,relative,maximum,'Fixed FP16 same-operand diagnostic guard; no tuning to the score')"," if enforce:assert relative<=.005 and maximum<=.1,(label,rms,relative,maximum,'Unchanged same-operand linear guard')")
s=s.replace('return dict(label=label,RMSE=rms,normalized_RMSE=relative,max_abs=maximum)','return dict(label=label,RMSE=rms,normalized_RMSE=relative,max_abs=maximum,diagnostic_only=not enforce)')
s=s.replace("numeric(vision,cv,'own full native FP16 vision CPU diagnostic')","numeric(vision,cv,'own full native FP16 vision CPU diagnostic',enforce=False)")
s=s.replace(" assert len(sys.argv)==(5 if diagnostic else 3)"," assert diagnostic and len(sys.argv)==5,'Read-only existing3 numerical diagnostic only; never an acceptance verifier'")
n='scripts/diagnostic_donut_existing453_all_head_KV_CPU_vision_only_v3_20261117.py';ast.parse(s);(P/Path(n).name).write_bytes(s.encode());h=hashlib.sha256(s.encode()).hexdigest()
v=dict(name=n,data=base64.b64encode(s.encode()).decode(),SHA256=h)
old=(P/'launch_Donut_existing453_full_value_no_accept_numeric_diagnostic_20261117_remote.py').read_text(encoding='utf-8')
old=old.replace('DonutInvoice_failed_runner_453_completed_NPU3_full_value_readonly_no_accept_numeric_diagnostic_v2_20261117','DonutInvoice_existing453_full_linear_KV_CPU_vision_diagnostic_only_v3_20261117').replace('run_Donut_existing453_full_numeric_no_accept_actual_20261117.py','run_Donut_existing453_full_linear_KV_vision_diagnostic_only_v3_actual_20261117.py').replace('scripts/verify_donut_invoice_own_native_fp16_v2_source_YAML_and_exact_caption_metrics_20261117.py',n).replace('d1130454f3b7328f8f1b65fc2be76573788ba5b590a60ade0a77a0a7a6ed8a28',h)
install="V="+repr(v)+"\nf=R/V['name'];raw=__import__('base64').b64decode(V['data']);assert hashlib.sha256(raw).hexdigest()==V['SHA256'];__import__('ast').parse(raw)\nif f.exists():assert f.read_bytes()==raw\nelse:f.write_bytes(raw)\n"
pos=old.index('from A100_alignment_incremental_monitor_');old=old[:pos]+install+old[pos:];ast.parse(old);(P/'launch_Donut_existing453_full_linear_KV_vision_diagnostic_v3_20261117_remote.py').write_bytes(old.encode());print(json.dumps(dict(diagnostic_source=n,SHA256=h,original_453_not_rerun_or_accepted=True,all_linear_head_KV_input_source_mask_position_checks_unchanged=True,CPU_vision_difference_collected_not_used_as_A100_score_or_extra_gate=True)))
