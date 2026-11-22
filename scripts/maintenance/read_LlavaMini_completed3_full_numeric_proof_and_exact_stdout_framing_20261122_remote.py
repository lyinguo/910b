import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'LlavaMini_own_original_FP16_2048_true3_v2_actual_reserved_devices_real_FLOCK_20261122';done=json.loads((J/'completion.json').read_bytes());report=Path(done['report_dir']);out=J/'fresh3/verification_stdout.json';s=out.read_text();decoder=json.JSONDecoder();docs=[];position=0
while position<len(s):
 while position<len(s) and s[position].isspace():position+=1
 if position==len(s):break
 value,end=decoder.raw_decode(s,position);docs.append(value);position=end
proof=docs[-1];capture=Path(json.loads((report/'operator_lineage.json').read_bytes())['actual_execution']['weight_loading_evidence']).parent;proof_path=capture/'independent_report_verification.json';assert json.loads(proof_path.read_bytes())==proof and len(docs)==4
print(json.dumps(dict(completion=done,stdout_SHA256=hashlib.sha256(out.read_bytes()).hexdigest(),stdout_bytes=out.stat().st_size,progress_docs=docs[:-1],proof=proof,proof_path=str(proof_path),proof_SHA256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),report_five_current_SHA256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in report.iterdir()},capture_home=str(capture),capture_archives=[dict(path=str(p),bytes=p.stat().st_size) for p in capture.glob('groups/*/complete_actual_values.npz')]),ensure_ascii=False))
