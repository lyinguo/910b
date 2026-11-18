"""Read-only publication gate against the user's local workbook full-precision A100 values."""
import pathlib,json,hashlib,decimal,math
D=pathlib.Path('/root/projects/comperror/workspace/diagnostics/batch_910b')
def check_local_A100_reference(model_id,task,score):
 if not isinstance(score,(int,float)) or not math.isfinite(score):raise ValueError('Nonfinite score')
 state=json.loads((D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json').read_text());p=pathlib.Path(state['local_A100_reference_snapshot']).resolve();assert p.is_relative_to(D.resolve());raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==state['local_A100_reference_snapshot_sha256'];ref=json.loads(raw)
 assert model_id in ref['models'],'Model absent from user-local target';r=ref['models'][model_id];assert task==r['task'],'User-local task mismatch';base=decimal.Decimal(r['A100_D']);assert base.is_finite() and base!=0,'Undefined A100 relative error'
 relative=abs(decimal.Decimal(str(score))-base)/abs(base)
 if relative>decimal.Decimal('.05'):raise ValueError('New result exceeds user-local full-precision A100 relative5%: '+str(relative))
 return {'model_id':model_id,'task':task,'score':score,'A100_local_full_precision':str(base),'relative_error':str(relative),'relative_error_percent':str(relative*100),'local_reference_snapshot':str(p),'local_reference_snapshot_sha256':h,'local_workbook_target':state['local_workbook_path']}
