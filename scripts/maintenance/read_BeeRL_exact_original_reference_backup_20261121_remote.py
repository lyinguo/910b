import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
expected='364cf16d7fca909bafa4c3fef0c042c595a9be7f6bdcea236d66c1278fb4e87b'
checked=[];found=[]
for p in B.glob('*/before_verified_write/*.xlsx'):
 if p.name!='C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx':continue
 h=hashlib.sha256(p.read_bytes()).hexdigest();checked.append({'path':str(p),'SHA256':h})
 if h==expected:found.append(str(p));break
assert found,'No exact immutable original reference backup; no replacement reference allowed'
print(json.dumps({'passed':True,'expected_SHA256':expected,'exact_backup':found[0],'checked_metadata':checked},ensure_ascii=False))
