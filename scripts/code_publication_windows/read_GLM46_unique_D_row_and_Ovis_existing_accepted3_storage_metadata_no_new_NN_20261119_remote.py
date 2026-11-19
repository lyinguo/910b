import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,xml.etree.ElementTree as ET
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
js=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
from batch_910b_support import workbook_rows
book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');rows,sheet,_=workbook_rows(book);ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:cells={e.get('r'):e.findtext('s:v',namespaces=ns) for e in ET.fromstring(z.read(sheet)).findall('.//s:c',ns)}
targets={}
for mid in ['zai-org/GLM-4.6V-Flash','AIDC-AI/Ovis-U1-3B']:
 row,=[x for x in rows if x['model_id']==mid];targets[mid]=dict(row,current_D=cells['D'+str(row['row'])])
O=B/'OvisU1_own_original_BF16_512_true3_v2_actual_empty_precision_guard_real_FLOCK_20261116';done=js(O/'completion.json');a,=done['accepted'];assert a['proof_sha256']==sha(a['proof'])=='221e4c3060bfce23733937f1679a7f89b0e58f0be148650738b99b0da016f39d';job=js(O/'job.json')
import yaml
cfg=yaml.safe_load((R/job['spec']['config']).read_bytes());proto=js(cfg['model']['kwargs']['protocol_file'])
S=B/'OvisU1_completed3_complete_standard_LZMA_NPZ_storage_diagnostic_only_20261118';diag={}
for p in S.glob('*.json'):
 if p.stat().st_size<=8*1024**2:diag[p.name]=dict(SHA256=sha(p),value=js(p))
front=Path(proto.get('own_frontend200',proto.get('frontend200','')));front_data=None
if front.is_file():
 records=[json.loads(s) for s in front.read_text().splitlines()];front_data=dict(path=str(front),SHA256=sha(front),count=len(records),first=records[0],keys=list(records[0]))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),book_SHA256=sha(book),targets=targets,own_Ovis_passed3=dict(job_path=str(O),job_SHA256=sha(O/'job.json'),accepted=a,job_metadata=job,config=cfg,protocol=proto,full_proof=js(a['proof'])),existing_LZMA_diagnostic=diag,existing_LZMA_NPZ_metadata=[dict(path=str(p),bytes=p.stat().st_size) for p in S.rglob('*.npz')],own_frontend=front_data,no_new_NN_or_CPU_learned_or_frontend_replay=True,no_file_changed=True),ensure_ascii=False))
