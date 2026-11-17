import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import json,hashlib,base64,zipfile,io,re,os,shutil,xml.etree.ElementTree as E
P=Path(__file__).parent;H=P/'CoderGemma4E4B_local_result_sync_verified176_20261117';plan=json.loads((H/'plan.json').read_bytes());target=Path(plan['target']);assert target.resolve().parent==Path('D:/code-lv').resolve()
sha=lambda b:hashlib.sha256(b).hexdigest()
old=target.read_bytes();before_stat=target.stat();assert sha(old)==plan['expected_before_sha256']==sha((H/'original_user_workbook.xlsx').read_bytes())
assert not (H/'local_publication_receipt.json').exists()
values=json.loads((H/'artifact_authored_G_values.json').read_bytes());cells={c['cell'] for c in plan['changes']};assert cells=={'G212'}
for c in plan['changes']:assert values[c['row']-2][0]==c['new_value']
q=json.loads((P/'R_Coder_verified176_server_book_for_local_CAS_20261117.json').read_bytes());new=base64.b64decode(q['book_base64'],validate=True);assert sha(new)==q['book_SHA256'] and q['server_receipt']['canonical_counts']['done']==176
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};sheet=plan['sheet_xml']
with zipfile.ZipFile(io.BytesIO(old)) as a,zipfile.ZipFile(io.BytesIO(new)) as b:
 assert a.namelist()==b.namelist() and a.comment==b.comment
 for x in a.infolist():
  y=b.getinfo(x.filename)
  for k in ('date_time','compress_type','external_attr','internal_attr','create_system','extra','comment','flag_bits','create_version','extract_version'):assert getattr(x,k)==getattr(y,k),(x.filename,k)
  if x.filename!=sheet:assert a.read(x.filename)==b.read(x.filename)
 s=a.read(sheet);t=b.read(sheet);ac={x.get('r'):E.tostring(x) for x in E.fromstring(s).findall('.//s:c',ns)};bc={x.get('r'):E.tostring(x) for x in E.fromstring(t).findall('.//s:c',ns)}
 assert {k for k in set(ac)|set(bc) if ac.get(k)!=bc.get(k)}==cells
 root=E.fromstring(t)
 for c in plan['changes']:
  x=next(x for x in root.findall('.//s:c',ns) if x.get('r')==c['cell']);assert Decimal(x.findtext('s:v',namespaces=ns))==Decimal(str(c['new_value']))
  pattern=rb'<c\b(?=[^>]*\br="'+c['cell'].encode()+rb'")[^>]*(?:/>|>.*?</c>)';s=re.sub(pattern,b'',s,flags=re.S);t=re.sub(pattern,b'',t,flags=re.S)
 assert s==t,'Any non-target raw worksheet bytes changed'
candidate=H/'server_exact_verified176_candidate.xlsx';candidate.write_bytes(new);assert sha(candidate.read_bytes())==q['book_SHA256']
backup=H/'local_CAS_immediate_backup.xlsx';shutil.copy2(target,backup);assert sha(backup.read_bytes())==sha(old)
assert sha(target.read_bytes())==sha(old) and (target.stat().st_size,target.stat().st_mtime_ns,target.stat().st_ctime_ns)==(before_stat.st_size,before_stat.st_mtime_ns,before_stat.st_ctime_ns)
tmp=target.with_name(target.name+'.verified176.tmp');assert tmp.parent.resolve()==target.parent.resolve() and not tmp.exists();tmp.write_bytes(new);assert sha(tmp.read_bytes())==q['book_SHA256'];os.replace(tmp,target);assert sha(target.read_bytes())==q['book_SHA256']
st=target.stat();receipt=dict(completed=True,UTC=datetime.now(timezone.utc).isoformat(),target=str(target),backup=str(backup),before_sha256=sha(old),after_sha256=sha(new),changed_G_cells=sorted(cells),only_target_G_changed=True,all_non_G_formulas_styles_rows_ZIP_metadata_and_members_preserved=True,all_prior175_C_B_D_G_preserved=True,artifact_tool_authored_target_numeric_values={c['cell']:c['new_value'] for c in plan['changes']},source_server_receipt_SHA256=q['server_receipt_SHA256'],fresh_stat_before=dict(size=before_stat.st_size,mtime_ns=before_stat.st_mtime_ns,ctime_ns=before_stat.st_ctime_ns),fresh_stat_after=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns))
(H/'local_publication_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
