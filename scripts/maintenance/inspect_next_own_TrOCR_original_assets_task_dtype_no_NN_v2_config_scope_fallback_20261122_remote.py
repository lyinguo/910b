"""Inspect only own pending OCR assets/deployment; do not borrow a vocabulary."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,zipfile,xml.etree.ElementTree as ET
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
targets=['DunnBC22/trocr-base-printed_captcha_ocr','dh-unibe/trocr-kurrent'];state=json.loads((D/'state.json').read_bytes())
try:paths=subprocess.run(['rg','--files','configs'],cwd=R,capture_output=True,text=True,check=True).stdout.splitlines()
except FileNotFoundError:paths=[str(p.relative_to(R)) for p in (R/'configs').rglob('*') if p.is_file() and p.suffix in ['.yaml','.yml']]
out=[]
for mid in targets:
 row=state['models'][mid];w=Path(row['weight_paths'][0]);assets={};listing=[]
 try:
  for p in sorted(w.iterdir()):
   if not p.is_file():listing.append(dict(name=p.name,is_file=False));continue
   s=p.stat();listing.append(dict(name=p.name,bytes=s.st_size))
   if p.suffix in ['.json','.txt','.model','.py'] and s.st_size<3*2**20:
    b=p.read_bytes();entry=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    if p.suffix in ['.json','.py']:
     try:entry['value']=json.loads(b)
     except ValueError:entry['value']=b.decode('utf8',errors='replace')
    assets[p.name]=entry
 except OSError as exc:listing.append(dict(actual_read_error=repr(exc)))
 configs={n:dict(SHA256=sha(R/n),complete_text=(R/n).read_text()) for n in paths if mid.replace('/','--') in Path(n).name and Path(n).suffix in ['.yaml','.yml']}
 out.append(dict(model_id=mid,own_weight_root=str(w),original_state_metadata=row,own_root_listing=listing,own_small_assets=assets,matching_source_configs=configs))
book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';assert sha(book)=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878';ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};rows=[]
with zipfile.ZipFile(book) as z:
 ss=[]
 if 'xl/sharedStrings.xml' in z.namelist():ss=[''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
 for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('m:sheetData/m:row',ns):
  cells={}
  for c in row.findall('m:c',ns):
   value=c.find('m:v',ns);v=value.text if value is not None else ''.join(c.find('m:is',ns).itertext()) if c.find('m:is',ns) is not None else None
   if c.get('t')=='s' and v is not None:v=ss[int(v)]
   cells[''.join(k for k in c.get('r') if k.isalpha())]=v
  if cells.get('C') in targets:rows.append(dict(physical_row=row.get('r'),cells=cells))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),own_candidates=out,actual_C_unique_workbook_rows=rows,no_model_or_CPU_learned_or_NPU_execution=True,no_original_assets_or_config_mutation=True,no_download_or_foreign_vocabulary=True),ensure_ascii=False))
