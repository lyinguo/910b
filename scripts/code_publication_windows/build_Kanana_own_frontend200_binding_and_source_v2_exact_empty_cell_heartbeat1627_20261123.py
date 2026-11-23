"""Build exact current dual-book binding and stage only useful new source."""
from pathlib import Path
import ast,base64,hashlib,json,re,zipfile
import xml.etree.ElementTree as ET
V=Path(__file__).resolve().parent
MID='kakaocorp/kanana-1.5-v-3b-instruct'
R='/root/projects/comperror'
def sha(b):return hashlib.sha256(b).hexdigest()
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
raw=book.read_bytes();st=book.stat();assert sha(raw)=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
    strings=[''.join(q.itertext()) for q in ET.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else []
    rows=[]
    for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('x:sheetData/x:row',ns):
        cells={}
        for c in row:
            t=c.find('x:v',ns);t=(t.text or '') if t is not None else ''
            if c.attrib.get('t')=='s':t=strings[int(t)]
            elif c.attrib.get('t')=='inlineStr':t=''.join(c.find('x:is',ns).itertext())
            cells[re.sub(r'\d+','',c.attrib.get('r',''))]=t
        if cells.get('C')==MID:rows.append(dict(row=int(row.attrib['r']),cells=cells))
assert len(rows)==1 and rows[0]['cells']['B']=='image-text-to-text' and rows[0]['cells'].get('G','')==''
q=dict(local_book=dict(SHA256=sha(raw),bytes=len(raw),mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns),local_row=rows[0])
pn='Kanana_own_frontend200_current_dual_binding_v1_plan_heartbeat1627_20261123.json'
qb=(json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode();(V/pn).write_bytes(qb)
mn='prepare_kanana_own_original_BF16_dedicated_frontend200_v1_heartbeat1627_20261123.py'
sources={'scripts/'+mn:(V/mn).read_bytes(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_bytes()}
for b in sources.values():ast.parse(b)
payload={p:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for p,b in sources.items()}
(V/'Kanana_own_frontend200_v1_primary_source_SHA_heartbeat1627_20261123.json').write_text(json.dumps({p:sha(b) for p,b in sources.items()},indent=2),encoding='utf8')
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,os,subprocess\nR=Path('/root/projects/comperror');payload=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,z in payload.items():\n b=base64.b64decode(z['base64']);assert hashlib.sha256(b).hexdigest()==z['SHA256'];ast.parse(b);p=R/n;p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\np=R/'workspace/diagnostics/batch_910b/heartbeat_handoff_plans'/"+repr(pn)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(qb))+"\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/"+mn+"'),str(p)],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier);(V/'stage_Kanana_own_frontend200_v1_heartbeat1627_20261123_remote.py').write_text(carrier,encoding='utf8')
print(json.dumps(dict(plan_SHA256=sha(qb),current_C_unique=rows[0],source_SHA256={p:sha(b) for p,b in sources.items()}),ensure_ascii=False))
