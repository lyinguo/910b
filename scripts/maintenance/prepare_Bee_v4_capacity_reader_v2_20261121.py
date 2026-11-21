from pathlib import Path
V=Path(__file__).resolve().parent
p=V/'read_Bee_v4_wait_actual_capacity_and_existing_dedup_sources_20261121_remote.py'
s=p.read_text('utf8')
a="names=subprocess.run(['rg','--files',str(R/'scripts')],capture_output=True,text=True,timeout=20)"
assert a in s
s=s.replace(a,"names=[str(p) for p in (R/'scripts').rglob('*.py')]").replace('for n in names.stdout.splitlines()','for n in names')
(V/'read_Bee_v4_wait_actual_capacity_and_existing_dedup_sources_v2_20261121_remote.py').write_text(s,encoding='utf8',newline='\n')
