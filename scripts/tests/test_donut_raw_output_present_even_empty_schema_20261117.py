import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[1])]
from verify_donut_invoice_existing200_v6_present_raw_field_full_numeric_diagnostic_20261117 import present_detail_raw
assert present_detail_raw({'raw_output':''})==''
assert present_detail_raw({'extra':{'raw_output':''}})==''
assert present_detail_raw({'raw_output':'real','extra':{'raw_output':'real'}})=='real'
for bad in [{},{'prediction':''},{'raw_output':None},{'raw_output':'x','extra':{'raw_output':''}}]:
 try:present_detail_raw(bad)
 except AssertionError:pass
 else:raise AssertionError('Missing or conflicting raw output was accepted')
print('7 real-presence/empty/conflict schema checks passed; no model forward')
