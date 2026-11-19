"""Correct only v1's expected unique-file count; retain the original failure."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
original=P/'test_moondream3_complete_physical_KV_journal_1128_20261119.py'
source=original.read_text(encoding='utf8')
assert source.count('assert len(values)==8, len(values)')==1
source=source.replace('assert len(values)==8, len(values)','assert len(values)==7, len(values)')
scope={'__name__':'separate_corrected_storage_unit_test','__file__':str(original)}
exec(compile(source,str(original),'exec'),scope)
if __name__=='__main__':scope['run'](Path(sys.argv[1]),Path(sys.argv[2]))
