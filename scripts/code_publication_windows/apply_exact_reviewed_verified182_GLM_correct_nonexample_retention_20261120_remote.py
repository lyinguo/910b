import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
assert hashlib.sha256((R/'scripts/maintenance/retain_verified182_GLM_accepted_correct_nonexamples_continue_Single_serial_20261120.py').read_bytes()).hexdigest()=='bc494d076a8ba99a97d381095d282829e8e640fa18760e9a309f81c0667e9902'
from maintenance import retain_verified182_GLM_accepted_correct_nonexamples_continue_Single_serial_20261120 as m
print(json.dumps(m.apply('fceeb6a183684448cfccdd8d5a7cf863837dbf9981d9e836f7c0514e45df6ab7'),ensure_ascii=False))
