import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import os
import runpy

R=Path('/root/projects/comperror')
source=R/'scripts/select_existing_original_resource_queue_1245_20261117.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='27daea622ea40c92d1182895e6dcd1a9d5803fac9cc0b7a7d224a4f80dd52494'
os.chdir('/root')
sys.path[:0]=[str(R),str(R/'scripts')]
runpy.run_path(str(source),run_name='__main__')
