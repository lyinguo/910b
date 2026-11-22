import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util
R=Path('/root/projects/comperror');p=R/'scripts/select_existing_original_resource_queue_1245_20261117.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='27daea622ea40c92d1182895e6dcd1a9d5803fac9cc0b7a7d224a4f80dd52494'
sys.path[:0]=[str(R),str(R/'scripts')]
spec=importlib.util.spec_from_file_location('frozen_original_resource_selector_readonly',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert hashlib.sha256(module.ORIGINAL_RESOURCE_GATE).hexdigest()==module.ORIGINAL_RESOURCE_GATE_SHA256=='bc810ce69760512bde98329e7085d1e458a7c074ce0326b41f0edb95d8a7771b'
exec(compile(module.ORIGINAL_RESOURCE_GATE,'frozen_original_resource_metadata_gate_only_no_dispatch','exec'),{})
