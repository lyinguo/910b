"""Create a separate load-only diagnostic; preserve the failed v1 unchanged."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path('D:/code-lv/.tmp/batch_910b_20261030')
old=P/'diagnostic_moondream3_own_original_HF_actual_loading_precision_no_forward_1106_20261119.py'
source=old.read_text(encoding='utf8')
assert source.count("json.dumps(value,ensure_ascii=False,indent=2)")==1
source=source.replace("json.dumps(value,ensure_ascii=False,indent=2)","json.dumps(value,ensure_ascii=False,indent=2,default=lambda x: sorted(x) if isinstance(x,set) else (_ for _ in ()).throw(TypeError(type(x).__name__)))")
source=source.replace("Moondream3_own_original_HF_actual_loading_precision_CPU_no_forward_1106_20261119","Moondream3_own_original_HF_actual_loading_precision_CPU_no_forward_v2_json_sets_1116_20261119")
source=source.replace("print(json.dumps(review,ensure_ascii=False))","print(json.dumps(dict(passed=review['passed'],actual_original_loaded_physical_dtypes=dtypes,source_tensors=len(records),actual_vision_parameter_dtype=review['actual_vision_parameter_dtype'],input_and_loaded_vision_dtype_match=review['input_and_loaded_vision_dtype_match'],actual_NEW_model_forwards=0),ensure_ascii=False))")
name='diagnostic_moondream3_own_original_HF_actual_loading_precision_no_forward_v2_json_sets_1116_20261119.py'
ast.parse(source)
b=source.encode('utf8');(P/name).write_bytes(b)
record=dict(relative='scripts/'+name,SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
deploy="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,runpy\nDATA="+repr(record)+"\nR=Path('/root/projects/comperror');b=base64.b64decode(DATA['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==DATA['SHA256'];ast.parse(b);p=R/DATA['relative'];assert not p.exists();p.write_bytes(b);sys.path[:0]=[str(R),str(R/'scripts')];runpy.run_path(str(p),run_name='__main__')\n"
(P/'deploy_execute_Moondream3_original_HF_load_precision_v2_1116_20261119_remote.py').write_text(deploy,encoding='utf8',newline='\n')
print(json.dumps({k:v for k,v in record.items() if k!='base64'}))
