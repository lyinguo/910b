"""Load explicitly configured project algorithm/metric components."""
import hashlib,importlib.util,pathlib,sys

def load_component(definition):
    filename,classname=definition.rsplit(':',1)
    root=pathlib.Path(__file__).resolve().parents[2]
    path=(root/filename).resolve()
    if not any(path.is_relative_to(root/part) for part in ('universal_eval','scripts')):
        raise ValueError('Component code must be in project source, not reports/diagnostics')
    name='_eval_component_'+hashlib.sha256(str(path).encode()+path.read_bytes()).hexdigest()[:20]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return getattr(sys.modules[name],classname)
