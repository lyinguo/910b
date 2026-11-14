"""Read-only adapter-to-frozen-source contract guard; no model import, process or workbook mutation."""
import ast,hashlib,json,pathlib
R=pathlib.Path('/root/projects/comperror')
def validate_alignment_adapter_spec(spec,expected_model_id,config=None):
 cfg=config if config is not None else json.loads((R/spec['config']).read_text())
 assert cfg['model']['name']==cfg['model']['kwargs']['protocol']['model_id']==expected_model_id,'Wrong target model identity'
 rel,name=cfg['model']['adapter'].rsplit(':',1);path=pathlib.PurePosixPath(rel)
 assert not path.is_absolute() and '..' not in path.parts and rel in spec['source_sha256'],'Configured model adapter is absent from reviewed source manifest: '+rel
 raw=(R/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==spec['source_sha256'][rel],'Adapter source changed'
 tree=ast.parse(raw.decode());assert name in {n.name for n in tree.body if isinstance(n,ast.ClassDef)},'Configured adapter class absent in declared own source'
 dependencies=[]
 for n in tree.body:
  if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('universal_eval.model.adapters.'):
   dep=n.module.replace('.','/')+'.py';assert dep in spec['source_sha256'],'Unfrozen local parent adapter: '+dep
   assert hashlib.sha256((R/dep).read_bytes()).hexdigest()==spec['source_sha256'][dep];dependencies.append(dep)
 return {'passed':True,'model_id':expected_model_id,'adapter':cfg['model']['adapter'],'reviewed_parent_adapters':dependencies}
