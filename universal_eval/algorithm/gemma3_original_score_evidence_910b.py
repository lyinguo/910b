"""Retain the original MCScore calculation and emit its actual selected raw label."""
from universal_eval.algorithm.mc_score import MCScoreAlgorithm

class Gemma3OriginalScoreEvidence910B(MCScoreAlgorithm):
 def run(self,model,samples,**kwargs):
  predictions,references,extras,statistics=super().run(model,samples,**kwargs)
  for sample,prediction,extra in zip(samples,predictions,extras):
   execution=model.get_sample_execution(sample.get('prompt',''),sample.get('media_path'))
   if execution.get('error') is None:
    assert prediction==execution['raw_output']==max(extra['scores'],key=extra['scores'].get)
    extra['raw_output']=execution['raw_output']
   extra['execution']=execution
  statistics['actual_inference_batch_size']=1
  statistics['original_MCScore_calculation_unchanged']=True
  return predictions,references,extras,statistics
