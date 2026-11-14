"""Original MCScore mathematics with full execution evidence; a selected label is explicitly not generated text."""
from universal_eval.algorithm.mc_score import MCScoreAlgorithm

class GRMRA100HighNineProbabilityEvidence1726(MCScoreAlgorithm):
 def run(self,model,samples,**kwargs):
  predictions,references,extras,statistics=super().run(model,samples,**kwargs)
  for sample,prediction,extra in zip(samples,predictions,extras):
   original_raw=extra['raw_output'];assert original_raw is None
   execution=model.get_sample_execution(sample.get('prompt',''),sample.get('media_path'))
   extra['original_mc_score_raw_output']=original_raw
   if execution.get('error') is None:
    assert prediction==execution['selected_candidate']==max(extra['scores'],key=extra['scores'].get)
    extra['raw_output']=prediction
    extra['raw_output_represents']='selected_probability_candidate_not_generated_text'
   extra['execution']=execution
  statistics['actual_inference_batch_size']=1
  statistics['original_MCScore_calculation_unchanged']=True
  statistics['original_mc_score_raw_None_retained_separately']=True
  return predictions,references,extras,statistics
