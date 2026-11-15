"""Original OCR Greedy computation/extraction with complete native execution attached."""
from universal_eval.algorithm.inference import GreedyInference
from universal_eval.metric.ocr import CERMetric

class DotsOriginalGreedyEvidence(GreedyInference):
 @property
 def algorithm_name(self):return 'dotsocr_original_greedy_evidence'
 def run(self,model,samples,**kwargs):
  predictions,references,extras,statistics=super().run(model,samples,**kwargs)
  assert len(predictions)==len(references)==len(extras)==len(samples)
  for sample,extra in zip(samples,extras):
   prompt=self._add_answer_instruction(sample['prompt']) if kwargs.get('extract_answer',True) else sample['prompt']
   extra['actual_execution']=model.get_sample_execution(prompt,sample.get('media_path'))
  return predictions,references,extras,statistics

class DotsOriginalCER(CERMetric):
 @property
 def supported_tasks(self):return super().supported_tasks+['image-text-to-text']
