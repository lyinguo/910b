"""Original Greedy OCR with the standard pipeline execution callback exactly once."""
from universal_eval.algorithm.inference import GreedyInference
from universal_eval.metric.ocr import CERMetric

class DotsOriginalGreedyEvidenceV2(GreedyInference):
 @property
 def algorithm_name(self):return 'dotsocr_original_greedy_evidence'

class DotsOriginalCERV2(CERMetric):
 @property
 def supported_tasks(self):return super().supported_tasks+['image-text-to-text']
