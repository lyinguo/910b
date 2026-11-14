"""Explicit-choice extraction; never turn an English article 'a' into option A."""
import re
from universal_eval.algorithm.inference import GreedyInference
from universal_eval.metric.interface import IMetric

def explicit_option(raw):
    if not isinstance(raw,str):return None
    rules=[r'<answer>\s*([A-E])\s*</answer>',r'\\boxed\{\s*([A-E])\s*\}',r'(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*:?|:)\s*[\'"(\[]*\s*([A-E])(?=[\s.\'"\])]|$)',r'(?:correct\s+)?option\s+(?:is\s+)?[\'"(\[]*([A-E])\b',r'(?:choose|select|pick)\s+[\'"(\[]*([A-E])\b',r'^\s*([A-E])\s*[.)](?=\s|$)',r'^\s*([A-E])\s*$']
    for rule in rules:
        matches=re.findall(rule,raw,flags=re.I)
        if matches:return matches[-1].upper()
    # A separate, complete answer line is unambiguous; ordinary isolated words
    # inside a paragraph are not searched or assigned a label.
    lines=re.findall(r'(?m)^\s*([A-E])\s*$',raw)
    return lines[-1].upper() if lines else None

class TinyStrictGreedy(GreedyInference):
    @property
    def algorithm_name(self):return 'tinyllava_native_greedy_explicit_mc'
    def _extract_answer(self,raw_output,is_multiple_choice=False):
        return explicit_option(raw_output) if is_multiple_choice else None

class StrictMCAccuracy(IMetric):
    @property
    def metric_name(self):return 'Strict_MC_Accuracy'
    @property
    def supported_tasks(self):return ['visual-question-answering','image-text-to-text']
    def compute(self,predictions,references,extra_info=None):
        if len(predictions)!=len(references):raise ValueError('Prediction/reference counts differ')
        correct=sum(isinstance(p,str) and re.fullmatch(r'[A-E]',p.strip().upper()) is not None and p.strip().upper()==str(r).strip().upper() for p,r in zip(predictions,references))
        return correct/len(predictions) if predictions else 0.0
