"""Use explicit MC answers with each model's unmodified generated text retained."""
from universal_eval.algorithm.tinyllava_strict_mc import TinyStrictGreedy

class NativeVLMStrictGreedy(TinyStrictGreedy):
    @property
    def algorithm_name(self):return 'native_vlm_greedy_explicit_mc'
