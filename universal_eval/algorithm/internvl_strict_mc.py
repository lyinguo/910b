"""Use the reviewed explicit-choice parser without changing shared defaults."""
from universal_eval.algorithm.tinyllava_strict_mc import TinyStrictGreedy

class InternVLStrictGreedy(TinyStrictGreedy):
    @property
    def algorithm_name(self):
        return 'internvl_qwen2_native_greedy_explicit_mc'
