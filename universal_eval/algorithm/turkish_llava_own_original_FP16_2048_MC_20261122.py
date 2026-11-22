"""Original Greedy prompt bytes and established explicit-choice Accuracy rule."""
from universal_eval.algorithm.gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113 import E4BActualOriginalHighTenMC
from universal_eval.algorithm.tinyllava_strict_mc import StrictMCAccuracy

class OwnTurkishLlavaGreedy(E4BActualOriginalHighTenMC):
    @property
    def algorithm_name(self):return 'turkish_llava_own_original_FP16_greedy2048'

class OwnTurkishLlavaAccuracy(StrictMCAccuracy):
    @property
    def metric_name(self):return 'Accuracy'
