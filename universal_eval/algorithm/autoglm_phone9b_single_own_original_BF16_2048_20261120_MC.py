"""Original saved Greedy prompt bytes with the established explicit-choice rule."""
from universal_eval.algorithm.gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113 import E4BActualOriginalHighTenMC
from universal_eval.algorithm.tinyllava_strict_mc import StrictMCAccuracy

class OwnAutoGLMPhone9BGreedy(E4BActualOriginalHighTenMC):
    @property
    def algorithm_name(self):return 'autoglm_phone9b_original_existing_thinkingFalse_greedy_explicit_MC'

class OwnAutoGLMPhone9BAccuracy(StrictMCAccuracy):
    @property
    def metric_name(self):return 'Accuracy'
