"""Original saved Greedy prompt bytes with the established explicit-choice rule."""
from universal_eval.algorithm.gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113 import E4BActualOriginalHighTenMC
from universal_eval.algorithm.tinyllava_strict_mc import StrictMCAccuracy

class OwnGLM41ThinkingGreedy(E4BActualOriginalHighTenMC):
    @property
    def algorithm_name(self):return 'glm41thinking_original_VLLM8192_greedy_explicit_MC'

class OwnGLM41ThinkingAccuracy(StrictMCAccuracy):
    @property
    def metric_name(self):return 'Accuracy'
