"""Original saved Greedy prompt bytes with the established explicit-choice rule."""
from universal_eval.algorithm.gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113 import E4BActualOriginalHighTenMC
from universal_eval.algorithm.tinyllava_strict_mc import StrictMCAccuracy

class OwnPhi35OriginalGreedy(E4BActualOriginalHighTenMC):
    @property
    def algorithm_name(self):return 'phi35_original_greedy_explicit_MC'

class OwnPhi35Accuracy(StrictMCAccuracy):
    @property
    def metric_name(self):return 'Accuracy'
