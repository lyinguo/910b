"""Original Greedy prompt bytes; original choice-probability result and strict Accuracy."""
from universal_eval.algorithm.gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113 import E4BActualOriginalHighTenMC
from universal_eval.algorithm.tinyllava_strict_mc import StrictMCAccuracy
class OwnGLM41BaseProbabilityInference(E4BActualOriginalHighTenMC):
 @property
 def algorithm_name(self):return 'glm41base_own_original_option_text_mean_logprob'
class OwnGLM41BaseAccuracy(StrictMCAccuracy):
 @property
 def metric_name(self):return 'Accuracy'
