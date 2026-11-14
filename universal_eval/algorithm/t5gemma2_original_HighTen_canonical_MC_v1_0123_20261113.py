from universal_eval.algorithm.native_explicit_mc_v2 import NativeExplicitMCV2
class T5Gemma2OriginalCanonicalMC(NativeExplicitMCV2):
    # The own adapter applies the exact original HighTen canonical _text branch.
    # The model text is checked against all 200 original actual AA input hashes.
    @staticmethod
    def _add_answer_instruction(prompt):return prompt
