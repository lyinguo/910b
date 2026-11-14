from universal_eval.algorithm.native_explicit_mc_v2 import NativeExplicitMCV2,explicit_choice
from universal_eval.algorithm.inference import GreedyInference
class E4BActualOriginalHighTenMC(NativeExplicitMCV2):
    # Exact frozen original A100 GreedyInference transform; no new instruction.
    _add_answer_instruction=staticmethod(GreedyInference._add_answer_instruction)
    def run(self,model,samples,**kwargs):
        predictions,references,extra,statistics=super().run(model,samples,**kwargs)
        invalid=[]
        for i,row in enumerate(extra):
            if explicit_choice(row.get('raw_output')) is None:predictions[i]='';invalid.append(i)
        statistics.update(strict_invalid_choice_indices=invalid,strict_invalid_choice_count=len(invalid),strict_invalid_policy='Full raw generation retained; invalid extracted choice empty; complete denominator retained',original_GreedyInference_answer_transform_preserved=True)
        return predictions,references,extra,statistics
