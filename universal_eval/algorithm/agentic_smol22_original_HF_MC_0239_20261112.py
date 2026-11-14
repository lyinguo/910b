from universal_eval.algorithm.native_explicit_mc_v2 import NativeExplicitMCV2
class AgenticSmol22OriginalMC(NativeExplicitMCV2):

    @staticmethod
    def _add_answer_instruction(prompt: str) -> str:
        """
        在 prompt 末尾追加指令，引导模型输出简短答案。

        选择题（含 "Choose the correct option"）:
          追加 "Reply with only the letter of the correct option."

        其他题型:
          追加 "Provide only the final answer."
        """
        if 'Choose the correct option' in prompt:
            return prompt + '\n\nReply with only the letter of the correct option.'
        if 'Final Answer:' in prompt or 'Answer:' in prompt:
            return prompt
        return prompt + '\n\nProvide only the final answer.'
