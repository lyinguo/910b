# -*- coding: utf-8 -*-
"""SmolDocling-256M-preview 判别式选择题适配器（docling-project/SmolDocling-256M-preview）。

根因（历史 Accuracy=0.335）: SmolDocling 是文档转换模型（SmolVLM-256M-Instruct 的
Docling/DocTags 微调版），面对 mmbench 选择题按"文档转录"习惯退化输出枚举串
"A. B. C. D. E." / "Answer: A. B. C. D. E."（200 样本中约半数），提取器取到开头
字母 A，形成 A-bias 低分。32 样本离线实测:
  - 生成式 + 触发式补全（Idefics3BaseVLM 式）: 3/16，模型输出 span 而非字母，不适用;
  - 生成式 + span_to_option 词覆盖率映射: 4/16，改善有限;
  - 判别式打分（teacher-forcing 各选项文本平均 logprob 取 argmax）: 16/32 (~0.50)。
故采用判别式多选解码: 对每个选项文本以 "The correct answer is" 触发前缀做
teacher-forcing，取平均 logprob 最高的选项字母。这是标准 LM 多选题评测协议
（lm-eval 风格 loglikelihood ranking），非正则宽松提取，不改动数据集/GT。

实现要点:
  - 剥离推理层追加的指令后缀（"Answer the question by providing only the option
    letter ..." / "Provide only the final answer."），避免污染最后一个选项文本与
    打分前缀（与 paligemma_transformers.py 同思路）。
  - 不用 chat_template（SmolDocling 模板为 Docling 转换优化，选择题下触发枚举
    退化），采用 processor 单轮契约: "<image>" + 题目 + "\n\nThe correct answer is"。
  - pixel_values 对齐模型 dtype（bf16 权重 vs fp32 像素）。
  - 无法解析选项（非选择题）时回退 TransformersVLM 生成式链路。
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from universal_eval.model.adapters.transformers_vlm import TransformersVLM

logger = logging.getLogger("universal_eval.model.adapters.smoldocling_discriminative")

# 推理层 (algorithm/inference.py _add_answer_instruction) 追加的指令后缀
_EVAL_INSTRUCTION_SUFFIXES = (
    re.compile(r"\n\nAnswer the question by providing only the option letter.*$", re.S),
    re.compile(r"\n\nProvide only the final answer\.\s*$"),
)

# 选择题选项行（"A. xxx" / "B) xxx"）
_OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-E])\s*[.)]\s+")

_TRIGGER = "\n\nThe correct answer is"

# letter-argmax 模式触发：dev-40 实测 " Answer:" 短触发仅 0.40（A-bias 重），
# 与 option 模式共用 "\n\nThe correct answer is" 触发后末位字母 argmax 得 0.50
_LETTER_TRIGGER = _TRIGGER


def _parse_options(prompt: str) -> List[Tuple[str, str]]:
    """解析 [(字母, 选项文本), ...]，仅保留从 A 开始连续递增的最长块。"""
    marks = list(_OPTION_LINE_RE.finditer(prompt))
    opts: List[Tuple[str, str]] = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(prompt)
        opts.append((m.group(1), prompt[start:end].strip()))
    result: List[Tuple[str, str]] = []
    expect = "A"
    for letter, text in opts:
        if letter == expect and text:
            result.append((letter, text))
            expect = chr(ord(expect) + 1)
    return result


def _strip_eval_instruction(prompt: str) -> str:
    for pat in _EVAL_INSTRUCTION_SUFFIXES:
        prompt = pat.sub("", prompt)
    return prompt


class SmolDoclingDiscriminativeVLM(TransformersVLM):
    """SmolDocling 判别式（选项 logprob argmax）选择题适配器。"""

    @property
    def inference_framework(self) -> str:
        return "Transformers"

    def __init__(self, *args: Any, scoring: str = "option", **kwargs: Any) -> None:
        # scoring="option": 各选项文本 teacher-forcing 平均 logprob argmax（原协议，主协议）
        # scoring="letter": 单次前向（"\n\nThe correct answer is" 触发后缀），末位 logits
        #                   在选项字母 token 上 argmax。dev-40 略优(0.50)但全量证伪(0.32，
        #                   字母先验坍缩)，SmolDocling 上不建议启用。
        self._scoring = str(scoring or "option").lower()
        self._letter_token_ids: Dict[str, List[int]] = {}
        super().__init__(*args, **kwargs)
        logger.info(
            "SmolDoclingDiscriminativeVLM: 判别式打分解码 (scoring=%s)", self._scoring
        )

    def load(self, model_path: str, **kwargs: Any) -> None:
        super().load(model_path, **kwargs)
        tok = getattr(self._processor, "tokenizer", None) or self._tokenizer
        for ch in "ABCDE":
            ids: List[int] = []
            for s in (ch, " " + ch):
                for i in tok(s, add_special_tokens=False)["input_ids"]:
                    if i not in ids:
                        ids.append(i)
            self._letter_token_ids[ch] = ids
        logger.info(
            "[smoldocling-disc] 字母 token ids: %s", self._letter_token_ids
        )

    def _prepare(self, text: str, image: Any) -> Dict[str, Any]:
        import torch

        inputs = self._processor(text=[text], images=[[image]], return_tensors="pt")
        device = self._device if self._device not in ("auto", "cpu") else "cuda:0"
        model_dtype = next(self._model.parameters()).dtype
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                if v.is_floating_point() and v.dtype != model_dtype:
                    v = v.to(model_dtype)
            out[k] = v
        return out

    def _score_continuation(self, prefix: str, image: Any, base_len: int, cont: str) -> float:
        """teacher-forcing: 返回续写 cont 在 prefix 之后的平均 logprob。

        与离线验证一致：对 prefix+cont 整段 tokenize，续写区长度按 cont 独立
        tokenize 的长度计（start = full_len - n_cont）。
        """
        import torch

        tok = getattr(self._processor, "tokenizer", None) or self._tokenizer
        full_inputs = self._prepare(prefix + cont, image)
        input_ids = full_inputs["input_ids"]
        n_cont = len(tok(cont, add_special_tokens=False)["input_ids"])
        if n_cont <= 0 or input_ids.shape[1] <= base_len:
            return float("-inf")
        fwd = dict(full_inputs)
        fwd["use_cache"] = False
        with torch.no_grad():
            logits = self._model(**fwd).logits[0]
        logp = torch.log_softmax(logits.float(), dim=-1)
        start = max(input_ids.shape[1] - n_cont, 1)
        total = 0.0
        cnt = 0
        for t in range(start, input_ids.shape[1]):
            total += logp[t - 1, input_ids[0, t].item()].item()
            cnt += 1
        return total / max(cnt, 1)

    def _letter_argmax(self, prefix: str, image: Any, letters: List[str]) -> str:
        """letter 模式: prefix+" Answer:" 单次前向，末位 logits 在选项字母 token 上 argmax。"""
        import torch

        inputs = self._prepare(prefix + _LETTER_TRIGGER, image)
        fwd = dict(inputs)
        fwd["use_cache"] = False
        with torch.no_grad():
            logits = self._model(**fwd).logits[0]
        last = torch.log_softmax(logits.float(), dim=-1)[-1]
        del logits
        scores = {}
        for ch in letters:
            ids = self._letter_token_ids.get(ch) or []
            if ids:
                scores[ch] = max(last[i].item() for i in ids)
        if not scores:
            return ""
        best = max(scores, key=lambda k: scores[k])
        logger.debug("[smoldocling-disc-letter] %s %s", best, {k: round(v, 3) for k, v in scores.items()})
        return best

    def predict(self, prompt: str, media_path: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        from PIL import Image

        image = None
        if media_path and os.path.exists(media_path):
            try:
                image = Image.open(media_path).convert("RGB")
            except Exception as e:
                logger.warning(f"smoldocling-disc: 图片加载失败 {media_path}: {e}")

        base_prompt = _strip_eval_instruction(prompt or "")
        options = _parse_options(base_prompt) if image is not None else []
        if not options:
            logger.debug("smoldocling-disc: 未解析到选项，回退生成式链路")
            return super().predict(prompt, media_path)

        try:
            if self._scoring == "letter":
                letter_prefix = "<image>" + base_prompt.rstrip()
                best_letter = self._letter_argmax(letter_prefix, image, [l for l, _ in options])
                if best_letter:
                    return best_letter
                logger.debug("smoldocling-disc: letter 模式无有效字母，回退 option 打分")
            prefix = "<image>" + base_prompt.rstrip() + _TRIGGER
            base_len = self._prepare(prefix, image)["input_ids"].shape[1]
            best_letter, best_score = None, float("-inf")
            for letter, opt_text in options:
                s = self._score_continuation(prefix, image, base_len, " " + opt_text)
                if s > best_score:
                    best_letter, best_score = letter, s
            logger.debug(
                "[smoldocling-disc] %s (%.3f, %d options)", best_letter, best_score, len(options)
            )
            return best_letter or ""
        except Exception as e:
            logger.error(f"smoldocling-disc: 判别式打分异常，回退生成式: {e}")
            return super().predict(prompt, media_path)

    def batch_predict(
        self, batch: List[Tuple[str, Optional[str]]]
    ) -> List[Union[str, Dict[str, Any]]]:
        results = []
        for prompt, media_path in batch:
            try:
                results.append(self.predict(prompt, media_path))
            except Exception as e:
                logger.error(f"smoldocling-disc: 逐条推理异常: {e}")
                results.append("")
        return results
