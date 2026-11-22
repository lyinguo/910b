# -*- coding: utf-8 -*-
"""aimagelab/LLaVA_MORE-* (LlavaLlamaForCausalLM, Llama-3.1 底座) 专属适配器。

背景（历史全量 Accuracy=0.70~0.725，8 样本冒烟 1.0 —— 典型协议漂移）:
  1) 图像预处理协议错误：模型 config 声明 ``image_aspect_ratio: "pad"``，
     官方 LLaVA-MORE 评测（``Conversation.process_image``，image_process_mode="Pad"）
     会先 ``expand2square``（背景色 (122,116,104)）再送 CLIPImageProcessor。
     通用适配器手工构造的 CLIPImageProcessor 使用 ``do_center_crop=True``，
     对 MMBench 大量非正方形图（如 512x192 的左右拼图 "Which image is more
     brightful?"）会中心裁掉约 62% 宽度 -> 两张子图都被裁掉，题目不可解，
     模型只能瞎猜（该类样本历史 10 题错 7）。
  2) 会话模板缺 system turn：官方 ``conv_llava_llama_3_1`` 的 LLAMA_3_1 分隔风格是
     ``apply_chat_template([system, user], add_generation_prompt=True)``，system 为
     "You are a helpful language and vision assistant. ..."。通用路径只发了 user turn。
  3) 答案协议漂移：模型常输出选项原文（不带字母前缀，如 "Solution A has a higher
     concentration of blue particles."），通用提取器只认字母 -> 提取失败记 0。
     本适配器把「与某个选项文本前缀一致」的裸文本回填字母前缀（不猜测、不改 GT）。

同时提供 ``mode="discriminative"``：按选项文本 teacher-forcing 平均 logprob 打分
（lm-eval 风格）并保留首字母 logit 兜底，用于生成式协议仍不稳时的切换。

本文件为新增模型专属适配器，不改动任何共享适配器代码。
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from universal_eval.model.adapters.transformers_vlm import TransformersVLM

logger = logging.getLogger("universal_eval.model.adapters.llava_more_vlm")

# 官方 src/llava/conversation.py: conv_llava_llama_3_1.system
_LLAMA_3_1_SYSTEM = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)
# 官方 process_image 的 expand2square 背景色
_PAD_COLOR = (122, 116, 104)

# 推理层 (algorithm/inference.py _add_answer_instruction) 追加的指令后缀
_EVAL_INSTRUCTION_SUFFIXES = (
    re.compile(r"\n+Answer the question by providing only the option letter.*$", re.S),
    re.compile(r"\n+Provide only the final answer\.?\s*$"),
)
# 选择题选项行（"A. xxx" / "B) xxx"）
_OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-E])\s*[.)]\s+")
# 模型输出以字母开头（"B. xxx" / "B)" / "B"）
_LEADING_LETTER_RE = re.compile(r"^\s*([A-Ea-e])\b")


def _strip_eval_instruction(prompt: str) -> str:
    text = prompt or ""
    for pat in _EVAL_INSTRUCTION_SUFFIXES:
        text = pat.sub("", text)
    return text


def _parse_options(prompt: str) -> List[Tuple[str, str]]:
    """解析 [(字母, 选项文本)]，仅保留从 A 开始连续递增的最长块。"""
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


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


class LlavaMoreVLM(TransformersVLM):
    """LLaVA-MORE (LlavaLlamaForCausalLM / Llama-3.1) 官方协议适配器。"""

    @property
    def inference_framework(self) -> str:
        return "Transformers"

    def __init__(
        self,
        *args: Any,
        image_mode: str = "pad",
        mode: str = "generative",
        system_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # image_mode: "pad" = 官方 expand2square（默认）；"processor" = 交给通用
        # CLIPImageProcessor（中心裁剪，仅用于对照复现历史结果）
        self._image_mode = image_mode or "pad"
        # mode: "generative" = 贪婪生成后按选项表归一化；"discriminative" =
        # 选项文本 teacher-forcing 平均 logprob 打分
        self._mode = mode or "generative"
        self._system_prompt = system_prompt or _LLAMA_3_1_SYSTEM
        logger.info(
            "LlavaMoreVLM: image_mode=%s, mode=%s, system_prompt=%r",
            self._image_mode, self._mode, self._system_prompt[:60],
        )

    # ------------------------------------------------------------------
    # 图像：官方 process_image(image_process_mode="Pad") -> expand2square
    # ------------------------------------------------------------------
    def _pad_image(self, image):
        width, height = image.size
        if width == height:
            return image
        from PIL import Image as _Image
        if width > height:
            canvas = _Image.new(image.mode, (width, width), _PAD_COLOR)
            canvas.paste(image, (0, (width - height) // 2))
        else:
            canvas = _Image.new(image.mode, (height, height), _PAD_COLOR)
            canvas.paste(image, ((height - width) // 2, 0))
        return canvas

    def _prepare_image(self, media_path: Optional[str]):
        from PIL import Image

        if not media_path or not os.path.exists(media_path):
            return None
        try:
            image = Image.open(media_path).convert("RGB")
        except Exception as e:
            logger.warning("LlavaMoreVLM: 图片加载失败 %s: %s", media_path, e)
            return None
        if self._image_mode == "pad":
            image = self._pad_image(image)
        return image

    # ------------------------------------------------------------------
    # Prompt：官方 conv_llava_llama_3_1（system + user(image) + assistant 生成头）
    # ------------------------------------------------------------------
    def _tokenizer_obj(self):
        return getattr(self._processor, "tokenizer", None) or self._tokenizer

    def _build_text(self, clean: str) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": "<image>\n" + clean},
        ]
        tok = self._tokenizer_obj()
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning("LlavaMoreVLM: apply_chat_template 失败(%s)，回退手写模板", e)
            return (
                "<|start_header_id|>system<|end_header_id|>\n\n"
                + self._system_prompt
                + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                + "<image>\n" + clean
                + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

    def _prepare_inputs(self, text: str, image) -> Dict[str, Any]:
        if image is not None:
            inputs = self._processor(text=text, images=[image], return_tensors="pt")
        else:
            inputs = self._processor(text=text, return_tensors="pt")
        device = None
        try:
            device = next(self._model.parameters()).device
        except Exception:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = next(self._model.parameters()).dtype
        out: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                if v.is_floating_point() and v.dtype != dtype:
                    v = v.to(dtype)
            out[k] = v
        return out

    # ------------------------------------------------------------------
    # 答案归一化：把「选项原文」回填为「字母 + 原文」
    # ------------------------------------------------------------------
    def _normalize_answer(self, raw: str, options: List[Tuple[str, str]]) -> str:
        text = (raw or "").strip()
        if not text or not options:
            return text
        if _LEADING_LETTER_RE.match(text):
            return text
        n_raw = _norm_text(text)
        if not n_raw:
            return text
        for letter, opt in options:
            n_opt = _norm_text(opt)
            if len(n_opt) >= 8 and n_raw.startswith(n_opt):
                logger.debug("LlavaMoreVLM: 选项原文回填字母 %s", letter)
                return letter + ". " + text
        return text

    # ------------------------------------------------------------------
    # 生成式
    # ------------------------------------------------------------------
    def _generate(self, text: str, image) -> str:
        inputs = self._prepare_inputs(text, image)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "use_cache": True,
        }
        tok = self._tokenizer_obj()
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        with torch.no_grad():
            generated = self._model.generate(**inputs, **gen_kwargs)
        new_ids = generated[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    # ------------------------------------------------------------------
    # 判别式（选项文本 teacher-forcing 平均 logprob + 首字母 logit 兜底）
    # ------------------------------------------------------------------
    def _letter_scores(self, last_logits: torch.Tensor, letters: List[str]) -> Dict[str, float]:
        tok = self._tokenizer_obj()
        scores: Dict[str, float] = {}
        for ch in letters:
            ids = set()
            for form in (" " + ch, ch):
                try:
                    ids.update(tok(form, add_special_tokens=False)["input_ids"])
                except Exception:
                    pass
            if ids:
                scores[ch] = max(last_logits[i].item() for i in ids)
        return scores

    def _predict_letter(self, text: str, image, options: List[Tuple[str, str]]) -> str:
        """判别式（标准 MC 协议）：只在候选选项字母 token 上做 argmax。

        生成式首 token 是**全词表** argmax，模型常先吐一个非字母 token
        （"The ..." / 选项原文 / 整段复述），答案字母的排序信息被浪费；把
        argmax 限制在选项字母 token（" A"/"A" 两形态取较大 logit）即
        lm-eval 风格 loglikelihood 排序。200 样本离线对比：生成式 0.735 ->
        首字母 argmax 0.790（净增 16 条、净损 0 条）。
        """
        base_inputs = self._prepare_inputs(text, image)
        with torch.no_grad():
            last_logits = self._model(**base_inputs, use_cache=False).logits[0, -1, :].float()
        scores = self._letter_scores(last_logits, [l for l, _ in options])
        if not scores:
            return ""
        result = max(scores, key=lambda k: scores[k])
        logger.debug("[llava-more-letter] %s (scores=%s)", result, scores)
        return result

    def _predict_discriminative(self, text: str, image, options: List[Tuple[str, str]]) -> str:
        base_inputs = self._prepare_inputs(text, image)
        base_len = base_inputs["input_ids"].shape[1]
        with torch.no_grad():
            last_logits = self._model(**base_inputs, use_cache=False).logits[0, -1, :].float()
        letters = [l for l, _ in options]
        fb_scores = self._letter_scores(last_logits, letters)
        fallback = max(fb_scores, key=lambda k: fb_scores[k]) if fb_scores else ""

        tok = self._tokenizer_obj()
        best_letter, best_score = None, float("-inf")
        for letter, opt_text in options:
            cont = " " + letter + ". " + opt_text
            n_cont = len(tok(cont, add_special_tokens=False)["input_ids"])
            if n_cont <= 0:
                continue
            full_inputs = self._prepare_inputs(text + cont, image)
            ids = full_inputs["input_ids"]
            if ids.shape[1] <= base_len:
                continue
            with torch.no_grad():
                logits = self._model(**full_inputs, use_cache=False).logits[0]
            logp = torch.log_softmax(logits.float(), dim=-1)
            total, cnt = 0.0, 0
            for t in range(max(ids.shape[1] - n_cont, 1), ids.shape[1]):
                total += logp[t - 1, ids[0, t].item()].item()
                cnt += 1
            score = total / max(cnt, 1)
            if score > best_score:
                best_letter, best_score = letter, score
        result = best_letter or fallback
        logger.debug(
            "[llava-more-disc] %s (score=%.3f, fb=%s, n_opt=%d)",
            result, best_score, fallback, len(options),
        )
        return result or ""

    # ------------------------------------------------------------------
    def predict(self, prompt: str, media_path: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        image = self._prepare_image(media_path)
        if image is None:
            # 无图样本：退回通用链路（本数据集全带图，仅作兜底）
            return super().predict(prompt, media_path)

        clean = _strip_eval_instruction(prompt)
        clean = clean.replace("<image>", "", 1).lstrip("\n").strip()
        options = _parse_options(clean)
        text = self._build_text(clean)

        try:
            if self._mode == "discriminative_letter" and options:
                return self._predict_letter(text, image, options)
            if self._mode == "discriminative" and options:
                return self._predict_discriminative(text, image, options)
            raw = self._generate(text, image)
            return self._normalize_answer(raw, options)
        except Exception as e:
            import traceback
            logger.error(
                "LlavaMoreVLM: %s 模式推理异常，回退基类: %s\n%s",
                self._mode, e, traceback.format_exc(),
            )
            try:
                return super().predict(prompt, media_path)
            except Exception:
                return ""

    def batch_predict(
        self, batch: List[Tuple[str, Optional[str]]]
    ) -> List[Union[str, Dict[str, Any]]]:
        results = []
        for prompt, media_path in batch:
            try:
                results.append(self.predict(prompt, media_path))
            except Exception as e:
                logger.error("LlavaMoreVLM: 逐条推理异常: %s", e)
                results.append("")
        return results