# -*- coding: utf-8 -*-
"""OS-Copilot/OS-Atlas-Base-4B 续写概率打分适配器。

背景：OS-Atlas-Base-4B 是 InternVL 系列的 base（非指令）模型，greedy 自由生成
输出为长文本续写（冒烟 8 样本仅 1 个恰好含字母，Accuracy 0.125），答案提取不可靠。
改为 mc_score：一次前向取 "Answer:" 后下一位置的 log-softmax，对候选选项字母的
首 token 取分，最高分为预测（同 cliptagger/gemma4-base 已验证思路）。
"""

import logging
from typing import Any, Dict, List, Optional

from universal_eval.model.adapters.high_nine_internvl_reference import InternVLAdapter

logger = logging.getLogger("universal_eval.model.adapters.osatlas_mc_score")


class OSAtlasMCScore(InternVLAdapter):
    """InternVL 系列 base 模型的选择题续写概率打分。"""

    _SCORE_TAIL = "\n\nAnswer:"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("max_new_tokens", 2048)
        super().__init__(*args, **kwargs)

    def load(self, model_path: str, **kwargs: Any) -> None:
        import os
        import torch.distributed as dist
        self._owns_process_group = not dist.is_initialized()
        # Load before initializing distributed: the old remote Phi3 config has
        # no tp_plan and cannot enter Transformers' distributed allocator warmup.
        super().load(model_path, **kwargs)
        if self._owns_process_group:
            dist.init_process_group('gloo', init_method=f'file:///tmp/a100_high_nine_osatlas_pg_{os.getpid()}', rank=0, world_size=1)

    def unload(self) -> None:
        super().unload()
        import torch.distributed as dist
        if getattr(self, '_owns_process_group', False) and dist.is_initialized():
            dist.destroy_process_group()

    def score_choice(
        self,
        prompt: str,
        options: List[str],
        media_path: Optional[str] = None,
        image: Any = None,
    ) -> Dict[str, float]:
        import torch

        if image is None:
            if media_path is None:
                raise ValueError("score_choice 需要 media_path 或已加载的 image")
            from PIL import Image

            image = Image.open(media_path).convert("RGB")

        tokenizer = self._tokenizer
        model = self._model

        # 1. 图像预处理（复用 InternVL 动态切图）
        pixel_values, num_patches = self._preprocess_internvl(image)
        pixel_values = pixel_values.to(self._device)
        model_dtype = next(model.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(model_dtype)

        # 2. 构造带图像占位的 input_ids
        text_prompt = (prompt or "").rstrip() + self._SCORE_TAIL
        query = self._build_internvl_query(text_prompt, image_size=None)
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        num_image_token = int(getattr(model, "num_image_token", 256))
        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * (num_image_token * num_patches)
            + "</img>"
        )
        query = query.replace("<image>", image_tokens, 1)
        input_ids = tokenizer(query, return_tensors="pt")["input_ids"].to(self._device)

        # 3. 一次前向取最后位置 log-softmax
        #    远程代码 modeling_internvl_chat.forward 需要 image_flags：
        #    squeeze(-1) 后形状须与 vit_embeds 第 0 维(num_patches)匹配，
        #    全 1 表示所有瓦片均参与，否则 None.squeeze / 掩码不匹配报错。
        #    forward 内用 self.img_context_token_id 定位图像 token 做 embed 注入，
        #    该属性默认 None（官方 chat 流程才会设置），直接调 forward 前必须赋值。
        model.img_context_token_id = img_context_token_id
        image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=self._device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_flags=image_flags,
            )
        log_probs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)

        # 4. 对每个选项字母首 token 打分
        scores: Dict[str, float] = {}
        for opt in options:
            cand = str(opt).strip().upper()
            best = None
            for variant in (cand, " " + cand, "\n" + cand):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if not ids:
                    continue
                val = float(log_probs[ids[0]].item())
                if best is None or val > best:
                    best = val
            scores[cand] = best if best is not None else float("-inf")
        return scores
