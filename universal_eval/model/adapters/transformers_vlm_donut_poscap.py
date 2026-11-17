"""Donut 系（VisionEncoderDecoderModel + donut-swin/mBART）decoder 位置越界保护适配器。

适用模型（两者同架构、同报错）：
  - katanaml-org/invoices-donut-model-v1
  - selvakumarcts/sk_invoice_receipts

真因（2026-09-21 实测定位）：
  decoder 为 mBART（`MBartForCausalLM`），`config.decoder.max_position_embeddings = 1536`，
  而 `MBartLearnedPositionalEmbedding` 内部 `offset = 2`，实际位置嵌入矩阵为 1538 行
  （`dec.embed_positions.weight.shape == (1538, 1024)`），即 decoder 序列总长度上限为 1536。
  注意：这两个模型的 `len(tokenizer)`（57566 / 57596）与
  `model.config.decoder.vocab_size`（57566 / 57596）完全一致，
  **不存在 token id 越界**；唯一越界点是位置嵌入。

  yaml 原 `max_new_tokens: 4096` 远超 1536。这类发票/收据专用 Donut 模型在通用图像
  caption 数据集上会退化进入重复循环（如 "ORDER BE ORDER BE ..."）而不输出 EOS，
  decoder 长度一路增长越过 1536 → 位置嵌入 index_select 越界 →
  CUDA device-side assert：
      /pytorch/aten/src/ATen/native/cuda/Indexing.cu:1515:
      indexSelectSmallIndex: Assertion `srcIndex < srcSelectDimSize` failed.
  该断言会污染整个 CUDA context，此后所有样本都报
  "device-side assert triggered"（历史现象：前 8 条成功、之后 192 条全失败）。

修复策略（最小侵入）：
  不修改公共适配器 `transformers_vlm.py`，不改模型权重，仅包装 `model.generate`，
  把本次生成的 decoder 总长度限制在位置嵌入可寻址范围内：
      effective_max_new_tokens = min(max_new_tokens, limit - prefix_len - 1)
  其中 limit 由 `config.decoder.max_position_embeddings` 与实际位置嵌入行数
  （减去 offset）交叉校验取小值，prefix_len 为 Donut 分支传入的
  `decoder_input_ids` 长度。yaml 中的 `max_new_tokens: 2048` 仍作为上限生效。
"""

import logging
from typing import Any, Optional

from universal_eval.model.adapters.transformers_vlm import TransformersVLM

logger = logging.getLogger(
    "universal_eval.model.adapters.transformers_vlm_donut_poscap"
)


class TransformersVLMPosCap(TransformersVLM):
    """TransformersVLM + decoder 位置嵌入长度上限保护（防止 index_select 越界）。"""

    def load(self, model_path: str, **kwargs: Any) -> None:
        super().load(model_path, **kwargs)

        try:
            limit = self._resolve_decoder_pos_limit()
        except Exception as _e:  # 探测失败不应阻断加载
            logger.warning(f"decoder 位置上限探测失败，跳过越界保护: {_e}")
            return

        if not limit or limit <= 0:
            logger.info("未检测到 decoder 位置嵌入上限，越界保护不启用")
            return

        self._pos_cap_limit = limit
        model = self._model
        _orig_generate = model.generate

        def _capped_generate(*args, **gen_kwargs):
            _max_new = gen_kwargs.get("max_new_tokens")
            if _max_new is not None:
                _prefix = 0
                _dec = gen_kwargs.get("decoder_input_ids")
                if _dec is not None and hasattr(_dec, "shape") and _dec.dim() == 2:
                    _prefix = int(_dec.shape[1])
                _cap = max(1, limit - _prefix - 1)
                if int(_max_new) > _cap:
                    logger.info(
                        f"位置越界保护: decoder 总长上限 {limit}, prompt 前缀 {_prefix} -> "
                        f"max_new_tokens {int(_max_new)} 裁剪为 {_cap}"
                    )
                    gen_kwargs["max_new_tokens"] = _cap
            return _orig_generate(*args, **gen_kwargs)

        model.generate = _capped_generate
        logger.info(
            f"已启用 Donut/mBART decoder 位置越界保护: decoder 序列总长上限 {limit}"
        )

    def _resolve_decoder_pos_limit(self) -> Optional[int]:
        """返回 decoder 位置嵌入可寻址的最大序列长度。"""
        model = self._model
        limit: Optional[int] = None

        _dec_cfg = getattr(getattr(model, "config", None), "decoder", None)
        _mpe = getattr(_dec_cfg, "max_position_embeddings", None) if _dec_cfg else None
        if _mpe:
            limit = int(_mpe)

        try:
            _dec = model.get_decoder()
            _inner = getattr(_dec, "get_decoder", None)
            if callable(_inner):
                _dec = _inner()
            _ep = getattr(_dec, "embed_positions", None)
            if _ep is not None and hasattr(_ep, "weight"):
                _rows = int(_ep.weight.shape[0]) - int(getattr(_ep, "offset", 0))
                limit = _rows if limit is None else min(limit, _rows)
        except Exception as _e:
            logger.warning(f"读取位置嵌入矩阵失败，仅用配置值: {_e}")

        return limit