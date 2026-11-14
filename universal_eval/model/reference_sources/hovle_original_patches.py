"""Original HoVLE runtime patches, promoted out of workspace diagnostics.

Only the three methods consumed by apply_original_runtime_patches are included.
Their source segments and ASTs are unchanged. Historical snapshots remain external
evidence; this is the versioned executable copy for the new source-layout YAML.
"""
from __future__ import annotations
import torch
import logging
logger = logging.getLogger(__name__)

def _patch_hovle_prepare_inputs(self) -> None:
        """修正 transformers 5.x 下 HoVLE prepare_inputs_for_generation 的两处问题。

        1) prefill 分支误判（仅 pixel_shuffle_loc=post 的 HD 变体受影响）：
           transformers 5.x 的 `_prefill()` 会带着一个**空 cache** 调用
           `prepare_inputs_for_generation`；HoVLE 远程代码只判断
           `past_key_values is not None`，于是 prefill 被当成 decode：
           返回 `input_embeds`（长度=LLM 序列 3592）配 `hd_attention_mask`
           （长度=hd 序列 13576），HolisticEmbedding 编码器随即报
           "Attention mask should be of size (1,1,3592,3592), but is [1,1,3592,13576]"。
           此处把「空 cache」归一化为 None，使其走 prefill 分支。
           非 HD 变体在 prefill 误判时两个 mask 长度一致，不受影响，故保持原行为。

        2) decode 阶段 position_ids 未裁剪：
           transformers 5.x 的 `_update_model_kwargs_for_generation` 会随解码
           逐 token 延长 `position_ids`（解码步长度 = 3593），而 HoVLE 的
           `prepare_inputs_for_generation` 只在 `position_ids is None` 时才按
           attention_mask 重算并裁剪，于是把整段 (1,3593) 的 position_ids
           传给只含 1 个 token 的 LLM 输入。`apply_rotary_pos_emb` 中
           `cos[position_ids]` 形状 (1,1,3593,128) 与 query (1,16,1,128) 广播，
           把 query 撑成 (1,16,3593,128)，最终 attn_weights=(1,16,3593,7185)，
           触发 "Attention weights should be of size (1,16,1,3593)"。
           此处按解码输入长度把 position_ids/hd_position_ids 裁到末尾。
        """
        model = self._model
        embedding_config = getattr(getattr(model, "embedding_model", None), "config", None)
        is_post = getattr(embedding_config, "pixel_shuffle_loc", None) == "post"
        orig = model.prepare_inputs_for_generation
        if getattr(orig, "_universal_eval_empty_cache_patched", False):
            return

        def _prepare(input_ids, past_key_values=None, **kwargs):
            if is_post and past_key_values is not None:
                empty = False
                try:
                    if hasattr(past_key_values, "get_seq_length"):
                        empty = past_key_values.get_seq_length() == 0
                    else:
                        pk = past_key_values[-1][0]
                        empty = pk is None or pk.shape[2] == 0
                except Exception:
                    empty = False
                if empty:
                    past_key_values = None
            out = orig(input_ids, past_key_values=past_key_values, **kwargs)
            # 解码阶段：position_ids 必须与对应输入的序列长度一致，否则 RoPE 广播
            if out.get("past_key_values") is not None:
                emb = out.get("input_embeds")
                if emb is not None:
                    pos = out.get("position_ids")
                    if pos is not None and pos.shape[1] != emb.shape[1]:
                        out["position_ids"] = pos[:, -emb.shape[1]:]
                hd_emb = out.get("hd_input_embeds")
                hd_pos = out.get("hd_position_ids")
                if (
                    hd_emb is not None
                    and hd_pos is not None
                    and hd_pos.shape[1] != hd_emb.shape[1]
                ):
                    out["hd_position_ids"] = hd_pos[:, -hd_emb.shape[1]:]
            return out

        _prepare._universal_eval_empty_cache_patched = True
        model.prepare_inputs_for_generation = _prepare
        logger.info(
            "HoVLE: 已修正 prepare_inputs_for_generation（空 cache 判定 + decode position_ids 裁剪）"
        )

def _patch_hovle_causal_mask(self) -> None:
        """把 Holistic Embedding 注意力的 O(q^2) 因果掩码构造换成 MLU 友好实现。

        远程代码 `InternLM2Attention.forward` 在收到 2D padding mask 时用
        `torch.zeros(bsz, 1, q_len, q_len)` + `torch.triu_indices(q_len, q_len)`
        + 高级索引赋值构造因果掩码。HD 变体（pixel_shuffle_loc=post）的 hd 序列
        q_len 可达 1.3 万，`triu_indices` 产生的 int64 索引张量在 MLU 上触发
        `CNRT error: "Device-side assert triggered"` 并污染设备，导致该样本及其后
        所有样本推理失败（response 全空）。
        此处用包装器在进入原 forward 之前把 2D mask 直接展开成 4D 因果掩码
        （广播比较，不产生索引张量），原 forward 的 2D 分支因此不再命中。
        """
        import torch

        def _make_wrapper(orig_forward):
            def _forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                **kwargs,
            ):
                if attention_mask is not None and attention_mask.dim() == 2:
                    bsz, q_len = hidden_states.shape[0], hidden_states.shape[1]
                    # kv_len 必须由 past_key_value + q_len 推出：transformers 只更新
                    # attention_mask，不会更新自定义的 hd_attention_mask，解码阶段后者
                    # 长度会比实际 kv 少 1，直接当 kv_len 会形状不匹配。
                    past_len = 0
                    if past_key_value is not None and past_key_value[0] is not None:
                        past_len = past_key_value[0].shape[-2]
                    kv_len = past_len + q_len
                    q_pos = torch.arange(q_len, device=hidden_states.device) + past_len
                    k_pos = torch.arange(kv_len, device=hidden_states.device)
                    causal = k_pos[None, :] > q_pos[:, None]
                    attention_mask = torch.zeros(
                        bsz,
                        1,
                        q_len,
                        kv_len,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ).masked_fill_(causal, -65504.0)
                return orig_forward(
                    self,
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )

            return _forward

        patched = 0
        seen = set()
        for module in self._model.modules():
            cls = type(module)
            if cls in seen or "forward" not in cls.__dict__:
                continue
            seen.add(cls)
            if "holistic_embedding" not in getattr(cls, "__module__", ""):
                continue
            # 只处理注意力类（RMSNorm/MLP 等同模块类不能接受 attention_mask）。
            if "Attention" not in cls.__name__:
                continue
            # flash 类由 _patch_flash_attn_to_eager 负责。
            if "_flash_attention_forward" in cls.__dict__:
                continue
            if getattr(cls, "_universal_eval_causal_mask_patched", False):
                continue
            cls.forward = _make_wrapper(cls.__dict__["forward"])
            cls._universal_eval_causal_mask_patched = True
            patched += 1
        if patched:
            logger.info(
                f"InternVL: 已替换 {patched} 个 HoVLE 注意力类的 2D→4D 因果掩码构造 (MLU)"
            )

def _patch_hovle(self) -> None:
        """针对 HoVLE 模型 on-the-fly 修补其 chat()/batch_chat() 方法。

        HoVLE 的官方 chat() 内部使用 .cuda() 把张量搬到 CUDA，而 MLU 环境
        (torch 未编译 CUDA) 会抛 "Torch not compiled with CUDA enabled"。
        此处将 .cuda() 全部替换为 .to(device)，使 chat() 在 MLU 上可用。
        """
        if not self.is_hovle():
            return
        model = self._model
        device = self._device
        logger.info("HoVLE: 检测到 Holistic Embedding 架构，修补 chat() 的 .cuda() 调用(MLU)")

        import inspect
        import textwrap
        import types

        def _patch_method(method_name: str):
            orig = getattr(model, method_name, None)
            if orig is None:
                return
            src = inspect.getsource(orig)
            # 将 .cuda() 替换为 .to(device)，使张量落到 MLU 而非 CUDA
            new_src = src.replace(".cuda()", f".to('{device}')")
            # HoVLE 的 chat() 体内 super() 是零参数形式，依赖 __class__ 自由变量 cell。
            # on-the-fly exec 重新编译后该 cell 丢失，触发 "super(): __class__ cell not found"。
            # 同时各类的 PreTrainedModel 在某些 MLU 兼容环境可能被移除 generate。
            # 因此把 super().generate( 显式改写为 self.generate(——实例调用走正常 MRO，
            # 既不依赖 __class__ cell，也不引用具体父类名，与 super().generate() 语义等价。
            new_src = new_src.replace("super().generate(", "self.generate(")
            # transformers 5.x 的 generate() 仅返回新生成的 token（不含 prompt），
            # 而 HoVLE 官方 chat() 按 4.x 语义做 [:, input_ids.shape[1]:] 切片。
            # 实测：input_ids=3592、out.shape=(1,2048) 时仍会切片，把回答开头整段
            # 切掉（response 从句子中间开始）；prompt 长度 >= max_new_tokens 时更会
            # 把 2048 个新 token 全部切掉，response 恒为空。
            # 改为按内容判定：仅当返回序列确实以 prompt 开头（4.x 语义）才切片。
            new_src = new_src.replace(
                "generation_output = generation_output[:, input_ids.shape[1]:]",
                "generation_output = (generation_output[:, input_ids.shape[1]:] "
                "if (generation_output.shape[1] >= input_ids.shape[1] and "
                "torch.equal(generation_output[0, :input_ids.shape[1]], input_ids[0])) "
                "else generation_output)",
            )
            # 该模型生成到 <|im_end|> 时不会停止（eos_token_id 未被 transformers 5.x
            # 生成循环采纳），会继续输出重复的 <|im_end|> 直到 max_new_tokens。
            # 保留特殊 token 参与解码，再由 chat() 自带的
            # response.split(template.sep)[0] 截断到第一个 <|im_end|>，
            # 等价于 4.x 下"遇到 eos 即停"的行为。
            new_src = new_src.replace(
                "tokenizer.batch_decode(generation_output, skip_special_tokens=True)",
                "tokenizer.batch_decode(generation_output, skip_special_tokens=False)",
            )
            new_src = textwrap.dedent(new_src)
            ns = dict(orig.__globals__)
            exec(compile(new_src, "<hovle_patch>", "exec"), ns)
            patched = ns[method_name]
            setattr(model, method_name, types.MethodType(patched, model))
            logger.info(f"HoVLE: 已修补 {method_name}()")

        _patch_method("chat")
        _patch_method("batch_chat")

        # transformers 6.x 的 generate() 在 _prepare_cache_for_generation() 阶段会访问
        # config.num_hidden_layers 来初始化 DynamicCache；而 HoVLE 的 InternVLChatConfig
        # 顶层并不暴露该属性（层数位于 embedding_config/language_model.config 中），
        # 导致 generate 报 "'InternVLChatConfig' object has no attribute 'num_hidden_layers'"。
        # 此处为 config 实例补一个动态属性：优先取语言模型(实际生成载体)的层数，否则回退 embedding 层数。
        if not hasattr(model.config, "num_hidden_layers"):
            n_layers = None
            lm = getattr(model, "language_model", None)
            if lm is not None and getattr(getattr(lm, "config", None), "num_hidden_layers", None):
                n_layers = lm.config.num_hidden_layers
            else:
                ec = getattr(model.config, "embedding_config", None)
                if ec is not None and getattr(ec, "num_hidden_layers", None):
                    n_layers = ec.num_hidden_layers
            if n_layers is not None:
                model.config.num_hidden_layers = n_layers
                logger.info(f"HoVLE: 为 config 补充 num_hidden_layers={n_layers}")
