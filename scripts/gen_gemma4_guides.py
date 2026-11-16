#!/usr/bin/env python3
"""生成 Gemma4 组模型的 adaptation guide（从最新评测报告读取指标）。"""
import json, os, time

BASE = 'workspace/reports'
OUT = 'configs/Gemma4ForConditionalGeneration'
CONV = '/mnt2/afs/converted_models'

def latest_report(stem_dash):
    rdir = f'{BASE}/{stem_dash.replace("--", "_")}/image-text-to-text'
    if not os.path.isdir(rdir):
        return None
    for d in sorted(os.listdir(rdir), reverse=True):
        rp = os.path.join(rdir, d, 'eval_report.json')
        if os.path.exists(rp):
            try:
                return json.load(open(rp))
            except Exception:
                continue
    return None

def fmt_metrics(stem):
    j = latest_report(stem)
    if not j:
        return '- （暂无报告）', ''
    m = j['metrics']['Accuracy']
    st = j['statistics']
    total = st['total_samples']
    n = st['success_count']
    el = st['elapsed_seconds']
    if total >= 200:
        line = f'- 全量评测(200样本): Accuracy **{m}**（{n}/200 成功，耗时 {el:.0f}s）'
    else:
        line = f'- 冒烟测试(8样本): Accuracy **{m}**（{n}/8 成功，耗时 {el:.0f}s）'
    return line, m

MLX_TP = {'gemma-4-E4B-it-MLX-8bit': 2, 'gemma-4-26b-a4b-it-MLX-8bit': 2,
          'gemma-4-26b-a4b-it-UD-MLX-3bit': 2, 'gemma-4-26b-a4b-it-UD-MLX-4bit': 2,
          'gemma-4-31b-it-MLX-8bit': 2, 'gemma-4-31b-it-UD-MLX-4bit': 2}

def mlx_guide(org, stem):
    repo = f'{org}/{stem}'
    conv = f'{CONV}/{stem}'
    tp = MLX_TP.get(stem, 2)
    moe = '- **MoE 专家融合**: 26B-A4B 模型的 MLX 权重使用 `switch_glu.{gate,up,down}_proj` 存储，'\
          '需融合为 HF 的 `experts.gate_up_proj`（gate 与 up 沿输出维拼接）与 `experts.down_proj`；'\
          '专家权重分散在不同 safetensors 分片，需跨分片收集。否则模型输出乱码。\n' if '26b' in stem else ''
    l1, m1 = fmt_metrics(f'{org}--{stem}')
    lines = f"""# {repo} 适配指南

## 评测结果
- 任务类型: image-text-to-text（mmbench 选择题，Accuracy 指标）
{l1}
- 推理框架: Transformers 5.14.1（bf16，device_map=auto，TP={tp}，2×A100 40GB）
- 适配器: `universal_eval/model/adapters/transformers_vlm.py:TransformersVLM`
- 可运行权重路径: `{conv}`

## 适配思路总结
原始权重为 MLX 量化格式（U32 packed 量化核 + BF16 scales/biases 的 affine 量化），
vLLM / Transformers 均无法直接加载，Linux/A100 上也没有 MLX 运行时。
方案：用 `scripts/convert_mlx_quant_to_hf.py` 离线转换为标准 HF safetensors（反量化为 BF16），
再用通用 TransformersVLM 适配器加载评测。

### 转换要点（convert_mlx_quant_to_hf.py）
1. **位流解包**: MLX 将 W-bit 量化权重按小端位序打包为 U32 数组，支持 2/3/4/6/8-bit；
   3-bit 等不能整除 32 的位宽跨 word 连续打包，需按位流整体解包。
2. **反量化**: `w = q * scales + biases`（group_size=32/64，scales/biases 为 BF16），转 BF16 保存。
3. **Key remap**: `language_model.model.* -> model.language_model.*`、
   `vision_tower.* -> model.vision_tower.*`、`multi_modal_projector.* -> model.multi_modal_projector.*`。
{moe}
4. **补 preprocessor_config.json**: 转换目录若缺该文件，Gemma4Processor 无法展开 `<|image|>` 占位符，
   多模态推理报 `'bool' object has no attribute 'all'`；写入
   `{{"processor_class": "Gemma4Processor"}}` 即可修复。
5. 生成 `model.safetensors.index.json` 分片索引与 `.complete` 完成标记，支持断点续转。

## 关键配置
```yaml
model:
  name: {repo}
  adapter: universal_eval/model/adapters/transformers_vlm.py:TransformersVLM
  model_path: {conv}
  download: false
  kwargs:
    device: auto          # device_map=auto 分布到 2 张卡
    max_new_tokens: 2048
    torch_dtype: bfloat16
```

## 平台迁移注意事项
- 转换一次即可复用：产物为标准 HF BF16 权重，任何支持 Gemma4ForConditionalGeneration 的
  框架（Transformers / vLLM）都可直接加载。
- 反量化后体积约为原 MLX 权重的 2 倍（BF16），26B/31B 级别需 2×40GB 显存（device_map=auto）。
- 3bit/4bit/8bit 为 MLX 有损量化，精度差异来自量化本身而非适配问题；
  可用同 base 的 BF16 模型对转换结果做余弦相似度交叉验证（≈1.0）。
- tokenizer/processor 全部来自原 MLX 目录（tokenizer.json、chat_template.jinja 等），直接复制。
"""
    open(f'{OUT}/{org}--{stem}_adaptation_guide.md', 'w').write(lines)
    print('written', f'{org}--{stem}_adaptation_guide.md')

def hlwq_guide():
    org, stem = 'caiovicentino1', 'Gemma-4-31B-Claude-Opus-HLWQ-Q5-Vision'
    conv = f'{CONV}/{org}--{stem}'
    l1, _ = fmt_metrics(f'{org}--{stem}')
    txt = f"""# {org}/{stem} 适配指南

## 评测结果
- 任务类型: image-text-to-text（mmbench 选择题，Accuracy 指标）
{l1}
- 推理框架: Transformers 5.14.1（bf16，device_map=auto，TP=2，2×A100 40GB，占用 ~66GB）
- 适配器: `universal_eval/model/adapters/transformers_vlm.py:TransformersVLM`
- 可运行权重路径: `{conv}`

## 适配思路总结
原始权重为自研 HLWQ INT4 量化格式（`model_int4.pt` + `hlwq_config.json`），
量化布局为 torchao `TensorCoreTiledLayout`（AffineQuantizedTensor 的一种 packed layout），
标准 Transformers/vLLM 均不识别。
方案：用 `scripts/convert_hlwq_int4_to_hf.py` 反量化为 BF16 后按 HF 分片保存，再以 TransformersVLM 评测。

### 转换要点（convert_hlwq_int4_to_hf.py）
1. `torch.load(model_int4.pt)` 得到 `AffineQuantizedTensor`（TensorCoreTiledLayout，
   int4 数据 + scale/zero_point），调用 `.dequantize()` 直接还原 BF16 权重。
2. 反量化需在 GPU 上分块进行（每次一块 Linear），避免一次性 60GB+ 峰值；
   注意 CUDA_VISIBLE_DEVICES 重映射后设备序号从 0 开始。
3. 保留原目录的 tokenizer / chat_template / preprocessor/processor 配置；
   生成 `model.safetensors.index.json`（15 分片，共 65.4GB）与 `.complete` 标记。

## 关键配置
```yaml
model:
  name: {org}/{stem}
  adapter: universal_eval/model/adapters/transformers_vlm.py:TransformersVLM
  model_path: {conv}
  download: false
  kwargs:
    device: auto
    max_new_tokens: 2048
    torch_dtype: bfloat16
```

## 平台迁移注意事项
- 反量化后为 31B BF16（~66GB），必须 2×40GB 或 1×80GB 显存；若目标平台显存不足，
  可在转换脚本中将 `.dequantize()` 替换为目标量化方案（如 GPTQ/AWQ）再保存。
- torchao 版本相关：`is_torchao_available` 的版本门槛检查已在适配器中 patch
  （torchao 0.11 环境），迁移时注意 torchao 与 torch 版本匹配。
- 该模型为社区微调 + INT4 量化版，冒烟/全量答案以 `<answer>X</answer>` 标签或首字母形式输出，
  评测框架的 extract_answer 逻辑可直接解析。
"""
    open(f'{OUT}/{org}--{stem}_adaptation_guide.md', 'w').write(txt)
    print('written hlwq guide')

def onnx_guide(size):
    org, stem = 'onnx-community', f'gemma-4-{size}-it-ONNX'
    src = f'/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration/{org}/{stem}'
    l1, _ = fmt_metrics(f'{org}--{stem}')
    txt = f"""# {org}/{stem} 适配指南

## 评测结果
- 任务类型: image-text-to-text（mmbench 选择题，Accuracy 指标）
{l1}
- 推理框架: onnxruntime 1.23.2（CPUExecutionProvider，手写推理循环）
- 适配器: `universal_eval/model/adapters/gemma4_ort.py:Gemma4ORTModel`
- 模型路径: `{src}`

## 适配思路总结
onnx-community 的导出为 HF-optimum 风格三组件 ONNX：
`embed_tokens.onnx`（input_ids → inputs_embeds + per_layer_inputs）、
`vision_encoder.onnx`（pixel_values + image_position_ids → image_features）、
`decoder_model_merged.onnx`（inputs_embeds + per_layer_inputs + past_key_values.* → logits + present.*）。
它使用 `past_key_values.N.key` 命名与 `per_layer_inputs`（Gemma4 E 系列 per-layer embeddings），
与 onnxruntime-genai 的 `past_key_values.N.decoder.key` 约定不兼容且 ORT GenAI 0.11.4
不支持 per-layer embeddings —— 因此手写 onnxruntime 生成循环适配器。

### 适配器要点（gemma4_ort.py）
1. **处理器**: `AutoProcessor.from_pretrained` 加载 Gemma4Processor（tokenizer + 图像处理器），
   chat template 生成含 `<|image|>`（id=258880）的输入序列。
2. **图像融合**（与 transformers Gemma4ForConditionalGeneration 一致）:
   input_ids 中 image_token 位置先替换为 pad 再查嵌入表；vision_encoder 输出的 image_features
   行数与 image token 数严格一致，masked_scatter 回这些位置；per_layer_inputs 保持 pad 查表值不变。
3. **生成循环**: prefill（零长度 past，num_logits_to_keep=1）→ 逐 token 解码，
   past/present 按名称映射（present.N.* → past_key_values.N.*），greedy argmax，
   eos 集合来自 generation_config.json（[1, 106, 50]）。
4. **混合精度**: "fp16" 导出实际为混合精度（embed/IO fp32，KV cache fp16），
   馈送张量按 session 输入声明 dtype 逐一 cast。

### ONNX Runtime 兼容性坑（重要）
- 图为 opset 21：默认图优化会引入 CUDA EP 无 kernel 的 `Squeeze-21` 节点导致会话初始化失败；
  `graph_optimization_level=ORT_DISABLE_ALL` 后可正常按原始图分配 EP。
- 融合算子 `GroupQueryAttention`（com.microsoft）的 CUDA kernel 不支持
  `position_ids` / `attention_bias` 输入 → decoder 目前只能跑 CPU EP
  （官方 README 示例同样使用 CPUExecutionProvider）。其余会话（embed/vision）在 GPU 空闲时可用 CUDA。
- CPU 上 fp32 图远快于 fp16 模拟：E2B 24 线程 prefill ~170 tok/s、decode ~2.7 tok/s。

## 关键配置
```yaml
model:
  name: {org}/{stem}
  adapter: universal_eval/model/adapters/gemma4_ort.py:Gemma4ORTModel
  model_path: {src}
  download: false
  kwargs:
    device: cpu           # decoder 的 GQA CUDA kernel 限制，见上
    max_new_tokens: 2048
    variant: fp16         # CPU 下自动回落到 fp32 图
    threads: 24
```

## 平台迁移注意事项
- 若目标平台 onnxruntime 升级到支持 GQA(position_ids/attention_bias) 的版本，
  可将 device 改为 cuda 并用 fp16 图获得 GPU 加速；适配器会自动尝试 CUDA 并在失败时回落 CPU。
- E2B/E4B 均含 audio_encoder（本评测未使用音频输入，未加载该会话）。
- 图像处理器输出 `image_position_ids`（对应 vision 图的 `pixel_position_ids` 输入），注意键名映射。
"""
    open(f'{OUT}/{org}--{stem}_adaptation_guide.md', 'w').write(txt)
    print('written', f'{org}--{stem}_adaptation_guide.md')

mlx_guide('unsloth', 'gemma-4-E4B-it-MLX-8bit')
mlx_guide('unsloth', 'gemma-4-26b-a4b-it-MLX-8bit')
mlx_guide('unsloth', 'gemma-4-26b-a4b-it-UD-MLX-3bit')
mlx_guide('unsloth', 'gemma-4-26b-a4b-it-UD-MLX-4bit')
mlx_guide('unsloth', 'gemma-4-31b-it-MLX-8bit')
mlx_guide('unsloth', 'gemma-4-31b-it-UD-MLX-4bit')
hlwq_guide()
onnx_guide('E2B')
onnx_guide('E4B')
print('ALL GUIDES DONE')
