# 910B A100 对齐评测代码

这里保存当前 910B 评测入口、模型适配器、独立验收代码与模型 YAML。代码来自运行机器的明确依赖清单，不包含原 Gitee 仓库历史、权重、数据集、评测结果或凭据。

每款模型由独立 YAML 直接运行，例如：

```sh
python3 run.py --config configs/InternVLChatModel/OpenGVLab--HoVLE-HD-910b-original-BF16-source-layout-v5.yaml
```

运行需要评测机器现有的 Ascend/PyTorch 环境、原权重、数据集和 SHA 绑定的外部证据。克隆此仓库不会下载这些资源，也不会安装依赖。历史 YAML 中的绝对路径保留真实部署含义，不能直接解释为可在任意机器脱离资源运行。

- `universal_eval/`：框架、算法和模型实现。
- `scripts/`：调度、校验和维护代码。
- `configs/`：独立模型 YAML 与真正被读取的运行配置。
- `tests/`：代码发布边界及迁移检查。
- `workspace/`：仅运行数据、完整报告、轨迹和历史证据，整个目录不提交。

完整评测仍保存五个原始文件：`eval_report.json`、`sample_inputs.jsonl`、`sample_outputs.jsonl`、`run.log`、`operator_lineage.json`。源码和 YAML 不放进报告目录。历史报告引用的源码快照作为不可变证据保留；新执行代码放在源码目录。

本次将实际执行的 HoVLE 三个历史补丁迁至 `universal_eval/model/reference_sources/hovle_original_patches.py`，并新增相应协议与 YAML。函数原文和 AST 均与原来源一致；旧配置、旧报告和正在执行的冻结代码保持原字节。此迁移不代表新增一次真机评测或更新精度成绩。

## 每日代码提交

使用独立分支 `codex/910b-source-layout-20261114`，先不合并。原 910b 每 300 分钟检查继续；每个北京时间自然日首次检查时，审核当天的代码变化，有变化提交一次，没有变化不生成空提交。

发布采用 `configs/publication/code_only.json` 的明确路径清单。新增实现先放入源码目录，审核用途、依赖和内容后加入清单。外部报告、权重和数据不能通过扩展清单进入仓库。

```sh
python3 -B -m unittest discover -s tests -v
python3 -B scripts/publish_code_daily.py plan --source-root SOURCE --repo-root CHECKOUT --policy configs/publication/code_only.json
# 审核完整 plan/diff，并完成相应代码检查后：
python3 -B scripts/publish_code_daily.py publish --source-root SOURCE --repo-root CHECKOUT --policy configs/publication/code_only.json --expected-plan-sha PLAN_SHA
```

`SOURCE` 是本次逐文件核 SHA 后的代码快照，`CHECKOUT` 是该 GitHub 分支的独立工作副本。工具拒绝未审核路径、工作区产物、权重、二进制大文件和明显凭据，精确暂存清单中的变化，并核对推送后的远端提交。不执行合并或强制推送。推送失败时保留本地提交，先排查认证或远端变化，不制造重复提交。

## Unsloth Gemma-3n-E2B-it

独立入口为 `configs/Gemma3nForConditionalGeneration/unsloth--gemma-3n-E2B-it-native-BF16-chat-2048-20261115.yaml`。它使用本款原有 BF16 权重、自带对话模板、2048 生成预算、Greedy 和 EOS `[1,106]`，以指定表格 D 列成绩比较 5% 相对误差。该实现采用 Ascend 原生 HF SDPA、外层 10 条及内层逐条计算；与保存的 vLLM 部署记录之间的运行时边界会在报告中披露，不声称捕获了历史 A100 引擎或张量。

共享 MobileNet `.ir` 文件仅包含形状和算子代码，所有学习参数都由评测机器上的本款权重装载。发布策略逐 SHA 审核该文件，并拒绝图中内嵌张量常量。只读 NPZ 索引器用于独立复核，不修改正在运行的旧验收或模型计算。配置检查和 CPU 输入准备不代表已完成真实 NPU 小样本或 200 条评测；这些阶段及五个完整报告仍须实际执行和验收。
