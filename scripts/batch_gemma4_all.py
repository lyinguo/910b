#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma4ForConditionalGeneration 未评测模型批量评测调度器

按模型大小分组并发，最大化 4 张 A100 利用率：
  - 单卡模型 (tp=1): 4 个不同模型并发，各占 1 张卡
  - 双卡模型 (tp=2): 2 个不同模型并发，各占 2 张卡
  - 四卡模型 (tp=4): 1 个模型独占 4 张卡

每个模型：生成配置 -> 8 样本冒烟测试 -> 200 样本全量评测 -> 验证
"""

import os
import subprocess
import time
import json
import yaml

PROJECT_ROOT = "/mnt/afs/comp_error"
MODEL_BASE = "/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration"
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs", "Gemma4ForConditionalGeneration")
MMBENCH = "/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl"
REPORT_ROOT = os.path.join(PROJECT_ROOT, "workspace", "reports")
LOG = os.path.join(REPORT_ROOT, "gemma4_batch_all.log")
PY = "/usr/bin/python"

# 待评测模型: (org/model, tp)
MODELS = [
    ("DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe-HERETIC-UNCENSORED-Thinking", 1),
    ("Jackrong/Gemopus-4-E4B-it", 1),
    ("cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit", 1),
    ("huihui-ai/Huihui-gemma-4-E2B-it-abliterated", 1),
    ("huihui-ai/Huihui4-8B-A4B", 1),
    ("huihui-ai/Huihui4-8B-A4B-v2", 1),
    ("llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic", 1),
    ("unsloth/gemma-4-E4B-it-unsloth-bnb-4bit", 1),
    ("z-lab/gemma-4-31B-it-PARO", 1),
    ("RedHatAI/gemma-4-31B-it-NVFP4", 1),
    ("RedHatAI/gemma-4-31B-it-FP8-block", 1),
    ("cyankiwi/gemma-4-31B-it-AWQ-8bit", 1),
    ("AuriAetherwiing/G4-31B-Musica-v1", 2),
    ("aifeifei798/Gemma-4-31B-Cognitive-Unshackled", 2),
    ("aifeifei798/Gemma-4-Queen-31B-it", 2),
    ("huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2", 2),
    ("jenerallee78/gemma-4-26B-A4B-it-ara-abliterated", 2),
    ("sophosympatheia/Glistening-Gem-31B-v1.0", 2),
    ("trohrbaugh/gemma-4-31b-it-heretic-ara", 2),
    ("virtuous7373/Gemma-4-Harmonia-31B", 2),
    ("TeichAI/gemma-4-26B-A4B-it-Claude-Opus-Distill-v2", 2),
    ("trohrbaugh/gemma-4-26B-A4B-it-heretic-ara", 2),
    ("huihui-ai/Huihui4-48B-A4B-abliterated", 4),
]

def load_evaluated():
    ev = set()
    path = os.path.join(REPORT_ROOT, "eval_summary.csv")
    if os.path.exists(path):
        for ln in open(path).readlines()[1:]:
            ev.add(ln.split(",")[0].strip())
    return ev

EVALUATED = load_evaluated()

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def config_name(model_name):
    return model_name.replace("/", "--")

def ensure_config(model_name, tp):
    cn = config_name(model_name)
    path = os.path.join(CONFIG_DIR, f"{cn}.yaml")
    mpath = os.path.join(MODEL_BASE, model_name)
    cfg = {
        "name": cn,
        "description": f"{model_name} (Gemma4ForConditionalGeneration) 全流程评测管线",
        "task_type": "image-text-to-text",
        "sample_range": "0:199",
        "data": {"preprocess": True, "apply_template": False, "filter_empty": True,
                 "dataset_path": MMBENCH},
        "model": {"name": model_name,
                  "adapter": "universal_eval/model/adapters/gemma4_vllm.py:Gemma4VLLMModel",
                  "model_path": mpath, "download": False,
                  "kwargs": {"tensor_parallel_size": tp, "max_tokens": 2048,
                             "max_model_len": 4096, "gpu_memory_utilization": 0.9,
                             "trust_remote_code": True}},
        "algorithm": {"name": "greedy_inference", "batch_size": 2,
                      "kwargs": {"extract_answer": True}},
        "metrics": {"override": ["Accuracy"]},
        "baseline": {"score": 0.0, "threshold": 0.05},
        "output": {"verbose": True},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path

def run_smoke(cfg_path, model_name, gpus):
    out = f"smoke_gemma4_{config_name(model_name)}.json"
    env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "VLLM_USE_V1": "0"}
    cmd = [PY, "run.py", "--config", cfg_path, "--sample_range", "0:7", "--output", out]
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True,
                           text=True, timeout=1800)
        output = (r.stdout or "") + (r.stderr or "")
        return (r.returncode == 0 and "推理失败" not in output), output
    except subprocess.TimeoutExpired:
        return False, "SMOKE TIMEOUT"

def run_full(cfg_path, model_name, gpus):
    """gpus 为该模型分配的 GPU 列表。"""
    env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "VLLM_USE_V1": "0"}
    cards = len(gpus)
    cmd = [PY, "scripts/parallel_model_eval.py", "--config", cfg_path,
           "--cards", str(cards), "--gpus-per-shard", str(cards), "--sample-range", "0:199"]
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True,
                           text=True, timeout=7200)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "FULL TIMEOUT"

def verify_report(model_name):
    cn = config_name(model_name)
    base = os.path.join(REPORT_ROOT, cn, "image-text-to-text")
    if not os.path.isdir(base):
        return False, base
    dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    if not dirs:
        return False, base
    latest = os.path.join(base, dirs[-1])
    rlog, rep, olin = os.path.join(latest, "run.log"), os.path.join(latest, "eval_report.json"), os.path.join(latest, "operator_lineage.json")
    if not (os.path.exists(rlog) and os.path.exists(rep) and os.path.exists(olin)):
        return False, latest
    if "推理失败" in open(rlog, encoding="utf-8").read():
        return False, latest
    d = json.load(open(rep, encoding="utf-8"))
    total = (d.get("statistics") or {}).get("total_samples")
    acc = (d.get("metrics") or {}).get("Accuracy")
    return (total == 200 and acc is not None and acc > 0), latest

def process_concurrent(batch, results):
    """并发处理一批 (model, tp, gpus) 的完整流程。"""
    procs = {}
    for model, tp, gpus in batch:
        cfg = ensure_config(model, tp)
        log(f"[冒烟] {model} gpus={gpus}")
        env_base = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "VLLM_USE_V1": "0"}
        out = f"smoke_gemma4_{config_name(model)}.json"
        cmd = [PY, "run.py", "--config", cfg, "--sample_range", "0:7", "--output", out]
        p = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env={**os.environ, **env_base},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs[model] = (p, tp, gpus, cfg)
    # wait smoke
    passed = {}
    for model, (p, tp, gpus, cfg) in procs.items():
        so, se = p.communicate(timeout=1800)
        outp = (so or "") + (se or "")
        # 冒烟成功判定：不依赖 returncode（run.py 在 gap 超基线时返回 exit 1 但推理成功），
        # 仅检查推理是否真失败 + 报告是否成功生成
        ok = ("推理失败" not in outp
              and "评测报告已保存至" in outp
              and "推理循环完成" in outp)
        passed[model] = ok
        if not ok:
            log(f"[冒烟失败] {model}\n{outp[-1200:]}")
        else:
            log(f"[冒烟通过] {model}")
    # full eval for smoke-passed, concurrently
    full_procs = {}
    for model, (p, tp, gpus, cfg) in procs.items():
        if not passed[model]:
            results[model] = "SMOKE_FAIL"
            continue
        log(f"[全量] {model} gpus={gpus}")
        cards = len(gpus)
        envf = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "VLLM_USE_V1": "0"}
        cmd = [PY, "scripts/parallel_model_eval.py", "--config", cfg,
               "--cards", str(cards), "--gpus-per-shard", str(cards), "--sample-range", "0:199"]
        full_procs[model] = (subprocess.Popen(cmd, cwd=PROJECT_ROOT, env={**os.environ, **envf},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), cfg)
    for model, (p, cfg) in full_procs.items():
        so, se = p.communicate(timeout=7200)
        outp = (so or "") + (se or "")
        ok, rep_dir = verify_report(model)
        if ok:
            log(f"[成功] {model} -> {rep_dir}")
            results[model] = "PASS"
        else:
            log(f"[失败] {model}\n{outp[-1200:]}")
            results[model] = "FAIL"

def main():
    log("=== Gemma4 批量评测启动 ===")
    todo = [(m, tp) for m, tp in MODELS if m not in EVALUATED]
    skipped = [(m, tp) for m, tp in MODELS if m in EVALUATED]
    log(f"待评测: {len(todo)}, 已评测跳过: {len(skipped)}")
    for m, tp in skipped:
        log(f"  跳过(已评测): {m}")

    # 分组
    tp1 = [x for x in todo if x[1] == 1]
    tp2 = [x for x in todo if x[1] == 2]
    tp4 = [x for x in todo if x[1] == 4]

    results = {}
    # 处理单卡组 (4 并发)
    for i in range(0, len(tp1), 4):
        grp = tp1[i:i+4]
        batch = []
        for j, (m, tp) in enumerate(grp):
            batch.append((m, tp, [j]))  # gpu j
        log(f"--- 单卡批 ({len(grp)} 模型) ---")
        process_concurrent(batch, results)
    # 处理双卡组 (2 并发)
    for i in range(0, len(tp2), 2):
        grp = tp2[i:i+2]
        batch = []
        for j, (m, tp) in enumerate(grp):
            batch.append((m, tp, [j*2, j*2+1]))  # gpu 0,1 / 2,3
        log(f"--- 双卡批 ({len(grp)} 模型) ---")
        process_concurrent(batch, results)
    # 处理四卡组
    for m, tp in tp4:
        log("--- 四卡批 ---")
        process_concurrent([(m, tp, [0,1,2,3])], results)

    log("=== 批量评测完成 ===")
    for m, s in results.items():
        log(f"  {m}: {s}")

if __name__ == "__main__":
    main()