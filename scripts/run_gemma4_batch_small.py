#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma4 分批小步评测执行器

用法:
    python scripts/run_gemma4_batch_small.py "org/model1:tp" "org/model2:tp" ...

每个模型自动完成: 生成配置 -> 8 样本冒烟 -> 200 样本全量 -> 报告验证。
按 tp 分组并发，充分利用 4 张 A100：
    tp=1: 4 个并发（各 1 卡）
    tp=2: 2 个并发（各 2 卡）
    tp=4: 1 个（独占 4 卡）
"""

import os
import subprocess
import time
import sys
import json
import yaml

PROJECT_ROOT = "/mnt/afs/comp_error"
MODEL_BASE = "/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration"
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs", "Gemma4ForConditionalGeneration")
MMBENCH = "/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl"
REPORT_ROOT = os.path.join(PROJECT_ROOT, "workspace", "reports")
PY = "/usr/bin/python"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)

def config_name(model_name):
    return model_name.replace("/", "--")

def ensure_config(model_name, tp):
    cn = config_name(model_name)
    path = os.path.join(CONFIG_DIR, f"{cn}.yaml")
    mpath = os.path.join(MODEL_BASE, model_name)
    if not os.path.isdir(mpath):
        log(f"  [警告] 模型目录不存在: {mpath}")
        return None
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

def verify_report(model_name):
    cn = config_name(model_name)
    base = os.path.join(REPORT_ROOT, cn, "image-text-to-text")
    if not os.path.isdir(base):
        return False, base, {}
    dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    if not dirs:
        return False, base, {}
    latest = os.path.join(base, dirs[-1])
    rlog, rep, olin = os.path.join(latest, "run.log"), os.path.join(latest, "eval_report.json"), os.path.join(latest, "operator_lineage.json")
    if not (os.path.exists(rlog) and os.path.exists(rep) and os.path.exists(olin)):
        return False, latest, {}
    if "推理失败" in open(rlog, encoding="utf-8").read():
        return False, latest, {}
    d = json.load(open(rep, encoding="utf-8"))
    total = (d.get("statistics") or {}).get("total_samples")
    acc = (d.get("metrics") or {}).get("Accuracy")
    stats = d.get("statistics") or {}
    return (total == 200 and acc is not None and acc > 0), latest, stats

def run_batch(batch):
    """batch: list of (model, tp, gpus). 并发执行冒烟+全量。"""
    procs = {}
    for model, tp, gpus in batch:
        cfg = ensure_config(model, tp)
        if cfg is None:
            results[model] = "NO_MODEL_DIR"
            continue
        log(f"[冒烟] {model} tp={tp} gpus={gpus}")
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
        ok = ("推理失败" not in outp
              and "评测报告已保存至" in outp
              and "推理循环完成" in outp)
        passed[model] = ok
        if not ok:
            log(f"[冒烟失败] {model}\n{outp[-1500:]}")
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
        ok, rep_dir, stats = verify_report(model)
        if ok:
            acc = stats.get("metrics", {}) if False else None
            log(f"[成功] {model} -> {rep_dir}")
            results[model] = "PASS"
        else:
            log(f"[失败] {model}\n{outp[-1500:]}")
            results[model] = "FAIL"

results = {}

def main():
    specs = sys.argv[1:]
    if not specs:
        log("用法: run_gemma4_batch_small.py 'org/model:tp' ...")
        return
    todo = []
    for s in specs:
        if ":" in s:
            m, tp = s.rsplit(":", 1)
            todo.append((m, int(tp)))
        else:
            todo.append((s, 1))
    log(f"=== 本批 {len(todo)} 个模型 ===")
    for m, tp in todo:
        log(f"  - {m} (tp={tp})")
    # 分组并发
    tp1 = [(m, tp) for m, tp in todo if tp == 1]
    tp2 = [(m, tp) for m, tp in todo if tp == 2]
    tp4 = [(m, tp) for m, tp in todo if tp == 4]
    for i in range(0, len(tp1), 4):
        grp = tp1[i:i+4]
        batch = [(m, tp, [j % 4]) for j, (m, tp) in enumerate(grp)]
        log(f"--- 单卡批 ({len(grp)}) ---")
        run_batch(batch)
    for i in range(0, len(tp2), 2):
        grp = tp2[i:i+2]
        batch = [(m, tp, [j*2, j*2+1]) for j, (m, tp) in enumerate(grp)]
        log(f"--- 双卡批 ({len(grp)}) ---")
        run_batch(batch)
    for m, tp in tp4:
        log("--- 四卡批 ---")
        run_batch([(m, tp, [0, 1, 2, 3])])
    log("=== 本批完成 ===")
    for m, s in results.items():
        log(f"  {m}: {s}")

if __name__ == "__main__":
    main()