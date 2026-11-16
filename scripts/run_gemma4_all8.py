#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma4ForConditionalGeneration 未评测模型 —— 8 卡动态 GPU 池调度器

依据 workspace/gemma4_evaluable.json 权威清单（decided_adapter + tp）：
  - 评测前检查模型是否完整（status==EVALUABLE）+ 是否已成功评测（report_valid）
  - 每个模型：生成正确 adapter 配置 -> 8 样本冒烟 -> 200 样本全量 -> 报告验证
  - 动态 GPU 池：空闲 GPU>=tp 即启动，充分利用全部 8 卡
  - vLLM 适配器用 tensor_parallel_size=tp；transformers 适配器用 device_map=auto
"""

import os
import subprocess
import time
import json
import yaml
import threading
from datetime import datetime

PROJECT_ROOT = "/mnt/afs/comp_error"
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs", "Gemma4ForConditionalGeneration")
MMBENCH = "/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl"
REPORT_ROOT = os.path.join(PROJECT_ROOT, "workspace", "reports")
MANIFEST = os.path.join(PROJECT_ROOT, "workspace", "gemma4_evaluable.json")
LOG = os.path.join(REPORT_ROOT, "gemma4_all8.log")
PY = "/mnt/afs/miniconda3/envs/lvzhiqiang/bin/python"
NUM_GPUS = 8

ADAPTER_MAP = {
    "gemma4_vllm.py:Gemma4VLLMModel": ("universal_eval/model/adapters/gemma4_vllm.py:Gemma4VLLMModel", True),
    "gemma4_base_vllm.py:Gemma4BaseVLLMModel": ("universal_eval/model/adapters/gemma4_base_vllm.py:Gemma4BaseVLLMModel", True),
    "transformers_vlm.py:TransformersVLM": ("universal_eval/model/adapters/transformers_vlm.py:TransformersVLM", False),
}


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def config_name(model_name):
    return model_name.replace("/", "--")


def ensure_config(entry):
    """为模型生成正确 adapter 的配置，返回 config_path。"""
    model_name = entry["name"]
    tp = entry["tp"]
    cn = config_name(model_name)
    path = os.path.join(CONFIG_DIR, f"{cn}.yaml")
    mpath = entry["path"]
    adapter_path, is_vllm = ADAPTER_MAP[entry["decided_adapter"]]

    if is_vllm:
        mcfg = {
            "name": cn,
            "description": f"{model_name} (Gemma4ForConditionalGeneration) 全流程评测管线",
            "task_type": "image-text-to-text",
            "sample_range": "0:199",
            "data": {"preprocess": True, "apply_template": False, "filter_empty": True,
                     "dataset_path": MMBENCH},
            "model": {"name": model_name, "adapter": adapter_path,
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
    else:
        mcfg = {
            "name": cn,
            "description": f"{model_name} (Gemma4ForConditionalGeneration) 全流程评测管线",
            "task_type": "image-text-to-text",
            "sample_range": "0:199",
            "data": {"preprocess": True, "apply_template": False, "filter_empty": True,
                     "dataset_path": MMBENCH},
            "model": {"name": model_name, "adapter": adapter_path,
                      "model_path": mpath, "download": False,
                      "kwargs": {"device": "auto", "max_new_tokens": 2048,
                                 "torch_dtype": "auto"}},
            "algorithm": {"name": "greedy_inference", "batch_size": 10,
                          "kwargs": {"extract_answer": True}},
            "metrics": {"override": ["Accuracy"]},
            "baseline": {"score": 0.0, "threshold": 0.05},
            "output": {"verbose": True},
        }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mcfg, f, allow_unicode=True, sort_keys=False)
    return path


def verify_report(model_name):
    # 报告目录命名存在两种约定：/ -> "--"（config_name）与 / -> "_"，需同时检查
    bases = [
        os.path.join(REPORT_ROOT, config_name(model_name), "image-text-to-text"),
        os.path.join(REPORT_ROOT, model_name.replace("/", "_"), "image-text-to-text"),
    ]
    base = next((b for b in bases if os.path.isdir(b)), bases[0])
    if not os.path.isdir(base):
        # 兼容 models 目录名，去掉可能的 "image-text-to-text" 层级差异
        for b in bases:
            if os.path.isdir(b):
                base = b
                break
        else:
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
    try:
        d = json.load(open(rep, encoding="utf-8"))
    except Exception:
        return False, latest, {}
    total = (d.get("statistics") or {}).get("total_samples")
    acc = (d.get("metrics") or {}).get("Accuracy")
    return (total == 200 and acc is not None and acc > 0), latest, d


def run_smoke(entry, gpus):
    model = entry["name"]
    cfg = ensure_config(entry)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["VLLM_USE_V1"] = "0"
    out = f"smoke_gemma4_{config_name(model)}.json"
    cmd = [PY, "run.py", "--config", cfg, "--sample_range", "0:7", "--output", out]
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                           capture_output=True, text=True, timeout=2400)
        outp = (r.stdout or "") + (r.stderr or "")
        ok = ("推理失败" not in outp
              and "评测报告已保存至" in outp
              and "推理循环完成" in outp)
        return ok, outp
    except subprocess.TimeoutExpired:
        return False, "SMOKE TIMEOUT"


def run_full(entry, gpus):
    model = entry["name"]
    cfg = ensure_config(entry)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["VLLM_USE_V1"] = "0"
    cards = len(gpus)
    cmd = [PY, "scripts/parallel_model_eval.py", "--config", cfg,
           "--cards", str(cards), "--gpus-per-shard", str(cards), "--sample-range", "0:199"]
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                           capture_output=True, text=True, timeout=36000)
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "FULL TIMEOUT"


def process_model(entry, results):
    model = entry["name"]
    try:
        ok, outp = run_smoke(entry, entry["_gpus"])
        if not ok:
            log(f"[冒烟失败] {model}\n{outp[-1500:]}")
            results[model] = "SMOKE_FAIL"
            return
        log(f"[冒烟通过] {model}")
        outp = run_full(entry, entry["_gpus"])
        ok, rep_dir, _d = verify_report(model)
        if ok:
            log(f"[成功] {model} -> {rep_dir}")
            results[model] = "PASS"
        else:
            log(f"[失败] {model}\n{outp[-1500:]}")
            results[model] = "FAIL"
    except Exception as e:
        log(f"[异常] {model}: {e}")
        results[model] = "EXC"


def main():
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    todo = [e for e in manifest if e.get("status") == "EVALUABLE" and not e.get("report_valid")]
    skipped = [e for e in manifest if e.get("status") != "EVALUABLE" or e.get("report_valid")]
    log(f"=== Gemma4 8卡调度启动: 待评测 {len(todo)}, 跳过 {len(skipped)} ===")
    for e in todo:
        log(f"  待评测 tp={e['tp']} {e['decided_adapter'].split(':')[0]:22s} {e['name']}")

    free_gpus = set(range(NUM_GPUS))
    results = {}
    queue = list(todo)
    active = {}
    lock = threading.Lock()

    def launch(e):
        gpus = list(free_gpus)[:e["tp"]]
        for g in gpus:
            free_gpus.discard(g)
        e["_gpus"] = gpus
        t = threading.Thread(target=process_model, args=(e, results), daemon=True)
        t.start()
        active[e["name"]] = (t, gpus)
        log(f"[启动] {e['name']} tp={e['tp']} gpus={gpus}")

    while queue or active:
        with lock:
            done = [m for m, (t, g) in active.items() if not t.is_alive()]
            for m in done:
                _t, g = active.pop(m)
                for _g in g:
                    free_gpus.add(_g)
                log(f"[释放] {m} -> 空闲 {sorted(free_gpus)}")
        with lock:
            for e in list(queue):
                if len(free_gpus) >= e["tp"]:
                    queue.remove(e)
                    launch(e)
        if queue or active:
            time.sleep(10)

    log("=== 全部模型评测完成 ===")
    for m, s in results.items():
        log(f"  {m}: {s}")


if __name__ == "__main__":
    main()