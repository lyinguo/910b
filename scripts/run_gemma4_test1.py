#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma4ForConditionalGeneration —— gemma4_test_1.txt 未评测模型 专用调度器

只处理 gemma4_test_1.txt 中 status==EVALUABLE 且尚无有效报告(report_valid=False)
的模型，在 8 卡 GPU 池上动态并行调度。复用 run_gemma4_all8 的配置生成/冒烟/全量/
报告验证逻辑（这些逻辑此前已验证可用）。

每个模型:
  8 样本冒烟 -> 200 样本全量 -> 报告验证
GPU 池: 空闲 GPU >= tp 即启动；tp=2 的模型并发 4 个，占满 8 卡。
"""
import os
import sys
import time
import json
import threading
from datetime import datetime

PROJECT_ROOT = "/mnt/afs/comp_error"
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_gemma4_all8 import config_name, ensure_config, verify_report  # noqa

MANIFEST = os.path.join(PROJECT_ROOT, "workspace", "gemma4_evaluable.json")
TEST_LIST = "/mnt/afs/yerk/model_test/gemma4_test_1.txt"
LOG = os.path.join(PROJECT_ROOT, "workspace", "reports", "gemma4_test1.log")
REPORT_ROOT = os.path.join(PROJECT_ROOT, "workspace", "reports")
PY = "/mnt/afs/miniconda3/envs/lvzhiqiang/bin/python"
NUM_GPUS = 8
SMOKE_TIMEOUT = 7200
MAX_ATTEMPTS = 5


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_free_gpus():
    try:
        out = subprocess_run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"])
    except Exception:
        return list(range(NUM_GPUS))
    free = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and int(parts[1]) < 500:
            free.append(int(parts[0]))
    return free


def subprocess_run(cmd):
    import subprocess
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout


def run_smoke(entry, gpus):
    model = entry["name"]
    cfg = ensure_config(entry)
    import subprocess
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    out = f"test1_smoke_{config_name(model)}.json"
    cmd = [PY, "run.py", "--config", cfg, "--sample_range", "0:7", "--output", out]
    logf = os.path.join(REPORT_ROOT, f"test1_smoke_stdout_{config_name(model)}.log")
    with open(logf, "w", encoding="utf-8") as fo:
        try:
            r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                               stdout=fo, stderr=subprocess.STDOUT, timeout=SMOKE_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, "SMOKE TIMEOUT"
    try:
        with open(logf, encoding="utf-8") as fo:
            outp = fo.read()
    except Exception:
        outp = ""
    ok = ("推理失败" not in outp
          and "评测报告已保存至" in outp and "推理循环完成" in outp)
    return ok, outp[-800:]


def run_full(entry, gpus):
    import subprocess
    model = entry["name"]
    cfg = ensure_config(entry)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    cards = len(gpus)
    cmd = [PY, "scripts/parallel_model_eval.py", "--config", cfg,
           "--cards", str(cards), "--gpus-per-shard", str(cards), "--sample-range", "0:199"]
    logf = os.path.join(REPORT_ROOT, f"test1_full_stdout_{config_name(model)}.log")
    with open(logf, "w", encoding="utf-8") as fo:
        try:
            r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                               stdout=fo, stderr=subprocess.STDOUT, timeout=90000)
        except subprocess.TimeoutExpired:
            return "FULL TIMEOUT"
    try:
        with open(logf, encoding="utf-8") as fo:
            return fo.read()
    except Exception:
        return ""


def process_model(entry, gpus, results):
    model = entry["name"]
    ok, outp = run_smoke(entry, gpus)
    if not ok:
        log(f"[冒烟失败] {model} gpus={gpus}\n{outp[-400:]}")
        results[model] = "SMOKE_FAIL"
        return
    log(f"[冒烟通过] {model} gpus={gpus}")
    run_full(entry, gpus)
    ok, rep_dir, _d = verify_report(model)
    if ok:
        log(f"[成功] {model} -> {rep_dir}")
        results[model] = "PASS"
    else:
        log(f"[失败] {model} 全量未通过 rep_dir={rep_dir}")
        results[model] = "FAIL"


def main():
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    test1 = [l.strip() for l in open(TEST_LIST) if l.strip()]
    by_name = {e["name"]: e for e in manifest}
    todo = []
    for m in test1:
        e = by_name.get(m)
        if e is None:
            log(f"[跳过/未知] {m} 不在 manifest")
            continue
        if e.get("status") != "EVALUABLE":
            log(f"[跳过] status={e.get('status')} {m} ({e.get('incomplete_reason','')})")
            continue
        ok, _rep, _ = verify_report(m)
        if ok:
            log(f"[跳过/已完成] {m}")
            continue
        todo.append(e)
    log(f"=== gemma4_test_1 调度启动: 待评测 {len(todo)} ===")
    for e in todo:
        log(f"  tp={e['tp']} {e['name']}")

    free_gpus = set(get_free_gpus())
    results = {}
    queue = list(todo)
    active = {}
    lock = threading.Lock()

    def launch(e):
        gpus = sorted(free_gpus)[:e["tp"]]
        for g in gpus:
            free_gpus.discard(g)
        t = threading.Thread(target=process_model, args=(e, gpus, results), daemon=True)
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

    log("=== gemma4_test_1 全部处理完毕 ===")
    for m, s in results.items():
        log(f"  {m}: {s}")


if __name__ == "__main__":
    main()
