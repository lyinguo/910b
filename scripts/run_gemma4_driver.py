#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma4ForConditionalGeneration 剩余待评测模型 —— 并行自动调度驱动

特性:
  - 依据 workspace/gemma4_evaluable.json 权威清单（decided_adapter + tp）
  - 跳过已成功评测(report_valid)与当前正在运行(in-flight)的模型
  - 每个模型 fork 一个子进程独立执行「8样本冒烟 -> 200样本全量 -> 报告验证」
  - 父进程并发把所有空闲 GPU 填满，充分利用 8 卡
  - 优先调度 vLLM 模型(快)，再调度 Transformers 模型(慢)
  - 冒烟超时后清理遗留进程，失败模型重试上限 5 次后放弃
"""
import os
import re
import time
import json
import glob
import signal
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = "/mnt/afs/comp_error"
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs", "Gemma4ForConditionalGeneration")
MMBENCH = "/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl"
REPORT_ROOT = os.path.join(PROJECT_ROOT, "workspace", "reports")
MANIFEST = os.path.join(PROJECT_ROOT, "workspace", "gemma4_evaluable.json")
LOG = os.path.join(REPORT_ROOT, "gemma4_driver.log")
RESULT_DIR = os.path.join(REPORT_ROOT, "_driver_results")
PY = "/mnt/afs/miniconda3/envs/lvzhiqiang/bin/python"
NUM_GPUS = 8
SMOKE_TIMEOUT = 7200   # 2 小时冒烟超时（26B/31B MLX 模型 8 样本较慢）
MAX_ATTEMPTS = 5

os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_ROOT)
from scripts.run_gemma4_all8 import config_name, ensure_config, verify_report  # noqa

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


def get_free_gpus():
    """用 nvidia-smi 返回空闲 GPU 列表（显存使用 < 500 MiB 视为空闲）。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return list(range(NUM_GPUS))
    free = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and int(parts[1]) < 500:
            free.append(int(parts[0]))
    return free


def get_inflight_models():
    """从进程表解析正在运行的 run.py --config，返回模型名集合。"""
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return set()
    inflight = set()
    for line in out.splitlines():
        m = re.search(r"run\.py --config \S+?/([^/\\]+)\.yaml", line)
        if m:
            inflight.add(m.group(1).replace("--", "/"))
    return inflight


def result_file(mn):
    return os.path.join(RESULT_DIR, config_name(mn) + ".txt")


def write_result(mn, code):
    with open(result_file(mn), "w", encoding="utf-8") as f:
        f.write(code)


def read_result(mn):
    p = result_file(mn)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    return None


def kill_config_processes(cfg_name):
    """杀掉针对某 config 的 run.py / vllm 遗留进程（冒烟超时清理）。"""
    pat = re.compile(re.escape(cfg_name))
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return
    for line in out.splitlines():
        if "run.py --config" in line and cfg_name in line:
            try:
                pid = int(line.split()[1])
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


def run_smoke(entry, gpus):
    model = entry["name"]
    cfg = ensure_config(entry)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    out = f"driver_smoke_{config_name(model)}.json"
    cmd = [PY, "run.py", "--config", cfg, "--sample_range", "0:7", "--output", out]
    logf = os.path.join(REPORT_ROOT, f"driver_smoke_stdout_{config_name(model)}.log")
    with open(logf, "w", encoding="utf-8") as fo:
        try:
            r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                               stdout=fo, stderr=subprocess.STDOUT, timeout=SMOKE_TIMEOUT)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            kill_config_processes(config_name(model))
            return False, "SMOKE TIMEOUT"
    outp = ""
    try:
        with open(logf, encoding="utf-8") as fo:
            outp = fo.read()
    except Exception:
        pass
    # 注意: run.py 返回码受 passed(达标)影响，低准确率模型会返回 1，
    # 冒烟测试只验证管线无错误运行，不要求准确率，故不依赖 rc==0。
    ok = ("推理失败" not in outp
          and "评测报告已保存至" in outp and "推理循环完成" in outp)
    return ok, outp[-1200:]


def run_full(entry, gpus):
    model = entry["name"]
    cfg = ensure_config(entry)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    cards = len(gpus)
    cmd = [PY, "scripts/parallel_model_eval.py", "--config", cfg,
           "--cards", str(cards), "--gpus-per-shard", str(cards), "--sample-range", "0:199"]
    logf = os.path.join(REPORT_ROOT, f"driver_full_stdout_{config_name(model)}.log")
    with open(logf, "w", encoding="utf-8") as fo:
        try:
            r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                               stdout=fo, stderr=subprocess.STDOUT, timeout=72000)
        except subprocess.TimeoutExpired:
            kill_config_processes(config_name(model))
            return "FULL TIMEOUT"
    try:
        with open(logf, encoding="utf-8") as fo:
            return fo.read()
    except Exception:
        return ""


def run_one_model(entry, gpus):
    """子进程入口：冒烟 -> 全量 -> 报告验证。返回 PASS/FAIL/其他。"""
    model = entry["name"]
    ok, outp = run_smoke(entry, gpus)
    if not ok:
        log(f"[冒烟失败] {model} gpus={gpus}\n{outp[-500:]}")
        return "SMOKE_FAIL"
    log(f"[冒烟通过] {model} gpus={gpus}")
    run_full(entry, gpus)
    ok, rep_dir, _d = verify_report(model)
    if ok:
        log(f"[成功] {model} -> {rep_dir}")
        return "PASS"
    log(f"[失败] {model} 全量评测未通过，报告目录 {rep_dir}")
    return "FAIL"


def main():
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    inflight = get_inflight_models()
    todo = [e for e in manifest
            if e.get("status") == "EVALUABLE"
            and e["name"] not in inflight
            and not os.path.exists(result_file(e["name"]))
            and not verify_report(e["name"])[0]]
    todo.sort(key=lambda e: (ADAPTER_MAP[e["decided_adapter"]][1] is False, e["name"]))
    log(f"=== Gemma4 并行驱动启动: 待评测 {len(todo)} ===")
    for e in todo:
        vllm = "vLLM" if ADAPTER_MAP[e["decided_adapter"]][1] else "TF "
        log(f"  {vllm} tp={e['tp']} {e['name']}")

    free_gpus = set(get_free_gpus())
    active = {}      # name -> (pid, tp, gpus)
    attempts = {}    # name -> int
    while todo or active:
        # 0) 周期核对实际空闲 GPU，拾取本驱动之外(fcc 外部 in-flight)释放的卡
        busy = set()
        for _pid, _tp, g in active.values():
            busy.update(g)
        free_gpus = set(get_free_gpus()) - busy
        # 1) 收割已完成的子进程
        for mn in list(active):
            pid, _tp, gpus = active[mn]
            try:
                done_pid, _st = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                done_pid = pid
            if done_pid == pid:
                res = read_result(mn)
                free_gpus.update(gpus)
                del active[mn]
                log(f"[完成] {mn} res={res} gpus={gpus} 空闲 {sorted(free_gpus)}")
                if res != "PASS":
                    attempts[mn] = attempts.get(mn, 0) + 1
                    if attempts[mn] >= MAX_ATTEMPTS:
                        log(f"[放弃] {mn} 超过 {MAX_ATTEMPTS} 次尝试")
                    else:
                        # 重新入队重试
                        entry = next((ee for ee in todo if ee["name"] == mn), None)
                        if entry is None:
                            entry = next((ee for ee in manifest if ee["name"] == mn), None)
                        if entry is not None:
                            todo.append(entry)
        # 2) 分派新模型填充空闲 GPU
        for e in list(todo):
            if e["name"] in active:
                continue
            if len(free_gpus) >= e["tp"]:
                gpus = sorted(free_gpus)[:e["tp"]]
                for g in gpus:
                    free_gpus.discard(g)
                write_result(e["name"], "RUNNING")
                pid = os.fork()
                if pid == 0:
                    try:
                        code = run_one_model(e, gpus)
                        write_result(e["name"], code)
                    except Exception as ex:
                        write_result(e["name"], f"EXC:{ex}")
                    os._exit(0)
                active[e["name"]] = (pid, e["tp"], gpus)
                todo.remove(e)
                log(f"[启动] {e['name']} tp={e['tp']} gpus={gpus}")
        if todo or active:
            time.sleep(10)

    log("=== 全部待评测模型处理完毕 ===")


if __name__ == "__main__":
    main()