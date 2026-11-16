#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma4 批量评测调度器

自动分类模型（标准vLLM / MLX-transformers / bnb-transformers），生成配置并按
GPU 需求分组并发调度。跳过已成功评测模型。

用法:
    python scripts/gemma4_scheduler.py --batch 1-card
    python scripts/gemma4_scheduler.py --batch 2-card
    python scripts/gemma4_scheduler.py --list
"""

import os, sys, csv, glob, json, time, subprocess, argparse

PROJECT_ROOT = "/mnt/afs/comp_error"
MODEL_BASE = "/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration"
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs", "Gemma4ForConditionalGeneration")
SUMMARY = os.path.join(PROJECT_ROOT, "workspace/reports/eval_summary.csv")
REPORT_ROOT = os.path.join(PROJECT_ROOT, "workspace/reports")
MMBENCH = "/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl"
PY = "/usr/bin/python"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def weight_mb(mpath):
    total = 0
    for p in glob.glob(os.path.join(mpath, "*.safetensors")):
        try: total += os.path.getsize(p)
        except: pass
    return total / (1024*1024)

def is_success_report(model_name):
    cn = model_name.replace('/', '--')
    base = os.path.join(REPORT_ROOT, cn, "image-text-to-text")
    if not os.path.isdir(base): return False
    for d in sorted(os.listdir(base)):
        rep_dir = os.path.join(base, d)
        if not os.path.isdir(rep_dir): continue
        rlog = os.path.join(rep_dir, "run.log")
        rep = os.path.join(rep_dir, "eval_report.json")
        olin = os.path.join(rep_dir, "operator_lineage.json")
        if not (os.path.exists(rlog) and os.path.exists(rep) and os.path.exists(olin)): continue
        try:
            if "推理失败" in open(rlog, encoding='utf-8').read(): continue
            j = json.load(open(rep, encoding='utf-8'))
            total = (j.get('statistics') or {}).get('total_samples')
            acc = (j.get('metrics') or {}).get('Accuracy')
            if total == 200 and acc is not None and acc > 0: return True
        except Exception: continue
    return False

def get_success_set():
    success = set()
    if os.path.exists(SUMMARY):
        with open(SUMMARY, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if 'Gemma4' in row.get('model_architecture','') and row.get('passed','')=='True':
                    success.add(row['model_name'])
    return success

def classify(model_name, mpath):
    """返回 (adapter_key, tp_groups)"""
    wmb = weight_mb(mpath)
    lower = model_name.lower()
    # MLX 量化模型
    if 'mlx' in lower or ('-mlx' in lower) or ('ud-mlx' in lower) or 'supergemma' in lower:
        # 查看 config.json 判断 quantization
        cfgp = os.path.join(mpath, "config.json")
        quant = ""
        if os.path.exists(cfgp):
            try:
                cfg = json.load(open(cfgp))
                quant = str(cfg.get('quantization_config') or '')
                qm = (cfg.get('quantization_config') or {}).get('quant_method','')
                qm = qm or ''
            except Exception: qm = ''
        # MLX 模型按原始精度加载，按权重大小决定卡数
        if wmb > 30000: tp = 2
        elif wmb > 20000: tp = 2
        else: tp = 1
        return "transformers_mlx", tp
    # bnb 4bit
    if 'bnb' in lower or '4bit' in lower and 'unsloth' in lower:
        return "transformers_bnb", 1
    # PARO 特殊
    if 'paro' in lower:
        return "skip_paro", 0
    # GGUF 跳过
    if 'gguf' in lower or 'gguf' in model_name.lower():
        return "skip_gguf", 0
    # ONNX 跳过
    if 'onnx' in lower:
        return "skip_onnx", 0
    # 标准 vLLM 模型
    if wmb > 30000: tp = 2
    elif wmb > 18000: tp = 2
    else: tp = 1
    return "vllm", tp

def build_config(model_name, adapter_key, tp):
    cn = model_name.replace('/', '--')
    mpath = os.path.join(MODEL_BASE, model_name)
    if adapter_key == "vllm":
        adapter = "universal_eval/model/adapters/gemma4_vllm.py:Gemma4VLLMModel"
        kwargs = {"tensor_parallel_size": tp, "max_tokens": 2048,
                  "max_model_len": 4096, "gpu_memory_utilization": 0.9,
                  "trust_remote_code": True}
    else:  # transformers
        adapter = "universal_eval/model/adapters/transformers_vlm.py:TransformersVLM"
        kwargs = {"device": "auto", "max_new_tokens": 2048, "torch_dtype": "auto"}
    cfg = {
        "name": cn,
        "description": f"{model_name} (Gemma4ForConditionalGeneration) 全流程评测管线",
        "task_type": "image-text-to-text",
        "sample_range": "0:199",
        "data": {"preprocess": True, "apply_template": False, "filter_empty": True,
                 "dataset_path": MMBENCH},
        "model": {"name": model_name, "adapter": adapter, "model_path": mpath,
                  "download": False, "kwargs": kwargs},
        "algorithm": {"name": "greedy_inference", "batch_size": 2,
                      "kwargs": {"extract_answer": True}},
        "metrics": {"override": ["Accuracy"]},
        "baseline": {"score": 0.0, "threshold": 0.05},
        "output": {"verbose": True},
    }
    path = os.path.join(CONFIG_DIR, f"{cn}.yaml")
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path

def collect_pending():
    success = get_success_set()
    pending = []
    for org in sorted(os.listdir(MODEL_BASE)):
        orgp = os.path.join(MODEL_BASE, org)
        if not os.path.isdir(orgp) or org.startswith('.'): continue
        for name in sorted(os.listdir(orgp)):
            mp = os.path.join(orgp, name)
            if not os.path.isdir(mp) or name.startswith('.'): continue
            full = f"{org}/{name}"
            wmb = weight_mb(mp)
            if wmb <= 50: continue
            if not os.path.exists(os.path.join(mp, "config.json")): continue
            if full in success or is_success_report(full): continue
            adapter_key, tp = classify(full, mp)
            if adapter_key.startswith("skip"): continue
            pending.append({"name": full, "wmb": wmb, "adapter": adapter_key, "tp": tp})
    return pending

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=["1-card","2-card","list"], default="list")
    ap.add_argument("--model", help="指定单个模型")
    ap.add_argument("--smoke", action="store_true", help="只做冒烟测试")
    args = ap.parse_args()

    pending = collect_pending()
    if args.model:
        pending = [p for p in pending if p['name'] == args.model]
    if args.batch == "list":
        log(f"=== 待评测 {len(pending)} 个 ===")
        for p in pending:
            log(f"  {p['name']:75s} wMB={p['wmb']:8.1f} adapter={p['adapter']} tp={p['tp']}")
        return
    if args.batch == "1-card":
        batch = [p for p in pending if p['tp'] == 1]
    else:
        batch = [p for p in pending if p['tp'] == 2]
    log(f"=== 本批 {len(batch)} 个 (tp={args.batch}) ===")
    for p in batch:
        log(f"  {p['name']} adapter={p['adapter']}")

if __name__ == "__main__":
    main()