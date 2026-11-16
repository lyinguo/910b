#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, glob, json

BASE = "/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration"
SUMMARY = "/mnt/afs/comp_error/workspace/reports/eval_summary.csv"
REPORT_ROOT = "/mnt/afs/comp_error/workspace/reports"

def weight_mb(mpath):
    total = 0
    for p in glob.glob(os.path.join(mpath, "*.safetensors")):
        try: total += os.path.getsize(p)
        except: pass
    return total / (1024*1024)

def has_config(mpath):
    return os.path.exists(os.path.join(mpath, "config.json"))

def is_success_report(model_name):
    """检查该模型是否已有成功的200样本报告。报告目录用单下划线分隔 org/name。"""
    cn = model_name.replace('/', '_')
    base = os.path.join(REPORT_ROOT, cn, "image-text-to-text")
    if not os.path.isdir(base): return False
    for d in sorted(os.listdir(base)):
        rep_dir = os.path.join(base, d)
        if not os.path.isdir(rep_dir): continue
        rlog = os.path.join(rep_dir, "run.log")
        rep = os.path.join(rep_dir, "eval_report.json")
        olin = os.path.join(rep_dir, "operator_lineage.json")
        if not (os.path.exists(rlog) and os.path.exists(rep) and os.path.exists(olin)):
            continue
        try:
            if "推理失败" in open(rlog, encoding='utf-8').read():
                continue
            j = json.load(open(rep, encoding='utf-8'))
            total = (j.get('statistics') or {}).get('total_samples')
            acc = (j.get('metrics') or {}).get('Accuracy')
            if total == 200 and acc is not None and acc > 0:
                return True
        except Exception:
            continue
    return False

# 已成功评测集合：CSV passed=True 或 实际报告成功
success = set()
for m in os.listdir(BASE):
    pass
with open(SUMMARY, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if 'Gemma4' in row.get('model_architecture','') and row.get('passed','')=='True':
            success.add(row['model_name'])

models = []
for org in sorted(os.listdir(BASE)):
    orgp = os.path.join(BASE, org)
    if not os.path.isdir(orgp): continue
    for name in sorted(os.listdir(orgp)):
        mp = os.path.join(orgp, name)
        if not os.path.isdir(mp): continue
        full = f"{org}/{name}"
        wmb = weight_mb(mp)
        cfg = has_config(mp)
        rep_success = is_success_report(full)
        models.append({"name": full, "wmb": wmb, "cfg": cfg,
                       "csv_success": full in success, "rep_success": rep_success})

print(f"总模型目录数: {len(models)}", flush=True)
print(f"\n=== 待评测候选 (未成功评测, 有权重>50MB, 有config) ===", flush=True)
pending = 0
for m in models:
    if not (m['csv_success'] or m['rep_success']) and m['cfg'] and m['wmb'] > 50:
        pending += 1
        print(f"  {m['name']:75s} wMB={m['wmb']:10.1f}", flush=True)
print(f"真正待评测数: {pending}", flush=True)

print(f"\n=== 已有成功报告但CSV未标记 (应视为成功, 跳过) ===", flush=True)
for m in models:
    if m['rep_success'] and not m['csv_success'] and m['cfg'] and m['wmb'] > 50:
        print(f"  {m['name']:75s} wMB={m['wmb']:10.1f}", flush=True)