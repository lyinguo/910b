#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mlx-community/gemma-4-31b-bf16 适配器加载 + 单条推理冒烟测试。"""
import os, sys, json, time
sys.path.insert(0, "/mnt/afs/comp_error")

from universal_eval.model.adapters.gemma4_transformers_mlx import Gemma4TransformersMLXModel

MODEL_PATH = "/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration/mlx-community/gemma-4-31b-bf16"
DATASET = "/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl"

def main():
    model = Gemma4TransformersMLXModel(max_new_tokens=64, gpus=(0, 1), max_memory_per_gpu="36GiB")
    t0 = time.time()
    print(">>> LOADING MODEL...", flush=True)
    model.load(MODEL_PATH)
    print(f">>> MODEL LOADED in {time.time()-t0:.1f}s", flush=True)

    # 读取前 2 个样本
    samples = []
    with open(DATASET, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= 2:
                break
    dataset_dir = os.path.dirname(DATASET)
    for i, s in enumerate(samples):
        prompt = s.get("question") or s.get("prompt") or s.get("text", "")
        rel = s.get("image_path") or s.get("media_path") or ""
        img = rel if os.path.isabs(rel) else os.path.join(dataset_dir, rel)
        print(f">>> SAMPLE {i}: prompt={prompt[:80]!r} img={img}", flush=True)
        out = model.predict(prompt, img)
        print(f">>> OUTPUT {i}: {out!r}", flush=True)
    model.unload()
    print(">>> DONE", flush=True)

if __name__ == "__main__":
    main()
