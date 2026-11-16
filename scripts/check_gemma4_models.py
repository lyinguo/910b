import csv, os
base = "/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration"
csv_path = "/mnt/afs/comp_error/workspace/reports/eval_summary.csv"
done = set()
with open(csv_path, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["model_architecture"] == "Gemma4ForConditionalGeneration":
            done.add(r["model_name"])
allm = []
for org in sorted(os.listdir(base)):
    op = os.path.join(base, org)
    if not os.path.isdir(op): continue
    for m in sorted(os.listdir(op)):
        mp = os.path.join(op, m)
        if os.path.isdir(mp) and ".offload_cache" not in m:
            allm.append((f"{org}/{m}", mp))
for m, mp in allm:
    if m in done: continue
    try:
        files = os.listdir(mp)
    except Exception as e:
        print(f"{m} | ERR {e}"); continue
    has_sf = any(f.endswith(".safetensors") for f in files)
    has_cfg = "config.json" in files
    mlx = any(f.endswith(".scales") or f.endswith(".biases") or f.endswith(".mlx") or f.endswith(".npz") for f in files)
    gguf = any(f.endswith(".gguf") for f in files)
    onnx = any(f.endswith(".onnx") for f in files)
    name = m.split("/")[-1]
    is_base = "it" not in name.lower()
    print(f"{m} | sf={has_sf} cfg={has_cfg} mlx={mlx} gguf={gguf} onnx={onnx} base={is_base}")
