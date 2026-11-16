import csv, os
csv_path = "/mnt/afs/comp_error/workspace/reports/eval_summary.csv"
base = "/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration"
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
        if os.path.isdir(os.path.join(op, m)):
            allm.append(f"{org}/{m}")
pending = [m for m in allm if m not in done]
print("=== 已评测成功 (Gemma4ForConditionalGeneration) 数量:", len(done))
print("=== 待评测 ===")
for m in pending:
    print(m)
