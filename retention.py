import json, glob, os, statistics
from collections import defaultdict

SWEEP_DIR = os.path.expanduser("~/quant-validation/sweep-results")
BASELINE = "Qwen2.5-Coder-1.5B-Instruct"
QUANT_TAGS = ["awq", "gguf", "gptq", "exl2"]

METRIC = {
    "gsm8k": "exact_match,flexible-extract",
    "humaneval_instruct": "pass@1,create_test",
}

def is_baseline(model):
    m = model.lower()
    return BASELINE.lower() in m and not any(t in m for t in QUANT_TAGS)

def label(model_args):
    for tag in QUANT_TAGS + [BASELINE.lower()]:
        if tag in model_args.lower():
            return tag
    return model_args[:40]

scores = defaultdict(list)
for path in glob.glob(os.path.join(SWEEP_DIR, "**", "results_*.json"), recursive=True):
    try:
        j = json.load(open(path))
        model = str(j["config"]["model_args"])
        for task, res in j["results"].items():
            key = METRIC.get(task)
            if key and key in res:
                scores[(model, task)].append(res[key])
    except Exception as e:
        print(f"skip {path}: {e}")

baseline_mean = {}
for (model, task), vals in scores.items():
    if is_baseline(model):
        baseline_mean[task] = statistics.mean(vals)

for task in set(t for _, t in scores):
    if task not in baseline_mean:
        print(f"WARNING: no baseline found for {task}")

print(f"{'model':<10} {'task':<20} {'n':>2} {'mean':>7} {'std':>7} {'retention':>10}")
print("-" * 62)
for (model, task), vals in sorted(scores.items()):
    n = len(vals)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if n > 1 else 0.0
    base = baseline_mean.get(task)
    ret = f"{100*mean/base:.0f}%" if base else "—"
    print(f"{label(model):<10} {task:<20} {n:>2} {mean:>7.3f} {std:>7.3f} {ret:>10}")
