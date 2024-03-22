import glob
import json

import pandas as pd


filenames = glob.glob("*summary.json")

results = []

for filename in filenames:
    with open(filename) as f:
        summary = json.load(f)
        metrics = summary["metrics"]
        iterations = metrics["iterations"]["values"]["count"]
        d = {
            "concurrent users": metrics["vus"]["values"]["max"],
            "average input tokens": metrics["input_tokens"]["values"]["count"] / iterations,
            "throughput (t/s)": metrics["tokens"]["values"]["rate"],
            "prefill throughput (t/s)": metrics["input_tokens"]["values"]["rate"],
            "decode throughput (t/s)": metrics["new_tokens"]["values"]["rate"],
            "average latency (ms)": metrics["time_per_token"]["values"]["avg"],
        }
        results.append(pd.DataFrame.from_dict(d, orient="index").transpose())

df = pd.concat(results)
df.to_csv("tgi-results.csv", index=False)
